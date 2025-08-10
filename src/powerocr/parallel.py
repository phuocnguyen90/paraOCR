# powerocr/parallel.py

import json
import gc
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from multiprocessing import Pool, Manager, cpu_count
from collections import defaultdict

import fitz
from PIL import Image
import numpy as np
import easyocr
import torch
from tqdm import tqdm

from .config import OCRConfig
from .models import OCRResult, OCRTask
from .utils import is_native_text_good_quality
from .pdf_processor import get_pdf_processor
from .processors import worker_dispatcher, worker_render_text_page, worker_process_table_page, worker_process_image_page
from .gpu_worker import initialize_gpu_worker, process_gpu_batch

class PerformanceTracker:
    def __init__(self):
        self.cpu_times: Dict[str, float] = defaultdict(float)
        self.gpu_times: Dict[str, float] = defaultdict(float)
    def add_cpu_time(self, source_path: str, duration: float):
        self.cpu_times[source_path] += duration
    def attribute_gpu_batch_time(self, meta_data: List[Dict], batch_duration: float):
        if not meta_data: return
        avg_time_per_page = batch_duration / len(meta_data)
        for meta in meta_data: self.gpu_times[meta['source_path']] += avg_time_per_page
    def get_final_metrics(self, source_path: str, total_pages: int, start_time: float) -> Dict:
        total_duration = time.perf_counter() - start_time
        cpu_total = self.cpu_times.get(source_path, 0)
        gpu_total = self.gpu_times.get(source_path, 0)
        return {"total_duration_seconds": round(total_duration, 4), "cpu_render_total_seconds": round(cpu_total, 4), "cpu_render_avg_sec_per_page": round(cpu_total / total_pages, 4) if total_pages > 0 else 0, "gpu_ocr_total_seconds": round(gpu_total, 4), "gpu_ocr_avg_sec_per_page": round(gpu_total / total_pages, 4) if total_pages > 0 else 0}

class OCREngine:
    def __init__(self, languages: List[str], gpu: bool = True, beamsearch: bool = False):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        if beamsearch: self.reader.beamsearch = True
        self.device = 'CUDA' if gpu and torch.cuda.is_available() else 'CPU'

        print(f"--- EasyOCR Engine on {self.device} (Beam Search: {beamsearch}, Batch Support: {self.has_batch_support}) ---")
    def process_images_in_batch(self, images: List[Image.Image]) -> List[str]:
        if not images: return []
        try:
            if self.has_batch_support:
                results = self.reader.readtext([np.array(img) for img in images], detail=0, paragraph=True)
                return ["\n".join(res) for res in results]
            else:
                return ["\n".join(self.reader.readtext(np.array(img), detail=0, paragraph=True)) for img in images]
        except Exception as e:
            print(f"\n  [GPU Error] OCR batch processing failed: {e}")
            return [""] * len(images)

def master_worker(task: Dict) -> Dict:
    task_type = task.get("processing_type")
    start_time = time.perf_counter()
    if task_type == "text_ocr": result = worker_render_text_page(task)
    elif task_type == "table": result = worker_process_table_page(task)
    elif task_type == "image": result = worker_process_image_page(task)
    else: task["error"] = f"Unknown processing type: {task_type}"; result = task
    result["duration_seconds"] = time.perf_counter() - start_time
    return result

# --- MAIN RUNNER CLASS ---
class OCRRunner:
    def __init__(self, config: OCRConfig):
        self.config = config
        # The runner no longer has its own engine. Engines live in the GPU workers.
        self.pdf_processor = get_pdf_processor(config.pdf_engine)

    def _log_error(self, source_path: str, reason: str):
        if not self.config.error_log_path: return
        with open(self.config.error_log_path, 'a', encoding='utf-8') as f:
            log_entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source_path": source_path, "error_reason": reason}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _log_performance(self, metric: Dict):
        if not self.config.log_performance: return
        with open(self.config.performance_log_path, 'a', encoding='utf-8') as f:
            log_entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **metric}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _create_page_tasks(self, tasks: List[OCRTask], outfile, perf_tracker: PerformanceTracker, file_start_times: Dict) -> List[Dict]:
        page_level_tasks = []
        for task in tqdm(tasks, desc="Scanning source files"):
            file_path = task.source_path
            file_path_str = str(file_path)
            file_start_times[file_path_str] = time.perf_counter()
            if file_path.suffix.lower() == '.pdf':
                scan_start_time = time.perf_counter()
                native_text = self.pdf_processor.get_native_text(file_path)
                scan_duration = time.perf_counter() - scan_start_time
                perf_tracker.add_cpu_time(file_path_str, scan_duration)
                if native_text and len(native_text) >= self.config.min_native_text_chars and \
                   is_native_text_good_quality(native_text, self.config.dictionary, self.config.native_text_quality_threshold):
                    final_metrics = perf_tracker.get_final_metrics(file_path_str, 1, file_start_times[file_path_str])
                    self._log_performance({"metric_type": "file_processed", "source_path": file_path_str, "processing_method": "native_text", **final_metrics})
                    result_obj = OCRResult(source_path=file_path_str, total_pages=1, content=[{"type": "text", "data": native_text}])
                    outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                    if self.config.export_txt: self._write_txt_file(file_path, native_text)
                    continue
            try:
                page_count = 1
                if file_path.suffix.lower() == '.pdf':
                    with fitz.open(file_path) as doc: page_count = len(doc)
                if page_count > 0:
                    for i in range(page_count):
                        page_level_tasks.append({"source_path": file_path_str, "page_num": i, "total_pages": page_count, "dpi": self.config.dpi, "temp_dir": str(self.config.temp_dir)})
            except Exception as e:
                self._log_error(file_path_str, f"Failed to open/count pages: {e}")
        return page_level_tasks

    def run(self, tasks: List[OCRTask]):
        perf_tracker = PerformanceTracker()
        file_start_times: Dict[str, float] = {}

        # The Manager allows us to have a shared dictionary for progress tracking across processes
        with Manager() as manager, open(self.config.output_path, 'a', encoding='utf-8') as outfile:
            
            # --- STAGE 1: SCANNING & NATIVE TEXT EXTRACTION ---
            # This stage happens in the main process. It's fast.
            # It filters out easy cases and creates a definitive list of pages needing heavy processing.
            page_level_tasks = self._create_page_tasks(tasks, outfile, perf_tracker, file_start_times)
            if not page_level_tasks:
                print("No new pages found for OCR processing."); return
            print(f"Created {len(page_level_tasks)} page-level tasks for parallel processing.")

            # --- STAGE 2: LAYOUT ANALYSIS & DISPATCHING ---
            # A pool of CPU workers analyzes each page layout and tags it with a processing type.
            print("\n--- Dispatching tasks for layout analysis ---")
            tagged_tasks = []
            with Pool(processes=self.config.num_workers) as pool:
                results_iterator = pool.imap_unordered(worker_dispatcher, page_level_tasks)
                for result in tqdm(results_iterator, total=len(page_level_tasks), desc="Analyzing Page Layouts"):
                    tagged_tasks.append(result)

            # --- STAGE 3: ROUTING & PROGRESS TRACKER SETUP ---
            # We sort the tagged tasks into different queues for different worker types.
            text_render_queue = [t for t in tagged_tasks if t.get("processing_type") == "text_ocr"]
            # (Future queues for tables/images would be created here)

            # The progress tracker is the master dictionary for aggregating final results.
            progress_tracker = manager.dict()
            for task in tagged_tasks:
                path_str = task["source_path"]
                if path_str not in progress_tracker:
                    progress_tracker[path_str] = manager.list([None] * task["total_pages"])
                if task.get("processing_type") == "error":
                    self._log_error(path_str, task.get("error", "Dispatcher failed"))
                    progress_tracker[path_str][task["page_num"]] = {"type": "error", "data": task.get("error")}

            # --- STAGE 4: PARALLEL PROCESSING (CPU RENDERING + MULTIPLE GPU WORKERS) ---
            print(f"\n--- Starting parallel OCR with {self.config.num_gpu_workers} GPU workers ---")
            
            # This outer pool is for CPU-bound rendering tasks.
            with Pool(processes=self.config.num_workers) as render_pool:
                # This inner pool is for GPU-bound OCR tasks.
                # The `initializer` function is the key: it creates one OCREngine per worker.
                with Pool(processes=self.config.num_gpu_workers, 
                          initializer=initialize_gpu_worker, 
                          initargs=(self.config.languages, self.config.beamsearch)) as gpu_pool:

                    render_iterator = render_pool.imap_unordered(worker_render_text_page, text_render_queue)
                    
                    async_gpu_results = []
                    image_batch_buffer = []
                    meta_batch_buffer = []

                    for result in tqdm(render_iterator, total=len(text_render_queue), desc="Rendering Pages (CPU)"):
                        if result.get("error"):
                            self._log_error(result["source_path"], result["error"]); continue
                        
                        perf_tracker.add_cpu_time(result["source_path"], result.get("duration_seconds", 0))
                        
                        image_batch_buffer.append(result["temp_path"])
                        meta_batch_buffer.append(result)

                        if len(image_batch_buffer) >= self.config.gpu_batch_size:
                            job = gpu_pool.apply_async(process_gpu_batch, (image_batch_buffer,))
                            async_gpu_results.append((job, meta_batch_buffer))
                            image_batch_buffer, meta_batch_buffer = [], []

                    if image_batch_buffer: # Submit the final, partial batch
                        job = gpu_pool.apply_async(process_gpu_batch, (image_batch_buffer,))
                        async_gpu_results.append((job, meta_batch_buffer))

                    # --- STAGE 5: ASYNCHRONOUS AGGREGATION ---
                    # We wait for the GPU results to come back and aggregate them.
                    print("\n--- Aggregating GPU results ---")
                    for job, meta_data in tqdm(async_gpu_results, desc="Aggregating Batches"):
                        start_time = time.perf_counter()
                        ocr_texts = job.get() # This blocks until this specific batch is done
                        duration = time.perf_counter() - start_time
                        perf_tracker.attribute_gpu_batch_time(meta_data, duration)

                        # Update the shared progress tracker with the OCR'd text
                        for i, text in enumerate(ocr_texts):
                            meta = meta_data[i]
                            progress_tracker[meta["source_path"]][meta["page_num"]] = {"type": "text", "data": text}
                        
                        # Clean up temp files for this batch
                        for meta in meta_data: Path(meta["temp_path"]).unlink()

            # --- STAGE 6: FINAL WRITE ---
            # Convert the Manager.dict back to a regular dict for the final write.
            self._write_final_results(dict(progress_tracker), outfile, file_start_times, perf_tracker)

    def _write_final_results(self, progress_tracker, outfile, file_start_times, perf_tracker):
        print("\n--- Aggregating and writing final results ---")
        for path_str, pages in progress_tracker.items():
            if not all(p is not None for p in pages):
                self._log_error(path_str, "Processing did not complete for all pages.")
            
            result_obj = OCRResult(source_path=path_str, total_pages=len(pages), content=list(pages))
            outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
            
            if all(p is not None for p in pages):
                final_metrics = perf_tracker.get_final_metrics(path_str, len(pages), file_start_times.get(path_str, 0))
                self._log_performance({"metric_type": "file_processed", "source_path": path_str, "processing_method": "ocr_parallel", "total_pages": len(pages), **final_metrics})
                
                if self.config.export_txt:
                    full_text = "\n\n--- PAGE BREAK ---\n\n".join([p.get('data', '') for p in pages if p and p.get('type') == 'text'])
                    if full_text: self._write_txt_file(Path(path_str), full_text)

    def _write_txt_file(self, source_path: Path, text: str):
        try:
            txt_filename = source_path.stem + ".txt"
            output_path = source_path.parent / txt_filename
            with open(output_path, 'w', encoding='utf-8') as f: f.write(text)
        except Exception as e:
            self._log_error(str(source_path), f"Failed to write discrete .txt file: {e}")