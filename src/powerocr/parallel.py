# powerocr/parallel.py
import json
import gc
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from multiprocessing import Pool
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

class PerformanceTracker:
    """A stateful class to track performance metrics for each file."""
    def __init__(self):
        # Stores total CPU time spent rendering pages for each file
        self.cpu_times: Dict[str, float] = defaultdict(float)
        # Stores total GPU time attributed to pages for each file
        self.gpu_times: Dict[str, float] = defaultdict(float)

    def add_cpu_time(self, source_path: str, duration: float):
        self.cpu_times[source_path] += duration

    def attribute_gpu_batch_time(self, meta_data: List[Dict], batch_duration: float):
        """Attributes the GPU batch time back to the individual files in the batch."""
        if not meta_data: return
        # Calculate the average GPU time per page in this specific batch
        avg_time_per_page = batch_duration / len(meta_data)
        for meta in meta_data:
            self.gpu_times[meta['source_path']] += avg_time_per_page

    def get_final_metrics(self, source_path: str, total_pages: int, start_time: float) -> Dict:
        """Aggregates and calculates the final metrics for a completed file."""
        total_duration = time.perf_counter() - start_time
        cpu_total = self.cpu_times.get(source_path, 0)
        gpu_total = self.gpu_times.get(source_path, 0)
        return {
            "total_duration_seconds": round(total_duration, 4),
            "cpu_render_total_seconds": round(cpu_total, 4),
            "cpu_render_avg_sec_per_page": round(cpu_total / total_pages, 4) if total_pages > 0 else 0,
            "gpu_ocr_total_seconds": round(gpu_total, 4),
            "gpu_ocr_avg_sec_per_page": round(gpu_total / total_pages, 4) if total_pages > 0 else 0,
        }


# --- OCREngine Class (Correct) ---
class OCREngine:
    def __init__(self, languages: List[str], gpu: bool = True, beamsearch: bool = False):
        print("--- Initializing OCR Engine ---")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        if beamsearch:
            self.reader.beamsearch = True
        self.device = 'CUDA' if gpu and torch.cuda.is_available() else 'CPU'
        self.has_batch_support = hasattr(self.reader, 'readtext_batch')
        print(f"--- EasyOCR Engine on {self.device} (Beam Search: {beamsearch}, Batch Support: {self.has_batch_support}) ---")

    def process_images_in_batch(self, images: List[Image.Image]) -> List[str]:
        if not images: return []
        try:
            if self.has_batch_support:
                results = self.reader.readtext_batch([np.array(img) for img in images], detail=0, paragraph=True)
                return ["\n".join(res) for res in results]
            else:
                return ["\n".join(self.reader.readtext(np.array(img), detail=0, paragraph=True)) for img in images]
        except Exception as e:
            print(f"\n  [GPU Error] OCR batch processing failed: {e}")
            return [""] * len(images)

# --- Worker Function (Correct) ---
def worker_process_file(task_with_config: tuple) -> Dict:
    start_time = time.perf_counter()
    task, config_dict = task_with_config
    config = OCRConfig.from_dict(config_dict)
    file_path = task.source_path
    dictionary = config.dictionary
    try:
        if file_path.suffix.lower() == '.pdf':
            with fitz.open(file_path) as doc:
                native_text = "\n".join(page.get_text("text") for page in doc).strip()
                if len(native_text) >= config.min_native_text_chars and is_native_text_good_quality(native_text, dictionary, config.native_text_quality_threshold):
                    duration = time.perf_counter() - start_time
                    return {"source_path": str(file_path), "type": "native_text", "text": native_text, "error": None, "duration_seconds": duration}
                page_count = len(doc)
                if not page_count:
                    duration = time.perf_counter() - start_time
                    return {"source_path": str(file_path), "type": "error", "error": "PDF has zero pages", "duration_seconds": duration}
                temp_paths = []
                for page in doc:
                    pix = page.get_pixmap(dpi=config.dpi)
                    temp_img_path = config.temp_dir / f"{uuid.uuid4()}.png"
                    pix.save(temp_img_path)
                    temp_paths.append(str(temp_img_path))
                duration = time.perf_counter() - start_time
                return {"source_path": str(file_path), "type": "ocr", "page_count": page_count, "temp_paths": temp_paths, "error": None, "duration_seconds": duration}
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            img = Image.open(file_path)
            temp_img_path = config.temp_dir / f"{uuid.uuid4()}.png"
            img.save(temp_img_path, format='PNG')
            duration = time.perf_counter() - start_time
            return {"source_path": str(file_path), "type": "ocr", "page_count": 1, "temp_paths": [str(temp_img_path)], "error": None, "duration_seconds": duration}
        else:
            duration = time.perf_counter() - start_time
            return {"source_path": str(file_path), "type": "error", "error": f"Unsupported file type: {file_path.suffix}", "duration_seconds": duration}
    except Exception as e:
        duration = time.perf_counter() - start_time
        return {"source_path": str(file_path), "type": "error", "error": f"Failed during rendering/reading: {str(e)}", "duration_seconds": duration}


# --- MAIN RUNNER CLASS ---
class OCRRunner:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.engine = OCREngine(config.languages, gpu=torch.cuda.is_available(), beamsearch=config.beamsearch)

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

    def run(self, tasks: list[OCRTask]):
        image_batch_buffer = []
        meta_batch_buffer = []
        ocr_progress_tracker: Dict[str, List[Optional[str]]] = {}
        perf_tracker = PerformanceTracker()
        file_start_times: Dict[str, float] = {}

        with open(self.config.output_path, 'a', encoding='utf-8') as outfile, \
             Pool(processes=self.config.num_workers) as pool:

            config_dict = self.config.to_dict()
            tasks_with_config = [(task, config_dict) for task in tasks]
            results_iterator = pool.imap_unordered(worker_process_file, tasks_with_config)
            
            for result_dict in tqdm(results_iterator, total=len(tasks), desc="Processing Files (CPU)"):
                if not result_dict: continue
                source_path_str = result_dict.get("source_path", "Unknown")
                if source_path_str not in file_start_times:
                    file_start_times[source_path_str] = time.perf_counter()
                perf_tracker.add_cpu_time(source_path_str, result_dict.get("duration_seconds", 0))

                result_type = result_dict.get("type", "error")
                error_msg = result_dict.get("error")

                if result_type == "error":
                    self._log_error(source_path_str, error_msg); continue
                
                # FIX #1: Handle native text results immediately
                if result_type == "native_text":
                    text = result_dict["text"]
                    result_obj = OCRResult(source_path=source_path_str, text=text, method="native_text_good_quality")
                    outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n"); outfile.flush()

                    final_metrics = perf_tracker.get_final_metrics(source_path_str, 1, file_start_times[source_path_str])
                    self._log_performance({
                        "metric_type": "file_processed",
                        "source_path": source_path_str,
                        "processing_method": "native_text",
                        **final_metrics
                    })

                    if self.config.export_txt and text:
                        self._write_txt_file(Path(source_path_str), text)
                    continue

                # FIX #2: Add OCR pages to the batch buffer
                if result_type == "ocr":
                    page_count = result_dict["page_count"]
                    if source_path_str not in ocr_progress_tracker:
                        ocr_progress_tracker[source_path_str] = [None] * page_count
                    for page_num, temp_path in enumerate(result_dict["temp_paths"]):
                        image_batch_buffer.append(temp_path)
                        meta_batch_buffer.append({"source_path": source_path_str, "page_num": page_num})
                
                # Process the GPU batch when full
                if len(image_batch_buffer) >= self.config.gpu_batch_size:
                    self._process_and_write_batch(image_batch_buffer, meta_batch_buffer, ocr_progress_tracker, outfile, perf_tracker, file_start_times)
                    image_batch_buffer.clear(); meta_batch_buffer.clear(); gc.collect()


            # Process the final, potentially incomplete batch
            if image_batch_buffer:
                self._process_and_write_batch(image_batch_buffer, meta_batch_buffer, ocr_progress_tracker, outfile, perf_tracker, file_start_times)

    def _process_and_write_batch(self, image_paths: List[str], meta_data: List[Dict], progress_tracker: Dict, outfile, perf_tracker: PerformanceTracker, file_start_times: Dict):
        if not image_paths: return

        images_to_process = [Image.open(p) for p in image_paths]
        
        start_time = time.perf_counter()
        ocr_results = self.engine.process_images_in_batch(images_to_process)
        duration = time.perf_counter() - start_time
        
        # --- NEW: Attribute GPU time back to the files ---
        perf_tracker.attribute_gpu_batch_time(meta_data, duration)

        for p in image_paths:
            try: Path(p).unlink()
            except OSError: pass

        for idx, text in enumerate(ocr_results):
            meta = meta_data[idx]
            source_path = meta['source_path']
            page_num = meta['page_num']
            if source_path in progress_tracker:
                progress_tracker[source_path][page_num] = text
        
        completed_paths = []
        for path_str, pages in list(progress_tracker.items()):
            if all(p is not None for p in pages):
                full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
                result_obj = OCRResult(source_path=path_str, text=full_text, method=f"easyocr_{self.engine.device.lower()}_parallel")
                outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                outfile.flush()
                
                # --- NEW: Log final performance for OCR'd files ---
                final_metrics = perf_tracker.get_final_metrics(path_str, len(pages), file_start_times[path_str])
                self._log_performance({
                    "metric_type": "file_processed",
                    "source_path": path_str,
                    "processing_method": "ocr_parallel",
                    "total_pages": len(pages),
                    **final_metrics
                })

                if self.config.export_txt and full_text:
                    self._write_txt_file(Path(path_str), full_text)
                
                completed_paths.append(path_str)
        
        for path_str in completed_paths:
            del progress_tracker[path_str]

    def _write_txt_file(self, source_path: Path, text: str):
        try:
            txt_filename = source_path.stem + ".txt"
            output_path = source_path.parent / txt_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            error_reason = f"Failed to write discrete .txt file: {e}"
            print(f"\n[Warning] {error_reason} for {source_path}")
            self._log_error(str(source_path), error_reason)