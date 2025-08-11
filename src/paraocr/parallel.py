# paraOCR/parallel.py
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from collections import defaultdict

import fitz
from tqdm import tqdm


from .config import OCRConfig
from .models import OCRResult, OCRTask
from .utils import is_native_text_good_quality
from .pdf_processor import get_pdf_processor
from .processors import (
    worker_dispatcher,
    worker_render_text_page,
    worker_process_table_page,
    worker_process_image_page,
)
from .gpu_worker import initialize_gpu_worker, process_gpu_batch
from .logger import configure_worker_logging

logger = logging.getLogger("paraocr")

class PerformanceTracker:
    def __init__(self):
        self.cpu_times: Dict[str, float] = defaultdict(float)
        self.gpu_times: Dict[str, float] = defaultdict(float)
        self.gpu_page_counts: Dict[str, int] = defaultdict(int)

    def add_cpu_time(self, source_path: str, duration: float):
        self.cpu_times[source_path] += duration

    def attribute_gpu_batch_time(self, meta_data: List[Dict], batch_duration: float):
        if not meta_data:
            return
        avg_time_per_page = batch_duration / max(1, len(meta_data))
        for meta in meta_data:
            source_path = meta["source_path"]
            self.gpu_times[source_path] += avg_time_per_page
            self.gpu_page_counts[source_path] += 1

    def get_final_metrics(self, source_path: str, total_pages: int, start_time: float) -> Dict:
        total_duration = time.perf_counter() - start_time
        cpu_total = self.cpu_times.get(source_path, 0.0)
        gpu_total = self.gpu_times.get(source_path, 0.0)
        gpu_pages_processed = self.gpu_page_counts.get(source_path, 0)
        return {
            "wall_clock_total_seconds": round(total_duration, 4),
            "cpu_render_total_seconds": round(cpu_total, 4),
            "gpu_ocr_work_seconds": round(gpu_total, 4),
            "gpu_avg_sec_per_page": round(gpu_total / gpu_pages_processed, 4) if gpu_pages_processed > 0 else 0,
        }

def master_worker(task: dict) -> dict:
    """
    Multiplex to the correct worker based on processing_type,
    also attach a duration field.
    """
    task_type = task.get("processing_type")
    start_time = time.perf_counter()

    if task_type == "text_ocr":
        result = worker_render_text_page(task)
    elif task_type == "table":
        result = worker_process_table_page(task)
    elif task_type == "image":
        result = worker_process_image_page(task)
    else:
        task["error"] = f"Unknown processing type received by master_worker, {task_type}"
        result = task

    result["duration_seconds"] = time.perf_counter() - start_time
    return result

# --- MAIN RUNNER CLASS ---
class OCRRunner:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.pdf_processor = get_pdf_processor(config.pdf_engine)

    def _log_error(self, source_path: str, reason: str):
        if not self.config.error_log_path:
            return
        try:
            self.config.error_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.error_log_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "source_path": source_path,
                    "error_reason": reason,
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to write error log")

    def _log_performance(self, metric: Dict):
        if not self.config.log_performance:
            return
        path = self.config.performance_log_path
        if not path:
            path = Path(str(self.config.output_path)).with_suffix(".perf.jsonl")
            self.config.performance_log_path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                log_entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **metric}
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to write performance log")

    def _create_page_tasks(
        self,
        tasks: List[OCRTask],
        outfile,
        perf_tracker: PerformanceTracker,
        file_start_times: Dict[str, float],
    ) -> List[Dict]:
        page_level_tasks: List[Dict] = []
        for task in tqdm(tasks, desc="Scanning source files"):
            file_path = task.source_path
            file_path_str = str(file_path)
            file_start_times[file_path_str] = time.perf_counter()

            if file_path.suffix.lower() == ".pdf":
                scan_start_time = time.perf_counter()
                native_text = self.pdf_processor.get_native_text(file_path)
                scan_duration = time.perf_counter() - scan_start_time
                perf_tracker.add_cpu_time(file_path_str, scan_duration)

                if (
                    native_text
                    and len(native_text) >= self.config.min_native_text_chars
                    and is_native_text_good_quality(
                        native_text, self.config.dictionary, self.config.native_text_quality_threshold
                    )
                ):
                    final_metrics = perf_tracker.get_final_metrics(file_path_str, 1, file_start_times[file_path_str])
                    self._log_performance(
                        {
                            "metric_type": "file_processed",
                            "source_path": file_path_str,
                            "processing_method": "native_text",
                            **final_metrics,
                        }
                    )
                    result_obj = OCRResult(
                        source_path=file_path_str, total_pages=1, content=[{"type": "text", "data": native_text}]
                    )
                    outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                    if self.config.export_txt:
                        self._write_txt_file(file_path, native_text)
                    continue

            try:
                page_count = 1
                if file_path.suffix.lower() == ".pdf":
                    with fitz.open(file_path) as doc:
                        page_count = len(doc)
                if page_count > 0:
                    for i in range(page_count):
                        page_level_tasks.append(
                            {
                                "source_path": file_path_str,
                                "page_num": i,
                                "total_pages": page_count,
                                "dpi": self.config.dpi,
                                "temp_dir": str(self.config.temp_dir),
                            }
                        )
            except Exception as e:
                self._log_error(file_path_str, f"Failed to open or count pages, {e}")

        return page_level_tasks

    # paraOCR/parallel.py, inside the OCRRunner class
    def _build_backend_kwargs(cfg: OCRConfig) -> Dict[str, Any]:
        kw = dict(cfg.ocr_backend_kwargs or {})
        if "languages" not in kw and "lang" not in kw:
            kw["languages"] = cfg.languages
        if "gpu" not in kw and "use_gpu" not in kw:
            kw["gpu"] = True
            kw["use_gpu"] = True
        return kw

    def run(self, tasks: List[OCRTask]):
        logger.info("Run started")
        perf_tracker = PerformanceTracker()
        file_start_times: Dict[str, float] = {}
        ctx = mp.get_context("spawn")

        with ctx.Manager() as manager, open(self.config.output_path, "a", encoding="utf-8") as outfile:
            # STAGE 1: scanning and native text extraction
            logger.progress("scan start", extra={"phase": "scan", "pct": 5})
            page_level_tasks = self._create_page_tasks(tasks, outfile, perf_tracker, file_start_times)
            
            if not page_level_tasks:
                logger.info("No new pages found for OCR processing")
                logger.progress("done", extra={"phase": "done", "pct": 100})
                return

            logger.info("Created %d page level tasks for parallel processing", len(page_level_tasks))
            logger.progress(
                "page tasks created",
                extra={"phase": "dispatch", "total_pages": len(page_level_tasks), "pct": 20},
            )

            # STAGE 2: layout analysis and dispatching
            logger.info("Dispatching tasks for layout analysis")
            logger.progress("dispatch start", extra={"phase": "dispatch", "pct": 10})

            tagged_tasks: List[Dict] = []
            with ctx.Pool(processes=self.config.num_workers, 
                initializer=configure_worker_logging, 
                initargs=(self.config.log_queue,)) as dispatch_pool:
                dispatched = 0
                total = len(page_level_tasks)
                results_iterator = dispatch_pool.imap_unordered(
                    worker_dispatcher, page_level_tasks, chunksize=16
                )
                for result in tqdm(results_iterator, total=len(page_level_tasks), desc="Analyzing Page Layouts"):
                    tagged_tasks.append(result)
                    dispatched += 1
                    logger.progress(
                        "dispatch progress",
                        extra={"phase": "dispatch", "current": dispatched, "total": total}
                    )
                    

            # STAGE 3: routing
            text_render_queue = [t for t in tagged_tasks if t.get("processing_type") == "text_ocr"]
            table_queue = [t for t in tagged_tasks if t.get("processing_type") == "table"]
            image_queue = [t for t in tagged_tasks if t.get("processing_type") == "image"]

            progress_tracker = manager.dict()
            for task in tagged_tasks:
                path_str = task["source_path"]
                if path_str not in progress_tracker:
                    progress_tracker[path_str] = manager.list([None] * task["total_pages"])
                if task.get("processing_type") == "error":
                    self._log_error(path_str, task.get("error", "Dispatcher failed"))
                    progress_tracker[path_str][task["page_num"]] = {
                        "type": "error",
                        "data": task.get("error"),
                    }
            if not text_render_queue: # Assuming only text for now
                logger.info("No pages were identified for OCR processing after dispatching.")
                self._write_final_results(dict(progress_tracker), outfile, file_start_times, perf_tracker)
                return

            # STAGE 4: parallel processing
            logger.info(
                "Starting parallel processing with %s GPU workers",
                self.config.num_gpu_workers,
            )
            logger.progress("render start", extra={"phase": "render", "pct": 30})
            rendered = 0
            total_render = len(text_render_queue)

            # This is the key: two separate, dedicated pools that run concurrently.
            with ctx.Pool(processes=self.config.num_workers, initializer=configure_worker_logging,initargs=(self.config.log_queue,)) as render_pool, \
                 ctx.Pool(processes=self.config.num_gpu_workers, 
                      initializer=initialize_gpu_worker, 
                      initargs=(self.config.log_queue,self.config.ocr_backend, self.config.ocr_backend_kwargs)) as gpu_pool:

                # --- STAGE 4a: Submit CPU Rendering Jobs ---
                # The main process streams rendering tasks to the render_pool.
                render_iterator = render_pool.imap_unordered(worker_render_text_page, text_render_queue, chunksize=16)
                
                async_gpu_results = []
                image_batch_buffer = []; meta_batch_buffer = []

                # This loop consumes results from the render_pool as they become available.
                for result in tqdm(render_iterator, total=len(text_render_queue), desc="Rendering Pages (CPU)"):
                    if result.get("error"):
                        self._log_error(result["source_path"], result["error"]); continue
                    
                    perf_tracker.add_cpu_time(result["source_path"], result.get("duration_seconds", 0.0))
                    
                    image_batch_buffer.append(result["temp_path"])
                    meta_batch_buffer.append(result)
                    rendered += 1
                    # Granular progress during render
                    logger.progress(
                        "render progress",
                        extra={"phase": "render", "current": rendered, "total": total_render}
                    )

                    # When the buffer is full, submit it to the GPU pool asynchronously.
                    if len(image_batch_buffer) >= self.config.gpu_batch_size:
                        job = gpu_pool.apply_async(process_gpu_batch, (image_batch_buffer,))
                        async_gpu_results.append((job, meta_batch_buffer))
                        image_batch_buffer, meta_batch_buffer = [], []

                # Submit the final, partial batch to the GPU pool
                if image_batch_buffer:
                    job = gpu_pool.apply_async(process_gpu_batch, (image_batch_buffer,))
                    async_gpu_results.append((job, meta_batch_buffer))

                # --- STAGE 4b: Asynchronous Aggregation of GPU Results ---
                # While the main process managed the CPU renderers, the GPU pool was working.
                # Now, we wait for all the submitted GPU jobs to complete.
                logger.info("Aggregating GPU results")
                logger.progress("aggregate start", extra={"phase": "aggregate", "pct": 85})
                aggregated = 0
                total_batches = len(async_gpu_results)

                for job, meta_data in tqdm(async_gpu_results, desc="Aggregating Batches"):
                    try:
                        ocr_texts, gpu_duration = job.get()
                    except Exception as e:
                        for meta in meta_data: self._log_error(meta["source_path"], f"GPU batch failed: {e}")
                        ocr_texts, gpu_duration = [""] * len(meta_data), 0.0

                    perf_tracker.attribute_gpu_batch_time(meta_data, gpu_duration)

                    # Update the shared progress tracker with the OCR'd text
                    for i, text in enumerate(ocr_texts):
                        meta = meta_data[i]
                        progress_tracker[meta["source_path"]][meta["page_num"]] = {"type": "text", "data": text}
                    
                    # Cleanup temp files for this batch
                    for meta in meta_data:
                        try: Path(meta["temp_path"]).unlink(missing_ok=True)
                        except Exception as e: self._log_error(meta["source_path"], f"Temp cleanup failed: {e}")

                    aggregated += 1
                    logger.progress(
                        "aggregate progress",
                        extra={"phase": "aggregate", "current": aggregated, "total": total_batches}
                    )

            # STAGE 5: final write
            logger.info("Aggregating and writing final results")
            logger.progress("final write", extra={"phase": "final", "pct": 95})
            self._write_final_results(dict(progress_tracker), outfile, file_start_times, perf_tracker)

            logger.info("Run finished")
            logger.progress("done", extra={"phase": "done", "pct": 100})

    def _write_final_results(self, progress_tracker, outfile, file_start_times, perf_tracker):
        for path_str, pages in progress_tracker.items():
            if not all(p is not None for p in pages):
                self._log_error(path_str, "Processing did not complete for all pages")

            result_obj = OCRResult(source_path=path_str, total_pages=len(pages), content=list(pages))
            outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")

            if all(p is not None for p in pages):
                final_metrics = perf_tracker.get_final_metrics(
                    path_str, len(pages), file_start_times.get(path_str, 0.0)
                )
                self._log_performance(
                    {
                        "metric_type": "file_processed",
                        "source_path": path_str,
                        "processing_method": "ocr_parallel",
                        "total_pages": len(pages),
                        **final_metrics,
                    }
                )

                if self.config.export_txt:
                    full_text = "\n\n--- PAGE BREAK ---\n\n".join(
                        [p.get("data", "") for p in pages if p and p.get("type") == "text"]
                    )
                    if full_text:
                        self._write_txt_file(Path(path_str), full_text)

    def _write_txt_file(self, source_path: Path, text: str):
        try:
            txt_filename = source_path.stem + ".txt"
            output_path = source_path.parent / txt_filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            self._log_error(str(source_path), f"Failed to write discrete txt file, {e}")