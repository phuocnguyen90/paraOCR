# paraOCR/parallel.py
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from collections import defaultdict
import hashlib
from collections import deque

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

def _cache_key(source_path: str, page_num: int, dpi: int, pdf_engine: str, version: str="v1") -> str:
    h = hashlib.sha1()
    h.update(f"{source_path}|{page_num}|{dpi}|{pdf_engine}|{version}".encode("utf-8"))
    return h.hexdigest()

def _render_png_path(temp_dir: Path, key: str) -> Path:
    return temp_dir / "cache" / "render" / key[:2] / f"{key}.png"

def _page_txt_path(temp_dir: Path, key: str) -> Path:
    return temp_dir / "cache" / "pages" / key[:2] / f"{key}.txt"

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
# --- NEW at top of parallel.py (with your other imports) ---
import os, json, hashlib
from dataclasses import asdict

def _file_cache_key(source_path: str, dpi: int, pdf_engine: str, version: str = "v1") -> str:
    h = hashlib.sha1()
    h.update(f"{source_path}|{dpi}|{pdf_engine}|{version}".encode("utf-8"))
    return h.hexdigest()

def _manifest_path(temp_dir: Path, key: str) -> Path:
    return temp_dir / "cache" / "manifest" / key[:2] / f"{key}.json"

def _file_signature(p: Path) -> dict:
    try:
        st = p.stat()
        return {"size": st.st_size, "mtime": int(st.st_mtime)}
    except Exception:
        return {"size": -1, "mtime": -1}

def _config_fingerprint(dpi: int, pdf_engine: str, cache_version: str) -> dict:
    return {"dpi": int(dpi), "pdf_engine": str(pdf_engine), "cache_version": str(cache_version)}


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
    @staticmethod
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
        completed_files = set()

        use_cache = getattr(self.config, "use_cache", True)
        keep_render_cache = getattr(self.config, "keep_render_cache", True)
        cache_version = getattr(self.config, "cache_version", "v1")

        with ctx.Manager() as manager, open(self.config.output_path, "a", encoding="utf-8") as outfile:
            # ---------------- STAGE 1: build work lists ----------------
            logger.progress("scan start", extra={"phase": "scan", "pct": 5})

            # Two buckets:
            #   - tagged_tasks_ready: pages that already have routing (from manifest) -> go straight to stage 3/4
            #   - page_level_tasks_to_dispatch: pages that still need layout analysis
            tagged_tasks_ready: List[Dict] = []
            page_level_tasks_to_dispatch: List[Dict] = []

            cfg_fp = _config_fingerprint(self.config.dpi, self.config.pdf_engine, cache_version)

            for task in tqdm(tasks, desc="Scanning source files", disable=False):
                file_path = task.source_path
                file_path_str = str(file_path)
                file_start_times[file_path_str] = time.perf_counter()

                # Try manifest first
                key = _file_cache_key(file_path_str, self.config.dpi, self.config.pdf_engine, cache_version)
                manifest_p = _manifest_path(self.config.temp_dir, key)
                sig_now = _file_signature(file_path)

                manifest_ok = False
                manifest = None
                if use_cache and manifest_p.exists():
                    try:
                        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
                        if (
                            manifest.get("source_path") == file_path_str
                            and manifest.get("config_fp") == cfg_fp
                            and manifest.get("signature") == sig_now
                        ):
                            manifest_ok = True
                    except Exception:
                        manifest_ok = False

                if manifest_ok:
                    # Shortcut 1: native text already finalized earlier
                    if manifest.get("processing_method") == "native_text":
                        result_obj = OCRResult(
                            source_path=file_path_str, total_pages=1,
                            content=[{"type": "text", "data": manifest.get("native_text", "")}]
                        )
                        outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                        if self.config.export_txt:
                            self._write_txt_file(file_path, manifest.get("native_text", ""))
                        # perf log
                        final_metrics = perf_tracker.get_final_metrics(file_path_str, 1, file_start_times[file_path_str])
                        self._log_performance({
                            "metric_type": "file_processed",
                            "source_path": file_path_str,
                            "processing_method": "native_text",
                            **final_metrics,
                        })
                        continue

                    # Shortcut 2: reuse page routing (skip dispatch)
                    for pg in manifest.get("pages", []):
                        tagged_tasks_ready.append({
                            "source_path": file_path_str,
                            "page_num": int(pg["page_num"]),
                            "total_pages": int(manifest.get("total_pages", 1)),
                            "dpi": int(cfg_fp["dpi"]),
                            "temp_dir": str(self.config.temp_dir),
                            "processing_type": pg["processing_type"],
                        })
                    continue

                # Fall back to original logic (no valid manifest)
                # (a) optional native text fast path
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
                        # write manifest as "native_text"
                        if use_cache:
                            manifest = {
                                "source_path": file_path_str,
                                "signature": sig_now,
                                "config_fp": cfg_fp,
                                "processing_method": "native_text",
                                "native_text": native_text,
                            }
                            manifest_p.parent.mkdir(parents=True, exist_ok=True)
                            manifest_p.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

                        final_metrics = perf_tracker.get_final_metrics(file_path_str, 1, file_start_times[file_path_str])
                        self._log_performance({
                            "metric_type": "file_processed",
                            "source_path": file_path_str,
                            "processing_method": "native_text",
                            **final_metrics,
                        })
                        result_obj = OCRResult(
                            source_path=file_path_str, total_pages=1,
                            content=[{"type": "text", "data": native_text}]
                        )
                        outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                        if self.config.export_txt:
                            self._write_txt_file(file_path, native_text)
                        continue

                # (b) page enumeration (lightweight)
                try:
                    page_count = 1
                    if file_path.suffix.lower() == ".pdf":
                        with fitz.open(file_path) as doc:
                            page_count = len(doc)
                    if page_count > 0:
                        for i in range(page_count):
                            page_level_tasks_to_dispatch.append({
                                "source_path": file_path_str,
                                "page_num": i,
                                "total_pages": page_count,
                                "dpi": self.config.dpi,
                                "temp_dir": str(self.config.temp_dir),
                            })
                except Exception as e:
                    self._log_error(file_path_str, f"Failed to open or count pages, {e}")

            # STAGE 2: layout analysis and dispatching
            tagged_tasks: List[Dict] = []
            if page_level_tasks_to_dispatch:
                logger.info("Dispatching %d pages for layout analysis", len(page_level_tasks_to_dispatch))
                logger.progress("dispatch start", extra={"phase": "dispatch", "pct": 10})

                with ctx.Pool(processes=self.config.num_workers,
                            initializer=configure_worker_logging,
                            initargs=(self.config.log_queue,)) as dispatch_pool:
                    dispatched = 0
                    total = len(page_level_tasks_to_dispatch)
                    results_iterator = dispatch_pool.imap_unordered(
                        worker_dispatcher, page_level_tasks_to_dispatch, chunksize=16
                    )
                    for result in tqdm(results_iterator, total=total, desc="Analyzing Page Layouts"):
                        tagged_tasks.append(result)
                        dispatched += 1
                        logger.progress(
                            "dispatch progress",
                            extra={"phase": "dispatch", "current": dispatched, "total": total}
                        )

                # Write/refresh manifests for files covered by dispatch
                if use_cache and tagged_tasks:
                    pages_by_file: Dict[str, dict] = {}
                    for t in tagged_tasks:
                        sp = t["source_path"]
                        d = pages_by_file.setdefault(sp, {
                            "source_path": sp,
                            "signature": _file_signature(Path(sp)),
                            "config_fp": cfg_fp,
                            "processing_method": "ocr_parallel",
                            "total_pages": t["total_pages"],
                            "pages": [],
                        })
                        d["pages"].append({"page_num": t["page_num"], "processing_type": t.get("processing_type", "text_ocr")})
                    for sp, mani in pages_by_file.items():
                        key = _file_cache_key(sp, self.config.dpi, self.config.pdf_engine, cache_version)
                        mpth = _manifest_path(self.config.temp_dir, key)
                        mpth.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            mpth.write_text(json.dumps(mani, ensure_ascii=False), encoding="utf-8")
                        except Exception:
                            logger.exception("Failed to write manifest for %s", sp)

            else:
                logger.info("Skipped layout analysis: all pages covered by cached manifests")

            # Merge cached-tagged pages + freshly dispatched pages
            if tagged_tasks:
                tagged_tasks_ready.extend(tagged_tasks)

            if not tagged_tasks_ready:
                logger.info("No pages were identified for OCR processing after dispatch/cache.")
                logger.progress("done", extra={"phase": "done", "pct": 100})
                return
                    

            # STAGE 3: routing


            progress_tracker = manager.dict()
            for task in tagged_tasks_ready:
                path_str = task["source_path"]
                if path_str not in progress_tracker:
                    progress_tracker[path_str] = manager.list([None] * task["total_pages"])
                if task.get("processing_type") == "error":
                    self._log_error(path_str, task.get("error", "Dispatcher failed"))
                    progress_tracker[path_str][task["page_num"]] = {
                        "type": "error",
                        "data": task.get("error"),
                    }
                # --- cache-aware filtering before Stage 4 ---
            text_render_queue = [t for t in tagged_tasks_ready if t.get("processing_type") == "text_ocr"]
            table_queue = [t for t in tagged_tasks_ready if t.get("processing_type") == "table"]
            image_queue = [t for t in tagged_tasks_ready if t.get("processing_type") == "image"]

            new_text_render_queue = []
            cached_text_pages = 0
            for t in text_render_queue:
                key = _cache_key(t["source_path"], t["page_num"], t["dpi"], self.config.pdf_engine, cache_version)
                t["cache_key"] = key  # used by the render worker to reuse PNGs
                if use_cache:
                    # if page text already cached -> adopt it and skip both render+GPU
                    txtp = _page_txt_path(self.config.temp_dir, key)
                    if txtp.exists():
                        try:
                            progress_tracker[t["source_path"]][t["page_num"]] = {"type": "text", "data": txtp.read_text(encoding="utf-8")}
                            cached_text_pages += 1
                            continue
                        except Exception as e:
                            self._log_error(t["source_path"], f"Failed reading cached page text: {e}")
                new_text_render_queue.append(t)

            text_render_queue = new_text_render_queue
            if not text_render_queue and cached_text_pages > 0:
                logger.info("All pages satisfied from cache (%d pages).", cached_text_pages)

            if not text_render_queue: # Assuming only text for now
                logger.info("No new pages require rendering/OCR; writing files completed from cache.")
                
                # Explicitly check for files that were just completed by the cache logic above
                final_progress = dict(progress_tracker)
                for path_str, pages in final_progress.items():
                    if path_str not in completed_files and all(p is not None for p in pages):
                        logger.info("File %s complete from cache. Writing to output.", path_str)
                        self._write_single_file_result(
                            path_str, list(pages), outfile, file_start_times, perf_tracker
                        )
                        completed_files.add(path_str)
                
                # The final loop at the very end of the `run` method will handle any incomplete
                # files, so we can simply return here.
                logger.info("All tasks were resolved from cache or had no pages to process.")
                return # Exit early as there is no parallel work to do

            # STAGE 4: parallel processing
            logger.info(
                "Starting parallel processing with %s GPU workers",
                self.config.num_gpu_workers,
            )
            logger.progress("render start", extra={"phase": "render", "pct": 30})
            rendered = 0
            total_render = len(text_render_queue)
            final_backend_kwargs = OCRRunner._build_backend_kwargs(self.config)

            # This is the key: two separate, dedicated pools that run concurrently.
            with ctx.Pool(processes=self.config.num_workers, initializer=configure_worker_logging,initargs=(self.config.log_queue,)) as render_pool, \
                 ctx.Pool(processes=self.config.num_gpu_workers, 
                      initializer=initialize_gpu_worker, 
                      initargs=(self.config.log_queue,self.config.ocr_backend, final_backend_kwargs)) as gpu_pool:

                # --- STAGE 4a: Submit CPU Rendering Jobs ---
                # The main process streams rendering tasks to the render_pool.
                render_iterator = render_pool.imap_unordered(worker_render_text_page, text_render_queue, chunksize=16)
                
                async_gpu_results = []
                pending = []  # queue of (AsyncResult job, meta_batch)
                image_batch_buffer = []; meta_batch_buffer = []

                total_render = len(text_render_queue)
                pbar_render = tqdm(render_iterator, total=total_render, desc="Rendering Pages (CPU)")

                for result in pbar_render:
                    if result.get("error"):
                        self._log_error(result["source_path"], result["error"])
                        continue

                    perf_tracker.add_cpu_time(result["source_path"], result.get("duration_seconds", 0.0))
                    rendered += 1
                    logger.progress("render progress", extra={"phase": "render", "current": rendered, "total": total_render})

                    result["cache_key"] = result.get("cache_key") or _cache_key(
                        result["source_path"], result["page_num"], result["dpi"], self.config.pdf_engine, cache_version
                    )

                    image_batch_buffer.append(result["temp_path"])
                    meta_batch_buffer.append(result)

                    if len(image_batch_buffer) >= self.config.gpu_batch_size:
                        job = gpu_pool.apply_async(process_gpu_batch, (image_batch_buffer,))
                        pending.append((job, meta_batch_buffer))
                        image_batch_buffer, meta_batch_buffer = [], []
                        # We no longer call _drain_ready here. Draining will be handled later.

                # final partial batch
                if image_batch_buffer:
                    job = gpu_pool.apply_async(process_gpu_batch, (image_batch_buffer,))
                    pending.append((job, meta_batch_buffer))
                

                logger.info("Waiting for remaining GPU batches...")
                logger.progress("aggregate start", extra={"phase": "aggregate", "pct": 85})
                # drain all remaining in order of completion
                total_batches = len(pending)
                with tqdm(total=total_batches, desc="Processing OCR (GPU)") as pbar_gpu:
                    while pending:
                        # Find all completed jobs
                        ready_indices = [i for i, (job, _) in enumerate(pending) if job.ready()]

                        if not ready_indices:
                            time.sleep(0.05)  # Avoid busy-waiting
                            continue

                        # Process all ready jobs, iterating backwards to safely pop from the list
                        for i in sorted(ready_indices, reverse=True):
                            job, meta_data = pending.pop(i)
                            try:
                                ocr_texts, gpu_duration = job.get()
                                perf_tracker.attribute_gpu_batch_time(meta_data, gpu_duration)

                                for text_idx, text in enumerate(ocr_texts):
                                    meta = meta_data[text_idx]
                                    progress_tracker[meta["source_path"]][meta["page_num"]] = {"type": "text", "data": text}
                                    if use_cache:
                                        key = meta.get("cache_key")
                                        if key:
                                            txtp = _page_txt_path(self.config.temp_dir, key)
                                            try:
                                                _ensure_parent(txtp)
                                                txtp.write_text(text or "", encoding="utf-8")
                                            except Exception as e:
                                                self._log_error(meta["source_path"], f"Cache write failed: {e}")

                                if not keep_render_cache:
                                    for meta in meta_data:
                                        try:
                                            Path(meta["temp_path"]).unlink(missing_ok=True)
                                        except Exception:
                                            pass # Logging this error can be noisy, so it's optional

                            except Exception as e:
                                logger.error("A GPU batch failed: %s", e)
                                for meta in meta_data:
                                    self._log_error(meta["source_path"], f"GPU batch failed: {e}")
                                    progress_tracker[meta["source_path"]][meta["page_num"]] = {"type": "error", "data": f"GPU batch failed: {e}"}
                            finally:
                                pbar_gpu.update(1)
                                affected_files = {meta["source_path"] for meta in meta_data}
                                for path_str in affected_files:
                                    if path_str in completed_files:
                                        continue
                                    
                                    current_pages = progress_tracker[path_str]
                                    if all(p is not None for p in current_pages):
                                        logger.info("File %s complete. Writing to output.", path_str)
                                        self._write_single_file_result(
                                            path_str, list(current_pages), outfile, file_start_times, perf_tracker
                                        )
                                        completed_files.add(path_str)
                
            # STAGE 5: final write
            logger.info("Aggregating and writing final results")
            logger.progress("final write", extra={"phase": "final", "pct": 95})
            # self._write_final_results(dict(progress_tracker), outfile, file_start_times, perf_tracker)
            final_progress = dict(progress_tracker)
            for path_str, pages in final_progress.items():
                if path_str not in completed_files:
                    logger.warning("File %s did not complete successfully. Writing final state.", path_str)
                    self._write_single_file_result(
                        path_str, list(pages), outfile, file_start_times, perf_tracker
                    )
            logger.info("Run finished")
            logger.progress("done", extra={"phase": "done", "pct": 100})

    # Deprecated method pending removal
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

    def _write_single_file_result(self, path_str, pages, outfile, file_start_times, perf_tracker):
        """Writes the result for a single, completed file and logs performance."""
        try:
            # Note: If OCRResult is a dataclass, use `asdict(result_obj)`
            result_obj = OCRResult(source_path=path_str, total_pages=len(pages), content=pages)
            outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
            outfile.flush()  # Make sure it's written to disk immediately

            final_metrics = perf_tracker.get_final_metrics(
                path_str, len(pages), file_start_times.get(path_str, 0.0)
            )
            self._log_performance({
                "metric_type": "file_processed",
                "source_path": path_str,
                "processing_method": "ocr_parallel",
                "total_pages": len(pages),
                **final_metrics,
            })

            if self.config.export_txt:
                full_text = "\n\n--- PAGE BREAK ---\n\n".join(
                    [p.get("data", "") for p in pages if p and p.get("type") == "text"]
                )
                if full_text:
                    self._write_txt_file(Path(path_str), full_text)
        except Exception as e:
            logger.error("Failed to write result for %s: %s", path_str, e)
            self._log_error(path_str, f"Failed during final result writing: {e}")