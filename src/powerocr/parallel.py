# powerocr/parallel.py

import json
import gc
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from multiprocessing import Pool
import fitz
from PIL import Image
import numpy as np
import easyocr
import torch
from tqdm import tqdm

from .config import OCRConfig
from .models import OCRResult, OCRTask 
from .utils import is_native_text_good_quality

def process_table_page(page_task: Dict) -> Dict:
    # In the future, this would use a library like Camelot or a vision model.
    # For now, it returns a placeholder.
    print(f"\n[Table Processor] Stub for page {page_task['page_num']} of {page_task['source_path']}")
    time.sleep(0.5) # Simulate work
    return {**page_task, "result_type": "table", "content": "<table>...placeholder...</table>"}

def process_image_page(page_task: Dict) -> Dict:
    # In the future, this could extract and save the image.
    print(f"\n[Image Processor] Stub for page {page_task['page_num']} of {page_task['source_path']}")
    time.sleep(0.1) # Simulate work
    return {**page_task, "result_type": "image", "content": "/path/to/extracted_image.png"}

def process_text_page(page_task: Dict) -> Dict:
    return

def worker_dispatcher(page_task: Dict) -> Dict:
    """
    CPU-bound worker that performs quick layout analysis and tags the page task.
    """
    file_path_str = page_task["source_path"]
    page_num = page_task["page_num"]
    
    try:
        with fitz.open(file_path_str) as doc:
            page = doc.load_page(page_num)
            
            # 1. Check for tables using PyMuPDF's built-in tool
            tables = page.find_tables()
            if len(tables.tables) > 0:
                # If tables are found, tag it for the table processor
                return {**page_task, "processing_type": "table"}

            # 2. Check for significant images (larger than a certain % of page area)
            images = page.get_images(full=True)
            page_area = page.rect.width * page.rect.height
            for img in images:
                xref = img[0]
                bbox = page.get_image_bbox(img)
                img_area = bbox.width * bbox.height
                if page_area > 0 and (img_area / page_area) > 0.25: # If image is > 25% of page
                    return {**page_task, "processing_type": "image"}

            # 3. If none of the above, it's a standard text OCR task
            return {**page_task, "processing_type": "text_ocr"}

    except Exception as e:
        return {**page_task, "processing_type": "error", "error": f"Dispatcher failed on page {page_num}: {e}"}

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

def worker_process_file(task_with_config: tuple) -> Dict:
    """CPU-bound worker that now also measures its execution time."""
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
                    return {"source_path": str(file_path), "type": "native_text", "text": native_text, "error": None}
                page_count = len(doc)
                if not page_count: return {"source_path": str(file_path), "type": "error", "error": "PDF has zero pages"}
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
            return {"source_path": str(file_path), "type": "ocr", "page_count": 1, "temp_paths": [str(temp_img_path)], "error": None}
        
        else:
            return {"source_path": str(file_path), "type": "error", "error": f"Unsupported file type: {file_path.suffix}"}
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        return {"source_path": str(file_path), "type": "error", "error": f"Failed during rendering/reading: {str(e)}","duration_seconds": duration}

class OCRRunner:
    """Orchestrates the parallel OCR pipeline."""
    def __init__(self, config: OCRConfig):
        self.config = config
        self.engine = OCREngine(config.languages, gpu=torch.cuda.is_available(), beamsearch=config.beamsearch)

    def _log_error(self, source_path: str, reason: str):
        
        with open(self.config.error_log_path, 'a', encoding='utf-8') as f:
            error_entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),"source_path": source_path, "error_reason": reason}
            f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
    
    def _log_performance(self, metric: Dict):
        """Appends a structured performance metric to the log file."""
        if not self.config.log_performance:
            return
        with open(self.config.performance_log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **metric
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def _write_txt_file(self, source_path: Path, text: str):
        """Creates a .txt file in the same directory as the source file."""
        try:
            # Create a new filename by replacing the original extension with .txt
            txt_filename = source_path.stem + ".txt"
            output_path = source_path.parent / txt_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            # Log an error but don't crash the whole pipeline
            error_reason = f"Failed to write discrete .txt file: {e}"
            print(f"\n[Warning] {error_reason} for {source_path}")
            self._log_error(str(source_path), error_reason)

    # --- The run method now correctly accepts the tasks list ---
    def run(self, tasks: list[OCRTask]):
        """Executes the parallel OCR process on a list of tasks."""

        
        image_batch_buffer = []
        meta_batch_buffer = []
        ocr_progress_tracker: Dict[str, List[Optional[str]]] = {}

        with open(self.config.output_path, 'a', encoding='utf-8') as outfile, \
             Pool(processes=self.config.num_workers) as pool:

            config_dict = self.config.to_dict()
            tasks_with_config = [(task, config_dict) for task in tasks]

            results_iterator = pool.imap_unordered(worker_process_file, tasks_with_config)
            
            for result_dict in tqdm(results_iterator, total=len(tasks), desc="Processing Files (CPU)"):
                
                if not result_dict: continue
                source_path_str = result_dict.get("source_path", "Unknown")
                source_path = Path(source_path_str) 
                result_type = result_dict.get("type", "error")
                error_msg = result_dict.get("error")

                if result_dict and "duration_seconds" in result_dict:
                                    self._log_performance({
                                        "metric_type": "cpu_render_file",
                                        "source_path": result_dict.get("source_path"),
                                        "duration_seconds": round(result_dict["duration_seconds"], 4)
                                    })

                if result_type == "error":
                    self._log_error(source_path_str, error_msg)
                    continue
                if result_type == "native_text":
                    text = result_dict["text"]
                    result_obj = OCRResult(source_path=source_path_str, text=text, method="native_text")
                    outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                    outfile.flush()
                    if self.config.export_txt and text:
                        self._write_txt_file(source_path, text)                    
                    continue

                if result_type == "ocr":
                    page_count = result_dict["page_count"]
                    if source_path_str not in ocr_progress_tracker:
                        ocr_progress_tracker[source_path_str] = [None] * page_count
                    for page_num, temp_path in enumerate(result_dict["temp_paths"]):
                        image_batch_buffer.append(temp_path)
                        meta_batch_buffer.append({"source_path": source_path_str, "page_num": page_num})

                if len(image_batch_buffer) >= self.config.gpu_batch_size:
                    self._process_and_write_batch(image_batch_buffer, meta_batch_buffer, ocr_progress_tracker, outfile)
                    image_batch_buffer.clear()
                    meta_batch_buffer.clear()
                    gc.collect()

            if image_batch_buffer:
                self._process_and_write_batch(image_batch_buffer, meta_batch_buffer, ocr_progress_tracker, outfile)

    def _process_and_write_batch(self, image_paths: List[str], meta_data: List[Dict], progress_tracker: Dict, outfile):

        images_to_process = [Image.open(p) for p in image_paths]
        start_time = time.perf_counter()

        ocr_results = self.engine.process_images_in_batch(images_to_process)

        duration = time.perf_counter() - start_time
        self._log_performance({
            "metric_type": "gpu_ocr_batch",
            "duration_seconds": round(duration, 4),
            "batch_size": len(image_paths),
            "throughput_pages_sec": round(len(images_to_process) / duration, 2) if duration > 0 else 0
        })

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
                result_obj = OCRResult(source_path=path, text=full_text, method=f"easyocr_{self.engine.device.lower()}_parallel")
                outfile.write(json.dumps(result_obj.__dict__, ensure_ascii=False) + "\n")
                outfile.flush()
                if self.config.export_txt and full_text:
                    self._write_txt_file(Path(path_str), full_text)
                
                completed_paths.append(path_str)

                completed_paths.append(path)
        for path in completed_paths:
            del progress_tracker[path]