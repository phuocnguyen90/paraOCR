# paraOCR/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm
import json

from .config import OCRConfig
from .utils import load_dictionary, load_processed_ids 
from .models import OCRTask 

__all__ = ["collect_tasks", "run_pipeline", "main"]

def collect_tasks(config: OCRConfig) -> list[OCRTask]:
    print("--- Collecting and filtering tasks ---")
    processed_paths = set()
    if not config.force_rerun:
        processed_paths = load_processed_ids(config.output_path)
        if processed_paths:
            print(f"Found {len(processed_paths)} previously processed files to skip.")
    tasks = []
    ignore_keywords_lower = [k.lower() for k in config.ignore_keywords]
    if not config.input_dir.exists():
        print(f"Error: Input directory does not exist: {config.input_dir}")
        return []
    all_files_in_dir = list(config.input_dir.rglob("*"))
    print(f"Scanning {len(all_files_in_dir)} total paths in input directory...")
    for file_path in tqdm(all_files_in_dir, desc="Filtering tasks"):
        if not file_path.is_file(): continue
        file_path_str = str(file_path)
        if file_path_str in processed_paths: continue
        filename_lower = file_path.name.lower()
        if any(keyword in filename_lower for keyword in ignore_keywords_lower): continue
        if filename_lower.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            tasks.append(OCRTask(source_path=file_path))
    return tasks


def run_pipeline(config: OCRConfig):
    """
    The main workhorse function. It takes a config object and runs the entire pipeline.
    This can be called directly from other Python scripts.
    """
    from .parallel import OCRRunner
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.error_log_path:
        config.error_log_path.parent.mkdir(parents=True, exist_ok=True)

    print("--- Starting ParaOCR ---")
    print(f"Input Directory: {config.input_dir}")
    print(f"Output File: {config.output_path}")
    print(f"CPU Workers: {config.num_workers} | GPU Workers: {config.num_gpu_workers} | GPU Batch Size: {config.gpu_batch_size} | DPI: {config.dpi}")
    print("-------------------------")

    tasks_to_run = collect_tasks(config)
    if tasks_to_run:
        print(f"\nStarting OCR process on {len(tasks_to_run)} new files.")
        runner = OCRRunner(config)
        runner.run(tasks_to_run)
    else:
        print("\nNo new files to process based on current settings. All tasks are complete.")
    print("\n--- ParaOCR Processing Complete ---")

def main():
    parser = argparse.ArgumentParser(description="paraOCR: High-performance file OCR.")
    
    # --- Arguments ---
    parser.add_argument("-i", "--input-dir", type=Path, required=True, help="Directory containing files to OCR.")
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="Path to save the output results JSONL file.")
    parser.add_argument("-w", "--workers", type=int, help="Number of CPU worker processes for rendering.")
    parser.add_argument("-b", "--gpu-batch-size", type=int, help="Number of images to send to the GPU in one batch.")
    parser.add_argument("-d", "--dpi", type=int, help="DPI to use for rendering PDF pages.")
    parser.add_argument("-l", "--languages", nargs='+', default=['vi', 'en'], help="List of language codes for EasyOCR.")
    parser.add_argument("--beamsearch", action='store_true', help="Enable beam search in EasyOCR (slower, more accurate).")
    parser.add_argument("--ignore-keyword", action='append', dest='ignore_keywords', help="A keyword in a filename to ignore (can be used multiple times).")
    parser.add_argument("--force-rerun", action='store_true', help="Force reprocessing of all files, ignoring previous results.")
    parser.add_argument("--error-log-path", type=Path, help="Path to save the error log JSONL file.")
    parser.add_argument("--pdf-engine", type=str, default="pymupdf", choices=["pymupdf"], help="The underlying engine to use for PDF processing.") 
    parser.add_argument("--export-txt", action='store_true', help="Also export a discrete .txt file for each document in its source directory.")
    parser.add_argument("--ocr-backend", type=str,
                        default="paraocr.ocr_backends.easyocr_backend.EasyOCREngine")
    parser.add_argument("--ocr-backend-kwargs", type=str, default="{}",
                        help='JSON dict for backend init kwargs, for example {"languages": ["vi","en"], "gpu": true}')
    perf_group = parser.add_argument_group('Performance Logging')
    perf_group.add_argument("--log-performance", action='store_true', help="Enable detailed performance logging to a file.")
    perf_group.add_argument("--performance-log-path", type=Path, help="Path to save the performance log JSONL file.")
    
    args = parser.parse_args()

    # normalize backend kwargs into a dict
    try:
        kwargs = json.loads(args.ocr_backend_kwargs) if isinstance(args.ocr_backend_kwargs, str) else args.ocr_backend_kwargs
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON for --ocr-backend-kwargs: {e}")

    # build config dict explicitly so types are right
    config = OCRConfig.from_dict({
        "input_dir": args.input_dir,
        "output_path": args.output_path,
        "error_log_path": args.error_log_path,
        "languages": args.languages,
        "ignore_keywords": args.ignore_keywords or [],
        "num_workers": args.workers,
        "gpu_batch_size": args.gpu_batch_size,
        "num_gpu_workers": None,  # let default stand unless you add a flag
        "dpi": args.dpi,
        "beamsearch": args.beamsearch,
        "force_rerun": args.force_rerun,
        "export_txt": args.export_txt,
        "log_performance": args.log_performance,
        "performance_log_path": args.performance_log_path,
        "pdf_engine": args.pdf_engine,
        "ocr_backend": args.ocr_backend,
        "ocr_backend_kwargs": kwargs,
    })

    run_pipeline(config)


if __name__ == '__main__':
    main()