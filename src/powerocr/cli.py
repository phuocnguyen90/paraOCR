# powerocr/cli.py

import argparse
from pathlib import Path
from tqdm import tqdm
import sys

from .config import OCRConfig
from .parallel import OCRRunner
from .utils import load_dictionary, load_processed_ids # <-- Import utils
from .models import OCRTask # <-- Import OCRTask model

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

def main():
    parser = argparse.ArgumentParser(description="PowerOCR: High-performance file OCR.")
    
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
    parser.add_argument("--pdf-engine", type=str, default="pymupdf", choices=["pymupdf"],
                        help="The underlying engine to use for PDF processing.") 
    parser.add_argument("--export-txt", action='store_true', help="Also export a discrete .txt file for each document in its source directory.")
    perf_group = parser.add_argument_group('Performance Logging')
    perf_group.add_argument("--log-performance", action='store_true', help="Enable detailed performance logging to a file.")
    perf_group.add_argument("--performance-log-path", type=Path, help="Path to save the performance log JSONL file.")
    
    args = parser.parse_args()

    # --- Create Config ---
    config_args = {k: v for k, v in vars(args).items() if v is not None}
    vi_dictionary = load_dictionary() 
    config_args['dictionary'] = vi_dictionary
    config = OCRConfig.from_dict(config_args)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.error_log_path: config.error_log_path.parent.mkdir(parents=True, exist_ok=True)


    print("--- Starting PowerOCR ---")
    print(f"Input Directory: {config.input_dir}")
    print(f"Output File: {config.output_path}")
    print(f"CPU Workers: {config.num_workers} | GPU Batch Size: {config.gpu_batch_size} | DPI: {config.dpi}")
    print("-------------------------")

    # --- THE CORRECTED WORKFLOW ---
    # 1. Collect and filter tasks first. This function now contains the resumability logic.
    tasks_to_run = collect_tasks(config)
    
    # 2. If there are tasks to do, create the runner and execute them.
    if tasks_to_run:
        print(f"\nStarting OCR process on {len(tasks_to_run)} new files.")
        runner = OCRRunner(config)
        runner.run(tasks_to_run)
    else:
        print("\nNo new files to process based on current settings. All tasks are complete.")

    print("\n--- PowerOCR Processing Complete ---")

if __name__ == '__main__':
    main()