import time
import argparse  # Added for CLI arguments
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# This is now just a default, can be overridden from the command line
DEFAULT_INPUT_DIR = Path("tests/data")
# OCR settings
LANGUAGES = ['vi', 'en']
USE_GPU = True
TARGET_DPI = 200  # Use the same DPI as your first pass for a fair comparison

def main():
    """
    A simple, single-threaded, sequential OCR script for benchmarking.
    Now accepts an --input-dir argument from the command line.
    """
    # 1. Set up and parse command-line arguments
    parser = argparse.ArgumentParser(
        description="A simple, single-threaded, sequential OCR script for benchmarking EasyOCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing files to OCR."
    )
    args = parser.parse_args()
    input_dir = args.input_dir

    print("--- Starting Vanilla OCR Benchmark ---")
    
    # Check if the provided directory exists
    if not input_dir.exists():
        print(f"\n[Error] Input directory does not exist: {input_dir.resolve()}")
        return

    print(f"Processing files in: {input_dir.resolve()}")

    # 2. Initialize the OCR Engine (this happens only once)
    print("Initializing EasyOCR Engine...")
    start_init = time.perf_counter()
    ocr_engine = easyocr.Reader(LANGUAGES, gpu=USE_GPU)
    init_duration = time.perf_counter() - start_init
    print(f"Engine initialized in {init_duration:.2f} seconds.")

    # 3. Collect all files to process from the specified directory
    files_to_process = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']
    ]

    if not files_to_process:
        print("No files found to process.")
        return

    print(f"\nFound {len(files_to_process)} files to process sequentially.")

    total_pages_processed = 0
    total_cpu_render_time = 0
    total_gpu_ocr_time = 0
    overall_start_time = time.perf_counter()

    # 4. Process each file one by one
    for file_path in tqdm(files_to_process, desc="Processing Files"):
        try:
            if file_path.suffix.lower() == '.pdf':
                with fitz.open(file_path) as doc:
                    for page_num, page in enumerate(doc):
                        # --- CPU Work ---
                        cpu_start = time.perf_counter()
                        pix = page.get_pixmap(dpi=TARGET_DPI)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        total_cpu_render_time += time.perf_counter() - cpu_start

                        # --- GPU Work ---
                        gpu_start = time.perf_counter()
                        img_array = np.array(img)
                        # Process one page at a time
                        _ = ocr_engine.readtext(img_array, detail=0, paragraph=True)
                        total_gpu_ocr_time += time.perf_counter() - gpu_start

                        total_pages_processed += 1
            else:  # Handle images
                # --- CPU Work (just opening) ---
                cpu_start = time.perf_counter()
                img = Image.open(file_path).convert("RGB")
                total_cpu_render_time += time.perf_counter() - cpu_start

                # --- GPU Work ---
                gpu_start = time.perf_counter()
                img_array = np.array(img)
                _ = ocr_engine.readtext(img_array, detail=0, paragraph=True)
                total_gpu_ocr_time += time.perf_counter() - gpu_start

                total_pages_processed += 1

        except Exception as e:
            print(f"\n[Error] Failed to process {file_path.name}: {e}")

    overall_duration = time.perf_counter() - overall_start_time

    # 5. Print the final benchmark report
    print("\n--- Vanilla Benchmark Report ---")
    print(f"Total Files Processed: {len(files_to_process)}")
    print(f"Total Pages Processed: {total_pages_processed}")
    print("-" * 30)
    print(f"Total Wall-Clock Time: {overall_duration:.2f} seconds")
    print("-" * 30)
    print(f"Total CPU Render Time:   {total_cpu_render_time:.2f} seconds")
    print(f"Total GPU OCR Time:      {total_gpu_ocr_time:.2f} seconds")
    print("-" * 30)
    if total_pages_processed > 0:
        print(f"Average Time per Page:   {(overall_duration / total_pages_processed):.2f} seconds")
        print(f"Throughput:              {(total_pages_processed / overall_duration):.2f} pages/second")
    print("--------------------------------")


if __name__ == "__main__":
    main()