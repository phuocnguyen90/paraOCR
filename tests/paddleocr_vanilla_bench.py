import numpy as np
import argparse
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int":   [np.int8, np.int16, np.int32, np.int64],
        "uint":  [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_],
    }


import os
import time
from pathlib import Path
from datetime import datetime
import json

import fitz  # PyMuPDF

from PIL import Image
from tqdm import tqdm

# PaddleOCR / PaddlePaddle
from paddleocr import PaddleOCR
import paddle
import cv2


# --- CONFIGURATION ---
# Default paths and settings
DEFAULT_INPUT_DIR = Path("tests/data")
DEFAULT_OUTPUT_JSONL = Path("tests/paddleocr_results.jsonl")

# OCR settings
LANGUAGES = ["vi", "en"]     # We'll pick a suitable PaddleOCR model from this
USE_GPU = True               # Will auto-fallback to CPU if CUDA build is unavailable
TARGET_DPI = 200             # Match your other benchmarks for apples-to-apples


def _pick_paddle_lang(langs) -> str:
    """
    PaddleOCR uses a single 'lang' model per Reader.
    This helper picks a reasonable model based on your list.
    """
    langs = [s.lower() for s in (langs or [])]
    if "vi" in langs:
        return "latin"
    if "en" in langs:
        return "en"
    if any(l in langs for l in ("fr", "de", "es", "it", "pt")):
        return "latin"
    return "en"


def _check_gpu_allowed(requested: bool) -> bool:
    """
    Verify whether PaddlePaddle is CUDA-enabled; if not, fall back to CPU.
    """
    if not requested:
        return False
    try:
        if not paddle.device.is_compiled_with_cuda():
            print("[WARN] PaddlePaddle was not compiled with CUDA. Falling back to CPU.")
            return False
        if paddle.device.cuda.device_count() < 1:
            print("[WARN] No visible CUDA devices. Falling back to CPU.")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Could not verify CUDA availability ({e}). Falling back to CPU.")
        return False


def _pil_to_cv_bgr(im: Image.Image) -> np.ndarray:
    """
    Convert a PIL RGB image to OpenCV BGR ndarray for PaddleOCR.
    """
    arr = np.array(im.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _serialize_paddle_result(paddle_result) -> list[dict]:
    """
    Convert PaddleOCR's complex output structure to a simple list of dicts.
    """
    lines = []
    if not paddle_result:
        return lines
    groups = paddle_result[0] if (len(paddle_result) == 1 and isinstance(paddle_result[0], list)) else paddle_result
    for item in groups or []:
        try:
            box, (txt, conf) = item
            bbox = [[float(x), float(y)] for (x, y) in box]
            lines.append({"text": str(txt), "confidence": float(conf), "bbox": bbox})
        except Exception:
            continue
    return lines


def main():
    print("--- Starting Vanilla PaddleOCR Benchmark (with JSONL export) ---")

    # 1. Set up and parse command-line arguments
    parser = argparse.ArgumentParser(
        description="A simple, single-threaded, sequential OCR script for benchmarking PaddleOCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing files to OCR."
    )
    parser.add_argument(
        "-o", "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Path to save the output results JSONL."
    )
    args = parser.parse_args()
    
    # Use the parsed arguments
    input_dir = args.input_dir
    output_jsonl = args.output_path

    if not input_dir.exists():
        print(f"[Error] Input directory does not exist: {input_dir.resolve()}")
        return

    # 2. Initialize PaddleOCR once
    paddle_lang = os.getenv("PADDLE_LANG") or _pick_paddle_lang(LANGUAGES)
    use_gpu = _check_gpu_allowed(USE_GPU)
    print(f"Initializing PaddleOCR (lang='{paddle_lang}', use_gpu={use_gpu})...")
    t0 = time.perf_counter()
    ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, use_gpu=use_gpu, show_log=False)
    print(f"Engine initialized in {time.perf_counter() - t0:.2f}s")

    # 3. Collect files from the correct directory
    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")
    ]
    if not files:
        print(f"No files found in {input_dir.resolve()}. Exiting.")
        return
    print(f"Found {len(files)} files to process.")

    total_pages = 0
    total_cpu_render = 0.0
    total_ocr = 0.0
    wall_start = time.perf_counter()

    # Ensure parent folder for output exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # 4. Process files and write results
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for fpath in tqdm(files, desc="Processing Files"):
            try:
                if fpath.suffix.lower() == ".pdf":
                    with fitz.open(fpath) as doc:
                        for page_idx, page in enumerate(doc):
                            c0 = time.perf_counter()
                            pix = page.get_pixmap(dpi=TARGET_DPI)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            total_cpu_render += time.perf_counter() - c0

                            o0 = time.perf_counter()
                            result = ocr.ocr(_pil_to_cv_bgr(img), cls=True)
                            total_ocr += time.perf_counter() - o0

                            lines = _serialize_paddle_result(result)
                            record = {
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "engine": "PaddleOCR", "source_path": str(fpath),
                                "page_index": page_idx, "lines": lines,
                                "text": "\n".join(l["text"] for l in lines if l.get("text")),
                            }
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_pages += 1
                else:
                    c0 = time.perf_counter()
                    img = Image.open(fpath).convert("RGB")
                    total_cpu_render += time.perf_counter() - c0

                    o0 = time.perf_counter()
                    result = ocr.ocr(_pil_to_cv_bgr(img), cls=True)
                    total_ocr += time.perf_counter() - o0

                    lines = _serialize_paddle_result(result)
                    record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "engine": "PaddleOCR", "source_path": str(fpath),
                        "page_index": 0, "lines": lines,
                        "text": "\n".join(l["text"] for l in lines if l.get("text")),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pages += 1

            except Exception as e:
                err = {"timestamp": datetime.utcnow().isoformat() + "Z", "source_path": str(fpath), "error": str(e)}
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")

    wall = time.perf_counter() - wall_start

    # 5. Summary
    print("\n--- Vanilla PaddleOCR Benchmark Report ---")
    print(f"Total Files Processed: {len(files)}")
    print(f"Total Pages Processed: {total_pages}")
    print("-" * 30)
    print(f"Total Wall-Clock Time:          {wall:.2f} s")
    print(f"Total CPU Render Time:          {total_cpu_render:.2f} s")
    print(f"Total PaddleOCR Time (GPU={use_gpu}): {total_ocr:.2f} s")
    print("-" * 30)
    if total_pages > 0:
        print(f"Average Time per Page:          {(wall / total_pages):.2f} s")
        print(f"Throughput:                     {(total_pages / wall):.2f} pages/s")
    print("--------------------------------")
    print(f"JSONL written to: {output_jsonl.resolve()}")


if __name__ == "__main__":
    main()