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
from typing import Any, Dict

import fitz  # PyMuPDF

from PIL import Image
from tqdm import tqdm

# PaddleOCR / PaddlePaddle
from paddleocr import PaddleOCR
import paddle
import cv2

# --- CONFIGURATION ---
DEFAULT_INPUT_DIR = Path("tests/data")
DEFAULT_OUTPUT_JSONL = Path("tests/paddleocr_results.jsonl")
LANGUAGES = ["vi", "en"]
USE_GPU = True
TARGET_DPI = 200

# --- HELPER FUNCTIONS ---

def _norm_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make kwargs compatible across PaddleOCR 2.x and 3.x.
    This is ported from the main paraOCR backend for consistency.
    """
    import paddle
    try:
        from paddleocr import __version__ as _pocv
        _maj = int(str(_pocv).split(".", 1)[0])
    except Exception:
        _maj = 2  # default assume 2.x

    k_in = dict(kwargs or {})

    # Language selection
    def _select_paddle_lang(langs: Any) -> str:
        if isinstance(langs, str): arr = [langs]
        elif isinstance(langs, (list, tuple, set)): arr = list(langs)
        else: arr = ["vi", "en"]
        arr = [str(x).lower() for x in arr if x]
        if "vi" in arr: return "la"
        if "en" in arr: return "en"
        return "en"

    langs = k_in.pop("languages", None) or k_in.pop("lang", None) or ["vi", "en"]
    lang = _select_paddle_lang(langs)

    # Device/GPU selection
    device = k_in.pop("device", None)
    want_gpu = k_in.pop("use_gpu", k_in.pop("gpu", None))
    if device is None:
        try:
            compiled = paddle.device.is_compiled_with_cuda()
            devs = paddle.device.cuda.device_count() if compiled else 0
            if want_gpu is None: want_gpu = bool(compiled and devs > 0)
            else: want_gpu = bool(want_gpu and compiled and devs > 0)
        except Exception:
            want_gpu = bool(want_gpu)
        device = "gpu:0" if want_gpu else "cpu"

    # Normalize other params based on version
    if _maj >= 3:
        out = {"lang": lang, "device": device}
    else:
        out = {"lang": lang, "use_gpu": device.startswith("gpu")}

    # Add common params, respecting version differences
    out["use_angle_cls" if _maj < 3 else "use_textline_orientation"] = True
    if _maj < 3:
        out["show_log"] = False

    return out

def _pil_to_cv_bgr(im: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to OpenCV BGR ndarray for PaddleOCR."""
    arr = np.array(im.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _serialize_paddle_result(paddle_result) -> list[dict]:
    """Convert PaddleOCR's complex output structure to a simple list of dicts."""
    lines = []
    if not paddle_result: return lines
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

    parser = argparse.ArgumentParser(description="A simple, single-threaded, sequential OCR script for benchmarking PaddleOCR.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing files to OCR.")
    parser.add_argument("-o", "--output-path", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Path to save the output results JSONL.")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_jsonl = args.output_path

    if not input_dir.exists():
        print(f"[Error] Input directory does not exist: {input_dir.resolve()}")
        return

    # 1. Prepare version-aware arguments for PaddleOCR
    print("Preparing PaddleOCR arguments...")
    initial_kwargs = {"languages": LANGUAGES, "use_gpu": USE_GPU}
    final_paddle_args = _norm_kwargs(initial_kwargs)
    
    print(f"Initializing PaddleOCR with: {final_paddle_args}")
    t0 = time.perf_counter()
    ocr = PaddleOCR(**final_paddle_args)
    print(f"Engine initialized in {time.perf_counter() - t0:.2f}s")

    # 2. Collect files
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")]
    if not files:
        print(f"No files found in {input_dir.resolve()}. Exiting.")
        return
    print(f"Found {len(files)} files to process.")

    total_pages = 0
    total_cpu_render = 0.0
    total_ocr = 0.0
    wall_start = time.perf_counter()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
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
                            record = {"source_path": str(fpath), "page_index": page_idx, "lines": lines}
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_pages += 1
                else: # Image file
                    c0 = time.perf_counter()
                    img = Image.open(fpath).convert("RGB")
                    total_cpu_render += time.perf_counter() - c0

                    o0 = time.perf_counter()
                    result = ocr.ocr(_pil_to_cv_bgr(img), cls=True)
                    total_ocr += time.perf_counter() - o0

                    lines = _serialize_paddle_result(result)
                    record = {"source_path": str(fpath), "page_index": 0, "lines": lines}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pages += 1
            except Exception as e:
                err = {"source_path": str(fpath), "error": str(e)}
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")

    wall = time.perf_counter() - wall_start

    # Summary
    print("\n--- Vanilla PaddleOCR Benchmark Report ---")
    print(f"Total Pages Processed: {total_pages}")
    print(f"Total Wall-Clock Time: {wall:.2f} s")
    if total_pages > 0:
        print(f"Throughput: {(total_pages / wall):.2f} pages/s")
    print("--------------------------------")
    print(f"JSONL written to: {output_jsonl.resolve()}")

if __name__ == "__main__":
    main()