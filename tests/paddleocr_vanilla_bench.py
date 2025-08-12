# benchmark_vanilla_paddleocr.py
import numpy as np
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
# Point this to a directory with a few test files (including the 100-page one)
INPUT_DIR = Path("tests\data")
OUTPUT_JSONL = Path("tests\paddleocr_results.jsonl")

# OCR settings
LANGUAGES = ["vi", "en"]     # We'll pick a suitable PaddleOCR model from this
USE_GPU = True               # Will auto-fallback to CPU if CUDA build is unavailable
TARGET_DPI = 200             # Match your EasyOCR benchmark for apples-to-apples


def _pick_paddle_lang(langs) -> str:
    """
    PaddleOCR uses a single 'lang' model per Reader.
    This helper picks a reasonable model based on your list.
    - Vietnamese is best covered by the 'latin' model.
    - English can use 'en'.
    """
    langs = [s.lower() for s in (langs or [])]
    if "vi" in langs:
        return "latin"
    if "en" in langs:
        return "en"
    # Fallbacks (customize as needed)
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
    PaddleOCR.ocr(img, cls=True) returns:
      [ [ [box_points, (text, conf)], ... ] ]
    We convert inner list to: [{"text": str, "confidence": float, "bbox": [[x,y],...]}]
    """
    lines = []
    if not paddle_result:
        return lines
    # Handle both single-image and batched output
    groups = paddle_result[0] if (len(paddle_result) == 1 and isinstance(paddle_result[0], list)) else paddle_result
    for item in groups:
        try:
            box, (txt, conf) = item
        except Exception:
            # Sometimes structure differs slightly; skip malformed entries
            continue
        # Ensure bbox is list of [x,y]
        bbox = [[float(x), float(y)] for (x, y) in box]
        lines.append({"text": str(txt), "confidence": float(conf), "bbox": bbox})
    return lines


def main():
    print("--- Starting Vanilla PaddleOCR Benchmark (with JSONL export) ---")
    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_JSONL.resolve()}")

    # 1) Initialize PaddleOCR once
    paddle_lang = os.getenv("PADDLE_LANG") or _pick_paddle_lang(LANGUAGES)
    use_gpu = _check_gpu_allowed(USE_GPU)
    print(f"Initializing PaddleOCR (lang='{paddle_lang}', use_gpu={use_gpu})...")
    t0 = time.perf_counter()
    ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, use_gpu=use_gpu, show_log=False)
    print(f"Engine initialized in {time.perf_counter() - t0:.2f}s")

    # 2) Collect files
    files = [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")
    ]
    if not files:
        print("No files found. Exiting.")
        return
    print(f"Found {len(files)} files.")

    total_pages = 0
    total_cpu_render = 0.0
    total_ocr = 0.0
    wall_start = time.perf_counter()

    # Ensure parent folder for output exists
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Open JSONL once; write one record per page/image
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for fpath in tqdm(files, desc="Processing Files"):
            try:
                if fpath.suffix.lower() == ".pdf":
                    with fitz.open(fpath) as doc:
                        for page_idx, page in enumerate(doc):
                            # CPU render
                            c0 = time.perf_counter()
                            pix = page.get_pixmap(dpi=TARGET_DPI)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            cpu_sec = time.perf_counter() - c0
                            total_cpu_render += cpu_sec

                            # OCR
                            o0 = time.perf_counter()
                            img_bgr = _pil_to_cv_bgr(img)
                            result = ocr.ocr(img_bgr, cls=True)
                            ocr_sec = time.perf_counter() - o0
                            total_ocr += ocr_sec

                            lines = _serialize_paddle_result(result)
                            record = {
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "engine": "PaddleOCR",
                                "engine_lang": paddle_lang,
                                "source_path": str(fpath),
                                "page_index": page_idx,
                                "image_size": {"width": pix.width, "height": pix.height},
                                "metrics": {
                                    "cpu_render_seconds": round(cpu_sec, 4),
                                    "ocr_seconds": round(ocr_sec, 4),
                                },
                                "lines": lines,  # list of {text, confidence, bbox}
                                # Optional: concatenated text for quick eyeballing
                                "text": "\n".join(l["text"] for l in lines if l.get("text")),
                            }
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_pages += 1
                else:
                    # Image file
                    c0 = time.perf_counter()
                    img = Image.open(fpath).convert("RGB")
                    w, h = img.size
                    cpu_sec = time.perf_counter() - c0
                    total_cpu_render += cpu_sec

                    o0 = time.perf_counter()
                    img_bgr = _pil_to_cv_bgr(img)
                    result = ocr.ocr(img_bgr, cls=True)
                    ocr_sec = time.perf_counter() - o0
                    total_ocr += ocr_sec

                    lines = _serialize_paddle_result(result)
                    record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "engine": "PaddleOCR",
                        "engine_lang": paddle_lang,
                        "source_path": str(fpath),
                        "page_index": 0,
                        "image_size": {"width": w, "height": h},
                        "metrics": {
                            "cpu_render_seconds": round(cpu_sec, 4),
                            "ocr_seconds": round(ocr_sec, 4),
                        },
                        "lines": lines,
                        "text": "\n".join(l["text"] for l in lines if l.get("text")),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pages += 1

            except Exception as e:
                err = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "engine": "PaddleOCR",
                    "source_path": str(fpath),
                    "error": str(e),
                }
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")

    wall = time.perf_counter() - wall_start

    # 3) Summary
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
    print(f"JSONL written to: {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()