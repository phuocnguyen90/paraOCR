# benchmark_vanilla_tesseract_vi_jsonl.py
# Single-threaded Tesseract benchmark (Vietnamese) + JSONL export

import os
import json
import time
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from tqdm import tqdm

import pytesseract as pt
pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- CONFIGURATION ---
INPUT_DIR = Path("tests/data")                      # folder with PDFs/images
OUTPUT_JSONL = Path("tesseract_vi_results.jsonl")   # JSONL out (one record per page/image)
TARGET_DPI = 200                                    # PDF render DPI (match your other benches)

# Tesseract settings
# - OEM: 3 = default (LSTM)
# - PSM: 6 = assume a uniform block of text (good default for documents)
TESSERACT_OEM = 3
TESSERACT_PSM = 6

# Language selection:
# Map your desired langs to Tesseract codes (eng, vie, fra, deu, spa, ...)
LANGS = ["vi", "en"]  # we'll map to 'vie+eng' below

# If Tesseract is not on PATH (Windows), set the binary path here or via env TESSERACT_CMD
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or None  # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --- Helpers ------------------------------------------------------------------

_TESS_MAP = {
    "vi": "vie",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
}

def _to_tesseract_lang(langs) -> str:
    if not langs:
        return "vie"  # default to Vietnamese
    codes = [_TESS_MAP.get(l.lower(), l.lower()) for l in langs]
    return "+".join(sorted(set(codes)))

def _ensure_tesseract_cmd():
    if TESSERACT_CMD:
        pt.pytesseract.tesseract_cmd = TESSERACT_CMD

def _image_to_words(img: Image.Image, lang: str) -> tuple[list[dict], float]:
    """
    Run Tesseract once to get per-word boxes + confidences.
    Returns (words, seconds).
    words: [{"text": str, "confidence": float(0..1), "bbox": [x1,y1,x2,y2]}]
    """
    cfg = f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}"
    t0 = time.perf_counter()
    # TSV with one row per word (and other levels); we’ll filter on conf>=0 and text != ''
    data = pt.image_to_data(img, lang=lang, config=cfg, output_type=pt.Output.DATAFRAME)
    sec = time.perf_counter() - t0

    words = []
    if data is not None and len(data) > 0:
        for _, row in data.iterrows():
            text = str(row.get("text", "") or "").strip()
            conf = row.get("conf", -1)
            if not text or conf is None or float(conf) < 0:
                continue
            x, y, w, h = (int(row.get("left", 0)), int(row.get("top", 0)),
                          int(row.get("width", 0)), int(row.get("height", 0)))
            words.append({
                "text": text,
                "confidence": float(conf) / 100.0,          # normalize to 0..1
                "bbox": [x, y, x + w, y + h],               # [x1, y1, x2, y2]
            })
    return words, sec

def _words_to_text(words: list[dict]) -> str:
    # Simple reconstruction: join words with spaces and preserve newlines if large y-gaps
    # (Keep it simple for benchmarking; quality inspection can use the words list.)
    return " ".join(w["text"] for w in words)


# --- Main ---------------------------------------------------------------------

def main():
    print("--- Starting Vanilla Tesseract (Vietnamese) Benchmark ---")
    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_JSONL.resolve()}")

    # Configure tesseract binary path if provided
    _ensure_tesseract_cmd()

    lang = _to_tesseract_lang(LANGS)
    print(f"Using Tesseract language(s): {lang}")

    # Collect files
    files = [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")
    ]
    if not files:
        print("No files found. Exiting.")
        return
    print(f"Found {len(files)} files.")

    total_pages = 0
    total_cpu_render = 0.0   # PDF rasterization time
    total_ocr = 0.0          # Tesseract OCR time
    wall_start = time.perf_counter()

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for fpath in tqdm(files, desc="Processing Files"):
            try:
                if fpath.suffix.lower() == ".pdf":
                    with fitz.open(fpath) as doc:
                        for page_idx, page in enumerate(doc):
                            # CPU render
                            c0 = time.perf_counter()
                            pix = page.get_pixmap(dpi=TARGET_DPI)  # RGB
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            cpu_sec = time.perf_counter() - c0
                            total_cpu_render += cpu_sec

                            # OCR (Tesseract is CPU-only)
                            words, ocr_sec = _image_to_words(img, lang=lang)
                            total_ocr += ocr_sec

                            record = {
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "engine": "Tesseract",
                                "engine_lang": lang,
                                "source_path": str(fpath),
                                "page_index": page_idx,
                                "image_size": {"width": pix.width, "height": pix.height},
                                "metrics": {
                                    "cpu_render_seconds": round(cpu_sec, 4),
                                    "ocr_seconds": round(ocr_sec, 4),
                                },
                                "words": words,                         # [{text, confidence, bbox}]
                                "text": _words_to_text(words),          # quick reconstruction
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

                    words, ocr_sec = _image_to_words(img, lang=lang)
                    total_ocr += ocr_sec

                    record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "engine": "Tesseract",
                        "engine_lang": lang,
                        "source_path": str(fpath),
                        "page_index": 0,
                        "image_size": {"width": w, "height": h},
                        "metrics": {
                            "cpu_render_seconds": round(cpu_sec, 4),
                            "ocr_seconds": round(ocr_sec, 4),
                        },
                        "words": words,
                        "text": _words_to_text(words),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pages += 1

            except Exception as e:
                err = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "engine": "Tesseract",
                    "source_path": str(fpath),
                    "error": str(e),
                }
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")

    wall = time.perf_counter() - wall_start

    # Summary
    print("\n--- Vanilla Tesseract Benchmark Report ---")
    print(f"Total Files Processed: {len(files)}")
    print(f"Total Pages Processed: {total_pages}")
    print("-" * 30)
    print(f"Total Wall-Clock Time:      {wall:.2f} s")
    print(f"Total CPU Render Time:      {total_cpu_render:.2f} s")
    print(f"Total Tesseract OCR Time:   {total_ocr:.2f} s")
    print("-" * 30)
    if total_pages > 0:
        print(f"Average Time per Page:      {(wall / total_pages):.2f} s")
        print(f"Throughput:                 {(total_pages / wall):.2f} pages/s")
    print("--------------------------------")
    print(f"JSONL written to: {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
