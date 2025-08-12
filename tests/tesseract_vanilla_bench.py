# Single-threaded Tesseract benchmark (Vietnamese) + JSONL export

import os
import json
import time
import argparse  # Added for CLI arguments
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from tqdm import tqdm

import pytesseract as pt

# --- CONFIGURATION ---
DEFAULT_INPUT_DIR = Path("tests/data")                      # Default folder with PDFs/images
DEFAULT_OUTPUT_JSONL = Path("tesseract_vi_results.jsonl")   # Default JSONL output path
TARGET_DPI = 200                                            # PDF render DPI

# Tesseract settings
TESSERACT_OEM = 3
TESSERACT_PSM = 6
LANGS = ["vi", "en"]  # Mapped to 'vie+eng' below

# If Tesseract is not on PATH, set the binary path here or via env TESSERACT_CMD
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or None  # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --- Helpers ------------------------------------------------------------------

_TESS_MAP = {
    "vi": "vie", "en": "eng", "fr": "fra", "de": "deu",
    "es": "spa", "it": "ita", "pt": "por", "nl": "nld",
}

def _to_tesseract_lang(langs) -> str:
    if not langs: return "vie"
    codes = [_TESS_MAP.get(l.lower(), l.lower()) for l in langs]
    return "+".join(sorted(set(codes)))

def _ensure_tesseract_cmd():
    if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
        pt.pytesseract.tesseract_cmd = TESSERACT_CMD
    # On Colab, pytesseract often finds the binary automatically if installed via apt-get

def _image_to_words(img: Image.Image, lang: str) -> tuple[list[dict], float]:
    """Runs Tesseract and returns structured word data."""
    cfg = f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}"
    t0 = time.perf_counter()
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
                "confidence": float(conf) / 100.0,
                "bbox": [x, y, x + w, y + h],
            })
    return words, sec

def _words_to_text(words: list[dict]) -> str:
    """Simple text reconstruction from word list."""
    return " ".join(w["text"] for w in words)


# --- Main ---------------------------------------------------------------------

def main():
    # 1. Set up and parse command-line arguments
    parser = argparse.ArgumentParser(
        description="A simple, single-threaded, sequential OCR script for benchmarking Tesseract.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing files to OCR.")
    parser.add_argument("-o", "--output-path", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Path to save the output results JSONL.")
    args = parser.parse_args()

    # Use the parsed arguments
    input_dir = args.input_dir
    output_jsonl = args.output_path
    
    print("--- Starting Vanilla Tesseract Benchmark ---")
    
    if not input_dir.exists():
        print(f"[Error] Input directory does not exist: {input_dir.resolve()}")
        return

    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_jsonl.resolve()}")

    _ensure_tesseract_cmd()
    lang = _to_tesseract_lang(LANGS)
    print(f"Using Tesseract language(s): {lang}")

    # 2. Collect files
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")]
    if not files:
        print(f"No files found in {input_dir.resolve()}. Exiting.")
        return
    print(f"Found {len(files)} files.")

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

                            words, ocr_sec = _image_to_words(img, lang=lang)
                            total_ocr += ocr_sec

                            record = {"source_path": str(fpath), "page_index": page_idx, "words": words}
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_pages += 1
                else: # Image file
                    c0 = time.perf_counter()
                    img = Image.open(fpath).convert("RGB")
                    total_cpu_render += time.perf_counter() - c0

                    words, ocr_sec = _image_to_words(img, lang=lang)
                    total_ocr += ocr_sec

                    record = {"source_path": str(fpath), "page_index": 0, "words": words}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pages += 1

            except Exception as e:
                err = {"source_path": str(fpath), "error": str(e)}
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")

    wall = time.perf_counter() - wall_start

    # 3. Summary
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
    print(f"JSONL written to: {output_jsonl.resolve()}")


if __name__ == "__main__":
    main()