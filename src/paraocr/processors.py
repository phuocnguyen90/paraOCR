# src/paraocr/processors.py
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
import hashlib

import fitz
from PIL import Image

logger = logging.getLogger("paraocr")

# --- 0. Helpers ---
def _render_png_path(temp_dir: Path, key: str) -> Path:
    return temp_dir / "cache" / "render" / key[:2] / f"{key}.png"
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)



# --- 1. Dispatcher worker ---
def worker_dispatcher(page_task: dict) -> dict:
    """
    Quick layout analysis and tagging for a single page.
    Returns the same dict with a 'processing_type' key set to one of
    'text_ocr', 'table', 'image', or 'error'.
    """
    start = time.perf_counter()
    file_path_str = page_task.get("source_path")
    page_num = int(page_task.get("page_num", 0))

    try:
        with fitz.open(file_path_str) as doc:
            page = doc.load_page(page_num)

            # Default to text
            processing_type = "text_ocr"

            # Table detection is relatively cheap but still non zero
            # Only call it for PDFs and when the page has a reasonable amount of text boxes
            try:
                tables = page.find_tables()
                if getattr(tables, "tables", []):
                    processing_type = "table"
            except Exception:
                # Best effort, stick to text if table detection fails
                logger.debug("Table detection failed on %s page %s", file_path_str, page_num)

            page_task["processing_type"] = processing_type
            return page_task

    except Exception as e:
        page_task["processing_type"] = "error"
        page_task["error"] = f"Dispatcher failed on page {page_num}, {e}"
        return page_task
    finally:
        page_task["duration_seconds"] = time.perf_counter() - start


# --- 2. Specialized workers ---
def worker_render_text_page(page_task: dict) -> dict:
    """
    Render one page to a PNG (deterministic cache location if cache_key provided)
    so the GPU OCR backend can consume it. Returns the task dict with:
      - 'temp_path' (str): path to the PNG (cached or newly rendered)
      - 'duration_seconds' (float)
      - 'error' (str) on failure
    Expected keys in page_task:
      source_path, page_num, dpi, temp_dir, (optional) cache_key
    """
    start = time.perf_counter()
    file_path = Path(page_task["source_path"])
    temp_dir  = Path(page_task["temp_dir"])
    dpi       = int(page_task.get("dpi", 200))
    page_num  = int(page_task.get("page_num", 0))
    cache_key = page_task.get("cache_key")  # runner should set this

    # deterministic cache path if key available
    png_cache_path = _render_png_path(temp_dir, cache_key) if cache_key else None

    try:
        # 0) Fast path: reuse cached render if available
        if png_cache_path and png_cache_path.exists():
            page_task["temp_path"] = str(png_cache_path)
            return page_task

        # 1) Ensure dirs
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = png_cache_path or (temp_dir / f"{uuid.uuid4()}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 2) Render
        if file_path.suffix.lower() == ".pdf":
            # Prefer matrix-based scaling (consistent across PyMuPDF versions)
            zoom = dpi / 72.0
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                pix.save(str(out_path))
        else:
            # Normalize images to RGB PNG (optionally you can resample to a target DPI)
            with Image.open(file_path) as im:
                im.convert("RGB").save(out_path, format="PNG")

        page_task["temp_path"] = str(out_path)
        return page_task

    except Exception as e:
        page_task["error"] = f"Text page rendering failed, {e}"
        return page_task

    finally:
        page_task["duration_seconds"] = time.perf_counter() - start

def worker_process_table_page(page_task: dict) -> dict:
    """
    Future placeholder for table extraction.
    Would return structured data such as CSV like rows.
    """
    start = time.perf_counter()
    try:
        page_task["content"] = [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]]
        page_task["content_type"] = "table"
        return page_task
    finally:
        page_task["duration_seconds"] = time.perf_counter() - start


def worker_process_image_page(page_task: dict) -> dict:
    """
    Future placeholder for image extraction.
    """
    start = time.perf_counter()
    try:
        page_task["content"] = f"/path/to/extracted/image_from_page_{page_task.get('page_num')}.png"
        page_task["content_type"] = "image"
        return page_task
    finally:
        page_task["duration_seconds"] = time.perf_counter() - start
