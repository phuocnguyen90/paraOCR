# src/paraocr/processors.py
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import fitz
from PIL import Image

logger = logging.getLogger("paraocr")


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
    Render one page to a temporary PNG to be consumed by the GPU OCR backend.
    Returns the task dict with 'temp_path' and 'duration_seconds'.
    """
    start = time.perf_counter()
    try:
        file_path = Path(page_task["source_path"])
        temp_dir = Path(page_task["temp_dir"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_img_path = temp_dir / f"{uuid.uuid4()}.png"

        if file_path.suffix.lower() == ".pdf":
            page_num = int(page_task["page_num"])
            dpi = int(page_task["dpi"])
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)
                pix.save(temp_img_path)
        else:
            # For image inputs, normalize to RGB and save as PNG
            with Image.open(file_path) as im:
                im.convert("RGB").save(temp_img_path, format="PNG")

        page_task["temp_path"] = str(temp_img_path)
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
