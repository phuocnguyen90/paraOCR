# paraOCR/processors.py

import time
from pathlib import Path
import fitz
from PIL import Image
import uuid

# --- 1. The Dispatcher Worker (Analyzes and Tags) ---
def worker_dispatcher(page_task: dict) -> dict:
    """
    CPU-bound worker that performs quick layout analysis and tags the page task.
    """
    file_path_str = page_task["source_path"]
    page_num = page_task["page_num"]
    
    try:
        with fitz.open(file_path_str) as doc:
            page = doc.load_page(page_num)
            
            # This logic can be expanded in the future.
            # For now, we are primarily identifying text pages.
            # We can add table/image detection here later.
            if len(page.find_tables().tables) > 0:
                page_task["processing_type"] = "table"
                return page_task

            # Default to text OCR
            page_task["processing_type"] = "text_ocr"
            return page_task

    except Exception as e:
        page_task["processing_type"] = "error"
        page_task["error"] = f"Dispatcher failed on page {page_num}: {e}"
        return page_task

# --- 2. Specialized Workers ---
def worker_render_text_page(page_task: dict) -> dict:
    """Renders a single page to a temporary image file for OCR."""
    try:
        file_path = Path(page_task["source_path"])
        temp_dir = Path(page_task["temp_dir"])
        temp_img_path = temp_dir / f"{uuid.uuid4()}.png"
        
        if file_path.suffix.lower() == '.pdf':
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_task["page_num"])
                pix = page.get_pixmap(dpi=page_task["dpi"])
                pix.save(temp_img_path)
        else: # Handle images
            Image.open(file_path).convert("RGB").save(temp_img_path, format='PNG')

        page_task["temp_path"] = str(temp_img_path)
        return page_task
    except Exception as e:
        page_task["error"] = f"Text page rendering failed: {e}"
        return page_task

def worker_process_table_page(page_task: dict) -> dict:
    """Future stub for a table extraction worker."""
    # In a real implementation, this would use a library like Camelot
    # and return structured data (e.g., a list of lists).
    page_task["content"] = [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]]
    page_task["content_type"] = "table"
    return page_task

def worker_process_image_page(page_task: dict) -> dict:
    """Future stub for an image extraction worker."""
    page_task["content"] = f"/path/to/extracted/image_from_page_{page_task['page_num']}.png"
    page_task["content_type"] = "image"
    return page_task