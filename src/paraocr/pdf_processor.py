# src/paraocr/pdf_processor.py
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image  # noqa, reserved for future use

logger = logging.getLogger("paraocr")


# --- Step 1, interface ---
class BasePDFProcessor(ABC):
    """
    Interface for any PDF processing engine.
    """

    @abstractmethod
    def get_native_text(self, file_path: Path) -> Optional[str]:
        """Extracts all text from a PDF, returns None on failure."""
        raise NotImplementedError

    @abstractmethod
    def render_pages_to_images(self, file_path: Path, dpi: int, temp_dir: Path) -> List[str]:
        """Renders all pages of a PDF to temporary image files and returns their paths."""
        raise NotImplementedError


# --- Step 2, concrete implementation with PyMuPDF ---
class PyMuPDFProcessor(BasePDFProcessor):
    """PDF processor that uses PyMuPDF."""

    def get_native_text(self, file_path: Path) -> Optional[str]:
        """
        Extract text using layout aware blocks.
        This is usually more reliable for complex PDFs than the plain text mode.
        """
        try:
            full_text_parts: List[str] = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    # sort=True gives reading order
                    blocks = page.get_text("blocks", sort=True)
                    # b[6] == 0 means text block
                    page_text = [b[4] for b in blocks if len(b) > 6 and b[6] == 0]
                    if page_text:
                        full_text_parts.append("\n".join(page_text))
            text = "\n".join(full_text_parts).strip()
            if not text:
                logger.debug("Native text extractor returned empty for %s", file_path)
                return None
            return text
        except Exception as e:
            logger.warning("PyMuPDF failed to extract native text from %s, %s", file_path.name, e)
            return None

    def render_pages_to_images(self, file_path: Path, dpi: int, temp_dir: Path) -> List[str]:
        """
        Render each page to a PNG image on disk and return the paths.
        """
        temp_paths: List[str] = []
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            with fitz.open(file_path) as doc:
                n = len(doc)
                if n == 0:
                    logger.debug("PDF has zero pages, %s", file_path)
                    return []
                for page in doc:
                    pix = page.get_pixmap(dpi=dpi)
                    temp_img_path = temp_dir / f"{uuid.uuid4()}.png"
                    pix.save(temp_img_path)  # pixmap owns its buffer, write directly
                    temp_paths.append(str(temp_img_path))
            return temp_paths
        except Exception as e:
            logger.warning("PyMuPDF failed to render %s to images, %s", file_path.name, e)
            return []  # fail soft so the caller can decide how to proceed


# --- Step 3, factory ---
def get_pdf_processor(engine_name: str = "pymupdf") -> BasePDFProcessor:
    """
    Create a PDF processor by name.
    """
    name = (engine_name or "").lower()
    if name == "pymupdf":
        return PyMuPDFProcessor()
    raise ValueError(f"Unknown PDF engine, '{engine_name}'. Supported engines, ['pymupdf']")
