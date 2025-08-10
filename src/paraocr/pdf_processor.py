# paraOCR/pdf_processor.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from PIL import Image
import fitz # PyMuPDF
import uuid # Moved import here

# --- Step 1: Define the Interface (The Contract) ---
class BasePDFProcessor(ABC):
    """
    Abstract Base Class defining the interface for any PDF processing engine.
    """
    @abstractmethod
    def get_native_text(self, file_path: Path) -> Optional[str]:
        """Extracts all text from a PDF, returns None on failure."""
        pass

    @abstractmethod
    def render_pages_to_images(self, file_path: Path, dpi: int, temp_dir: Path) -> List[str]:
        """Renders all pages of a PDF to temporary image files and returns their paths."""
        pass

# --- Step 2: Create the Concrete Implementation for PyMuPDF ---
class PyMuPDFProcessor(BasePDFProcessor):
    """A PDF processor that uses the PyMuPDF (Fitz) library."""

    def get_native_text(self, file_path: Path) -> Optional[str]:
        """
        Extracts text using a robust, layout-aware method ('blocks').
        This is much more reliable for complex PDFs than the simple 'text' method.
        """
        try:
            full_text = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    
                    blocks = page.get_text("blocks", sort=True) # Sort by position for reading order
                    page_text = [b[4] for b in blocks if b[6] == 0] # b[6] == 0 means it's a text block
                    full_text.append("\n".join(page_text))
            
            return "\n".join(full_text).strip()
        except Exception as e:
            # We can log this error if needed, but for now, we fail gracefully
            print(f"\n[Warning] PyMuPDF failed to extract native text from {file_path.name}: {e}")
            return None

    def render_pages_to_images(self, file_path: Path, dpi: int, temp_dir: Path) -> List[str]:
        """Renders all pages of a PDF to temporary image files and returns their paths."""
        temp_paths = []
        try:
            with fitz.open(file_path) as doc:
                if not len(doc):
                    return []
                for page in doc:
                    pix = page.get_pixmap(dpi=dpi)
                    temp_img_path = temp_dir / f"{uuid.uuid4()}.png"
                    pix.save(temp_img_path)
                    temp_paths.append(str(temp_img_path))
            return temp_paths
        except Exception as e:
            print(f"\n[Warning] PyMuPDF failed to render {file_path.name} to images: {e}")
            return [] # Return empty list on failure

# --- Step 3: Create a Factory to Select the Engine ---
def get_pdf_processor(engine_name: str = "pymupdf") -> BasePDFProcessor:
    """
    Factory function that returns an instance of the chosen PDF processor.
    """
    if engine_name.lower() == "pymupdf":
        return PyMuPDFProcessor()
    # In the future, you could add:
    # elif engine_name.lower() == "poppler":
    #     return PopplerProcessor()
    else:
        raise ValueError(f"Unknown PDF engine: '{engine_name}'. Supported engines: ['pymupdf']")