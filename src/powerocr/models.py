# powerocr/models.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict

@dataclass
class OCRTask:
    """Represents a single file to be processed."""
    source_path: Path


@dataclass
class PageTask:
    """Represents a single page to be processed."""
    source_path: str
    page_num: int
    total_pages: int
    dpi: int
    temp_dir: str
    # This field will be added by the dispatcher
    processing_type: str = "unknown"
    
@dataclass
class OCRResult:
    """Represents the final, aggregated output for a single document."""
    source_path: str
    total_pages: int
    # Content will be a list, with each element being the result for a page
    content: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
