# powerocr/models.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class OCRTask:
    """Represents a single file to be processed."""
    source_path: Path
    
@dataclass
class OCRResult:
    """Represents the output for a single processed file."""
    source_path: str
    text: Optional[str]
    method: str
    error: Optional[str] = None
