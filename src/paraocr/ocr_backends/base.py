# paraocr/ocr_backends/base.py
from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np

class BaseOCREngine(ABC):
    @abstractmethod
    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        """Return list of texts and total duration in seconds."""
        pass
