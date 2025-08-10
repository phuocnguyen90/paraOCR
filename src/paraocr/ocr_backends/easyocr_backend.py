# paraocr/ocr_backends/easyocr_backend.py
from typing import List, Tuple
import time
import numpy as np
import easyocr

from .base import BaseOCREngine

class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages=None, beamsearch=False, gpu=True):
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=gpu)
        if beamsearch:
            self.reader.beamsearch = True

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        out = []
        for img in images:
            result = self.reader.readtext(img, detail=0, paragraph=True)
            out.append("\n".join(result))
        dur = time.perf_counter() - start
        return out, dur
