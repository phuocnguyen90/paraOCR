# paraocr/ocr_backends/easyocr_backend.py
from typing import List, Tuple
import time
import numpy as np
from PIL import Image
import easyocr
from typing import List, Dict, Any, Tuple

from .base import BaseOCREngine

def _norm_langs_to_easyocr(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Accept languages or lang as inputs, prefer list
    langs = kwargs.pop("languages", None) or kwargs.pop("lang", None)
    if isinstance(langs, str):
        langs = [langs]
    if not langs:
        langs = ["vi", "en"]  # sensible default for your use case
    # EasyOCR wants a list
    kwargs["languages"] = langs
    return kwargs

class EasyOCREngine(BaseOCREngine):
    def __init__(self, **kwargs: Dict[str, Any]):
        k = _norm_langs_to_easyocr(dict(kwargs))
        use_gpu = k.pop("gpu", True)
        langs = k.pop("languages")
        beamsearch = k.pop("beamsearch", False)
        self.reader = easyocr.Reader(langs, gpu=use_gpu)
        if beamsearch:
            self.reader.beamsearch = True

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        out = []
        for arr in images:
            if isinstance(arr, Image.Image):
                arr = np.array(arr)
            lines = self.reader.readtext(arr, detail=0, paragraph=True)
            out.append("\n".join(lines))
        return out, time.perf_counter() - start