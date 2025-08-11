# paraocr/ocr_backends/paddleocr_backend.py
from typing import List, Tuple, Any, Dict
import time
import numpy as np
import cv2
from paddleocr import PaddleOCR

from .base import BaseOCREngine

def _norm_langs_to_paddle(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Accept languages or lang, then pick a single Paddle lang
    langs = kwargs.pop("languages", None) or kwargs.get("lang")
    if isinstance(langs, list):
        lang = "vi" if "vi" in langs else langs[0]
    elif isinstance(langs, str) and langs:
        lang = langs
    else:
        lang = "vi"  # default for your scenario
    kwargs["lang"] = lang
    return kwargs

class PaddleOCREngine(BaseOCREngine):
    def __init__(self, **kwargs: Dict[str, Any]):
        k = _norm_langs_to_paddle(dict(kwargs))
        use_angle_cls = k.pop("use_angle_cls", True)
        rec_batch_num = k.pop("rec_batch_num", 6)
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            rec_batch_num=rec_batch_num,
            show_log=False,
            **k
        )

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        texts: List[str] = []
        for img in images:
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
            result = self.ocr.ocr(bgr, cls=False)
            lines = []
            if isinstance(result, list):
                for page in result or []:
                    if page is None:
                        continue
                    for line in page:
                        try:
                            lines.append(line[1][0])
                        except Exception:
                            pass
            texts.append("\n".join(lines))
        return texts, round(time.perf_counter() - start, 6)
