# paraocr/ocr_backends/paddleocr_backend.py
from typing import List, Tuple, Any, Dict
import time
import numpy as np
import cv2
from paddleocr import PaddleOCR

from .base import BaseOCREngine

class PaddleOCREngine(BaseOCREngine):
    """
    Thin adapter over PaddleOCR. Works with your read_batch interface.
    Notes:
      lang='en' uses the Latin recognition model which already covers Vietnamese characters.
      If you have a dedicated Vietnamese model, pass lang='vi'.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        use_angle_cls: bool = False,
        rec_batch_num: int = 6,
        **kwargs: Dict[str, Any],
    ):
        # PaddleOCR expects BGR images. We convert in read_batch.
        # show_log=False keeps console quieter for your UI streaming.
        self.ocr = PaddleOCR(
            lang=lang,
            use_gpu=use_gpu,
            use_angle_cls=use_angle_cls,
            rec_batch_num=rec_batch_num,
            show_log=False,
            **kwargs,
        )

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        texts: List[str] = []

        for img in images:
            # Convert RGB to BGR if needed
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img

            # Paddle returns list of pages. Each page is a list of lines.
            # Each line is [box, (text, score)]
            result = self.ocr.ocr(bgr, cls=False)

            lines: List[str] = []
            if isinstance(result, list):
                for page in result:
                    if page is None:
                        continue
                    for line in page:
                        try:
                            txt = line[1][0]
                            lines.append(txt)
                        except Exception:
                            continue

            texts.append("\n".join(lines))

        dur = round(time.perf_counter() - start, 6)
        return texts, dur
