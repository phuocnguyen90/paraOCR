# paraocr/ocr_backends/paddleocr_backend.py
from __future__ import annotations

from typing import List, Tuple, Any, Dict
import time
import numpy as np

# PaddleOCR -> imgaug expects NumPy 1.x; if you're on NumPy 2.x this shim avoids import errors.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "float":   [np.float16, np.float32, np.float64],
        "int":     [np.int8, np.int16, np.int32, np.int64],
        "uint":    [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128],
        "others":  [np.bool_, np.bytes_, np.str_],
    }

import cv2
from PIL import Image
from paddleocr import PaddleOCR
import paddle

from .base import BaseOCREngine


def _select_paddle_lang(langs: Any) -> str:
    """
    PaddleOCR uses a single 'lang' per instance.
    For Vietnamese we generally get better coverage from 'latin'.
    """
    if isinstance(langs, str):
        arr = [langs]
    elif isinstance(langs, (list, tuple, set)):
        arr = list(langs)
    else:
        arr = ["vi", "en"]
    arr = [str(x).lower() for x in arr if x]

    if "en" in arr:
        return "en"
    if "vi" in arr:
        return "latin"
    if any(x in arr for x in ("fr", "de", "es", "it", "pt", "ro", "nl", "sv", "da", "no", "pl", "cs", "sk", "hu")):
        return "latin"
    return "en"


def _norm_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    k = dict(kwargs or {})

    # languages/lang -> single Paddle lang
    langs = k.pop("languages", None) or k.pop("lang", None) or ["vi", "en"]
    k["lang"] = _select_paddle_lang(langs)

    # GPU flags: accept gpu/use_gpu; auto-fallback to CPU if CUDA wheel/devices not available
    use_gpu = k.pop("use_gpu", k.pop("gpu", True))
    try:
        compiled = paddle.device.is_compiled_with_cuda()
        devs = paddle.device.cuda.device_count() if compiled else 0
        if not (use_gpu and compiled and devs > 0):
            use_gpu = False
    except Exception:
        use_gpu = False if use_gpu is None else bool(use_gpu)
    k["use_gpu"] = bool(use_gpu)

    # Reasonable defaults; allow override via kwargs
    k.setdefault("use_angle_cls", True)
    try:
        if "rec_batch_num" in k:
            k["rec_batch_num"] = int(k["rec_batch_num"])
        else:
            k["rec_batch_num"] = 6
    except Exception:
        k["rec_batch_num"] = 6
    k.setdefault("show_log", False)
    return k


def _ensure_bgr(img: Any) -> np.ndarray:
    """Accept PIL.Image | np.ndarray | path-like; return OpenCV BGR uint8."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            # Heuristic: assume RGB; convert to BGR
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    # Path-like fallback
    pil = Image.open(img).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class PaddleOCREngine(BaseOCREngine):
    def __init__(self, **kwargs: Dict[str, Any]):
        k = _norm_kwargs(kwargs)
        self.ocr = PaddleOCR(**k)

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        outs: List[str] = []
        for img in images:
            bgr = _ensure_bgr(img)
            # cls=True leverages angle classifier if enabled at init
            result = self.ocr.ocr(bgr, cls=True)
            lines: List[str] = []
            # result shape: [[ [box, (text, conf)], ... ]]
            if isinstance(result, list):
                groups = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
                for item in groups or []:
                    try:
                        txt = item[1][0]
                        if txt:
                            lines.append(str(txt))
                    except Exception:
                        pass
            outs.append("\n".join(lines))
        return outs, round(time.perf_counter() - start, 6)
