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
    """
    Make kwargs compatible across PaddleOCR 2.x and 3.x:
    - languages/lang -> single 'lang'
    - GPU: map to 'device' ('gpu:0' or 'cpu') for 3.x, 'use_gpu' (bool) for 2.x
    - Batch size: map 'rec_batch_num' <-> 'text_recognition_batch_size'
    - Angle cls: map 'use_angle_cls' <-> 'use_textline_orientation'
    - Return ONE dict ready for PaddleOCR(**dict)
    """
    import paddle
    try:
        from paddleocr import __version__ as _pocv
        _maj = int(str(_pocv).split(".", 1)[0])
    except Exception:
        _maj = 2  # assume 2.x if unknown

    k_in = dict(kwargs or {})

    # ----- language -----
    langs = k_in.pop("languages", None) or k_in.pop("lang", None) or ["vi", "en"]
    lang = _select_paddle_lang(langs)  # single string for Paddle

    # ----- device / gpu -----
    device = k_in.pop("device", None)
    want_gpu = k_in.pop("use_gpu", k_in.pop("gpu", None))
    if device is None:
        try:
            compiled = paddle.device.is_compiled_with_cuda()
            devs = paddle.device.cuda.device_count() if compiled else 0
            if want_gpu is None:
                want_gpu = bool(compiled and devs > 0)
            else:
                want_gpu = bool(want_gpu and compiled and devs > 0)
        except Exception:
            want_gpu = bool(want_gpu)
        device = "gpu:0" if want_gpu else "cpu"

    # ----- batch size -----
    bs = k_in.pop("text_recognition_batch_size", k_in.pop("rec_batch_num", 6))
    try:
        bs = int(bs)
    except Exception:
        bs = 6

    # ----- angle classifier -----
    angle = k_in.pop("use_textline_orientation", k_in.pop("use_angle_cls", True))
    angle = bool(angle)

    # ----- logging -----
    show_log = bool(k_in.pop("show_log", False))

    # Build the version-appropriate dict
    if _maj >= 3:
        out = {
            "lang": lang,
            "device": device,  # 'gpu:0' or 'cpu'
            "text_recognition_batch_size": bs,
            "use_textline_orientation": angle,
            "show_log": show_log,
        }
    else:
        out = {
            "lang": lang,
            "use_gpu": device.startswith("gpu"),
            "rec_batch_num": bs,
            "use_angle_cls": angle,
            "show_log": show_log,
        }

    # Allowlist a few extras if present (don’t forward unknowns that 3.x might reject)
    allow = [
        "det_model_dir", "rec_model_dir", "cls_model_dir",
        "det_db_box_thresh", "det_db_unclip_ratio",
        "max_text_length", "det_limit_side_len", "det_limit_type",
    ]
    for kk in allow:
        if kk in k_in:
            out[kk] = k_in[kk]

    return out


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
