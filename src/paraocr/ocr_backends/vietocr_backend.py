from typing import List, Tuple, Dict, Any
import time
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from .base import BaseOCREngine

class VietOCREngine(BaseOCREngine):
    """
    Adapter for VietOCR. Good baseline for Vietnamese text.
    """
    def __init__(self, config_name: str = "vgg_transformer", device: str = "cuda", **overrides: Dict[str, Any]):
        cfg = Cfg.load_config_from_name(config_name)
        cfg["device"] = device
        # common tweaks
        cfg["weights"] = overrides.get("weights", cfg["weights"])
        cfg["predictor"]["beamsearch"] = overrides.get("beamsearch", False)
        self.detector = None  # VietOCR does recognition only
        self.recognizer = Predictor(cfg)

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        out = []
        for arr in images:
            pil = Image.fromarray(arr)
            txt = self.recognizer.predict(pil)
            out.append(txt or "")
        dur = time.perf_counter() - start
        return out, dur
