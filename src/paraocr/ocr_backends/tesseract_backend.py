# paraocr/ocr_backends/tesseract_backend.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any
import os
import time

import numpy as np
from PIL import Image
import pytesseract as pt
# set default tesseract cmd
# utils_tesseract.py
import os, platform, shutil
from pathlib import Path

import re  # NEW

def _as_int(x, default: int) -> int:
    try:
        if isinstance(x, str):
            x = x.strip().rstrip(",}] ")
        return int(x)
    except Exception:
        m = re.search(r"-?\d+", str(x))
        return int(m.group()) if m else default
    

def resolve_tesseract_cmd() -> str | None:
    # 1) explicit env override
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and Path(cmd).exists():
        return cmd

    # 2) look on PATH
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd

    # 3) common fallbacks by OS
    system = platform.system()
    if system == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            "/opt/homebrew/bin/tesseract",   # Apple Silicon Homebrew
            "/usr/local/bin/tesseract",      # Intel Homebrew/MacPorts
        ]
    else:  # Linux and others
        candidates = [
            "/usr/bin/tesseract",            # apt/yum default
            "/usr/local/bin/tesseract",      # source install
            "/snap/bin/tesseract",           # snap
        ]

    for p in candidates:
        if Path(p).exists():
            return p
    return None

cmd = resolve_tesseract_cmd()
if cmd:
    pt.pytesseract.tesseract_cmd = cmd

from .base import BaseOCREngine


# Map your common codes to Tesseract's traineddata names
_TESS_LANG_MAP = {
    "vi": "vie",
    "en": "eng",
}


def _norm_langs_to_tesseract(kwargs: Dict[str, Any]) -> str:
    # Accept languages or lang; allow str or list
    langs = kwargs.pop("languages", None) or kwargs.pop("lang", None)
    if isinstance(langs, str):
        langs = [langs]
    if not langs:
        langs = ["vi", "en"]  # sensible default for your use case
    codes = [_TESS_LANG_MAP.get(str(l).lower(), str(l).lower()) for l in langs]
    return "+".join(sorted(set(codes)))


class TesseractOCREngine(BaseOCREngine):
    """
    Pytesseract-based backend.

    Kwargs supported (all optional):
      - languages / lang: list[str] or str → mapped to "eng", "vie", ...
      - tesseract_cmd: full path to tesseract binary (Windows)
      - tessdata_prefix: path to tessdata directory
      - oem: 0..3 (default 3 = LSTM)
      - psm: page segmentation mode (default 6 = uniform block of text)
      - preserve_interword_spaces: bool (default True)
      - extra_config: str of extra flags (appended to config string)
      - (ignored safely if present): gpu, use_gpu, beamsearch, model_storage_directory, download_enabled
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        k = dict(kwargs)  # don't mutate caller's dict

        # Ignore GPU-related flags (may be injected by the pipeline defaults)
        for junk in ("gpu", "use_gpu", "beamsearch", "model_storage_directory", "download_enabled"):
            k.pop(junk, None)

        # Binary path / tessdata (Windows-friendly)
        tesseract_cmd = k.pop("tesseract_cmd", None) or k.pop("tesseract_path", None)
        if tesseract_cmd:
            pt.pytesseract.tesseract_cmd = str(tesseract_cmd)
            if not os.path.exists(pt.pytesseract.tesseract_cmd):
                raise RuntimeError(f"Tesseract binary not found: {pt.pytesseract.tesseract_cmd}")



        tessdata_prefix = k.pop("tessdata_prefix", None)
        if tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_prefix)

        # Language(s)
        self.lang = _norm_langs_to_tesseract(k)

        # Tesseract config
        oem = _as_int(k.pop("oem", 3), 3)
        psm = _as_int(k.pop("psm", 6), 6)
        preserve_spaces = bool(k.pop("preserve_interword_spaces", True))
        extra_cfg = str(k.pop("extra_config", "")).strip()

        cfg_parts = [f"--oem {oem}", f"--psm {psm}"]
        if preserve_spaces:
            cfg_parts.append("-c preserve_interword_spaces=1")
        if extra_cfg:
            cfg_parts.append(extra_cfg)
        self._config = " ".join(cfg_parts)

        # Warm check (not critical; will raise later if missing)
        try:
            _ = pt.get_tesseract_version()
        except Exception:
            pass

    def _to_pil(self, img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            # assume RGB/BGR-ish uint8; pytesseract works fine with RGB PIL
            if img.ndim == 2:
                return Image.fromarray(img)
            # Heuristic: if last dim is 3/4 treat as RGB
            return Image.fromarray(img[..., :3])
        # As a last resort, try opening via PIL (path-like)
        return Image.open(img).convert("RGB")

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        start = time.perf_counter()
        texts: List[str] = []
        for im in images:
            pil_im = self._to_pil(im)
            txt = pt.image_to_string(pil_im, lang=self.lang, config=self._config)
            texts.append(txt.strip())
        return texts, time.perf_counter() - start
