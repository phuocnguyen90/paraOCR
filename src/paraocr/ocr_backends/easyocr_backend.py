# paraocr/ocr_backends/easyocr_backend.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any
import os
import time
import logging
import errno
import warnings
import numpy as np
from PIL import Image

# EasyOCR / Torch can be absent in CPU-only envs; import lazily
import easyocr


from .base import BaseOCREngine

logger = logging.getLogger("paraocr")


# -----------------------------
# Helpers
# -----------------------------

def _as_bool(x, default=True) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return default


def _norm_langs_to_easyocr(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Accept languages or lang and normalize to a list for EasyOCR."""
    k = dict(kwargs or {})
    langs = k.pop("languages", None) or k.pop("lang", None)
    if isinstance(langs, str):
        langs = [langs]
    if not langs:
        langs = ["vi", "en"]
    k["languages"] = [str(l).strip() for l in langs if l]
    return k


def _ensure_rgb_uint8(img: Any) -> np.ndarray:
    """Accept PIL.Image | np.ndarray | path-like; return RGB uint8 numpy."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            # Rescale if it looks like float [0..1]
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8, copy=False)
        if arr.ndim == 2:  # grayscale
            return np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3]
        # Fallback via PIL
        return np.array(Image.fromarray(arr).convert("RGB"))
    # Treat as path-like
    with Image.open(img) as im:
        return np.array(im.convert("RGB"))


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _acquire_file_lock(lock_path: str, timeout: float = 120.0, poll: float = 0.2):
    """
    Lightweight, dependency-free inter-process lock using atomic file create.
    Prevents concurrent EasyOCR model downloads across multiple workers.
    """
    start = time.perf_counter()
    lock_dir = os.path.dirname(lock_path) or "."
    os.makedirs(lock_dir, exist_ok=True)
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            if time.perf_counter() - start > timeout:
                # Give up—better to proceed than deadlock
                logger.warning("Model init lock timeout; proceeding without lock: %s", lock_path)
                return None
            time.sleep(poll)


def _release_file_lock(fd, lock_path: str):
    try:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if lock_path and os.path.exists(lock_path):
            os.unlink(lock_path)
    except Exception:
        # Non-fatal
        logger.debug("Failed to release lock: %s", lock_path, exc_info=True)


# -----------------------------
# Backend
# -----------------------------

class EasyOCREngine(BaseOCREngine):
    """
    Stable EasyOCR adapter for paraOCR.

    Supported kwargs (all optional):
      - languages / lang: list[str] | str (default ["vi","en"])
      - gpu / use_gpu: bool (defaults to True if CUDA is available, else False)
      - model_storage_directory: str (shared cache dir recommended)
      - user_network_directory: str
      - download_enabled: bool (default True)
      - recog_network, detector, recognizer, verbose, quantize, decoder, beam_width
      - (ignored safely if present): unknown keys are forwarded to Reader when possible
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        k = _norm_langs_to_easyocr(kwargs)

        # GPU choice with safe fallback
        want_gpu = _as_bool(k.pop("gpu", k.pop("use_gpu", True)), True)
        use_gpu = bool(want_gpu and _torch_cuda_available())

        # Reader options
        model_dir = k.pop("model_storage_directory", None)
        user_net_dir = k.pop("user_network_directory", None)
        download_enabled = _as_bool(k.pop("download_enabled", True), True)
        recog_network = k.pop("recog_network", "standard")
        detector = _as_bool(k.pop("detector", True), True)
        recognizer = _as_bool(k.pop("recognizer", True), True)
        verbose = _as_bool(k.pop("verbose", False), False)
        quantize = _as_bool(k.pop("quantize", False), False)

        # Decoder preferences for recognition; we pass them to readtext()
        # EasyOCR supports decoder='greedy' or 'beamsearch' with beamWidth
        decoder = str(k.pop("decoder", "greedy")).strip().lower()
        if decoder not in ("greedy", "beamsearch"):
            decoder = "greedy"
        beam_width = k.pop("beam_width", k.pop("beamWidth", 10))
        try:
            beam_width = int(beam_width)
        except Exception:
            beam_width = 10

        # Prevent concurrent model downloads via a simple inter-process lock
        # (safe to use a shared directory if provided; else falls back to ~/.cache)
        cache_root = model_dir or os.path.join(os.path.expanduser("~"), ".cache", "easyocr")
        lock_path = os.path.join(cache_root, "model_init.lock")
        fd = None
        try:
            fd = _acquire_file_lock(lock_path, timeout=180.0)

            # Init Reader (guarded)
            langs = k.pop("languages")
            self.reader = easyocr.Reader(
                langs,
                gpu=use_gpu,
                model_storage_directory=model_dir,
                user_network_directory=user_net_dir,
                recog_network=recog_network,
                download_enabled=download_enabled,
                detector=detector,
                recognizer=recognizer,
                verbose=verbose,
                quantize=quantize,
                # Forward any additional, compatible kwargs (best effort)
                **{kk: vv for kk, vv in k.items() if kk not in ("beamsearch",)}
            )
        except Exception as e:
            # Fallback: try CPU if GPU init failed
            if use_gpu:
                logger.warning("EasyOCR GPU init failed, falling back to CPU: %s", e)
                self.reader = easyocr.Reader(
                    k.get("languages", ["vi", "en"]),
                    gpu=False,
                    model_storage_directory=model_dir,
                    user_network_directory=user_net_dir,
                    recog_network=recog_network,
                    download_enabled=download_enabled,
                    detector=detector,
                    recognizer=recognizer,
                    verbose=verbose,
                    quantize=quantize,
                )
            else:
                # Re-raise on CPU failure
                raise
        finally:
            _release_file_lock(fd, lock_path)

        # Store decode prefs for read calls
        self._decoder = (decoder or "greedy").lower()
        if self._decoder not in ("greedy", "beamsearch"):
            self._decoder = "greedy"
        try:
            self._beam_width = int(beam_width)
        except Exception:
            self._beam_width = 10
        # clamp to a modest range to reduce numeric issues and CPU overhead
        self._beam_width = max(1, min(self._beam_width, 20))

    def read_batch(self, images: List[np.ndarray]) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        outputs: List[str] = []

        for idx, im in enumerate(images):
            try:
                rgb = _ensure_rgb_uint8(im)

                # main attempt (suppress runtime overflow spam)
                with np.errstate(over="ignore", invalid="ignore"):
                    with warnings.catch_warnings(record=True) as wlist:
                        warnings.simplefilter("always", category=RuntimeWarning)
                        lines = self.reader.readtext(
                            rgb,
                            detail=0,
                            paragraph=True,
                            decoder=self._decoder,
                            beamWidth=(self._beam_width if self._decoder == "beamsearch" else None),
                        )

                    # detect numeric warnings during beam search and retry greedily
                    if self._decoder == "beamsearch" and any(
                        ("overflow" in str(w.message).lower()) for w in wlist
                    ):
                        logger.warning("EasyOCR beamsearch overflow detected; retrying with greedy decoder on item %d", idx)
                        lines = self.reader.readtext(
                            rgb,
                            detail=0,
                            paragraph=True,
                            decoder="greedy",
                            beamWidth=None,
                        )

                if isinstance(lines, (list, tuple)):
                    outputs.append("\n".join([str(x) for x in lines if x]))
                else:
                    outputs.append(str(lines) if lines else "")

            except Exception as e:
                logger.error("EasyOCR failed on batch item %d: %s", idx, e, exc_info=True)
                outputs.append("")  # keep alignment

        return outputs, round(time.perf_counter() - t0, 6)
