# src/paraocr/gpu_worker.py
from __future__ import annotations

import logging
import os
import time
import importlib
from typing import List, Tuple, Any

import numpy as np
from PIL import Image
from .logger import configure_worker_logging 

logger = logging.getLogger("paraocr")

# One engine instance per worker process
ocr_engine: Any | None = None


def _import_obj(dotted: str):
    mod_path, _, attr = dotted.rpartition(".")
    if not mod_path or not attr:
        raise ImportError(f"Invalid backend path, {dotted}")
    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ImportError(f"Backend class not found, {dotted}") from e


def initialize_gpu_worker(log_queue, backend_path: str, backend_kwargs: dict):
    """
    Called once in each GPU worker process.
    Loads the backend class and creates the engine instance.
    """
    configure_worker_logging(log_queue)
    logger = logging.getLogger("paraocr")
    pid = os.getpid()
    logger.info("Initializing GPU worker, backend, %s, pid, %s", backend_path, pid)
    try:
        EngineCls = _import_obj(backend_path)
    except Exception:
        logger.exception("Cannot import backend, %s", backend_path)
        raise

    global ocr_engine
    try:
        ocr_engine = EngineCls(**(backend_kwargs or {}))
    except Exception:
        logger.exception("Backend initialization failed for %s", backend_path)
        raise

    logger.info("GPU worker ready, pid, %s", pid)


def process_gpu_batch(image_paths: List[str]) -> Tuple[List[str], float]:
    """
    Runs OCR on a batch of image file paths.
    Returns the recognized texts and the measured wall time for the call.
    """
    global ocr_engine
    if not ocr_engine:
        logger.error("GPU worker called before initialization")
        return ([""] * len(image_paths), 0.0)

    if not image_paths:
        return ([], 0.0)

    # Load images into memory once per batch
    arrays: List[np.ndarray] = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                # enforce a consistent mode for backends
                arrays.append(np.array(im.convert("RGB")))
        except Exception:
            logger.exception("Failed to open image, %s", p)
            arrays.append(None)  # keep alignment with image_paths

    start = time.perf_counter()
    try:
        # Backends should implement read_batch(List[np.ndarray]) -> List[str] or (List[str], float)
        result = ocr_engine.read_batch(arrays)  # type: ignore[attr-defined]
        if isinstance(result, tuple) and len(result) == 2:
            texts, backend_time = result
            duration = float(backend_time)
        else:
            texts = result
            duration = time.perf_counter() - start
    except Exception:
        logger.exception("GPU batch processing failed")
        # keep duration for visibility even on failure
        duration = time.perf_counter() - start
        texts = [""] * len(image_paths)

    # Replace None loaded slots with empty strings to keep alignment safe
    if len(texts) != len(image_paths):
        # be defensive if a backend returns a mismatched list
        logger.warning("Backend returned %d items for %d inputs", len(texts), len(image_paths))
        # pad or trim
        if len(texts) < len(image_paths):
            texts = list(texts) + [""] * (len(image_paths) - len(texts))
        else:
            texts = list(texts[: len(image_paths)])

    return texts, duration
