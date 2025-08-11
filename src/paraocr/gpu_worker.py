# paraocr/gpu_worker.py
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import importlib
import time

ocr_engine = None

def _import_obj(dotted: str):
    mod_path, _, attr = dotted.rpartition(".")
    if not mod_path or not attr:
        raise ImportError(f"Invalid backend path: {dotted}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)

def initialize_gpu_worker(backend_path: str, backend_kwargs: dict):
    print(f"Initializing GPU worker with backend {backend_path} in PID {os.getpid()}...")
    try:
        EngineCls = _import_obj(backend_path)
    except Exception as e:
        print(f"[Init Error] Cannot import backend {backend_path}: {e}")
        raise
    global ocr_engine
    ocr_engine = EngineCls(**(backend_kwargs or {}))
    print(f"GPU worker ready in PID {os.getpid()}")


def process_gpu_batch(image_paths: List[str]) -> Tuple[List[str], float]:
    global ocr_engine
    if not ocr_engine:
        return ([""] * len(image_paths), 0.0)
    start = time.perf_counter()
    texts: List[str] = []
    for p in image_paths:
        arr = np.array(Image.open(p))
        # backends read numpy arrays
        batch_texts, _ = ocr_engine.read_batch([arr])
        texts.append(batch_texts[0] if batch_texts else "")
    return texts, time.perf_counter() - start
