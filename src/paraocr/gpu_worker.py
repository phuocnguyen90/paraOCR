# paraocr/gpu_worker.py
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import importlib

_engine = None

def _import_obj(dotted_path: str):
    mod_path, _, obj_name = dotted_path.rpartition(".")
    mod = importlib.import_module(mod_path)
    return getattr(mod, obj_name)

def initialize_gpu_worker(backend_path: str, backend_kwargs: Dict[str, Any]):
    """Called once per process. Creates one engine instance in global state."""
    global _engine
    print(f"Initializing GPU worker with backend {backend_path} in PID {os.getpid()}...")
    EngineCls = _import_obj(backend_path)
    _engine = EngineCls(**(backend_kwargs or {}))
    print(f"GPU worker ready in PID {os.getpid()}.")

def process_gpu_batch(image_paths: List[str]) -> Tuple[List[str], float]:
    global _engine
    if _engine is None or not image_paths:
        return [""] * len(image_paths), 0.0
    imgs = [np.array(Image.open(p)) for p in image_paths]
    return _engine.read_batch(imgs)
