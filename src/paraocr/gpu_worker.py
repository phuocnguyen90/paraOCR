# paraOCR/gpu_worker.py

import os
from PIL import Image
import numpy as np
import easyocr
from typing import List, Tuple
import time

# --- Global variable to hold the model ---
# This is a key optimization: the model is loaded only ONCE when the worker process starts.
ocr_engine = None

def initialize_gpu_worker(languages: List[str], beamsearch: bool):
    """
    This function is called once per worker process to initialize its own OCR engine.
    """
    global ocr_engine
    print(f"Initializing GPU Worker (PID {os.getpid()})...")
    # Each worker gets its own instance of the easyocr.Reader.
    ocr_engine = easyocr.Reader(languages, gpu=True)
    if beamsearch:
        ocr_engine.beamsearch = True
    print(f"GPU Worker (PID {os.getpid()}) initialized successfully.")

def process_gpu_batch(image_paths: List[str]) -> Tuple[List[str], float]:
    """
    Now returns a tuple: (list_of_texts, duration_of_work).
    """
    global ocr_engine
    if not ocr_engine or not image_paths:
        return ([""] * len(image_paths), 0.0)
    
    start_time = time.perf_counter()
    try:
        # The logic for OCR is the same
        all_texts = []
        for path in image_paths:
            img_array = np.array(Image.open(path))
            result = ocr_engine.readtext(img_array, detail=0, paragraph=True)
            all_texts.append("\n".join(result))
        
        duration = time.perf_counter() - start_time
        return (all_texts, duration)

    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"\n[GPU Worker Error] Batch processing failed: {e}")
        return ([""] * len(image_paths), duration)