# powerocr/gpu_worker.py

import os
from PIL import Image
import numpy as np
import easyocr
from typing import List

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

def process_gpu_batch(image_paths: List[str]) -> List[str]:
    """
    The main work function for a GPU worker. It takes a list of image paths,
    loads them, and processes them one by one using its initialized engine.
    """
    global ocr_engine
    if not ocr_engine or not image_paths:
        return [""] * len(image_paths)
    
    try:
        # --- THIS IS THE CORRECTED AND ONLY LOGIC ---
        # The official easyocr.Reader works on one image at a time.
        # We loop through the batch and call readtext for each image.
        # The parallelism comes from having multiple GPU workers doing this loop simultaneously.
        
        all_texts = []
        for path in image_paths:
            # Load a single image into a numpy array
            img_array = np.array(Image.open(path))
            
            # Process the single image using the correct method
            result = ocr_engine.readtext(img_array, detail=0, paragraph=True)
            
            # Join the text parts and append to our results list
            all_texts.append("\n".join(result))
            
        return all_texts

    except Exception as e:
        print(f"\n[GPU Worker Error] Batch processing failed: {e}")
        return [""] * len(image_paths)