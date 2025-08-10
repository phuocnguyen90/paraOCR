# powerocr/utils.py

import json
from pathlib import Path
from typing import Set
from slugify import slugify
import pkg_resources

def load_processed_ids(output_path: Path) -> Set[str]:
    """
    Reads the OCR results file to find which files have already been processed.
    This makes the script resumable. It now expects the result to have 'source_path'.
    """
    processed_paths = set()
    if not output_path.exists():
        return processed_paths
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if 'source_path' in record:
                    processed_paths.add(record['source_path'])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed_paths

def load_dictionary() -> Set[str]:
    """
    Loads the comprehensive Vietnamese dictionary that is packaged with the library.
    """
    try:
        # This safely finds the path to the data file even after the package is installed
        filepath = pkg_resources.resource_filename('powerocr', 'vi_full.txt')
        with open(filepath, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except (FileNotFoundError, ModuleNotFoundError):
        print("[Warning] Comprehensive dictionary 'vi_full.txt' not found. Text quality check will be degraded.")
        return set()

# --- THIS IS THE MISSING FUNCTION ---
def is_native_text_good_quality(text: str, dictionary: Set[str], threshold: float) -> bool:
    """
    Performs a two-stage heuristic check to validate the quality of extracted native text.
    """
    if not text:
        return False

    # --- Stage 1: Character Script Analysis (Fast Check) ---
    # Define the set of characters we expect in Vietnamese/Latin script
    # This includes basic Latin alphabet, numbers, common punctuation, and Vietnamese-specific characters.
    vietnamese_chars = "aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz"
    valid_chars = set(vietnamese_chars + vietnamese_chars.upper() + "0123456789 .,;:!?-()[]{}'\"")
    
    total_chars = len(text)
    valid_char_count = sum(1 for char in text if char in valid_chars or char.isspace())
    
    # If less than 85% of characters are valid (or whitespace), it's likely garbage.
    if (valid_char_count / total_chars) < 0.85:
        return False # Fails the fast check

    # --- Stage 2: Comprehensive Dictionary Check (Slower, More Accurate Check) ---
    if not dictionary:
        return True # Cannot perform stage 2, so we optimistically pass

    words = text.lower().split()
    if len(words) < 5:
        return True # Not enough words to make a reliable judgment

    in_dict_count = sum(1 for word in words if word in dictionary)
    quality_ratio = in_dict_count / len(words)
    
    return quality_ratio >= threshold

def safe_fname(name: str, fallback: str = "file") -> str:
    """
    Creates a filesystem-safe filename from a given string.
    """
    name = (name or "").strip() or fallback
    if "." in name:
        base, ext = name.rsplit(".", 1)
        # Avoid slugifying a valid extension
        return f"{slugify(base)[:100]}.{ext}"
    else:
        return slugify(name)[:100]