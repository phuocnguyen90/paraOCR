# powerocr/utils.py

import json
from pathlib import Path
from typing import Set
from slugify import slugify

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

def load_dictionary(dict_path: Path) -> Set[str]:
    """Loads a dictionary of common words for quality checking."""
    if not dict_path.exists():
        # It's better to raise an error if a critical file is missing
        # or handle it gracefully in the CLI. For now, we'll print a warning.
        print(f"[Warning] Dictionary file not found at '{dict_path}'. Text quality check will be disabled.")
        return set()
    with open(dict_path, 'r', encoding='utf-8') as f:
        # Use a set for fast 'in' lookups
        return {line.strip().lower() for line in f if line.strip()}

# --- THIS IS THE MISSING FUNCTION ---
def is_native_text_good_quality(text: str, dictionary: Set[str], threshold: float) -> bool:
    """
    Checks if the extracted native text is likely real human language or scanner garbage.
    It calculates the ratio of words in the text that are also present in the dictionary.
    """
    # If no dictionary is provided, we can't perform the check, so we optimistically return True.
    if not dictionary:
        return True
        
    words = text.lower().split()
    
    # If there are very few words, it's hard to judge quality, so we can be lenient.
    if len(words) < 5:
        return True

    # Count how many of the words in the text are found in our dictionary
    in_dict_count = sum(1 for word in words if word in dictionary)
    
    # Calculate the ratio
    quality_ratio = in_dict_count / len(words)
    
    # Return True if the ratio meets or exceeds our quality threshold
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