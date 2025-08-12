# src/paraocr/utils.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Set, Optional, Union

# Prefer stdlib importlib.resources, fall back to pkg_resources when needed
try:
    from importlib.resources import files as ir_files  # Python 3.9+
except Exception:  # pragma: no cover
    ir_files = None  # type: ignore

try:
    import pkg_resources  # type: ignore
except Exception:  # pragma: no cover
    pkg_resources = None  # type: ignore

from slugify import slugify

logger = logging.getLogger("paraocr")


# ----------------------------
# Results and dictionary utils
# ----------------------------

def load_processed_ids(output_path: Path) -> Set[str]:
    """
    Read JSONL results and collect already processed source paths.
    Tolerates bad lines.
    """
    processed: Set[str] = set()
    if not output_path or not Path(output_path).exists():
        return processed
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    sp = rec.get("source_path")
                    if isinstance(sp, str) and sp:
                        processed.add(sp)
                except Exception:
                    continue
    except Exception as e:
        logger.warning("Failed to read processed ids from %s, %s", output_path, e)
    return processed


def _read_dict_lines(fp: Path) -> Set[str]:
    words: Set[str] = set()
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
    except Exception as e:
        logger.warning("Failed to read dictionary file %s, %s", fp, e)
    return words


def load_dictionary() -> Set[str]:
    """
    Load the packaged Vietnamese dictionary if available.
    Tries importlib.resources first, then pkg_resources.
    Returns an empty set when not found.
    """
    # Attempt importlib.resources on the installed package name
    candidates: list[Path] = []
    for pkg_name in ("paraocr", "paraOCR"):
        try:
            if ir_files:
                res = ir_files(pkg_name).joinpath("vi_full.txt")
                # res may be a Traversable, convert to local path when possible
                try:
                    p = Path(str(res))
                    if p.exists():
                        candidates.append(p)
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback to pkg_resources
    if not candidates and pkg_resources:
        for pkg_name in ("paraocr", "paraOCR"):
            try:
                p_str = pkg_resources.resource_filename(pkg_name, "vi_full.txt")
                p = Path(p_str)
                if p.exists():
                    candidates.append(p)
                    break
            except Exception:
                continue

    if candidates:
        return _read_dict_lines(candidates[0])

    logger.warning("Comprehensive dictionary vi_full.txt not found. Text quality check will be weaker.")
    return set()


def is_native_text_good_quality(text: str, dictionary: Set[str], threshold: float) -> bool:
    """
    Two stage heuristic to judge text quality.
      Stage 1, character validity ratio
      Stage 2, dictionary hit ratio when a dictionary is present
    """
    if not text:
        return False

    # Stage 1, character set check
    vietnamese_chars = (
        "aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩị"
        "jklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxy"
        "ýỳỷỹỵz"
    )
    valid_chars = set(vietnamese_chars + vietnamese_chars.upper() + "0123456789 .,;:!?-()[]{}'\"")
    total = len(text)
    valid = sum(1 for ch in text if ch in valid_chars or ch.isspace())
    if total == 0 or valid / total < 0.85:
        return False

    # Stage 2, dictionary ratio
    if not dictionary:
        return True

    # Simple tokenization, keep alphabetic sequences
    words = [w for w in re.findall(r"[A-Za-zÀ-ỹ]+", text.lower()) if w]
    if len(words) < 5:
        return True

    hits = sum(1 for w in words if w in dictionary)
    return (hits / len(words)) >= float(threshold)


def safe_fname(name: str, fallback: str = "file") -> str:
    """
    Create a filesystem safe name, preserve extension when present.
    """
    name = (name or "").strip() or fallback
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return f"{slugify(base)[:100]}.{ext}"
    return slugify(name)[:100]


