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


# ----------------------------
# Logging helpers for WebUI
# ----------------------------

PROGRESS = 25  # numeric level between INFO 20 and WARNING 30
logging.addLevelName(PROGRESS, "PROGRESS")


def progress(self, msg, *args, **kwargs):
    if self.isEnabledFor(PROGRESS):
        self._log(PROGRESS, msg, args, **kwargs)


logging.Logger.progress = progress  # type: ignore


class QueueTextHandler(logging.Handler):
    """Plain text log lines to a queue for Advanced/Basic log textbox."""
    def __init__(self, q):
        super().__init__()
        self.q = q
        # keep it clean & human-friendly
        self.setFormatter(logging.Formatter("%(message)s"))
    def emit(self, record):
        try:
            self.q.put(self.format(record) + "\n")
        except Exception:
            pass


class QueueEventHandler(logging.Handler):
    """
    Emit structured events for the UI.
    Used for progress bars and compact Basic view.
    """
    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord):
        try:
            evt = {
                "level": record.levelname,
                "msg": record.getMessage(),
                "phase": getattr(record, "phase", None),
                "pct": getattr(record, "pct", None),
                "current": getattr(record, "current", None),
                "total": getattr(record, "total", None),
                "file": getattr(record, "file", None),
            }
            self.q.put(evt)
        except Exception:
            pass

class OnlyLevelFilter(logging.Filter):
    """Pass only records with exactly this levelno."""
    def __init__(self, levelno: int):
        super().__init__()
        self.levelno = levelno
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.levelno
    
class ExcludeLevelFilter(logging.Filter):
    """Exclude records with exactly this levelno."""
    def __init__(self, levelno: int):
        super().__init__()
        self.levelno = levelno
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != self.levelno
    

from logging.handlers import RotatingFileHandler
import sys

def setup_logging(
    text_queue=None,
    event_queue=None,
    level: int = logging.INFO,
    *,
    file_path: Optional[Union[str, Path]] = None,
    file_level: Optional[int] = None,
    include_progress_in_file: bool = False,
    add_console_if_no_handlers: bool = True,
) -> logging.Logger:
    """
    Configure the 'paraocr' logger.

    - text_queue:      queue for human-readable log lines (INFO/DEBUG), PROGRESS is excluded.
    - event_queue:     queue for progress events; receives PROGRESS only.
    - level:           base level for text outputs (INFO for Basic, DEBUG for Advanced).
    - file_path:       if provided, write a persistent log file.
    - file_level:      level for file handler (defaults to `level` if None).
    - include_progress_in_file: write PROGRESS records to file when True (default False).
    - add_console_if_no_handlers: add a simple console handler if nothing else is attached.
    """
    logger = logging.getLogger("paraocr")
    logger.setLevel(logging.DEBUG)      # accept everything; handlers decide what to emit
    logger.propagate = False            # don't bubble to root

    # Clear existing handlers to avoid duplicates on repeated setup
    logger.handlers[:] = []

    handlers_attached = 0

    # File handler
    if file_path:
        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setLevel(file_level if file_level is not None else level)
        # Nice timestamped format for files
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        if not include_progress_in_file:
            fh.addFilter(ExcludeLevelFilter(PROGRESS))
        logger.addHandler(fh)
        handlers_attached += 1

    # Text queue handler (for the scrolling textbox)
    if text_queue is not None:
        th = QueueTextHandler(text_queue)
        th.setLevel(level)  # INFO for Basic, DEBUG for Advanced
        th.addFilter(ExcludeLevelFilter(PROGRESS))  # never show PROGRESS in text log
        logger.addHandler(th)
        handlers_attached += 1

    # Event queue handler (for the progress bar)
    if event_queue is not None:
        eh = QueueEventHandler(event_queue)
        eh.setLevel(PROGRESS)
        eh.addFilter(OnlyLevelFilter(PROGRESS))     # accept PROGRESS only
        logger.addHandler(eh)
        handlers_attached += 1

    # Optional console fallback (useful in CLI runs)
    if handlers_attached == 0 and add_console_if_no_handlers:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        ch.addFilter(ExcludeLevelFilter(PROGRESS))
        logger.addHandler(ch)

    return logger