# src/paraocr/logger.py

import logging
import sys
from pathlib import Path
from queue import Queue  # This is the thread-safe queue for the listener
from multiprocessing import Queue as MPQueue # The process-safe queue for workers
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Union, Optional

# --- Custom Log Level for Progress ---
PROGRESS = 25
logging.addLevelName(PROGRESS, "PROGRESS")

def progress(self, msg, *args, **kwargs):
    if self.isEnabledFor(PROGRESS):
        self._log(PROGRESS, msg, args, **kwargs)

logging.Logger.progress = progress

# --- Custom Handlers (for the Listener) ---
class UILogHandler(logging.Handler):
    """Feeds plain text log lines to a UI queue for the log textbox."""
    def __init__(self, q: Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(message)s"))
    def emit(self, record: logging.LogRecord):
        # We only format, the UI thread will add the newline
        self.q.put(self.format(record))

class UIEventHandler(logging.Handler):
    """Emits structured events to a UI queue for progress bars, etc."""
    def __init__(self, q: Queue):
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
            }
            self.q.put(evt)
        except Exception:
            self.handleError(record)

# --- Custom Filters ---
class OnlyLevelFilter(logging.Filter):
    def __init__(self, levelno: int):
        super().__init__()
        self.levelno = levelno
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.levelno

class ExcludeLevelFilter(logging.Filter):
    def __init__(self, levelno: int):
        super().__init__()
        self.levelno = levelno
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != self.levelno

# --- Main Configuration Function ---
def setup_logging(
    log_queue: MPQueue,
    *,
    text_ui_queue: Optional[Queue] = None,
    event_ui_queue: Optional[Queue] = None,
    level: int = logging.INFO,
    file_path: Optional[Union[str, Path]] = None,
    file_level: Optional[int] = None,
) -> QueueListener:
    """
    Sets up the logging listener architecture.

    Args:
        log_queue: The process-safe queue that all workers will log to.
        text_ui_queue: The thread-safe queue for the Gradio text log.
        event_ui_queue: The thread-safe queue for the Gradio progress events.
        level: The base logging level for UI and console outputs.
        file_path: Path to the persistent log file.
        file_level: The logging level for the file.

    Returns:
        A QueueListener instance. You must call .start() on it.
    """
    # --- Step 1: Create the actual handlers (file, UI, etc.) ---
    handlers = []
    
    # File handler
    if file_path:
        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        # Use RotatingFileHandler for robustness in long-running apps
        fh = RotatingFileHandler(fp, maxBytes=5*1024*1024, backupCount=2, encoding="utf-8")
        fh.setLevel(file_level if file_level is not None else level)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(processName)-15s | %(levelname)-8s | %(message)s"))
        fh.addFilter(ExcludeLevelFilter(PROGRESS))
        handlers.append(fh)

    # Text UI queue handler
    if text_ui_queue:
        th = UILogHandler(text_ui_queue)
        th.setLevel(level)
        th.addFilter(ExcludeLevelFilter(PROGRESS))
        handlers.append(th)

    # Event UI queue handler
    if event_ui_queue:
        eh = UIEventHandler(event_ui_queue)
        eh.setLevel(PROGRESS)
        eh.addFilter(OnlyLevelFilter(PROGRESS))
        handlers.append(eh)

    # --- Step 2: Create and return the listener ---
    # The listener pulls from the process-safe queue and pushes to the configured handlers.
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    return listener

def configure_worker_logging(log_queue: MPQueue):
    """
    Configures the logger for a worker process.
    This is called in the initializer of a multiprocessing.Pool.
    It removes all existing handlers and adds only a QueueHandler.
    """
    logger = logging.getLogger("paraocr")
    logger.setLevel(logging.DEBUG)
    
    # Remove any handlers that may have been inherited from the parent process
    logger.handlers.clear()
    
    # Add the QueueHandler. This is the only handler workers will have.
    qh = QueueHandler(log_queue)
    logger.addHandler(qh)