# src/paraocr/ui_utils.py
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import zipfile
import traceback
from pathlib import Path
from threading import Thread
from typing import Tuple, Optional, List, Dict, Any
from queue import Queue

import pandas as pd
import gradio as gr

from .config import OCRConfig
from .cli import run_pipeline
from .utils import load_dictionary
import logging
from multiprocessing import Manager
from .logger import setup_logging, configure_worker_logging, PROGRESS

# ---------------- Environment & Storage ----------------

class Storage:
    name: str
    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        raise NotImplementedError

class LocalStorage(Storage):
    name = "Local"
    def __init__(self, base: Optional[Path] = None):
        self.base = base or (Path.home() / "paraOCR_Workspace")
    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        base = self.base
        input_dir = base / "input_data"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return base, input_dir, output_dir, ""

class ColabTempStorage(Storage):
    name = "Colab Temporary"
    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        base = Path("/content/paraocr_workspace")
        input_dir = base / "input_data"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return base, input_dir, output_dir, ""

class ColabDriveStorage(Storage):
    name = "Google Drive"
    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            base = Path("/content/paraocr_workspace")
            input_dir = base / "input_data"
            output_dir = base / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            return base, input_dir, output_dir, "Google Drive is not mounted, using temporary storage in /content"
        base = drive_root / "paraOCR_Workspace"
        input_dir = base / "input_data"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return base, input_dir, output_dir, ""

class KaggleStorage(Storage):
    name = "Kaggle"
    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        base = Path("/kaggle/working/paraocr_workspace")
        input_dir = base / "input_data"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return base, input_dir, output_dir, ""

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def _storage_by_name(name: str) -> Storage:
    if name == "Google Drive":
        return ColabDriveStorage()
    if name == "Colab Temporary":
        return ColabTempStorage()
    if name == "Kaggle":
        return KaggleStorage()
    return LocalStorage()

def pick_workspace(storage_choice: str) -> Tuple[Path, Path, Path, str]:
    storage = _storage_by_name(storage_choice)
    base, input_dir, output_dir, notice = storage.ensure_workspace()
    return base, input_dir, output_dir, (notice or "")

# ---------------- Log Filtering & Progress ----------------

class UILogger:
    """Filters log messages for a cleaner UI."""
    def __init__(self, mode="Basic"):
        self.mode = mode
        self.basic_keywords = [
            "starting",
            "unzipping",
            "copying",
            "created",
            "dispatching",
            "aggregating",
            "writing final results",
            "processing complete",
            "error",
            "found results",
            "no results",
        ]
    def filter(self, line: str) -> Optional[str]:
        if self.mode == "Advanced":
            return line
        l = line.strip()
        low = l.lower()
        if any(k in low for k in self.basic_keywords):
            return l
        if "Rendering Pages (CPU):" in l or "Aggregating Batches:" in l:
            return l
        return None

class PhaseTracker:
    """
    Manages the state of the UI progress bar by consuming structured PROGRESS events.
    """
    def __init__(self):
        self.start_time = time.perf_counter()
        self.phase = "Starting"

        self._last_eta_update = 0.0
        self.phase_pct = 0.0

    def update_from_event(self, event: Dict):
        """Updates the tracker's state from a PROGRESS event from the queue."""
        self.phase = event.get("phase", self.phase).replace("_", " ").title()
        self.phase_pct = event.get("pct", self.phase_pct)

    def get_description(self) -> str:
        """Calculates and returns the full description string for the progress bar."""
        eta = self._get_eta_text()
        return f"{self.phase}{' | ' + eta if eta else ''}"
    
    
    def get_percent(self) -> float:
        return max(0.0, min(100.0, self.phase_pct))
    


    def _get_eta_text(self) -> Optional[str]:
        """Calculates the estimated time remaining."""
        now = time.perf_counter()
        if now - self._last_eta_update < 3.0:  # Throttle ETA updates
            return None
        self._last_eta_update = now

        elapsed = now - self.start_time
        if self.phase_pct < 5 or self.phase_pct > 98: # Don't show ETA at the very beginning or end
            return None
            
        # ETA calculation: (time_elapsed / progress) * (1 - progress)
        est_total_time = (elapsed / self.phase_pct) * 100
        remaining = max(0, est_total_time - elapsed)

        if remaining < 10:
            return "Finishing up..."
        
        return f"Est. {int(remaining)}s left"

# ---------------- Data helpers ----------------

def scan_for_results(results_jsonl_path: Path) -> pd.DataFrame:
    if not results_jsonl_path.exists():
        return pd.DataFrame(columns=["File Name", "Status", "Source Path"])
    rows: List[dict] = []
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                source_path = Path(data["source_path"])
                txt_path = source_path.parent / (source_path.stem + ".txt")
                status = "Success" if txt_path.exists() else "OCR done, TXT missing"
                rows.append({"File Name": source_path.name, "Status": status, "Source Path": str(source_path)})
            except Exception:
                continue
    return pd.DataFrame(rows)

def view_file_content(df, evt: gr.SelectData):
    if evt.index is None:
        return "Select a file in the table to view"
    row = df.iloc[evt.index[0]]
    source = Path(row["Source Path"])
    txt_path = source.parent / (source.stem + ".txt")
    return txt_path.read_text(encoding="utf-8") if txt_path.exists() else f"Missing {txt_path}"

# ---------------- Input prep ----------------

def prepare_input_directory(input_method, zip_file, multi_files, gdrive_path, current_input_dir) -> str:
    log = ""
    if input_method == "Upload a .zip file":
        if not zip_file:
            raise ValueError("Please upload a zip file.")
        zpath = Path(zip_file)
        log += f"Unzipping '{zpath.name}'...\n"
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(current_input_dir)
        log += "Unzip complete.\n"

    elif input_method == "Upload individual files":
        paths = multi_files or []
        if not paths:
            raise ValueError("Please upload one or more files.")
        allowed = {".pdf", ".png", ".jpg", ".jpeg"}
        to_copy = [Path(p) for p in paths if Path(p).suffix.lower() in allowed]
        log += f"Copying {len(to_copy)} uploaded files...\n"
        for p in to_copy:
            dst = current_input_dir / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        log += "Copy complete.\n"

    else:  # Use path to copy from
        if not gdrive_path:
            raise ValueError("Please provide a path to copy from.")
        src = Path(gdrive_path)
        if not src.exists():
            raise FileNotFoundError(f"Path not found: {src}")
        log += f"Copying files from '{src}'...\n"
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    dst = current_input_dir / p.relative_to(src)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
        else:
            shutil.copy2(src, current_input_dir / src.name)
        log += "Copy complete.\n"

    return log

# ---------------- Orchestrator: file-tailed logging (no flicker) ----------------

def render_progress_html(pct: float, text: str = "") -> str:
    pct = max(0, min(100, int(pct)))
    return (
        "<div style='height:8px;background:#eee;border-radius:6px;overflow:hidden'>"
        f"<div style='width:{pct}%;height:100%;background:#4f46e5'></div></div>"
        f"<div style='font-size:12px;margin-top:6px;color:#555'>{text}</div>"
    )

def pipeline_worker(config: OCRConfig):
    """
    This is the target for the background thread.
    It simply calls the main pipeline function. All logging from the pipeline
    will be automatically captured by the QueueListener system.
    """
    try:
        # We don't need to capture stdout anymore, the logger handles it.
        run_pipeline(config)
    except Exception:
        # Get the logger and log the crash
        import logging
        logger = logging.getLogger("paraocr")
        logger.critical("---  PIPELINE CRASHED  ---", exc_info=True)

def run_ocr_task(storage_choice, input_method, zip_file, multi_files, gdrive_path, langs, log_mode):
    """
    Streams UI updates by tailing a session log file written by the pipeline.
    """
    # --- 1. Setup Queues and the Logging Listener ---
    text_ui_queue = Queue()
    event_ui_queue = Queue()

    tracker = PhaseTracker()
    log_history = ""

    try:
        # --- 2. Setup Workspace & Inputs ---
        base, input_dir, output_dir, notice = pick_workspace(storage_choice)
        if notice:
            log_history += f"{notice}\n"

        run_ts = time.strftime("%Y%m%d-%H%M%S")
        current_input_dir = input_dir / run_ts
        current_input_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl_path = output_dir / f"results_{run_ts}.jsonl"
        error_log_path = output_dir / f"errors_{run_ts}.jsonl"
        session_log_path = output_dir / f"run_{run_ts}.log"

        log_history += prepare_input_directory(input_method, zip_file, multi_files, gdrive_path, current_input_dir)

        # --- 3. Configure and START the Logging System ---
        import logging
        from multiprocessing import Manager
        log_queue = Manager().Queue(-1) # The process-safe queue
        level = logging.DEBUG if log_mode == "Advanced" else logging.INFO

        listener = setup_logging(
            log_queue=log_queue,
            text_ui_queue=text_ui_queue,
            event_ui_queue=event_ui_queue,
            level=level,
            file_path=session_log_path
        )
        listener.start()
        # Get a logger instance for the main UI thread
        logger = logging.getLogger("paraocr")
        logger.info("Workspace: %s", base)
        logger.info("Starting OCR with languages: %s", ", ".join(langs or ["vi", "en"]))

        # --- 4. Start the Pipeline ---
        config = OCRConfig.from_dict({
            "input_dir": str(current_input_dir),
            "output_path": str(output_jsonl_path),
            "error_log_path": str(error_log_path),
            "export_txt": True,
            "num_workers": 2,
            "num_gpu_workers": 1,
            "gpu_batch_size": 8,
            "dpi": 200,
            "languages": langs or ["vi", "en"],
            "dictionary": load_dictionary(),
            "log_queue": log_queue,
        })
        pipeline_thread = Thread(target=pipeline_worker, args=(config,), daemon=True)
        pipeline_thread.start()

        # --- 5. Main UI Update Loop ---
        # This loop pulls from the thread-safe UI queues populated by the listener.
        while pipeline_thread.is_alive():
            # Drain the text queue for the log box
            while not text_ui_queue.empty():
                log_history += text_ui_queue.get_nowait() + "\n"
            
            # Drain the event queue for the progress bar
            while not event_ui_queue.empty():
                event = event_ui_queue.get_nowait()
                tracker.update_from_event(event)

            pct = tracker.get_percent()
            desc = tracker.get_description()
            eta = tracker._get_eta_text()

            # Yield a single, consolidated update
            yield {
                "log": log_history,
                "progress_html": render_progress_html(pct, f"{desc}{' — ' + eta if eta else ''}")
            }
            time.sleep(0.2) # UI update frequency

        pipeline_thread.join()
        listener.stop() # IMPORTANT: Stop the listener thread

        # --- 6. Finalization after thread completion ---
        while not text_ui_queue.empty():
            log_history += text_ui_queue.get_nowait() + "\n"
        while not event_ui_queue.empty():
            tracker.update_from_event(event_ui_queue.get_nowait())

        # --- 7. Scan for results and yield the final UI state ---
        log_history += "\n✅ Processing complete. Scanning for results...\n"
        
        results_df = scan_for_results(output_jsonl_path)
        if results_df.empty:
            log_history += "⚠️ No results found. Check the log for errors.\n"
        else:
            log_history += f"✨ Found {len(results_df)} processed documents. Displaying below.\n"

        yield {
            "log": log_history, 
            "results": results_df, 
            "progress_html": render_progress_html(100, "Done")
        }



    except Exception:
        log_history += "\n[UI Error]\n" + traceback.format_exc() + "\n"
        yield {"log": log_history, "progress_html": render_progress_html(0, "Error")}
