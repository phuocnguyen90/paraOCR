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
    
from collections import defaultdict
from dataclasses import dataclass

def _to_int_or_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            # guard NaN
            return None if (isinstance(v, float) and (v != v)) else int(v)
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None

@dataclass
class PhaseWindow:
    done: int = 0
    total: int | None = None
    started_at: float = 0.0
    ewma_secs_per_item: float | None = None  # smoothed duration
    _last_beat_time: float = 0.0
    _last_done_seen: int = 0

class PhaseTracker:
    """
    Manages the state of the UI progress bar by consuming structured PROGRESS events.
    """
    def __init__(self):
        self.start_time = time.perf_counter()
        self.phase_key = "scan"
        self.phase_name = "Starting"
        self._last_eta_text = ""
        self._last_eta_update = 0.0
        self.phase_pct = 0.0
        self.current = None
        self.total = None
        # tune these to your pipeline proportions
        self.phase_ranges = {
            "scan":      (  2, 10),
            "dispatch":  ( 10, 30),
            "render":    ( 30, 40),
            "aggregate": ( 40, 95),
            "final":     ( 95, 99),
            "done":      (100,100),
        }
        

        self.phases: dict[str, PhaseWindow] = defaultdict(PhaseWindow)

    def _enter_phase_if_needed(self, key: str):
        if key != self.phase_key:
            self.phase_key = key
            self.phase_name = key.replace("_", " ").title()
            w = self.phases[key]
            if not w.started_at:
                now = time.perf_counter()
                w.started_at = now
                w._last_beat_time = now
                w._last_done_seen = 0


    def update_from_event(self, e: dict):
        key = (e.get("phase") or self.phase_key or "scan").lower()
        self._enter_phase_if_needed(key)
        w = self.phases[key]

        # coerce totals
        tot = _to_int_or_none(e.get("total"))
        if tot is not None and tot >= 0:
            w.total = tot

        # coerce current and update EWMA using deltas between beats
        cur = _to_int_or_none(e.get("current"))
        if cur is not None and cur >= 0:
            now = time.perf_counter()
            delta_items = max(0, cur - w._last_done_seen)
            delta_time = max(1e-6, now - (w._last_beat_time or now))
            if delta_items > 0:
                inst = delta_time / float(delta_items)  # secs per item for this beat
                alpha = 0.25
                w.ewma_secs_per_item = inst if w.ewma_secs_per_item is None else (
                    alpha * inst + (1 - alpha) * w.ewma_secs_per_item
                )
                w._last_done_seen = cur
                w._last_beat_time = now
            w.done = cur  # always reflect latest done

        # explicit pct still takes precedence
        if isinstance(e.get("pct"), (int, float)):
            self.phase_pct = float(e["pct"])
            return

        # derive pct from phase range + counts
        lo, hi = self.phase_ranges.get(key, (0, 100))
        if w.total and w.total > 0:
            frac = max(0.0, min(1.0, w.done / w.total))
            self.phase_pct = lo + (hi - lo) * frac
        else:
            self.phase_pct = float(lo)
    
    def _eta_for_phase(self, key: str) -> str:
        w = self.phases[key]
        if not (w.total and w.ewma_secs_per_item is not None):
            return ""
        remaining = max(0, w.total - w.done)
        secs_left = remaining * max(0.0, w.ewma_secs_per_item)

        now = time.perf_counter()
        if now - self._last_eta_update < 0.5 and self._last_eta_text:
            return self._last_eta_text

        mins, secs = divmod(int(secs_left + 0.5), 60)
        txt = f"ETA {mins}m {secs}s" if mins else f"ETA {secs}s"
        self._last_eta_text = txt
        self._last_eta_update = now
        return txt

    def get_description(self) -> str:
        w = self.phases[self.phase_key]
        parts = [self.phase_name]
        if w.total is not None:
            parts.append(f"{w.done}/{w.total}")
        eta = self._eta_for_phase(self.phase_key)
        if eta:
            parts.append(eta)
        return " | ".join(parts)
    
    
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

def run_ocr_task(
    pdf_paths: List[Path],
    *,
    languages,
    log_mode,
    num_workers,
    num_gpu_workers,
    gpu_batch_size,
    cpu_chunk_size,
):
    """
    Streams UI updates by draining in-memory UI queues that the logging listener fills.
    Expects `pdf_paths` to already be a list of PDF Paths (ZIPs expanded by the WebUI).
    """
    import time, traceback, shutil, logging, dataclasses
    from queue import Queue as ThreadQueue, Empty
    from threading import Thread
    from multiprocessing import Manager
    from pathlib import Path as _Path

    # Local imports to avoid circulars on module load
    from .ui_utils import pick_workspace, render_progress_html, scan_for_results, PhaseTracker, in_colab

    # --- helpers --------------------------------------------------------------

    def _build_config(cfg_dict: dict):
        """
        Construct a real OCRConfig instance, filtering unknown keys so `from_dict` won't fail.
        Falls back to direct constructor if needed. Guarantees attributes like `pdf_engine` exist.
        """
        from .config import OCRConfig  # must exist in your project

        # Determine allowable field names
        try:
            if dataclasses.is_dataclass(OCRConfig):
                field_names = {f.name for f in dataclasses.fields(OCRConfig)}
            elif hasattr(OCRConfig, "__annotations__"):
                field_names = set(OCRConfig.__annotations__.keys())
            else:
                field_names = set(cfg_dict.keys())
        except Exception:
            field_names = set(cfg_dict.keys())

        filtered = {k: v for k, v in cfg_dict.items() if k in field_names}

        # If pdf_engine is a field but absent, set a safe default
        if "pdf_engine" in field_names and "pdf_engine" not in filtered:
            filtered["pdf_engine"] = "pymupdf"  # safe default; adjust if your project uses another

        # Try from_dict first for proper type coercions; then fall back to kwargs
        if hasattr(OCRConfig, "from_dict"):
            try:
                return OCRConfig.from_dict(filtered)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            return OCRConfig(**filtered)
        except Exception as e:
            # As a last resort, create a simple object with attribute access
            from types import SimpleNamespace
            # Ensure pdf_engine is present
            cfg = filtered.copy()
            cfg.setdefault("pdf_engine", "pymupdf")
            return SimpleNamespace(**cfg)

    def _pipeline_entry(config):
        """
        Wrapper to ensure the Thread target is a function, not the OCRRunner class itself.
        Discovers tasks from the config.input_dir and runs the pipeline.
        """
        from .parallel import OCRRunner
        from .models import OCRTask

        in_dir = _Path(config.input_dir)
        # Discover PDFs (you can add image types if needed)
        allowed = {".pdf"}
        tasks = [OCRTask(source_path=p) for p in sorted(in_dir.rglob("*")) if p.is_file() and p.suffix.lower() in allowed]

        runner = OCRRunner(config)
        runner.run(tasks)

    # --- 1) UI Queues & tracker ------------------------------------------------

    text_ui_queue: ThreadQueue[str] = ThreadQueue()
    event_ui_queue: ThreadQueue[dict] = ThreadQueue()

    tracker = PhaseTracker()
    log_history = ""

    try:
        # --- 2) Workspace & input prep ----------------------------------------
        if in_colab():
            storage_choice = "Google Drive" if Path("/content/drive/MyDrive").exists() else "Colab Temporary"
        else:
            storage_choice = "Local"

        base, input_dir, output_dir, notice = pick_workspace(storage_choice)
        if notice:
            log_history += f"{notice}\n"

        run_ts = time.strftime("%Y%m%d-%H%M%S")
        current_input_dir = input_dir / run_ts
        current_input_dir.mkdir(parents=True, exist_ok=True)

        output_jsonl_path = output_dir / f"results_{run_ts}.jsonl"
        error_log_path = output_dir / f"errors_{run_ts}.jsonl"
        session_log_path = output_dir / f"run_{run_ts}.log"

        # Copy given PDFs into this run's input dir
        pdf_paths = [p if isinstance(p, _Path) else _Path(p) for p in (pdf_paths or [])]
        pdf_paths = [p for p in pdf_paths if p.suffix.lower() == ".pdf" and p.exists()]
        if not pdf_paths:
            log_history += "No PDFs found in the provided paths.\n"
            yield {"log": log_history, "progress_html": render_progress_html(0, "No input")}
            return

        log_history += f"Preparing {len(pdf_paths)} PDF(s) for processing...\n"
        for p in pdf_paths:
            dst = current_input_dir / p.name
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
            except Exception:
                shutil.copy(p, dst)
        log_history += "Input preparation complete.\n"

        # --- 3) Logging system wiring -----------------------------------------
        log_queue = Manager().Queue(-1)  # process-safe queue
        level = logging.DEBUG if (log_mode or "Basic") == "Advanced" else logging.INFO

        from .logger import setup_logging
        listener = setup_logging(
            log_queue=log_queue,
            text_ui_queue=text_ui_queue,
            event_ui_queue=event_ui_queue,
            level=level,
            file_path=session_log_path,
        )
        listener.start()

        logger = logging.getLogger("paraocr")
        langs = languages or ["vi", "en"]
        logger.info("Workspace: %s", base)
        logger.info("Starting OCR with languages: %s", ", ".join(langs))
        logger.progress("scan start", extra={"phase": "scan", "pct": 2})

        # --- 4) Build config & start pipeline thread --------------------------
        cfg_dict = {
            "input_dir": str(current_input_dir),
            "output_path": str(output_jsonl_path),
            "error_log_path": str(error_log_path),
            "export_txt": True,
            "num_workers": int(num_workers),
            "num_gpu_workers": int(num_gpu_workers),
            "gpu_batch_size": int(gpu_batch_size),
            "dpi": 200,
            "languages": langs,
            "dictionary": {},  # will be set via load_dictionary if OCRConfig supports it
            "log_queue": log_queue,
            "cpu_chunk_size": int(cpu_chunk_size),
        }

        # Provide dictionary if your config expects it
        try:
            from .utils import load_dictionary
            cfg_dict["dictionary"] = load_dictionary()
        except Exception:
            pass

        config = _build_config(cfg_dict)

        # ensure required Path-like attributes exist (some pipelines expect Path objects)
        try:
            if hasattr(config, "output_path"):
                setattr(config, "output_path", _Path(getattr(config, "output_path")))
            if hasattr(config, "error_log_path"):
                setattr(config, "error_log_path", _Path(getattr(config, "error_log_path")))
        except Exception:
            pass

        pipeline_thread = Thread(target=_pipeline_entry, args=(config,), daemon=True)
        pipeline_thread.start()

        # --- 5) UI loop: drain queues & emit updates --------------------------
        while pipeline_thread.is_alive():
            # drain text logs
            try:
                while True:
                    log_history += text_ui_queue.get_nowait() + "\n"
            except Empty:
                pass
            # drain progress events
            try:
                while True:
                    event = event_ui_queue.get_nowait()
                    tracker.update_from_event(event)
            except Empty:
                pass

            pct = tracker.get_percent()
            desc = tracker.get_description()
            try:
                eta = tracker._get_eta_text()  # optional
            except Exception:
                eta = ""

            label = f"{desc}{(' — ' + eta) if eta else ''}"
            yield {
                "log": log_history,
                "progress_html": render_progress_html(pct, label),
            }
            time.sleep(0.2)

        pipeline_thread.join()
        listener.stop()

        # --- 6) Final drain ----------------------------------------------------
        try:
            while True:
                log_history += text_ui_queue.get_nowait() + "\n"
        except Empty:
            pass
        try:
            while True:
                tracker.update_from_event(event_ui_queue.get_nowait())
        except Empty:
            pass

        # --- 7) Results --------------------------------------------------------
        log_history += "\n✅ Processing complete. Scanning for results...\n"
        results_df = scan_for_results(output_jsonl_path)
        if results_df.empty:
            log_history += "⚠️ No results found. Check the log for errors.\n"
        else:
            log_history += f"✨ Found {len(results_df)} processed documents. Displaying below.\n"

        yield {
            "log": log_history,
            "results": results_df,
            "progress_html": render_progress_html(100, "Done"),
        }

    except Exception:
        log_history += "\n[UI Error]\n" + traceback.format_exc() + "\n"
        yield {"log": log_history, "progress_html": render_progress_html(0, "Error")}


# webui.py (top; or utils.py and import)
def _in_colab() -> bool:
    return "google.colab" in sys.modules

def _cpu_core_count() -> int:
    try:
        return os.cpu_count() or 2
    except Exception:
        return 2

def _gpu_total_vram_gb() -> float:
    # Prefer nvidia-smi (doesn't init CUDA)
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True)
        mibs = [float(x.strip()) for x in out.splitlines() if x.strip()]
        if mibs:
            return sum(mibs) / 1024.0
    except Exception:
        pass

    # Fallback to torch (may init CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            return sum(
                torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
            ) / (1024**3)
    except Exception:
        pass
    return 0.0

import math

def _suggest_defaults():
    colab = _in_colab()
    cpu = _cpu_core_count()
    vram = _gpu_total_vram_gb()
    # 1 GPU worker per ~4 GiB VRAM, minimum 0 if no GPU
    gpu_workers = int(max(0, math.floor(vram / 4.0)))

    if colab:
        # Your requested Colab baseline
        cpu_workers = 2
        gpu_workers = max(gpu_workers, 1)  # if there is *some* GPU
    else:
        # Your requested local baseline (bounded by cores/VRAM)
        cpu_workers = min(max(4, 1), cpu)  # at least 4 if available
        # If VRAM is tiny, keep at 0/1
        gpu_workers = max(min(gpu_workers, 2), 0)

    # Reasonable batch sizes; tweak to taste
    gpu_batch_size = 4 if gpu_workers <= 1 else 8
    cpu_chunk_size = 16

    return {
        "cpu_workers": max(1, min(cpu_workers, cpu)),
        "gpu_workers": gpu_workers,
        "gpu_batch_size": gpu_batch_size,
        "cpu_chunk_size": cpu_chunk_size,
        "vram_gb": round(vram, 2),
        "cpu_cores": cpu,
        "env": "Colab" if colab else "Local",
    }

def _is_zip(path: str | Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False
    
def _collect_inputs(uploaded_files, session_tmpdir: Path) -> list[Path]:
    """
    Accepts a list of gradio UploadedFiles OR a single one.
    Expands ZIPs into session_tmpdir and returns a flat list of PDF Paths.
    """
    if not uploaded_files:
        return []

    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    pdfs: list[Path] = []
    for f in uploaded_files:
        p = Path(getattr(f, "name", f))
        if _is_zip(p):
            with zipfile.ZipFile(p, "r") as zf:
                # Extract only PDFs; ignore other noise
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    if str(zi.filename).lower().endswith(".pdf"):
                        target = session_tmpdir / Path(zi.filename).name
                        with zf.open(zi) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        pdfs.append(target)
        else:
            # If a single PDF or other file — only accept PDFs
            if p.suffix.lower() == ".pdf":
                # Copy into session dir to isolate lifecycle
                target = session_tmpdir / p.name
                if str(p) != str(target):
                    try:
                        shutil.copy2(p, target)
                    except Exception:
                        shutil.copy(p, target)
                pdfs.append(target)
    # De‑dupe and sort for stability
    pdfs = sorted(set(pdfs), key=lambda x: x.name)
    return pdfs
