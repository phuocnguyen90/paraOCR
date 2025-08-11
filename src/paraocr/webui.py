# src/paraocr/webui.py

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
from queue import Queue, Empty
from typing import Tuple, Optional, List, Dict

import gradio as gr

from .config import OCRConfig
from .cli import run_pipeline
from .utils import load_dictionary

# ---------------- Environment detection ----------------

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def in_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

def in_azure() -> bool:
    # Placeholder, extend later
    return False

# ---------------- Storage adapters ----------------

class Storage:
    name: str

    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        """Return base, input_dir, output_dir, notice"""
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
            # fall back to temp
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
    # You can extend this later
    def ensure_workspace(self) -> Tuple[Path, Path, Path, str]:
        base = Path("/kaggle/working/paraocr_workspace")
        input_dir = base / "input_data"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return base, input_dir, output_dir, ""

# ---------------- Logging plumbing ----------------

class QueueWriter:
    """Line buffered writer that mirrors to the real stream and pushes lines to a queue."""
    def __init__(self, stream, q: Queue):
        self.stream = stream
        self.q = q
        self.buf = ""
    def write(self, s: str):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.q.put(line + "\n")
    def flush(self):
        self.stream.flush()

SENTINEL = object()

def pipeline_worker(config: OCRConfig, q: Queue):
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = QueueWriter(old_out, q)
    sys.stderr = QueueWriter(old_err, q)
    try:
        run_pipeline(config)
    except Exception:
        q.put("\n[Pipeline Error]\n" + traceback.format_exc() + "\n")
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        q.put(SENTINEL)

# ---------------- UI helpers ----------------

def scan_for_results(results_jsonl_path: Path):
    import pandas as pd
    if not results_jsonl_path.exists():
        return pd.DataFrame(columns=["File Name", "Status", "Source Path"])
    processed_files = []
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                source_path = Path(data["source_path"])
                txt_path = source_path.parent / (source_path.stem + ".txt")
                status = "Success" if txt_path.exists() else "OCR done, TXT missing"
                processed_files.append({
                    "File Name": source_path.name,
                    "Status": status,
                    "Source Path": str(source_path)
                })
            except Exception:
                continue
    import pandas as pd
    return pd.DataFrame(processed_files)

def view_file_content(df, evt: gr.SelectData):
    if evt.index is None:
        return "Select a file in the table to view"
    row = df.iloc[evt.index[0]]
    source = Path(row["Source Path"])
    txt_path = source.parent / (source.stem + ".txt")
    return txt_path.read_text(encoding="utf-8") if txt_path.exists() else f"Missing {txt_path}"

class UILogger:
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
            "estimated time",
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

class ETAEstimator:
    def __init__(self):
        self.start_ts = time.perf_counter()
        self.total_pages = None
        self.last_eta_msg_ts = 0.0
        self.progress_pct = 0.0
        self.sec_per_page_guess = 1.6

    def update_from_line(self, line: str):
        import re
        m = re.search(r"Created\s+(\d+)\s+page-level tasks", line)
        if m:
            self.total_pages = int(m.group(1))
            return
        m2 = re.search(r"Rendering Pages \(CPU\):\s+(\d+)%", line)
        if m2:
            self.progress_pct = max(self.progress_pct, float(m2.group(1)))
            return

    def eta_text(self) -> Optional[str]:
        now = time.perf_counter()
        if now - self.last_eta_msg_ts < 3.0:
            return None
        self.last_eta_msg_ts = now
        if self.total_pages:
            est_total = self.total_pages * self.sec_per_page_guess
            elapsed = now - self.start_ts
            if self.progress_pct > 0:
                done = self.progress_pct / 100.0
                remaining = max(0.0, est_total - elapsed) if done < 0.98 else 0.0
            else:
                remaining = max(0.0, est_total - elapsed)
            return f"Estimated time left, {int(remaining)} seconds"
        if now - self.start_ts > 5:
            return "Preparing tasks, estimating time soon"
        return None

# ---------------- UI core ----------------

def _pick_default_storage() -> str:
    if in_colab():
        return "Colab Temporary"
    if in_kaggle():
        return "Kaggle"
    return "Local"

def _storage_by_name(name: str) -> Storage:
    if name == "Google Drive":
        return ColabDriveStorage()
    if name == "Colab Temporary":
        return ColabTempStorage()
    if name == "Kaggle":
        return KaggleStorage()
    return LocalStorage()

def launch_webui():
    # available storages depend on env
    storage_choices: List[str]
    if in_colab():
        storage_choices = ["Google Drive", "Colab Temporary"]
    elif in_kaggle():
        storage_choices = ["Kaggle"]
    else:
        storage_choices = ["Local"]

    default_storage = _pick_default_storage()

    with gr.Blocks(theme=gr.themes.Soft(), title="paraOCR WebUI") as app:
        gr.Markdown("# paraOCR batch processing")
        gr.Markdown("Choose storage, add inputs, pick languages, start the run")

        with gr.Row():
            with gr.Column(scale=1):
                storage_choice = gr.Radio(
                    storage_choices,
                    value=default_storage,
                    label="Storage location"
                )
                if in_colab():
                    gr.Markdown("Tip, mount Drive with `from google.colab import drive; drive.mount('/content/drive')` for persistence")

                gr.Markdown("### 1. Input method")
                input_method = gr.Radio(
                    ["Upload a .zip file", "Upload individual files", "Use path to copy from"],
                    value="Upload a .zip file",
                    label="Input source"
                )

                zip_upload = gr.File(
                    label="Upload a .zip file",
                    file_types=[".zip"],
                    type="filepath",
                    visible=True
                )
                multi_file_upload = gr.Files(
                    label="Drop or select multiple files",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                    visible=False
                )
                gdrive_path_input = gr.Textbox(
                    label="Path to copy from",
                    placeholder="/content/drive/MyDrive/Docs or /content/some_folder or C:/data",
                    visible=False
                )

                language_selector = gr.CheckboxGroup(
                    choices=["vi", "en", "fr", "de", "es"],
                    value=["vi", "en"],
                    label="Languages"
                )

                log_mode_selector = gr.Radio(
                    choices=["Basic", "Advanced"],
                    value="Basic",
                    label="Log display mode"
                )

                start_button = gr.Button("Start processing", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 2. Live Log")
                progress_html = gr.HTML(value="<div style='height:8px;background:#eee;border-radius:6px;overflow:hidden'><div style='width:0%;height:100%'></div></div>")
                log_output = gr.Textbox(label="Processing Log", lines=16, interactive=False)

        with gr.Row():
            gr.Markdown("### 3. Processed files")
        with gr.Row():
            results_table = gr.DataFrame(
                headers=["File Name", "Status", "Source Path"],
                datatype=["str", "str", "str"],
                label="Click a row to view content",
                interactive=True
            )
        with gr.Row():
            gr.Markdown("### 4. View content")
        with gr.Row():
            text_viewer = gr.Textbox(label="File content", lines=20, interactive=False)

        # toggles
        def toggle_inputs(choice):
            if choice == "Upload a .zip file":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            if choice == "Upload individual files":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        input_method.change(
            fn=toggle_inputs,
            inputs=input_method,
            outputs=[zip_upload, multi_file_upload, gdrive_path_input]
        )

        # main handler
        def process_documents_wrapper(
            storage_choice,
            input_method,
            zip_file,
            multi_files,
            gdrive_path,
            langs,
            log_mode,                       # "Basic" or "Advanced"
            progress=gr.Progress(track_tqdm=True),
        ):
            import re
            from queue import Queue, Empty
            from threading import Thread

            # ---------- tiny progress bar renderer for the HTML component ----------
            def render_progress(pct: float, text: str = "") -> str:
                pct = max(0, min(100, int(pct)))
                return (
                    f"<div style='height:8px;background:#eee;border-radius:6px;overflow:hidden'>"
                    f"<div style='width:{pct}%;height:100%'></div></div>"
                    f"<div style='font-size:12px;margin-top:6px;color:#555'>{text}</div>"
                )

            # ---------- Basic vs Advanced filter ----------
            class UILogger:
                def __init__(self, mode="Basic"):
                    self.mode = mode
                    # key phrases for Basic mode
                    self.keep = [
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
                    self.last_emit = 0.0

                def filter(self, line: str) -> str | None:
                    if self.mode == "Advanced":
                        return line
                    l = line.strip()
                    low = l.lower()
                    if any(k in low for k in self.keep):
                        return l
                    return None

            # ---------- Phase and ETA tracker driven by your pipeline prints ----------
            class PhaseTracker:
                def __init__(self):
                    self.start = time.perf_counter()
                    self.total_pages = None
                    self.render_pct = 0.0
                    self.phase_pct = 0.0
                    self.last_eta_emit = 0.0
                    # phase weights, rough but steady
                    self.weights = {
                        "scan": 10,
                        "dispatch": 20,   # cumulative
                        "render": 90,     # render 10 to 90
                        "aggregate": 98,  # aggregate 90 to 98
                        "final": 99,
                        "done": 100,
                    }
                    self.sec_per_page_guess = 1.6  # Colab T4 friendly default

                def update_from_line(self, line: str):
                    # total pages discovered
                    m = re.search(r"Created\s+(\d+)\s+page-level tasks", line)
                    if m:
                        self.total_pages = int(m.group(1))
                        self.phase_pct = max(self.phase_pct, self.weights["dispatch"])
                        return

                    # rendering percent
                    m2 = re.search(r"Rendering Pages \(CPU\):\s+(\d+)%", line)
                    if m2:
                        rp = float(m2.group(1))
                        # map 0..100 render to 10..90 overall
                        self.render_pct = 10 + 0.8 * rp
                        self.phase_pct = max(self.phase_pct, self.render_pct)
                        return

                    if "--- Dispatching tasks" in line:
                        self.phase_pct = max(self.phase_pct, self.weights["dispatch"])
                        return
                    if "--- Starting parallel processing" in line:
                        self.phase_pct = max(self.phase_pct, 15)
                        return
                    if "--- Aggregating GPU results" in line:
                        self.phase_pct = max(self.phase_pct, self.weights["aggregate"])
                        return
                    if "--- Aggregating and writing final results" in line:
                        self.phase_pct = max(self.phase_pct, self.weights["final"])
                        return
                    if "Processing Complete" in line:
                        self.phase_pct = self.weights["done"]
                        return

                def eta_text(self) -> str | None:
                    # calm ETA, emit at most once every 10 seconds
                    now = time.perf_counter()
                    if now - self.last_eta_emit < 10.0:
                        return None
                    self.last_eta_emit = now

                    if self.total_pages:
                        est_total = self.total_pages * self.sec_per_page_guess
                        elapsed = now - self.start
                        remaining = max(0.0, est_total - elapsed)
                        # hide tiny remainders
                        if remaining < 8:
                            return None
                        return f"Estimated time left, {int(remaining)} seconds"
                    # early boot
                    if now - self.start > 6:
                        return "Preparing tasks, estimating time soon"
                    return None

                def percent(self) -> float:
                    return max(0.0, min(100.0, self.phase_pct))

                def desc(self) -> str:
                    p = self.percent()
                    if p < 10:
                        return "Scanning"
                    if p < 20:
                        return "Dispatching"
                    if p < 90:
                        return "Rendering pages"
                    if p < 99:
                        return "Aggregating"
                    if p < 100:
                        return "Writing results"
                    return "Done"

            logger = UILogger(mode=log_mode)
            tracker = PhaseTracker()

            # ---------- workspace pick ----------
            picked = pick_workspace(storage_choice)
            if len(picked) == 4:
                base, INPUT_DATA_DIR, OUTPUT_DIR, notice = picked
                log_history = ((notice or "") + "\n") if notice is not None else ""
            else:
                base, INPUT_DATA_DIR, OUTPUT_DIR = picked
                log_history = ""

            # ---------- run folders ----------
            run_ts = time.strftime("%Y%m%d-%H%M%S")
            current_input_dir = INPUT_DATA_DIR / run_ts
            current_input_dir.mkdir(parents=True, exist_ok=True)
            output_jsonl_path = OUTPUT_DIR / f"results_{run_ts}.jsonl"
            error_log_path = OUTPUT_DIR / f"errors_{run_ts}.jsonl"

            # ---------- inputs ----------
            if input_method == "Upload a .zip file":
                if not zip_file:
                    log_history += "Please upload a zip file\n"
                    yield log_history, gr.update(), render_progress(0, "Waiting for zip")
                    return
                log_history += "Unzipping input archive\n"
                with zipfile.ZipFile(zip_file) as zf:
                    zf.extractall(current_input_dir)
                log_history += "Copy complete\n"
            elif input_method == "Upload individual files":
                paths = multi_files or []
                if not paths:
                    log_history += "Please upload one or more PDF PNG or JPEG files\n"
                    yield log_history, gr.update(), render_progress(0, "Waiting for files")
                    return
                allowed = {".pdf", ".png", ".jpg", ".jpeg"}
                to_copy = [Path(p) for p in paths if Path(p).suffix.lower() in allowed]
                log_history += f"Copying {len(to_copy)} uploaded files\n"
                for p in to_copy:
                    dst = current_input_dir / p.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                log_history += "Copy complete\n"
            else:
                if not gdrive_path:
                    log_history += "Please provide a path to copy from\n"
                    yield log_history, gr.update(), render_progress(0, "Waiting for path")
                    return
                src = Path(gdrive_path)
                if not src.exists():
                    log_history += f"Path not found, {src}\n"
                    yield log_history, gr.update(), render_progress(0, "Waiting for path")
                    return
                log_history += "Copying files\n"
                for p in src.rglob("*"):
                    if p.is_file():
                        dst = current_input_dir / p.relative_to(src)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dst)
                log_history += "Copy complete\n"

            # ---------- config ----------
            selected_langs = langs or ["vi", "en"]
            backend_kwargs = {
                "languages": selected_langs,
                "gpu": True,
                "use_gpu": True,
                "use_angle_cls": True,
                "rec_batch_num": 6,
                "det_db_thresh": 0.25,
                "det_db_box_thresh": 0.35,
                "det_db_unclip_ratio": 2.2,
                "show_log": False,
            }

            config = OCRConfig.from_dict({
                "input_dir": str(current_input_dir),
                "output_path": str(output_jsonl_path),
                "error_log_path": str(error_log_path),
                "export_txt": True,
                "num_workers": 2,
                "num_gpu_workers": 1,
                "gpu_batch_size": 8,
                "dpi": 200,
                "languages": selected_langs,
                "ocr_backend_kwargs": backend_kwargs,
                "dictionary": load_dictionary(),
            })
            if config.log_performance and not config.performance_log_path:
                config.performance_log_path = Path(str(config.output_path)).with_suffix(".perf.jsonl")

            # ---------- worker thread ----------
            log_queue = Queue()
            t = Thread(target=pipeline_worker, args=(config, log_queue), daemon=True)
            t.start()

            log_history += f"Workspace, {base}\n"
            log_history += f"Starting OCR with languages, {', '.join(selected_langs)}\n"
            # initial progress pulse
            progress(0.05, desc="Starting")
            yield log_history, gr.update(), render_progress(5, "Starting")

            # ---------- stream logs ----------
            while True:
                try:
                    item = log_queue.get(timeout=0.5)
                    if item is SENTINEL:
                        break

                    tracker.update_from_line(item)

                    eta_line = tracker.eta_text()
                    flt = logger.filter(item)
                    if flt:
                        log_history += flt if flt.endswith("\n") else flt + "\n"

                    # update both the built in progress and our visible bar
                    pct = tracker.percent()
                    desc = tracker.desc()
                    progress(pct / 100.0, desc=desc)
                    bar_html = render_progress(pct, desc if not eta_line else f"{desc}, {eta_line}")

                    yield log_history, gr.update(), bar_html

                except Empty:
                    # idle pulse, keep the bar fresh
                    pct = tracker.percent()
                    desc = tracker.desc()
                    progress(pct / 100.0, desc=desc)
                    yield log_history, gr.update(), render_progress(pct, desc)

            t.join()

            # drain
            try:
                while True:
                    item = log_queue.get_nowait()
                    if item is SENTINEL:
                        continue
                    tracker.update_from_line(item)
                    flt = logger.filter(item)
                    if flt:
                        log_history += flt if flt.endswith("\n") else flt + "\n"
            except Empty:
                pass

            # final
            tracker.phase_pct = 100.0
            progress(1.0, desc="Done")
            log_history += "\nProcessing complete\n"
            results_df = scan_for_results(output_jsonl_path)
            if results_df.empty:
                log_history += "No results found, check the log above"
                yield log_history, results_df, render_progress(100, "Done")
            else:
                log_history += "Found results, displaying below"
                yield log_history, results_df, render_progress(100, "Done")

        # Compatibility shim for older notebooks
        def pick_workspace(storage_choice: str):
            # storage_choice can be "Google Drive", "Colab Temporary", "Kaggle", or "Local"
            storage = _storage_by_name(storage_choice)
            base, input_dir, output_dir, notice = storage.ensure_workspace()
            return base, input_dir, output_dir, notice or ""


        

        start_button.click(
            fn=process_documents_wrapper,
            inputs=[
                storage_choice,
                input_method,
                zip_upload,
                multi_file_upload,
                gdrive_path_input,
                language_selector,
                log_mode_selector
            ],
            outputs=[log_output, results_table, progress_html]
        )
        results_table.select(fn=view_file_content, inputs=results_table, outputs=text_viewer)

    # Colab users usually want share=True and debug=True
    app.launch(debug=True, share=in_colab())


# allow running directly
if __name__ == "__main__":
    launch_webui()
