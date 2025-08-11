# src/paraocr/webui.py
from __future__ import annotations

import os
import sys
import math
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import List, Iterable, Optional

import gradio as gr

from .ui_utils import (
    run_ocr_task,           # must be a generator that yields dicts with 'log'/'results'/'progress_html'
    view_file_content,      # (table -> text) callback
    in_colab,               # bool
)

# ───────────────────────────────────────────────────────────────────────────────
# Helpers: environment detection, GPU VRAM, ZIP/PDF collection, defaults
# ───────────────────────────────────────────────────────────────────────────────

def _in_colab() -> bool:
    return "google.colab" in sys.modules

def _cpu_core_count() -> int:
    try:
        return os.cpu_count() or 2
    except Exception:
        return 2

def _gpu_total_vram_gb() -> float:
    """
    Total visible GPU VRAM across devices (GiB). Tries torch, then nvidia-smi.
    """
    # Try PyTorch first
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            total_gib = 0.0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gib += props.total_memory / (1024 ** 3)  # bytes -> GiB
            return float(total_gib)
    except Exception:
        pass

    # Fallback to nvidia-smi
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        # nvidia-smi returns MiB per line
        mibs = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return (sum(mibs) / 1024.0) if mibs else 0.0
    except Exception:
        return 0.0

def _is_zip(path: str | Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False

def _collect_inputs(uploaded_files, session_tmpdir: Path) -> list[Path]:
    """
    Accepts a gradio UploadedFile or list thereof.
    - If it's a .zip, extracts PDFs into session_tmpdir.
    - If it's a PDF, copies it into session_tmpdir.
    Returns a flat, de-duplicated list of PDF Paths.
    """
    if not uploaded_files:
        return []
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    pdfs: list[Path] = []
    for f in uploaded_files:
        p = Path(getattr(f, "name", f))
        # Some Gradio backends keep a temp path under .name; fall back to .file if available
        if not p.exists() and hasattr(f, "file"):
            try:
                p = Path(f.file.name)
            except Exception:
                pass

        if _is_zip(p):
            try:
                with zipfile.ZipFile(p, "r") as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        if str(zi.filename).lower().endswith(".pdf"):
                            target = session_tmpdir / Path(zi.filename).name
                            with zf.open(zi) as src, open(target, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            pdfs.append(target)
            except Exception:
                # If extraction fails for any reason, skip this archive
                continue
        else:
            if p.suffix.lower() == ".pdf":
                target = session_tmpdir / p.name
                if str(p) != str(target):
                    try:
                        shutil.copy2(p, target)
                    except Exception:
                        shutil.copy(p, target)
                pdfs.append(target)

    # De-dupe and stable sort
    unique = sorted(set(pdfs), key=lambda x: x.name)
    return unique

def _suggest_defaults() -> dict:
    """
    Sensible defaults:
      - Colab: CPU=2, GPU>=1 if any GPU; else 0.
      - Local: CPU=4 (bounded by cores), GPU=2 (bounded by VRAM @ ~4GiB/worker).
    Also returns detected CPU cores and total VRAM for display.
    """
    colab = _in_colab()
    cores = _cpu_core_count()
    vram_gib = _gpu_total_vram_gb()

    # derive GPU workers: ~1 worker per 4 GiB VRAM
    vram_based_workers = int(max(0, math.floor(vram_gib / 4.0)))

    if colab:
        cpu_workers = 2
        gpu_workers = max(vram_based_workers, 1) if vram_gib > 0 else 0
    else:
        cpu_workers = min(max(4, 1), cores)           # baseline 4, bounded by cores
        gpu_workers = min(max(2, 0), max(vram_based_workers, 0))  # baseline 2, bounded by VRAM

    gpu_workers = max(0, gpu_workers)
    cpu_workers = max(1, min(cpu_workers, cores))

    # Reasonable batch sizes; tune as needed
    gpu_batch_size = 4 if gpu_workers <= 1 else 8
    cpu_chunk_size = 16

    return {
        "env": "Colab" if colab else "Local",
        "cpu_cores": cores,
        "vram_gb": round(vram_gib, 2),
        "cpu_workers": cpu_workers,
        "gpu_workers": gpu_workers,
        "gpu_batch_size": gpu_batch_size,
        "cpu_chunk_size": cpu_chunk_size,
    }

# ───────────────────────────────────────────────────────────────────────────────
# Orchestration callback for the Run button
# ───────────────────────────────────────────────────────────────────────────────

def _progress_shell() -> str:
    return (
        "<div style='height:8px;background:#eee;border-radius:6px;overflow:hidden'>"
        "<div id='paraocr-bar' style='width:0%;height:100%;background:#4f46e5;transition:width .2s ease'></div>"
        "</div><div id='paraocr-text' style='font-size:12px;margin-top:6px;color:#555'>Idle</div>"
    )

def _warn(msg: str) -> str:
    return f"⚠️ {msg}\n"

def handle_run(files, langs, log_mode, cpu_w, gpu_w, gpu_bs, cpu_chunk):
    """
    Gradio generator: prepares inputs, then streams updates from run_ocr_task.
    Expects run_ocr_task(...) to be a generator yielding dicts:
      { "log": str, "results": list|None, "progress_html": str }
    """
    # Per-run working dir
    session_tmpdir = Path(tempfile.mkdtemp(prefix="paraocr_"))
    log_prefix = ""
    try:
        pdfs = _collect_inputs(files, session_tmpdir)
        if not pdfs:
            yield _warn("No PDFs found in your upload."), gr.update(), _progress_shell()
            return

        # Sanity: auto-downgrade GPU workers if no VRAM detected
        detected_vram = _gpu_total_vram_gb()
        if int(gpu_w) > 0 and detected_vram <= 0:
            log_prefix += _warn("No GPU detected — forcing GPU workers to 0.")
            gpu_w = 0

        # Call into the pipeline
        stream = run_ocr_task(
            pdfs,
            languages=langs or [],
            log_mode=log_mode,
            num_workers=int(cpu_w),
            num_gpu_workers=int(gpu_w),
            gpu_batch_size=int(gpu_bs),
            cpu_chunk_size=int(cpu_chunk),
        )

        for update in stream:
            log_update = (log_prefix + (update.get("log") or "")).rstrip()
            log_prefix = ""  # prefix only once
            progress_update = update.get("progress_html") or _progress_shell()
            results_update = update.get("results", None)
            yield log_update, (results_update if results_update is not None else gr.update()), progress_update

    except TypeError as e:
        # Helpful hint if run_ocr_task signature wasn't updated yet
        msg = _warn(
            "run_ocr_task(...) signature seems incompatible with the new UI. "
            "Update it to accept (pdf_paths, languages, log_mode, num_workers, num_gpu_workers, gpu_batch_size, cpu_chunk_size)."
        )
        yield msg + f"\nDetails: {e}", gr.update(), _progress_shell()
    except Exception as e:
        yield _warn(f"Unexpected error: {e}"), gr.update(), _progress_shell()
    finally:
        # You may choose to keep artifacts; no cleanup here by default
        pass

# ───────────────────────────────────────────────────────────────────────────────
# Gradio App
# ───────────────────────────────────────────────────────────────────────────────

def launch_webui():
    # Storage options (kept for parity with your previous UI; not strictly required)
    if in_colab():
        storage_choices = ["Google Drive", "Colab Temporary"]
        default_storage = "Google Drive" if Path("/content/drive/MyDrive").exists() else "Colab Temporary"
    else:
        storage_choices = ["Local"]
        default_storage = "Local"

    defaults = _suggest_defaults()

    with gr.Blocks(theme=gr.themes.Soft(), title="paraOCR WebUI") as app:
        gr.Markdown("# paraOCR Batch Processing")
        gr.Markdown("Upload PDFs or a ZIP, tweak settings, and run OCR. Live logs + progress below.")

        with gr.Tabs():
            with gr.Tab("Inputs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        storage_choice = gr.Radio(storage_choices, value=default_storage, label="Storage location")
                        if in_colab():
                            gr.HTML("<span style='color:#666;font-size:12px'>Tip: mount Drive for persistence.</span>")

                        gr.Markdown("### 1) Input")
                        input_files = gr.File(
                            label="Upload PDFs or a ZIP (auto-detect & expand)",
                            file_count="multiple",
                            file_types=[".pdf", ".zip"],
                            interactive=True,
                        )

                        gr.Markdown("### 2) General Options")
                        language_selector = gr.CheckboxGroup(
                            choices=["vi", "en", "fr", "de", "es"],
                            value=["vi", "en"],
                            label="Languages",
                        )
                        log_mode = gr.Radio(
                            choices=["Basic", "Advanced"],
                            value="Basic",
                            label="Log display mode",
                        )

                        start_button = gr.Button("Start Processing", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 3) Live Log & Progress")
                        progress_html = gr.HTML(value=_progress_shell())
                        log_output = gr.Textbox(label="Processing Log", lines=18, interactive=False)

                gr.Markdown("### 4) Processed files")
                results_table = gr.DataFrame(
                    headers=["File Name", "Status", "Source Path"],
                    datatype=["str", "str", "str"],
                    label="Click a row to view content",
                    interactive=True,
                )

                gr.Markdown("### 5) View content")
                text_viewer = gr.Textbox(label="File content", lines=20, interactive=False)

            with gr.Tab("Settings"):
                env_info = gr.Markdown(
                    f"**Detected:** {defaults['env']} | "
                    f"CPU cores: {defaults['cpu_cores']} | "
                    f"VRAM: {defaults['vram_gb']} GiB",
                    visible=True,
                )

                cpu_workers = gr.Slider(
                    minimum=1,
                    maximum=_cpu_core_count(),
                    step=1,
                    value=defaults["cpu_workers"],
                    label="CPU Workers",
                )
                gpu_workers = gr.Slider(
                    minimum=0,
                    maximum=8,
                    step=1,
                    value=defaults["gpu_workers"],
                    label="GPU Workers",
                )
                gpu_batch_size = gr.Slider(
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=defaults["gpu_batch_size"],
                    label="GPU Batch Size (images/job)",
                )
                cpu_chunk_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    step=1,
                    value=defaults["cpu_chunk_size"],
                    label="CPU Chunk Size",
                )

        # Wire up actions
        start_button.click(
            fn=handle_run,
            inputs=[input_files, language_selector, log_mode, cpu_workers, gpu_workers, gpu_batch_size, cpu_chunk_size],
            outputs=[log_output, results_table, progress_html],
        )

        results_table.select(fn=view_file_content, inputs=results_table, outputs=text_viewer)

    app.launch(debug=True, share=in_colab())

if __name__ == "__main__":
    launch_webui()
