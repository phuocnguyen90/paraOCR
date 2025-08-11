# src/paraocr/webui.py
from __future__ import annotations

import gradio as gr
from pathlib import Path
from typing import List

from .ui_utils import (
    run_ocr_task,
    scan_for_results,
    view_file_content,
    in_colab,
)
import logging
from multiprocessing import Manager
from .logger import setup_logging, configure_worker_logging, PROGRESS



def launch_webui():
    # Storage options



    if in_colab():
        storage_choices = ["Google Drive", "Colab Temporary"]
        default_storage = "Google Drive" if Path("/content/drive/MyDrive").exists() else "Colab Temporary"
    else:
        storage_choices = ["Local"]
        default_storage = "Local"

    with gr.Blocks(theme=gr.themes.Soft(), title="paraOCR WebUI") as app:
        gr.Markdown("# paraOCR Batch Processing")
        gr.Markdown("Pick storage, add inputs, choose languages, and go.")

        with gr.Row():
            with gr.Column(scale=1):
                storage_choice = gr.Radio(storage_choices, value=default_storage, label="Storage location")
                if in_colab():
                    gr.HTML("<span style='color:#666;font-size:12px'>Tip: mount Drive for persistence.</span>")

                gr.Markdown("### 1. Input method")
                input_method = gr.Radio(
                    ["Upload a .zip file", "Upload individual files", "Use path to copy from"],
                    value="Upload a .zip file",
                    label="Input source",
                )

                zip_upload = gr.File(
                    label="Upload a .zip file",
                    file_types=[".zip"],
                    type="filepath",
                    visible=True,
                )
                multi_file_upload = gr.Files(
                    label="Drop or select multiple files (PDF/PNG/JPEG)",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                    visible=False,
                )
                gdrive_path_input = gr.Textbox(
                    label="Path to copy from",
                    placeholder="/content/drive/MyDrive/Docs or C:/data",
                    visible=False,
                )

                gr.Markdown("### 2. Settings")
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
                gr.Markdown("### 3. Live Log & Progress")
                progress_html = gr.HTML(
                    value=(
                        "<div style='height:8px;background:#eee;border-radius:6px;overflow:hidden'>"
                        "<div style='width:0%;height:100%;background:#4f46e5'></div></div>"
                        "<div style='font-size:12px;margin-top:6px;color:#555'>Idle</div>"
                    )
                )
                log_output = gr.Textbox(label="Processing Log", lines=16, interactive=False)

        with gr.Row():
            gr.Markdown("### 4. Processed files")
        with gr.Row():
            results_table = gr.DataFrame(
                headers=["File Name", "Status", "Source Path"],
                datatype=["str", "str", "str"],
                label="Click a row to view content",
                interactive=True,
            )

        with gr.Row():
            gr.Markdown("### 5. View content")
        with gr.Row():
            text_viewer = gr.Textbox(label="File content", lines=20, interactive=False)

        # Toggle inputs
        def toggle_inputs(choice):
            if choice == "Upload a .zip file":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            if choice == "Upload individual files":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        input_method.change(
            fn=toggle_inputs,
            inputs=input_method,
            outputs=[zip_upload, multi_file_upload, gdrive_path_input],
        )

        # Main handler
        def handle_run(storage, method, zip_f, multi_f, path_f, langs, log_mode):
            # Stream updates from the orchestrator
            for update in run_ocr_task(storage, method, zip_f, multi_f, path_f, langs, log_mode):
                log_update = update.get("log", "")
                progress_update = update.get("progress_html", "")
                results_update = update.get("results", None)
                yield log_update, results_update if results_update is not None else gr.update(), progress_update

        start_button.click(
            fn=handle_run,
            inputs=[storage_choice, input_method, zip_upload, multi_file_upload, gdrive_path_input, language_selector, log_mode],
            outputs=[log_output, results_table, progress_html],
        )

        results_table.select(fn=view_file_content, inputs=results_table, outputs=text_viewer)

    app.launch(debug=True, share=in_colab())

if __name__ == "__main__":
    launch_webui()
