# src/paraocr/cli.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
import multiprocessing as mp

from tqdm import tqdm
import time
import re, ast

from .config import OCRConfig
from .models import OCRTask
from .utils import load_processed_ids
from .logger import configure_worker_logging, setup_logging

__all__ = ["collect_tasks", "run_pipeline", "main"]

logger = logging.getLogger("paraocr")

# Helper

def _parse_backend_kwargs(val) -> dict:
    """
    Accept several syntaxes for --ocr-backend-kwargs:
      1) JSON (double quotes)                      {"languages":["vi","en"],"oem":3,"psm":6}
      2) JSON wrapped in single quotes             '{"languages":["vi","en"],"oem":3,"psm":6}'
      3) Python-literal dict with single quotes    {'languages': ['vi','en'], 'oem': 3, 'psm': 6}
      4) key=value pairs separated by , or ;       languages=vi,en;oem=3;psm=6
    """
    if isinstance(val, dict):
        return dict(val)
    if not isinstance(val, str):
        return {}

    s = val.strip()
    # Strip outer quotes like '"{...}"' or "'{...}'"
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        s = s[1:-1].strip()

    # Try strict JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try Python literal dict
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, dict):
            return lit
    except Exception:
        pass

    # Fallback: key=value pairs
    out: dict = {}
    parts = re.split(r"[;,]\s*", s)
    for part in parts:
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            continue

        # Clean key/value: remove quotes, stray braces/brackets, trailing commas
        k = k.strip().strip('"\'' ).lstrip("{[").rstrip("}]").strip()
        v = v.strip().strip('"\'' ).lstrip("{[").rstrip("}]").rstrip(",").strip()

        # List support like vi,en
        if "," in v and not re.search(r"[\{\}\[\]]", v):
            v = [x.strip().strip('"\'' ) for x in v.split(",") if x.strip()]

        # Coerce scalars
        if isinstance(v, str):
            low = v.lower()
            if low in ("true", "false"):
                v = (low == "true")
            else:
                # integers/floats (allow a trailing comma/bracket already stripped above)
                if re.fullmatch(r"-?\d+", v):
                    v = int(v)
                elif re.fullmatch(r"-?\d+\.\d*", v):
                    v = float(v)

        out[k] = v

    if out:
        return out

    raise SystemExit(f"Invalid --ocr-backend-kwargs. Could not parse: {val!r}")


def _normalize_common_backend_kwargs(d: dict) -> dict:
    """
    Backend-agnostic cleanup:
      - hyphen-case -> snake_case
      - lowercase keys
      - normalize languages: 'lang' -> 'languages', and "vi,en" -> ["vi","en"]
    """
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        key = k.strip().lower().replace("-", "_")
        out[key] = v

    # normalize languages
    if "languages" not in out and "lang" in out:
        out["languages"] = out.pop("lang")
    if "languages" in out:
        langs = out["languages"]
        if isinstance(langs, str):
            # split "vi,en" -> ["vi","en"]
            out["languages"] = [s.strip() for s in langs.split(",")] if "," in langs else [langs.strip()]
        elif isinstance(langs, (set, tuple)):
            out["languages"] = list(langs)
        # else: assume list/dict are already fine

    return out



def collect_tasks(config: OCRConfig) -> List[OCRTask]:
    logger.info("Collecting and filtering tasks")
    processed_paths = set()
    if not getattr(config, "force_rerun", False):
        processed_paths = load_processed_ids(config.output_path)
        if processed_paths:
            logger.info("Found %d previously processed files to skip", len(processed_paths))

    tasks: List[OCRTask] = []
    ignore_keywords_lower = [k.lower() for k in getattr(config, "ignore_keywords", [])]

    if not config.input_dir.exists():
        logger.error("Input directory does not exist, %s", config.input_dir)
        return []

    all_files_in_dir = list(config.input_dir.rglob("*"))
    logger.info("Scanning %d total paths in input directory", len(all_files_in_dir))

    for file_path in tqdm(all_files_in_dir, desc="Filtering tasks"):
        if not file_path.is_file():
            continue
        file_path_str = str(file_path)
        if file_path_str in processed_paths:
            continue
        filename_lower = file_path.name.lower()
        if any(keyword in filename_lower for keyword in ignore_keywords_lower):
            continue
        if filename_lower.endswith((".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            tasks.append(OCRTask(source_path=file_path))

    logger.info("Selected %d files for processing", len(tasks))
    return tasks


def run_pipeline(config: OCRConfig):
    """
    Build directories and run the pipeline (non-UI CLI).
    """
    from .parallel import OCRRunner  # local import to avoid import cycles

    # Ensure folders exist
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(config, "error_log_path", None):
        config.error_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ParaOCR")
    logger.info("Input directory, %s", config.input_dir)
    logger.info("Output file, %s", config.output_path)
    logger.info(
        "CPU workers, %s | GPU workers, %s | GPU batch size, %s | DPI, %s",
        getattr(config, "num_workers", None),
        getattr(config, "num_gpu_workers", None),
        getattr(config, "gpu_batch_size", None),
        getattr(config, "dpi", None),
    )

    tasks_to_run = collect_tasks(config)
    if tasks_to_run:
        logger.info("Starting OCR process on %d new files", len(tasks_to_run))
        runner = OCRRunner(config)
        runner.run(tasks_to_run)
    else:
        logger.info("No new files to process based on current settings, all tasks are complete")

    logger.info("ParaOCR processing complete")


# -------------------------------
# CLI parsing
# -------------------------------

def _build_run_parser(subparsers: argparse._SubParsersAction | argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Build the 'run' (pipeline) parser. If 'subparsers' is the root parser,
    this also works for legacy (no-subcommand) mode.
    """
    if isinstance(subparsers, argparse.ArgumentParser):
        p = subparsers
    else:
        p = subparsers.add_parser("run", help="Run the OCR pipeline in non-UI CLI mode")
    p.add_argument("-i", "--input-dir", type=Path, required=True, help="Directory containing files to OCR")
    p.add_argument("-o", "--output-path", type=Path, required=True, help="Path to save the output results JSONL")
    p.add_argument("-w", "--workers", type=int, help="Number of CPU worker processes for rendering")
    p.add_argument("-g", "--gpu-workers", type=int, help="Number of GPU worker processes")
    p.add_argument("-b", "--gpu-batch-size", type=int, help="Number of images to send to the GPU in one batch")
    p.add_argument("-d", "--dpi", type=int, help="DPI to use for rendering PDF pages")
    p.add_argument("-l", "--languages", nargs="+", default=["vi", "en"], help="Language codes for OCR")
    p.add_argument("--beamsearch", action="store_true", help="Enable beam search for OCR backends that support it")
    p.add_argument(
        "--ignore-keyword",
        action="append",
        dest="ignore_keywords",
        help="Keyword in filename to ignore, can be used multiple times",
    )
    p.add_argument("--force-rerun", action="store_true", help="Reprocess all files and ignore previous results")
    p.add_argument("--error-log-path", type=Path, help="Path to save the error log JSONL file")
    p.add_argument(
        "--pdf-engine",
        type=str,
        default="pymupdf",
        choices=["pymupdf"],
        help="Underlying engine for PDF processing",
    )
    p.add_argument("--export-txt", action="store_true", help="Also export a discrete txt per document")
    p.add_argument(
        "--ocr-backend",
        type=str,
        default="paraocr.ocr_backends.easyocr_backend.EasyOCREngine",
        help="Dotted path to an OCR backend class",
    )
    p.add_argument(
        "--ocr-backend-kwargs",
        type=str,
        default="{}",
        help='JSON dict for backend init kwargs, e.g. {"languages": ["vi","en"], "gpu": true}',
    )
    perf_group = p.add_argument_group("Performance logging")
    perf_group.add_argument("--log-performance", action="store_true", help="Enable performance logging to a file")
    perf_group.add_argument("--performance-log-path", type=Path, help="Path for the performance log JSONL file")
    return p


def _build_webui_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    wp = subparsers.add_parser("webui", help="Launch the paraOCR Web UI (Gradio)")
    wp.add_argument("--share", action="store_true", help="Create a public Gradio link (useful on remote/Colab)")
    wp.add_argument("--server-name", default="127.0.0.1", help="Host to bind (use 0.0.0.0 to expose on LAN)")
    wp.add_argument("--server-port", type=int, default=7860, help="Port to bind")
    wp.add_argument("--no-debug", action="store_true", help="Disable debug logs in the UI process")
    return wp


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="paraOCR — high performance OCR with optional Web UI")
    subparsers = parser.add_subparsers(dest="command")

    # subcommands
    run_parser = _build_run_parser(subparsers)
    _build_webui_parser(subparsers)

    # Legacy mode: if user passed no subcommand but provided -i/-o, treat as 'run'
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"run", "webui"}:
        # Try parsing as legacy 'run' (keeps backwards compatibility)
        legacy_parser = argparse.ArgumentParser(add_help=False)
        _build_run_parser(legacy_parser)
        try:
            args = legacy_parser.parse_args(argv)
            args.command = "run"
            return args
        except SystemExit:
            # Fall back to full parser to show proper help
            pass

    return parser.parse_args(argv)


# -------------------------------
# Entry points
# -------------------------------

def _launch_webui_from_cli(args: argparse.Namespace) -> None:
    """
    Launch Gradio Web UI. We keep parameters simple and delegate to webui.launch_webui().
    """
    try:
        from .webui import launch_webui, in_colab
    except Exception as e:
        raise SystemExit(f"Web UI components not available: {e}")

    # The current launch_webui() picks share=True automatically on Colab.
    # If user forces --share, we can set an env hint that your launch_webui reads
    # (optional; if you want to support server_name/port explicitly, expose them in launch_webui).
    import os
    if args.share:
        os.environ["PARAOCR_WEBUI_SHARE"] = "1"
    if args.server_name:
        os.environ["PARAOCR_WEBUI_SERVER_NAME"] = str(args.server_name)
    if args.server_port:
        os.environ["PARAOCR_WEBUI_SERVER_PORT"] = str(args.server_port)
    if args.no_debug:
        os.environ["PARAOCR_WEBUI_DEBUG"] = "0"

    # Simply call the UI launcher; it will read envs or fall back to defaults.
    launch_webui()


def _run_from_cli(args: argparse.Namespace) -> None:
    ctx = mp.get_context("spawn")
    log_queue = ctx.Manager().Queue(-1)
    # Derive a logfile next to the output JSONL (optional but handy)
    log_file = None
    try:
        if args.output_path:
            base_name = Path(args.output_path).with_suffix(".log").name
        else:
            base_name = f"paraocr_{time.strftime('%Y%m%d-%H%M%S')}.log"
        log_file = Path.cwd() / base_name
    except Exception:
        log_file = Path.cwd() / f"paraocr_{time.strftime('%Y%m%d-%H%M%S')}.log"

    listener = setup_logging(
        log_queue=log_queue,
        text_ui_queue=None,       # no Gradio here
        event_ui_queue=None,      # no progress events to UI
        level=logging.INFO,
        file_path=log_file,
    )
    listener.start()

    try:
        # Normalize backend kwargs
        backend_kwargs = _parse_backend_kwargs(args.ocr_backend_kwargs) \
            if isinstance(args.ocr_backend_kwargs, str) \
            else (dict(args.ocr_backend_kwargs) if args.ocr_backend_kwargs else {})
        backend_kwargs = _normalize_common_backend_kwargs(backend_kwargs)

        # Build config dict and prune None so dataclass defaults apply
        cfg_dict = {
            "input_dir": args.input_dir,
            "output_path": args.output_path,
            "error_log_path": args.error_log_path,
            "languages": args.languages,
            "ignore_keywords": args.ignore_keywords or [],
            "beamsearch": args.beamsearch,
            "force_rerun": args.force_rerun,
            "export_txt": args.export_txt,
            "log_performance": args.log_performance,
            "performance_log_path": args.performance_log_path,
            "pdf_engine": args.pdf_engine,
            "ocr_backend": args.ocr_backend,
            "ocr_backend_kwargs": backend_kwargs,
            # optional tuning fields
            "num_workers": args.workers,
            "num_gpu_workers": args.gpu_workers,
            "gpu_batch_size": args.gpu_batch_size,
            "dpi": args.dpi,
            # >>> IMPORTANT: give the pipeline the same process-safe queue <<<
            "log_queue": log_queue,
        }
        cfg_dict = {k: v for k, v in cfg_dict.items() if v is not None}

        config = OCRConfig.from_dict(cfg_dict)

        # Run the non-UI pipeline
        run_pipeline(config)

    finally:
        # Always stop the listener so the process can exit cleanly
        try:
            listener.stop()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None):
    args = _parse_args(argv)

    if args.command == "webui":
        _launch_webui_from_cli(args)
        return

    if args.command == "run":
        _run_from_cli(args)
        return

    # No subcommand and not legacy-compatible args → show help
    print("Usage:\n  paraocr run -i <input_dir> -o <output.jsonl> [options]\n  paraocr webui [--share] [--server-name 0.0.0.0] [--server-port 7860]")
    sys.exit(2)

# src/paraocr/cli.py
def webui_entry():
    # behaves like running: paraocr webui
    main(["webui"])


if __name__ == "__main__":
    main()
