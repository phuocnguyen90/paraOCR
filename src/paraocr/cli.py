# src/paraocr/cli.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

from .config import OCRConfig
from .models import OCRTask
from .utils import load_processed_ids, setup_logging


__all__ = ["collect_tasks", "run_pipeline", "main"]

logger = logging.getLogger("paraocr")


def collect_tasks(config: OCRConfig) -> List[OCRTask]:
    logger.info("Collecting and filtering tasks")
    processed_paths = set()
    if not config.force_rerun:
        processed_paths = load_processed_ids(config.output_path)
        if processed_paths:
            logger.info("Found %d previously processed files to skip", len(processed_paths))

    tasks: List[OCRTask] = []
    ignore_keywords_lower = [k.lower() for k in config.ignore_keywords]

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
    Build directories and run the pipeline.
    """
    from .parallel import OCRRunner  # local import to avoid import cycles

    # Ensure folders exist
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.error_log_path:
        config.error_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ParaOCR")
    logger.info("Input directory, %s", config.input_dir)
    logger.info("Output file, %s", config.output_path)
    logger.info(
        "CPU workers, %s | GPU workers, %s | GPU batch size, %s | DPI, %s",
        config.num_workers,
        config.num_gpu_workers,
        config.gpu_batch_size,
        config.dpi,
    )

    tasks_to_run = collect_tasks(config)
    if tasks_to_run:
        logger.info("Starting OCR process on %d new files", len(tasks_to_run))
        runner = OCRRunner(config)
        runner.run(tasks_to_run)
    else:
        logger.info("No new files to process based on current settings, all tasks are complete")

    logger.info("ParaOCR processing complete")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="paraOCR, high performance file OCR")

    parser.add_argument("-i", "--input-dir", type=Path, required=True, help="Directory containing files to OCR")
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="Path to save the output results JSONL")
    parser.add_argument("-w", "--workers", type=int, help="Number of CPU worker processes for rendering")
    parser.add_argument("-b", "--gpu-batch-size", type=int, help="Number of images to send to the GPU in one batch")
    parser.add_argument("-d", "--dpi", type=int, help="DPI to use for rendering PDF pages")
    parser.add_argument("-l", "--languages", nargs="+", default=["vi", "en"], help="Language codes for OCR")
    parser.add_argument("--beamsearch", action="store_true", help="Enable beam search for OCR backends that support it")
    parser.add_argument(
        "--ignore-keyword",
        action="append",
        dest="ignore_keywords",
        help="Keyword in filename to ignore, can be used multiple times",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Reprocess all files and ignore previous results")
    parser.add_argument("--error-log-path", type=Path, help="Path to save the error log JSONL file")
    parser.add_argument(
        "--pdf-engine",
        type=str,
        default="pymupdf",
        choices=["pymupdf"],
        help="Underlying engine for PDF processing",
    )
    parser.add_argument("--export-txt", action="store_true", help="Also export a discrete txt per document")
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default="paraocr.ocr_backends.easyocr_backend.EasyOCREngine",
        help="Dotted path to an OCR backend class",
    )
    parser.add_argument(
        "--ocr-backend-kwargs",
        type=str,
        default="{}",
        help='JSON dict for backend init kwargs, for example {"languages": ["vi","en"], "gpu": true}',
    )

    perf_group = parser.add_argument_group("Performance logging")
    perf_group.add_argument("--log-performance", action="store_true", help="Enable performance logging to a file")
    perf_group.add_argument("--performance-log-path", type=Path, help="Path for the performance log JSONL file")

    return parser.parse_args()


def main():
    # CLI logging to stdout
    setup_logging()  # no queue handlers, we will attach a console handler below
    if not logger.handlers:
        h = logging.StreamHandler(stream=sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)

    args = _parse_args()

    # Normalize backend kwargs
    try:
        backend_kwargs = (
            json.loads(args.ocr_backend_kwargs)
            if isinstance(args.ocr_backend_kwargs, str)
            else dict(args.ocr_backend_kwargs)
        )
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON for --ocr-backend-kwargs, {e}")

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
        "workers": args.workers,             # temp keys for pruning
        "gpu_batch_size": args.gpu_batch_size,
        "dpi": args.dpi,
    }

    # Map temp keys to real names if set
    if cfg_dict.pop("workers", None) is not None:
        cfg_dict["num_workers"] = args.workers
    if cfg_dict["gpu_batch_size"] is None:
        cfg_dict.pop("gpu_batch_size")
    if cfg_dict["dpi"] is None:
        cfg_dict.pop("dpi")

    # Remove explicit None values
    cfg_dict = {k: v for k, v in cfg_dict.items() if v is not None}

    config = OCRConfig.from_dict(cfg_dict)
    run_pipeline(config)


if __name__ == "__main__":
    main()
