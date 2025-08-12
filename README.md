# ParaOCR

**A high-performance, parallel pipeline for batch OCR processing of local files.**

`paraOCR` is a Python library for high-throughput OCR on large collections of files—PDFs, PNGs, JPEGs—at the speed your hardware allows. Built around EasyOCR, it adds production-grade features such as true batch processing, a parallel CPU/GPU architecture, and detailed logging.

Originally CLI-first, it now also ships with a Web UI for ease of use. After installing the library, launch it from the CLI with `paraocr webui` (or `python -m paraocr.cli webui`).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

-   **Parallel CPU & GPU Processing:** Maximizes hardware utilization for unparalleled speed.
-   **Multiple Concurrent GPU Workers:** Fully utilizes high-VRAM GPUs by running multiple OCR engines at once.
-   **Low & Stable RAM Usage:** A true streaming pipeline design ensures that even massive datasets can be processed without running out of system memory.
-   **Resumable & Robust:** The pipeline is fully resumable and gracefully handles corrupted files by logging them instead of crashing.
-   **Extensible Architecture:** Pluggable, content-aware processors (e.g., table and image extraction) with support for multiple OCR backends—LLM-assisted OCR on the roadmap.
-   **Detailed Performance Logging:** An optional flag generates a detailed performance log to help you tune parameters and identify bottlenecks.


## Performance Benchmarks


**Dataset:** 10 files (\~60 pages) • **Settings:** CPU=10, GPU=3, Batch=16, DPI=200, langs=`vi en`
*(Model download time excluded.)*

| Backend       | Mode        | Wall time    | Throughput   |
| ------------- | ----------- | ------------ | ------------ |
| **PaddleOCR** | **paraOCR** | **28.07 s**  | **2.14 p/s** |
| PaddleOCR     | vanilla     | 29.33 s      | 2.05 p/s     |
| **Tesseract** | **paraOCR** | **71.02 s**  | **0.85 p/s** |
| Tesseract     | vanilla     | 185.31 s     | 0.32 p/s     |
| **EasyOCR**   | **paraOCR** | **116.00 s** | **0.52 p/s** |
| EasyOCR       | vanilla     | 181.12 s     | 0.33 p/s     |

**Speedups (paraOCR vs. vanilla):**
Tesseract **\~2.6×**, EasyOCR **\~1.6×**, PaddleOCR **\~1.05×**.

> Note: PaddleOCR here uses `lang=latin` (fast; Vietnamese accuracy may be lower than VN-specific models).


---

## Installation

Ensure you have a compatible version of Python (3.8+ recommended).

1.  **Install PyTorch with GPU support (Recommended):**
    For the best performance, install a version of PyTorch that matches your system's CUDA version *before* installing paraOCR. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct command for your system.

    *Example for CUDA 12.1:*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Install paraOCR:**

    You can clone this repo and install this library using:
    ```bash
    git clone https://github.com/phuocnguyen90/paraOCR.git
    cd paraOCR
    pip install -e .
    ```

    Install the library directly from PyPI (WIP):
    ```bash
    pip install paraOCR
    ```

## Quickstart Guide

The primary command for paraOCR is `paraocr`. All processing is driven through this single entry point.

**Basic Usage:**

Process all supported files in a directory named `my_scanned_docs` and save the results to `ocr_results.jsonl`.

```bash
paraocr --input-dir ./my_scanned_docs --output-path ./ocr_results.jsonl
```

**Advanced Usage (Tuning for Performance):**

This command uses 10 CPU workers, a GPU batch size of 32, renders PDFs at a high-quality 300 DPI, and ignores any files with "screenshot" in the name.


```bash
paraOCR \
    --input-dir ./large_dataset \
    --output-path ./large_dataset_results.jsonl \
    --workers 10 \
    --gpu-batch-size 32 \
    --dpi 300 \
    --ignore-keyword screenshot
```

## Command-Line Interface (CLI) Reference

| Argument | Short | Description | Default |
|---|---|---|---|
| `--input-dir` | `-i` | **(Required)** Directory containing files to OCR. | - |
| `--output-path`| `-o` | **(Required)** Path to save the output results JSONL file. | - |
| `--workers` | `-w` | Number of CPU worker processes for rendering PDFs. | `cpu_count - 2` |
| `--gpu-batch-size` | `-b`| Number of images to send to the GPU in one batch. | `16` |
| `--dpi` | `-d` | DPI to use for rendering PDF pages. | `200` |
| `--languages` | `-l` | List of language codes for EasyOCR (e.g., `vi en ja`). | `vi en` |
| `--beamsearch` | | Enable beam search in EasyOCR (slower, more accurate). | `False` |
| `--ignore-keyword` | | A keyword in a filename to ignore (case-insensitive). Can be used multiple times. | `[]` |
| `--force-rerun` | | Force reprocessing of all files, ignoring previous results. | `False` |
| `--error-log-path` | | Path to save a log of files that failed to process. | `paraOCR_error_log.jsonl` |
| `--dictionary-path` | | Path to a dictionary file for native text quality checks. | `vi_dict.txt` |
| `--log-performance`| | Enable detailed performance logging to a file. | `False` |
| `--performance-log-path` | | Path to save the performance log JSONL file. | `paraOCR_performance_log.jsonl` |

    

## Output Format

The script produces a JSON Lines (`.jsonl`) file, where each line is a JSON object representing one processed document.

**Example `ocr_results.jsonl` entry:**
```json
{
    "source_path": "my_scanned_docs\\contracts\\contract_abc.pdf",
    "text": "HỢP ĐỒNG KINH TẾ\nBên A: Công ty TNHH Example...\n--- PAGE BREAK ---\n...Điều khoản 2: Thanh toán...",
    "method": "easyocr_cuda_parallel",
    "error": null
}
```

## Web UI
Launch the UI (local):
```bash
paraocr webui
# or explicitly host:
paraocr webui --server-name 0.0.0.0 --server-port 7860
```
In Google Colab (public share link):

```bash
!pip install -U git+https://github.com/phuocnguyen90/paraOCR.git
!paraocr webui --share
```

Alternatively, launch as a module or call the function:
```bash
python -m paraocr.cli webui --share
```
or inside Python:
```python
from paraocr.webui import launch_webui
launch_webui()
```

### Web UI Basics
Upload PDFs or a ZIP (ZIPs auto-expand).

Settings → adjust CPU workers, GPU workers, GPU batch size, CPU chunk size.

Start → watch live logs and progress/ETA; results appear in a table; click a row to preview text.

## Optimization

paraOCR includes a detailed performance logging feature to help you tune parameters for your specific hardware and dataset.

To enable it, add the `--log-performance` flag to your command:
```bash
paraOCR -i ./docs -o ./res.jsonl -w 8 -b 16 --log-performance
```
This will create a `paraOCR_performance_log.jsonl` file with structured timing data for every major operation. 

### Performance Log Metrics:

`cpu_render_file`: Measures the time a single CPU worker takes to process one source file (read, check native text, and/or render pages to temporary images). This helps identify if your CPU is the bottleneck.

`gpu_ocr_batch`: Measures the time the GPU takes to perform OCR on an entire batch of images. This log includes the batch size and the resulting throughput (pages/second).



## FAQ & Basic Debugging

**Q: How do I choose the right number of GPU workers `(--gpu-workers`)**

**A:**  Aach worker loads its own model; ~4 GB VRAM/worker is a good rule of thumb. If you see OOM, reduce batch size or workers..

**Q: The script is running, but my GPU utilization is low or has sharp peaks and valleys.**

**A:** Likely CPU-bound—raise `--workers` or lower `--dpi`.

**Q: I ran out of GPU memory (VRAM).**

**A:** This happens when the batch of images is too large to fit in your GPU's memory. Lower `--gpu-batch-size` (e.g., 8 → 4).


**Q: How do I handle files that failed to process?**

**A:** The script generates an error log (`paraOCR_error_log.jsonl` by default). This file contains a list of every document that failed and the reason why.
*   **Common reasons:** "file is corrupted", "format error: No default Layer config".
*   **Solution:** These files often need to be opened with a tool like Adobe Acrobat and re-saved to fix the corruption, or they may need to be manually reviewed. You can create a targeted second-pass run by pointing paraOCR at a directory containing only these failed files.

**Q: How do I pause/resume the process?**

**CLI:** Press **Ctrl+C** to stop safely. To resume, rerun the **exact same command** (same `--input-dir` and `--output-path`). The runner reads the existing results file and skips anything already processed.

* Want to redo everything? Add `--force-rerun`.
* If you hard-killed the job and the last JSONL line is truncated, just rerun—the loader is tolerant; worst case, delete the last partial line.

**Web UI:** Resume is not supported yet (WIP). For now, either:

* use the CLI with the same `--output-path` to resume, or
* re-upload only the files that still need processing.


## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or suggest a feature, or open a pull request with your improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.