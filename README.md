# paraOCR

**A high-performance, parallel pipeline for batch OCR processing of local files.**

paraOCR is a Python command-line tool designed to perform high-quality Optical Character Recognition (OCR) on thousands of local files (PDFs, PNGs, JPEGs) quickly and efficiently. It acts as a powerful wrapper around the excellent `easyocr` library, adding a robust parallel processing architecture to maximize the throughput of your hardware.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

-   **Parallel CPU & GPU Processing:** Maximizes hardware utilization for unparalleled speed.
-   **Multiple Concurrent GPU Workers:** Fully utilizes high-VRAM GPUs by running multiple OCR engines at once.
-   **Low & Stable RAM Usage:** A true streaming pipeline design ensures that even massive datasets can be processed without running out of system memory.
-   **Smart Text Extraction:** Automatically detects and uses high-quality text embedded in PDFs, falling back to powerful OCR only when necessary.
-   **Resumable & Robust:** The pipeline is fully resumable and gracefully handles corrupted files by logging them instead of crashing.
-   **Extensible Architecture:** Designed with hooks for future content-aware processors (e.g., table and image extraction).
-   **Detailed Performance Logging:** An optional flag generates a detailed performance log to help you tune parameters and identify bottlenecks.


## Performance Benchmarks

`paraOCR` leverages parallel processing to dramatically accelerate OCR tasks compared to a standard, sequential script. The following benchmarks were conducted on a system with an **Intel Core i5-12400 CPU** and an **NVIDIA GeForce RTX 3060 (12GB) GPU**.

#### Test 1: Large Document (1 file, 100 pages)

| Metric | Vanilla Script | `paraOCR` (Parallel) | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Total Time** | 419.4 sec | **200.9 sec** | **2.09x Faster** |
| **Throughput** | 0.24 pages/sec | **0.50 pages/sec** | **+108%** |

#### Test 2: Mixed Small Files (10 files, 60 pages total)

| Metric | Vanilla Script | `paraOCR` (Parallel) | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Total Time** | 176.1 sec | **112.5 sec** | **1.57x Faster** |
| **Throughput** | 0.34 pages/sec | **0.53 pages/sec** | **+56%** |

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

The primary command for paraOCR is `paraOCR`. All processing is driven through this single entry point.

**Basic Usage:**

Process all supported files in a directory named `my_scanned_docs` and save the results to `ocr_results.jsonl`.

```bash
paraOCR --input-dir ./my_scanned_docs --output-path ./ocr_results.jsonl
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

## Benchmarking & Optimization

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

**A:**  This is the most important new setting for performance. It depends entirely on your GPU's VRAM. Each GPU worker loads its own copy of the EasyOCR model into memory.

**Rule of Thumb:** Based on testing, a single GPU worker at 200 DPI and a batch size of 16 consumes approximately 4 GB of VRAM.

**Q: The script is running, but my GPU utilization is low or has sharp peaks and valleys.**

**A:** This is the classic sign of a **CPU bottleneck**. Your CPU cores cannot prepare images (render PDFs) fast enough to keep the GPU continuously fed.
*   **Solution 1 (Recommended):** Increase the number of CPU workers with the `-w` or `--workers` argument (e.g., `-w 10`).
*   **Solution 2:** Decrease the rendering complexity by using a lower `--dpi` (e.g., `-d 150`). This is a trade-off between speed and quality.

**Q: I ran out of GPU memory (VRAM).**

**A:** This happens when the batch of images is too large to fit in your GPU's memory.
*   **Solution:** Decrease the GPU batch size with the `-b` or `--gpu-batch-size` argument (e.g., `-b 8` or `-b 4`). High-DPI images consume significantly more VRAM.

**Q: I ran out of system memory (RAM).**

**A:** This should not happen with the current architecture. If it does, it means you are processing an extremely large number of files and the tracking objects are growing too large. Please open an issue on GitHub. A temporary workaround is to process your data in smaller sub-directories.

**Q: The script fails with a `ModuleNotFoundError` on first run.**

**A:** This usually means the package was not installed correctly.
*   **Solution:** Ensure you are in the correct virtual environment. Uninstall and reinstall the package:
    ```bash
    pip uninstall paraOCR
    pip install .
    ```

**Q: How do I handle files that failed to process?**

**A:** The script generates an error log (`paraOCR_error_log.jsonl` by default). This file contains a list of every document that failed and the reason why.
*   **Common reasons:** "file is corrupted", "format error: No default Layer config".
*   **Solution:** These files often need to be opened with a tool like Adobe Acrobat and re-saved to fix the corruption, or they may need to be manually reviewed. You can create a targeted second-pass run by pointing paraOCR at a directory containing only these failed files.

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or suggest a feature, or open a pull request with your improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.