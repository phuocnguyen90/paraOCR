here’s a drop-in README section you can paste in.

---

# Benchmarks

**Dataset:** `tests/data` (10 files ≈ 60 pages)
**Settings:** CPU workers = 10, GPU workers = 3, GPU batch size = 16, DPI = 200, languages = `vi en`
**Note:** model files were cached before timing (download time excluded).

### Results

| Backend       | Mode        | Wall time (s) | Pages | Throughput (pages/s) | Notes            |
| ------------- | ----------- | ------------- | ----- | -------------------- | ---------------- |
| **PaddleOCR** | **paraOCR** | **28.07**     | 60    | **2.14**             | `lang=latin`     |
| PaddleOCR     | vanilla     | 29.33         | 60    | 2.05                 |                  |
| **Tesseract** | **paraOCR** | **71.02**     | 60    | **0.85**             |                  |
| Tesseract     | vanilla     | 185.31        | 60    | 0.32                 |                  |
| **EasyOCR**   | **paraOCR** | **116.00**    | 60    | **0.52**             | decoder=`greedy` |
| EasyOCR       | vanilla     | 181.12        | 60    | 0.33                 |                  |

**Speedups (paraOCR vs vanilla):**

* Tesseract: \~**2.61×** (185.31 → 71.02 s)
* EasyOCR (greedy): \~**1.56×** (181.12 → 116.00 s)
* PaddleOCR: \~**1.05×** (29.33 → 28.07 s)

> ⚠️ Accuracy note: PaddleOCR’s fastest setting for this dataset used `lang=latin`. It’s very quick but may underperform on Vietnamese accuracy versus engines/models specialized for Vietnamese.

---

### Reproduce (paraOCR)

```bash
# PaddleOCR (adapter)
paraocr run -i tests/data -o paddle_paraocr.jsonl \
  -w 10 -g 3 -b 16 -d 200 --languages vi en \
  --ocr-backend paraocr.ocr_backends.paddleocr_backend.PaddleOCREngine \
  --ocr-backend-kwargs '{"languages":["vi","en"],"use_gpu":true}'
```

```bash
# Tesseract (adapter)
paraocr run -i tests/data -o tesseract_paraocr.jsonl \
  -w 10 -g 3 -b 16 -d 200 --languages vi en \
  --ocr-backend paraocr.ocr_backends.tesseract_backend.TesseractOCREngine \
  --ocr-backend-kwargs '{"languages":["vi","en"],"oem":3,"psm":6}'
```

```bash
# EasyOCR (adapter, greedy decoder for stability & fairness)
paraocr run -i tests/data -o easyocr_paraocr.jsonl \
  -w 10 -g 3 -b 16 -d 200 --languages vi en \
  --ocr-backend paraocr.ocr_backends.easyocr_backend.EasyOCREngine \
  --ocr-backend-kwargs '{"languages":["vi","en"],"gpu":true,"decoder":"greedy"}'
```

> Tip: to ensure fair timing, pre-warm model caches (e.g., `python -c "import easyocr; easyocr.Reader(['vi','en'], gpu=True)"`) before running.

### Reproduce (vanilla)

Use the simple single-threaded scripts in `tests/`:

* `tests/tesseract_vanilla_bench.py`
* `tests/paddleocr_vanilla_bench.py`
* `tests/easyocr_vanilla_bench.py`

Each script prints total wall time and pages processed; run them from the repo root so they pick up `tests/data`.
