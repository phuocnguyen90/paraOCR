# /content/bench_colab.py
from __future__ import annotations
import json, os, re, shlex, subprocess, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- paths & knobs ----
REPO_SRC = Path("/content/paraOCR-src")          # cloned repo for tests/data
INPUT_DIR = REPO_SRC / "tests" / "data"
OUT_DIR   = Path("/content/paraocr_bench/outputs")
REPORT    = Path("/content/paraocr_bench/bench_summary.json")

# Safe defaults for Colab T4
CPU_WORKERS = int(os.environ.get("PARAOCR_CPU", "2"))
GPU_WORKERS = int(os.environ.get("PARAOCR_GPU", "1"))  # 1 avoids CUDA re-init issues
GPU_BATCH   = int(os.environ.get("PARAOCR_BATCH", "16"))
DPI         = int(os.environ.get("PARAOCR_DPI", "200"))
LANGS       = ["vi", "en"]

# Backends (dotted paths in your package)
EASYOCR_BACKEND  = "paraocr.ocr_backends.easyocr_backend.EasyOCREngine"
TESS_BACKEND     = "paraocr.ocr_backends.tesseract_backend.TesseractOCREngine"
PADDLE_BACKEND   = "paraocr.ocr_backends.paddleocr_backend.PaddleOCREngine"

# Vanilla scripts (from the repo you cloned)
V_TESS = REPO_SRC / "tests" / "tesseract_vanilla_bench.py"
V_PADD = REPO_SRC / "tests" / "paddleocr_vanilla_bench.py"
V_EASY = REPO_SRC / "tests" / "easyocr_vanilla_bench.py"

# Shared model cache to prevent re-downloads across workers
EASYOCR_CACHE = "/content/.cache/easyocr"
os.makedirs(EASYOCR_CACHE, exist_ok=True)

@dataclass
class BenchRow:
    backend: str        # easyocr | paddleocr | tesseract
    mode: str           # vanilla | paraocr
    pages: Optional[int]
    wall_s: float
    out_jsonl: Optional[str] = None
    tail: Optional[str] = None
    notes: str = ""

def run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout, time.perf_counter() - t0

def parse_vanilla(stdout: str) -> Tuple[Optional[int], Optional[float]]:
    pages = None; wall = None
    m1 = re.search(r"Total Pages Processed:\s+(\d+)", stdout)
    if m1: pages = int(m1.group(1))
    m2 = re.search(r"Total Wall-Clock Time:\s+([\d\.]+)\s*s", stdout)
    if m2: wall = float(m2.group(1))
    return pages, wall

def count_pages_jsonl(path: Path) -> int:
    if not path.exists(): return 0
    pages = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "total_pages" in obj:
                pages += int(obj.get("total_pages") or 0)
    return pages

def prewarm():
    print("== Prewarm ==")
    # EasyOCR
    try:
        import easyocr
        easyocr.Reader(LANGS, gpu=True, model_storage_directory=EASYOCR_CACHE)
        print("  EasyOCR ready")
    except Exception as e:
        print("  EasyOCR prewarm skipped:", e)

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        import paddle
        use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        PaddleOCR(use_angle_cls=True, lang="la", use_gpu=use_gpu, show_log=False)
        print(f"  PaddleOCR ready (gpu={use_gpu})")
    except Exception as e:
        print("  PaddleOCR prewarm skipped:", e)

    # Tesseract
    try:
        import pytesseract
        print("  Tesseract:", pytesseract.get_tesseract_version())
    except Exception as e:
        print("  Tesseract check skipped:", e)

def run_vanilla_all() -> List[BenchRow]:
    rows: List[BenchRow] = []
    # Tesseract
    if V_TESS.exists():
        rc, out, dt = run([sys.executable, str(V_TESS)])
        pages, wall = parse_vanilla(out)
        rows.append(BenchRow("tesseract", "vanilla", pages, wall or dt,
                             tail="\n".join(out.splitlines()[-15:])))
    else:
        print("[warn] missing", V_TESS)
    # Paddle
    if V_PADD.exists():
        rc, out, dt = run([sys.executable, str(V_PADD)])
        pages, wall = parse_vanilla(out)
        rows.append(BenchRow("paddleocr", "vanilla", pages, wall or dt,
                             tail="\n".join(out.splitlines()[-15:])))
    else:
        print("[warn] missing", V_PADD)
    # EasyOCR
    if V_EASY.exists():
        rc, out, dt = run([sys.executable, str(V_EASY)])
        pages, wall = parse_vanilla(out)
        rows.append(BenchRow("easyocr", "vanilla", pages, wall or dt,
                             tail="\n".join(out.splitlines()[-15:])))
    else:
        print("[warn] missing", V_EASY)
    return rows

def paraocr_cmd(backend_path: str, backend_kwargs: Dict, out_jsonl: Path) -> List[str]:
    arg = json.dumps(backend_kwargs)
    return [
        "paraocr", "run",
        "-i", str(INPUT_DIR),
        "-o", str(out_jsonl),
        "-w", str(CPU_WORKERS),
        "-g", str(GPU_WORKERS),
        "-b", str(GPU_BATCH),
        "-d", str(DPI),
        "--languages", *LANGS,
        "--ocr-backend", backend_path,
        "--ocr-backend-kwargs", arg,
    ]

def run_paraocr_all() -> List[BenchRow]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[BenchRow] = []

    # EasyOCR (greedy decoder for stability/fairness)
    easy_out = OUT_DIR / "paraocr_easyocr.jsonl"
    cmd = paraocr_cmd(
        EASYOCR_BACKEND,
        {"languages": LANGS, "gpu": True, "decoder": "greedy",
         "model_storage_directory": EASYOCR_CACHE},
        easy_out,
    )
    print("\n[paraocr] easyocr:", " ".join(shlex.quote(x) for x in cmd))
    t0 = time.perf_counter()
    rc, out, _ = run(cmd)
    wall = time.perf_counter() - t0
    rows.append(BenchRow("easyocr", "paraocr", count_pages_jsonl(easy_out), wall,
                         str(easy_out), "\n".join(out.splitlines()[-15:]),
                         notes=f"rc={rc}" if rc else ""))

    # PaddleOCR (lang=latin; GPU if available)
    padd_out = OUT_DIR / "paraocr_paddleocr.jsonl"
    cmd = paraocr_cmd(
        PADDLE_BACKEND,
        {"languages": LANGS, "use_gpu": True, "rec_batch_num": 12},
        padd_out,
    )
    print("\n[paraocr] paddleocr:", " ".join(shlex.quote(x) for x in cmd))
    t0 = time.perf_counter()
    rc, out, _ = run(cmd)
    wall = time.perf_counter() - t0
    rows.append(BenchRow("paddleocr", "paraocr", count_pages_jsonl(padd_out), wall,
                         str(padd_out), "\n".join(out.splitlines()[-15:]),
                         notes=f"rc={rc}" if rc else ""))

    # Tesseract (CPU)
    tess_out = OUT_DIR / "paraocr_tesseract.jsonl"
    cmd = paraocr_cmd(
        TESS_BACKEND,
        {"languages": LANGS, "oem": 3, "psm": 6},
        tess_out,
    )
    print("\n[paraocr] tesseract:", " ".join(shlex.quote(x) for x in cmd))
    t0 = time.perf_counter()
    rc, out, _ = run(cmd)
    wall = time.perf_counter() - t0
    rows.append(BenchRow("tesseract", "paraocr", count_pages_jsonl(tess_out), wall,
                         str(tess_out), "\n".join(out.splitlines()[-15:]),
                         notes=f"rc={rc}" if rc else ""))

    return rows

def print_table(rows: List[BenchRow]) -> None:
    print("\n=== Benchmark Summary (Colab/T4 defaults: CPU=2, GPU=1, BATCH=16, DPI=200) ===")
    print("{:<11} {:<9} {:>7} {:>10} {:>12}".format("Backend","Mode","Pages","Wall(s)","Throughput"))
    print("-"*60)
    for r in rows:
        tp = (r.pages / r.wall_s) if (r.pages and r.wall_s) else None
        print("{:<11} {:<9} {:>7} {:>10.2f} {:>12}".format(
            r.backend, r.mode, r.pages if r.pages is not None else "—", r.wall_s,
            f"{tp:.2f}" if tp else "—"
        ))
    print("-"*60)
    print("Report JSON:", REPORT)

def main():
    print("Input dir:", INPUT_DIR)
    if not INPUT_DIR.exists():
        raise SystemExit(f"Missing dataset: {INPUT_DIR}")

    # Suggested for Colab to avoid CUDA-in-fork issues:
    os.environ.setdefault("PARAOCR_BENCH_GPU", str(GPU_WORKERS))
    os.environ.setdefault("PARAOCR_BENCH_CPU", str(CPU_WORKERS))

    prewarm()

    vanilla = run_vanilla_all()
    para    = run_paraocr_all()

    rows = vanilla + para
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    print_table(rows)

if __name__ == "__main__":
    main()
