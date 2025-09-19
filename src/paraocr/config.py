# paraOCR/config.py
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
import tempfile
from multiprocessing import cpu_count
from importlib.resources import files


def get_default_dictionary() -> Set[str]:
    # try packaged resource first
    try:
        txt = files("paraocr").joinpath("vi_full.txt").read_text(encoding="utf-8")
        return {line.strip().lower() for line in txt.splitlines() if line.strip()}
    except Exception:
        # fallback to local dev file if present
        p = Path("vi_full.txt")
        if p.exists():
            return {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
        return set()

@dataclass
class OCRConfig:
    """Configuration for a paraOCR processing run."""
    input_dir: Path
    output_path: Path
    error_log_path: Path = Path("paraOCR_error_log.jsonl")

    languages: List[str] = field(default_factory=lambda: ['vi', 'en'])
    ignore_keywords: List[str] = field(default_factory=list)

    num_workers: int = max(1, cpu_count() - 2)
    gpu_batch_size: int = 16
    num_gpu_workers: int = 3
    dpi: int = 200
    beamsearch: bool = False
    force_rerun: bool = False

    temp_dir: Path = Path(tempfile.gettempdir()) / "paraOCR_temp"
    export_txt: bool = True # Change default export to text to True for UX
    log_performance: bool = False
    performance_log_path: Path = Path("paraOCR_performance_log.jsonl")   


    use_cache: bool = True                # enable page/txt + render png cache
    keep_render_cache: bool = True        # keep rendered PNGs after OCR (faster resume)
    cache_version: str = "v1"             # bump to invalidate old cache entries

    process_tables: bool = False
    process_images: bool = False

    ocr_backend: str = "paraocr.ocr_backends.easyocr_backend.EasyOCREngine"
    ocr_backend_kwargs: Dict[str, Any] = field(default_factory=dict)

    
    min_native_text_chars: int = 100
    native_text_quality_threshold: float = 0.3
    dictionary: Set[str] = field(default_factory=get_default_dictionary)

    pdf_engine: str = "pymupdf"

    log_queue: Optional[Any] = None

    def to_dict(self):
        """Converts config to a dictionary suitable for multiprocessing (pickling)."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d


    @classmethod
    def from_dict(cls, config_dict: dict):
        d = dict(config_dict)

        # normalize path-like fields
        for key in ["input_dir", "output_path", "error_log_path", "temp_dir", "performance_log_path"]:
            if key in d and isinstance(d[key], str):
                d[key] = Path(d[key])

        # allow explicit None to mean use default
        for key in ["num_workers", "gpu_batch_size", "num_gpu_workers", "dpi", "performance_log_path"]:
            if d.get(key) is None:
                d.pop(key, None)  # <- safe pop

        cfg = cls(**d)

        # if logging is on but no path was provided, pick one next to results
        if cfg.log_performance and not cfg.performance_log_path:
            cfg.performance_log_path = Path(str(cfg.output_path)).with_suffix(".perf.jsonl")

        return cfg