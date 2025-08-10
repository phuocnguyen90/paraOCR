# powerocr/config.py
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Set
import tempfile
from multiprocessing import cpu_count

# This function is defined here so it can be used as a default_factory
def get_default_dictionary() -> Set[str]:
    default_dict_path = Path("vi_dict.txt")
    if default_dict_path.exists():
        with open(default_dict_path, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    return set()

@dataclass
class OCRConfig:
    """Configuration for a PowerOCR processing run."""
    input_dir: Path
    output_path: Path
    error_log_path: Path = Path("powerocr_error_log.jsonl")
    languages: List[str] = field(default_factory=lambda: ['vi', 'en'])
    ignore_keywords: List[str] = field(default_factory=list)
    num_workers: int = max(1, cpu_count() - 2)
    gpu_batch_size: int = 16
    num_gpu_workers: int = 3
    dpi: int = 200
    beamsearch: bool = False
    force_rerun: bool = False
    temp_dir: Path = Path(tempfile.gettempdir()) / "powerocr_temp"
    export_txt: bool = False
    log_performance: bool = False
    performance_log_path: Path = Path("powerocr_performance_log.jsonl")    
    process_tables: bool = False
    process_images: bool = False

    
    min_native_text_chars: int = 100
    native_text_quality_threshold: float = 0.3
    dictionary: Set[str] = field(default_factory=get_default_dictionary)

    pdf_engine: str = "pymupdf"

    def to_dict(self):
        """Converts config to a dictionary suitable for multiprocessing (pickling)."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    # --- THIS IS THE NEW, IMPORTANT METHOD ---
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Creates a config instance from a dictionary, correctly handling Path objects."""
        # Define which fields are expected to be paths
        path_fields = ['input_dir', 'output_path', 'error_log_path', 'temp_dir']
        
        # Convert string paths back to Path objects
        for field_name in path_fields:
            if field_name in config_dict and isinstance(config_dict[field_name], str):
                config_dict[field_name] = Path(config_dict[field_name])
        
        # Handle dictionary if it's passed as a path string
        if 'dictionary_path' in config_dict:
            dict_path = Path(config_dict.pop('dictionary_path'))
            if dict_path.exists():
                 with open(dict_path, 'r', encoding='utf-8') as f:
                    config_dict['dictionary'] = {line.strip().lower() for line in f if line.strip()}
            elif 'dictionary' not in config_dict:
                 config_dict['dictionary'] = set()

        return cls(**config_dict)