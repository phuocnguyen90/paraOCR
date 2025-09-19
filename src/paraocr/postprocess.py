# src/paraocr/postprocess.py
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# This is a third-party library: pip install gibberish-detector-redux
from gibberish_detector import GibberishDetector

from .config import OCRConfig
from .models import OCRResult

logger = logging.getLogger("paraocr")

@dataclass
class IntelligibilityResult:
    """Structured result for an intelligibility check."""
    passed: bool
    unintelligible_ratio: float
    reason: str | None = None

class Postprocessor:
    """
    Handles post-processing checks on completed OCR results, starting with
    an intelligibility (gibberish) check.
    """
    def __init__(self, config: OCRConfig):
        self.config = config
        self.threshold = getattr(config, "postprocess_intelligibility_threshold", 0.3)
        self.failure_log_path = getattr(config, "postprocess_failure_log_path", None)
        self.languages = config.languages or ["en", "vi"]
        
        self.detectors: Dict[str, GibberishDetector] = self._load_models()

    def _load_models(self) -> Dict[str, GibberishDetector]:
        """Loads the gibberish detector models for the specified languages."""
        detectors = {}
        # Assume models are in a 'models' subdirectory relative to this file
        models_dir = Path(__file__).parent / "models"
        
        for lang in self.languages:
            model_path = models_dir / f"{lang}.model"
            if not model_path.exists():
                logger.warning(
                    "Gibberish detector model not found for language '%s'. Expected at: %s",
                    lang, model_path
                )
                continue
            
            try:
                detector = GibberishDetector.create_from_file(str(model_path))
                detectors[lang] = detector
                logger.info("Loaded intelligibility model for language: %s", lang)
            except Exception as e:
                logger.error(
                    "Failed to load gibberish detector model for '%s' from %s: %s",
                    lang, model_path, e
                )
        
        if not detectors:
            logger.error("No intelligibility models were loaded. Disabling post-processing check.")
            
        return detectors

    def check_intelligibility(self, ocr_result: OCRResult) -> IntelligibilityResult:
        """
        Checks if the OCR text is largely intelligible based on language models.

        A line is considered "unintelligible" only if it is classified as gibberish
        by the detectors for ALL specified languages.
        """
        if not self.detectors:
            return IntelligibilityResult(passed=True, unintelligible_ratio=0.0)

        full_text = "\n".join(
            p.get("data", "") for p in ocr_result.content if p and p.get("type") == "text"
        )
        
        if not full_text.strip():
            return IntelligibilityResult(passed=True, unintelligible_ratio=0.0)

        total_chars = 0
        unintelligible_chars = 0

        lines = full_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            total_chars += len(line)
            
            # A line must be gibberish in ALL languages to be considered unintelligible
            is_unintelligible_line = True
            for lang, detector in self.detectors.items():
                if not detector.is_gibberish(line):
                    is_unintelligible_line = False
                    break # It's intelligible in at least one language, so we can stop checking
            
            if is_unintelligible_line:
                unintelligible_chars += len(line)
        
        if total_chars == 0:
            return IntelligibilityResult(passed=True, unintelligible_ratio=0.0)
            
        ratio = unintelligible_chars / total_chars
        
        if ratio > self.threshold:
            reason = (
                f"Unintelligible content ratio {ratio:.2f} "
                f"exceeds threshold {self.threshold}."
            )
            result = IntelligibilityResult(passed=False, unintelligible_ratio=ratio, reason=reason)
            self._log_failure(ocr_result.source_path, result)
            return result
        
        return IntelligibilityResult(passed=True, unintelligible_ratio=ratio)

    def _log_failure(self, source_path: str, result: IntelligibilityResult):
        """Logs a failed check to the specified failure log file."""
        if not self.failure_log_path:
            return
            
        try:
            self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source_path": source_path,
                "reason": result.reason,
                "unintelligible_ratio": round(result.unintelligible_ratio, 4)
            }
            with open(self.failure_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to write to the post-processing failure log")