# powerocr/exceptions.py
class PowerOCRError(Exception):
    """Base exception for the powerocr library."""
    pass

class FileProcessingError(PowerOCRError):
    """Raised when a single file fails to process."""
    pass