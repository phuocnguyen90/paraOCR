# paraOCR/exceptions.py
class paraOCRError(Exception):
    """Base exception for the paraOCR library."""
    pass

class FileProcessingError(paraOCRError):
    """Raised when a single file fails to process."""
    pass