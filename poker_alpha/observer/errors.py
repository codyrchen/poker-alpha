"""Observer error types."""


class ObserverDependencyError(ImportError):
    """An optional dependency (Pillow, mss, pytesseract...) is missing."""


class OCRUnavailable(ObserverDependencyError):
    """The requested OCR backend cannot run on this machine."""


class CalibrationError(ValueError):
    """The table could not be located or the calibration is invalid."""


def require_pil():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ObserverDependencyError(
            "the screen observer needs Pillow: pip install 'poker-alpha[vision]'"
        ) from exc
