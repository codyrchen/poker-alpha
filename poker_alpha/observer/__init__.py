"""Optional screen observer: screenshots -> noisy observations -> table state.

Kept separate from all poker logic: the solver, ranges and decision engine
never import this package. Image work needs the optional ``[vision]``
dependencies (Pillow; ``mss`` for live capture); OCR via Tesseract is a
further optional backend and is never faked when unavailable.

Use only where real-time assistance is permitted (private, play-money or
test games). The observer reads the screen; it never clicks, types, controls
a browser or submits actions. Where live assistance is prohibited, feed saved
screenshots / sessions through it for post-hand analysis instead.
"""

from .errors import CalibrationError, ObserverDependencyError, OCRUnavailable

__all__ = ["CalibrationError", "ObserverDependencyError", "OCRUnavailable"]
