"""Frame sources: where screenshots come from."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Sequence, Tuple, Union

from .errors import ObserverDependencyError, require_pil


class ScreenSource(Protocol):
    def capture(self):
        """Return the next frame as a ``PIL.Image.Image`` (RGB)."""
        ...


class ImageFileSource:
    """Frames from image files (fixtures, saved sessions) — replay-friendly."""

    def __init__(self, paths: Sequence[Union[str, Path]], loop: bool = False) -> None:
        require_pil()
        if not paths:
            raise ValueError("no image paths")
        self.paths = [Path(p) for p in paths]
        self.loop = loop
        self._i = 0

    def capture(self):
        from PIL import Image

        if self._i >= len(self.paths):
            if not self.loop:
                raise StopIteration("no more frames")
            self._i = 0
        img = Image.open(self.paths[self._i]).convert("RGB")
        self._i += 1
        return img


class ImageSequenceSource:
    """Frames from in-memory images (tests, synthetic sessions)."""

    def __init__(self, images: Sequence) -> None:
        self.images = list(images)
        self._i = 0

    def capture(self):
        if self._i >= len(self.images):
            raise StopIteration("no more frames")
        img = self.images[self._i]
        self._i += 1
        return img


class MSSScreenSource:
    """Live screen capture with ``mss`` (read-only; no input is ever sent).

    ``monitor`` is an mss monitor index (1 = primary) or a pixel box
    ``(left, top, width, height)``.
    """

    def __init__(self, monitor: Union[int, Tuple[int, int, int, int]] = 1) -> None:
        require_pil()
        try:
            import mss  # noqa: F401
        except ImportError as exc:
            raise ObserverDependencyError(
                "live capture needs mss: pip install 'poker-alpha[vision]'") from exc
        self.monitor = monitor

    def capture(self):
        import mss
        from PIL import Image

        with mss.mss() as sct:
            if isinstance(self.monitor, int):
                mon = sct.monitors[self.monitor]
            else:
                left, top, w, h = self.monitor
                mon = {"left": left, "top": top, "width": w, "height": h}
            shot = sct.grab(mon)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
