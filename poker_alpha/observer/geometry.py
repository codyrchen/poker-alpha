"""Capture geometry: the coordinate systems of the live observer (Phase 52).

=====================  ==============================================  ===========
system                 meaning                                          unit
=====================  ==============================================  ===========
monitor                one display from ``mss`` (``left, top, width,     screen
                       height``); ``left/top`` are global (a display    points
                       left of the main one has a negative ``left``)
capture rectangle      ``(left, top, width, height)`` RELATIVE to the   screen
                       monitor's top-left; ``None`` / zero size =       points
                       the whole monitor
global capture box     monitor origin + capture rectangle (what mss      screen
                       is asked to grab), see :func:`live.absolute_box`  points
captured image         the frame mss returns                            pixels
table box              ``(x0, y0, x1, y1)`` located in the captured      pixels
                       image (or a fixed ``table_bbox`` calibration)
normalized region      ``Region(x, y, w, h)`` as fractions of the        0..1 of the
                       table box                                         table box
=====================  ==============================================  ===========

On macOS, screen coordinates (and Cmd+Shift+4 read-outs) are *points*; a
Retina display returns 2 pixels per point, so a 640x400-point capture is a
1280x800-pixel image. Windows / Linux display scaling can give fractional
ratios (1.25, 1.5). Calibration regions are normalized to the located table,
so they do not care; a FIXED ``table_bbox`` is in captured-image pixels and
is only valid for the capture rectangle (and scale) it was set with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Rect = Tuple[int, int, int, int]      # left, top, width, height (points)
Box = Tuple[int, int, int, int]       # x0, y0, x1, y1


@dataclass(frozen=True)
class CaptureGeometry:
    """Maps between screen points and captured-image pixels for one capture."""

    monitor: Dict[str, int]
    rect: Optional[Rect]
    image_size: Tuple[int, int]           # (width, height) in pixels

    @property
    def global_box(self) -> Rect:
        from .live import absolute_box

        return absolute_box(self.monitor, self.rect)

    @property
    def points_size(self) -> Tuple[int, int]:
        g = self.global_box
        return g[2], g[3]

    @property
    def pixels_per_point(self) -> Tuple[float, float]:
        pw, ph = self.points_size
        return self.image_size[0] / pw, self.image_size[1] / ph

    @property
    def scale_label(self) -> str:
        sx, sy = self.pixels_per_point
        if abs(sx - sy) > 0.02:
            return f"non-uniform {sx:.2f}x{sy:.2f} px/pt (unexpected: check the capture)"
        if abs(sx - 2.0) < 0.02:
            return "2.00 px/pt (Retina)"
        return f"{sx:.2f} px/pt"

    def image_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Captured-image pixel -> global screen point."""
        gx, gy, _, _ = self.global_box
        sx, sy = self.pixels_per_point
        return gx + x / sx, gy + y / sy

    def screen_to_image(self, x: float, y: float) -> Tuple[float, float]:
        """Global screen point -> captured-image pixel."""
        gx, gy, _, _ = self.global_box
        sx, sy = self.pixels_per_point
        return (x - gx) * sx, (y - gy) * sy

    def box_to_monitor_rect(self, box: Box) -> Rect:
        """A pixel box in the image -> a capture rectangle (monitor-relative
        points) that would capture exactly that box: used to suggest a
        tighter crop around a located table."""
        x0, y0 = self.image_to_screen(box[0], box[1])
        x1, y1 = self.image_to_screen(box[2], box[3])
        ml, mt = self.monitor["left"], self.monitor["top"]
        return (int(round(x0 - ml)), int(round(y0 - mt)),
                int(round(x1 - x0)), int(round(y1 - y0)))

    def summary(self) -> Dict[str, object]:
        return {"monitor": dict(self.monitor), "capture_rect_points": self.rect,
                "global_box_points": self.global_box, "points_size": self.points_size,
                "image_size_pixels": self.image_size,
                "pixels_per_point": self.pixels_per_point, "scale": self.scale_label}


def suggested_crop(geometry: CaptureGeometry, table_box: Box, regions_box: Box,
                   margin: float = 0.04) -> Rect:
    """Capture rectangle (monitor-relative points) around the table plus its
    recognition regions with a margin — a hint for the user, never applied
    automatically."""
    l = min(table_box[0], regions_box[0])
    t = min(table_box[1], regions_box[1])
    r = max(table_box[2], regions_box[2])
    b = max(table_box[3], regions_box[3])
    m = int(round(margin * (table_box[2] - table_box[0])))
    w, h = geometry.image_size
    return geometry.box_to_monitor_rect((max(0, l - m), max(0, t - m),
                                         min(w, r + m), min(h, b + m)))
