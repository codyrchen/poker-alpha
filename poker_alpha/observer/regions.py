"""Normalized regions relative to a located table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Box = Tuple[int, int, int, int]  # left, top, right, bottom (pixels)


@dataclass(frozen=True)
class Region:
    """A rectangle in table-normalized coordinates (0..1 of table width/height)."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("region must have positive size")
        if self.x < -0.5 or self.y < -0.5 or self.x + self.w > 1.5 or self.y + self.h > 1.5:
            raise ValueError("region far outside the table")

    def to_pixels(self, table: Box) -> Box:
        left, top, right, bottom = table
        tw, th = right - left, bottom - top
        return (int(round(left + self.x * tw)), int(round(top + self.y * th)),
                int(round(left + (self.x + self.w) * tw)),
                int(round(top + (self.y + self.h) * th)))

    def to_list(self):
        return [self.x, self.y, self.w, self.h]


def crop(image, region: Region, table: Box):
    box = region.to_pixels(table)
    l, t, r, b = box
    W, H = image.size
    return image.crop((max(l, 0), max(t, 0), min(r, W), min(b, H)))
