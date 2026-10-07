"""PokerNow-style table layout adapter.

"PokerNow-style" means the general look of browser poker tables of that kind:
an oval felt with seats around it, stacks under player names, bets pushed
toward the middle, community cards in the centre and the hero's cards near
their seat. The default geometry here is generic and parameterized by seat
count; it is **not** measured from real PokerNow screenshots.

    Exact PokerNow visual accuracy is not validated without representative
    screenshots.

To support a real client: take screenshots, adjust the region boxes (all
table-normalized, so resolution-independent) in a saved
:class:`~poker_alpha.observer.calibration.TableCalibration`, learn glyph
templates with :meth:`TemplateOCR.from_samples` (or install Tesseract), and
add the screenshots plus their ground truth under ``tests/fixtures/observer``.

This module only *reads* pixels. There is no clicking, typing, browser
automation or action submission anywhere in the observer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .calibration import TableCalibration, color_mask, locate_table
from .cards import CardRecognizer, TemplateCardRecognizer
from .regions import Region, crop
from .text import (OCRBackend, TemplateOCR, fix_separators, foreground_mask,
                   parse_amount)


def default_layout(num_seats: int = 6, hero_seat: int = 0,
                   name: str = "pokernow-style-default") -> TableCalibration:
    """Generic oval layout with the hero at the bottom centre."""
    regions: Dict[str, Region] = {
        "pot": Region(0.42, 0.555, 0.16, 0.055),
    }
    for i in range(5):
        regions[f"board_{i}"] = Region(0.335 + i * 0.068, 0.36, 0.062, 0.17)
    cx, cy, rx, ry = 0.5, 0.5, 0.40, 0.37
    for seat in range(num_seats):
        k = (seat - hero_seat) % num_seats
        theta = math.pi / 2 + 2 * math.pi * k / num_seats
        sx, sy = cx + rx * math.cos(theta), cy + ry * math.sin(theta)
        bx, by = sx + 0.26 * (cx - sx), sy + 0.26 * (cy - sy)
        regions[f"seat{seat}_name"] = Region(sx - 0.07, sy - 0.04, 0.14, 0.035)
        regions[f"seat{seat}_stack"] = Region(sx - 0.07, sy, 0.14, 0.045)
        regions[f"seat{seat}_active"] = Region(sx - 0.07, sy + 0.05, 0.14, 0.012)
        regions[f"seat{seat}_cards"] = Region(sx - 0.035, sy - 0.10, 0.07, 0.05)
        regions[f"seat{seat}_bet"] = Region(bx - 0.05, by - 0.02, 0.10, 0.042)
        regions[f"seat{seat}_dealer"] = Region(bx + 0.055, by - 0.02, 0.026, 0.042)
        if seat == hero_seat:
            regions["hero_card_0"] = Region(sx + 0.08, sy - 0.15, 0.055, 0.15)
            regions["hero_card_1"] = Region(sx + 0.14, sy - 0.15, 0.055, 0.15)
    return TableCalibration(name=name, num_seats=num_seats, hero_seat=hero_seat,
                            regions=regions)


@dataclass(frozen=True)
class FieldReading:
    """One recognized value with its confidence and provenance."""

    value: object
    confidence: float
    region: str
    timestamp: Optional[float] = None


@dataclass
class FrameObservation:
    """Everything read from one frame, field by field."""

    timestamp: Optional[float]
    table_bbox: Tuple[int, int, int, int]
    fields: Dict[str, FieldReading] = field(default_factory=dict)

    def value(self, name: str, default=None):
        f = self.fields.get(name)
        return default if f is None else f.value

    def confidence(self, name: str) -> float:
        f = self.fields.get(name)
        return 0.0 if f is None else f.confidence


class PokerNowStyleAdapter:
    """Reads a :class:`FrameObservation` from a screenshot (see module doc)."""

    def __init__(self, calibration: TableCalibration,
                 amount_ocr: Optional[OCRBackend] = None,
                 stack_ocr: Optional[OCRBackend] = None,
                 card_recognizer: Optional[CardRecognizer] = None,
                 color_tolerance: int = 40) -> None:
        self.cal = calibration
        self.amount_ocr = amount_ocr or TemplateOCR("0123456789.,")
        self.stack_ocr = stack_ocr or TemplateOCR("0123456789.,ALIN")
        self.cards = card_recognizer or TemplateCardRecognizer(
            calibration.suit_colors)
        self.tol = color_tolerance

    # -- TableLayoutAdapter interface -----------------------------------------

    def locate_table(self, image):
        return locate_table(image, self.cal)

    def extract_regions(self, image) -> Dict[str, object]:
        bbox = self.locate_table(image)
        return {name: crop(image, r, bbox) for name, r in self.cal.regions.items()}

    # -- reading ---------------------------------------------------------------

    def _amount(self, img, name: str, ts, ocr) -> FieldReading:
        if not foreground_mask(img).any():
            return FieldReading(0.0, 1.0, name, ts)  # clearly empty
        res = ocr.read_text(img)
        if res.text == "ALLIN":
            return FieldReading("ALLIN", res.confidence, name, ts)
        value = parse_amount(fix_separators(res.text))
        if value is None:
            return FieldReading(None, 0.0, name, ts)
        return FieldReading(value, res.confidence, name, ts)

    def _color_share(self, img, color) -> float:
        if img.size[0] == 0 or img.size[1] == 0:
            return 0.0
        return float(color_mask(img, color, self.tol).mean())

    def read_frame(self, image, timestamp: Optional[float] = None) -> FrameObservation:
        bbox = self.locate_table(image)
        crops = {name: crop(image, r, bbox) for name, r in self.cal.regions.items()}
        obs = FrameObservation(timestamp, bbox)
        f = obs.fields
        f["pot"] = self._amount(crops["pot"], "pot", timestamp, self.amount_ocr)
        for i in range(5):
            r = self.cards.recognize(crops[f"board_{i}"])
            f[f"board_{i}"] = FieldReading(r.card, r.confidence, f"board_{i}", timestamp)
        for i in range(2):
            r = self.cards.recognize(crops[f"hero_card_{i}"])
            f[f"hero_card_{i}"] = FieldReading(r.card, r.confidence,
                                               f"hero_card_{i}", timestamp)
        dealer_scores = []
        active_scores = []
        for s in range(self.cal.num_seats):
            stack = self._amount(crops[f"seat{s}_stack"], f"seat{s}_stack",
                                 timestamp, self.stack_ocr)
            occupied = foreground_mask(crops[f"seat{s}_stack"]).any()
            f[f"seat{s}.occupied"] = FieldReading(bool(occupied), stack.confidence
                                                  if occupied else 1.0,
                                                  f"seat{s}_stack", timestamp)
            if stack.value == "ALLIN":
                f[f"seat{s}.all_in"] = FieldReading(True, stack.confidence,
                                                    f"seat{s}_stack", timestamp)
                stack = FieldReading(0.0, stack.confidence, stack.region, timestamp)
            else:
                f[f"seat{s}.all_in"] = FieldReading(False, stack.confidence,
                                                    f"seat{s}_stack", timestamp)
            f[f"seat{s}.stack"] = stack if occupied else FieldReading(
                None, 1.0, stack.region, timestamp)
            f[f"seat{s}.bet"] = self._amount(crops[f"seat{s}_bet"], f"seat{s}_bet",
                                             timestamp, self.amount_ocr)
            if s == self.cal.hero_seat:
                held = f["hero_card_0"].value is not None
                f[f"seat{s}.in_hand"] = FieldReading(
                    held, f["hero_card_0"].confidence, "hero_card_0", timestamp)
            else:
                share = self._color_share(crops[f"seat{s}_cards"],
                                          self.cal.card_back_color)
                f[f"seat{s}.in_hand"] = FieldReading(
                    share > 0.3, min(1.0, abs(share - 0.3) / 0.3),
                    f"seat{s}_cards", timestamp)
            dealer_scores.append(self._color_share(crops[f"seat{s}_dealer"],
                                                   self.cal.button_color))
            active_scores.append(self._color_share(crops[f"seat{s}_active"],
                                                   self.cal.highlight_color))
        order = np.argsort(dealer_scores)[::-1]
        top, second = dealer_scores[order[0]], dealer_scores[order[1]]
        f["dealer"] = FieldReading(int(order[0]) if top > 0.2 else None,
                                   float(min(1.0, max(0.0, top - second) / 0.5)),
                                   f"seat{int(order[0])}_dealer", timestamp)
        a = int(np.argmax(active_scores))
        f["actor"] = FieldReading(a if active_scores[a] > 0.5 else None,
                                  float(min(1.0, abs(active_scores[a] - 0.5) / 0.4)),
                                  f"seat{a}_active", timestamp)
        return obs
