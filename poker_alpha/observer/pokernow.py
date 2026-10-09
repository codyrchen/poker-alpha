"""PokerNow-style table layout adapter.

"PokerNow-style" means the general look of browser poker tables of that kind:
an oval felt with seats around it, stacks under player names, bets pushed
toward the middle, community cards in the centre and the hero's cards near
their seat. The default geometry here is generic and parameterized by seat
count; it is **not** measured from real PokerNow screenshots.

    Exact PokerNow visual accuracy is not validated without representative
    screenshots.

:func:`pokernow_hu_layout` is different: real PokerNow heads-up geometry,
aligned on one real annotated frame (``tests/fixtures/pokernow``), with
PokerNow-specific recognizers (``client="pokernow"``). One frame, also used
for tuning, is a regression check, not a validation.

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
from .cards import CardRecognizer, PokerNowCardRecognizer, TemplateCardRecognizer
from .regions import Region, crop
from .text import (OCRBackend, TemplateOCR, fix_separators, foreground_mask,
                   parse_amount, pokernow_ocr, segment_glyphs)


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
        bx, by = sx + 0.36 * (cx - sx), sy + 0.30 * (cy - sy)
        regions[f"seat{seat}_active"] = Region(sx - 0.07, sy - 0.058, 0.14, 0.012)
        regions[f"seat{seat}_name"] = Region(sx - 0.07, sy - 0.04, 0.14, 0.035)
        regions[f"seat{seat}_stack"] = Region(sx - 0.07, sy, 0.14, 0.045)
        regions[f"seat{seat}_cards"] = Region(sx - 0.035, sy - 0.115, 0.07, 0.05)
        if seat == hero_seat:
            # Hero: cards above the plate, bet to the left, button to the right.
            regions["hero_card_0"] = Region(sx - 0.062, sy - 0.235, 0.058, 0.16)
            regions["hero_card_1"] = Region(sx + 0.004, sy - 0.235, 0.058, 0.16)
            regions[f"seat{seat}_bet"] = Region(sx - 0.20, sy - 0.02, 0.10, 0.042)
        else:
            regions[f"seat{seat}_bet"] = Region(bx - 0.05, by - 0.02, 0.10, 0.042)
        # Button beside the seat plate, clear of every text region.
        regions[f"seat{seat}_dealer"] = Region(sx + 0.075, sy - 0.035, 0.024, 0.038)
    return TableCalibration(name=name, num_seats=num_seats, hero_seat=hero_seat,
                            regions=regions,
                            source=f"default_layout({num_seats}, {hero_seat}): generic oval, "
                                   "not measured from a real client")


# -- real PokerNow heads-up layout --------------------------------------------
#
# Measured on a real PokerNow heads-up frame (tests/fixtures/pokernow/raw/
# hu_preflop_0001.png). Coordinates below are pixels of that frame; the table
# box there (ellipse fitted to the green felt, ``table_detector="green_oval"``)
# is ``_HU_REF_TABLE``. Regions are stored relative to the table box, so any
# capture size / browser zoom works.
#
# In PokerNow heads-up both players sit along the bottom edge, below the felt.
# Each player "container" is the same: the hole cards on the left, the name /
# stack plate to their right, the street bet as a pill under the stack inside
# the plate, the dealer button above the card / plate junction. The plate
# turns pale yellow for the player to act.
_HU_REF_TABLE = (4, 9, 631, 312)
_HU_CONTAINER_X = {"left": 94.0, "right": 331.0}
_HU_SEAT = {                       # (x0, y0, x1, y1), x relative to the container
    "card_0": (3, 320, 46, 380),   # left (rear) hole card face
    "card_1": (49, 319, 97, 379),  # right (front) hole card face
    "cards": (13, 321, 88, 376),   # where an opponent's card backs show
    "name": (104, 335, 151, 349),
    "stack": (104, 348, 156, 365),
    "bet": (104, 362, 156, 379),   # the "+1.00" pill, with a small margin
    "dealer": (82, 281, 108, 302),
    "active": (153, 333, 177, 360),  # plate background right of the text
}
_HU_POT = (262, 51, 367, 78)
# The reference frame is preflop: the board slots below are placed in the
# felt centre at hole-card size but NOT yet verified against a real board.
_HU_BOARD = [(189 + 52 * i, 101, 235 + 52 * i, 166) for i in range(5)]


def _hu_region(x0: float, y0: float, x1: float, y1: float, dx: float = 0.0) -> Region:
    l, t, r, b = _HU_REF_TABLE
    w, h = r - l, b - t
    return Region((x0 + dx - l) / w, (y0 - t) / h, (x1 - x0) / w, (y1 - y0) / h)


def pokernow_hu_layout(hero_side: str = "right",
                       name: str = "pokernow-heads-up") -> TableCalibration:
    """Real PokerNow heads-up geometry: seat 0 = hero, seat 1 = opponent.

    ``hero_side`` is where the hero's plate is on screen ("right" = hero
    bottom-right, opponent bottom-left). The table bounds are the green felt,
    found each frame by hue (``table_detector="green_oval"``), so the layout
    follows browser zoom and window size.
    """
    if hero_side not in _HU_CONTAINER_X:
        raise ValueError("hero_side must be 'left' or 'right'")
    opp_side = "left" if hero_side == "right" else "right"
    regions: Dict[str, Region] = {"pot": _hu_region(*_HU_POT)}
    for i, box in enumerate(_HU_BOARD):
        regions[f"board_{i}"] = _hu_region(*box)
    for seat, side in ((0, hero_side), (1, opp_side)):
        dx = _HU_CONTAINER_X[side]
        for k in ("cards", "name", "stack", "bet", "dealer", "active"):
            regions[f"seat{seat}_{k}"] = _hu_region(*_HU_SEAT[k], dx=dx)
        if seat == 0:
            regions["hero_card_0"] = _hu_region(*_HU_SEAT["card_0"], dx=dx)
            regions["hero_card_1"] = _hu_region(*_HU_SEAT["card_1"], dx=dx)
    return TableCalibration(
        name=name, num_seats=2, hero_seat=0, regions=regions,
        source=f"pokernow_hu_layout(hero_side={hero_side!r}): measured on the real PokerNow "
               "tuning frame tests/fixtures/pokernow/raw/hu_preflop_0001.png",
        table_detector="green_oval", client="pokernow", pot_includes_bets=False,
        felt_color=(40, 130, 78),
        text_color=(250, 250, 250), highlight_color=(248, 252, 215),
        button_color=(226, 234, 250), card_back_color=(215, 125, 125),
        suit_colors={"s": (25, 25, 25), "h": (200, 30, 40),
                     "d": (200, 30, 40), "c": (25, 25, 25)})


LAYOUT_PRESETS = ("Generic layout", "PokerNow Heads-Up")


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
        if calibration.client == "pokernow":
            # Real-PokerNow recognizers (see pokernow_ocr / PokerNowCardRecognizer).
            self.amount_ocr = amount_ocr or pokernow_ocr()
            self.stack_ocr = stack_ocr or pokernow_ocr("0123456789.,+ALIN")
            self.cards = card_recognizer or PokerNowCardRecognizer()
        else:
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
        from .text import segment_glyphs

        bg = getattr(ocr, "background", "border")
        if segment_glyphs(foreground_mask(img, background=bg), drop_edge_blobs=True) is None:
            return FieldReading(0.0, 1.0, name, ts)  # clearly empty
        res = ocr.read_text(img)
        text, conf = res.text, res.confidence
        if text.startswith("+"):
            text = text[1:]      # PokerNow bet pills read "+1.00"
        if text == "ALLIN":
            return FieldReading("ALLIN", conf, name, ts)
        value = parse_amount(fix_separators(text))
        if value is None:
            return FieldReading(None, 0.0, name, ts)
        return FieldReading(value, conf, name, ts)

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
            occupied = segment_glyphs(foreground_mask(crops[f"seat{s}_stack"], background=getattr(
                self.stack_ocr, "background", "border")),
                                      drop_edge_blobs=True) is not None
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
