"""Card recognition from card-slot crops.

:class:`TemplateCardRecognizer` handles face-up cards rendered as a light
card face with the rank glyph in the upper part and a suit mark below it:

* presence — share of light "card face" pixels in the slot;
* rank — :class:`~poker_alpha.observer.text.TemplateOCR` over the upper part
  (ranks ``23456789TJQKA``; learn real glyphs with ``from_samples`` — a
  ``"10"`` rank glyph pair must be relabelled as ``T``);
* suit — the mean colour of the non-face "ink" pixels, matched to the
  calibration's per-suit colours. This requires a **four-colour deck**;
  with two-colour decks suits are ambiguous and the reading's confidence
  drops accordingly (shape templates are a TODO, not faked).

Confidence is the product of the rank match score and the suit colour
margin — match scores, not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from .text import TemplateOCR

RANKS = "23456789TJQKA"


@dataclass(frozen=True)
class CardReading:
    card: Optional[str]      # e.g. "As"; None if no face-up card
    present: bool
    confidence: float


class CardRecognizer(Protocol):
    def recognize(self, image) -> CardReading:
        ...


class TemplateCardRecognizer:
    def __init__(self, suit_colors: Dict[str, Tuple[int, int, int]],
                 rank_ocr: Optional[TemplateOCR] = None,
                 face_threshold: int = 215, min_face_fraction: float = 0.45,
                 rank_fraction: float = 0.5) -> None:
        self.suit_colors = {k: np.array(v, dtype=np.float64)
                            for k, v in suit_colors.items()}
        self.rank_ocr = rank_ocr or TemplateOCR(charset=RANKS)
        self.face_threshold = face_threshold
        self.min_face_fraction = min_face_fraction
        self.rank_fraction = rank_fraction

    def recognize(self, image) -> CardReading:
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
        if arr.size == 0:
            return CardReading(None, False, 0.0)
        face = arr.min(axis=2) >= self.face_threshold
        frac = float(face.mean())
        if frac < self.min_face_fraction:
            # Confidence that the slot is empty grows as the face share falls.
            return CardReading(None, False, float(min(1.0, 1.0 - frac /
                                                      self.min_face_fraction + 0.5)))
        h, w = arr.shape[:2]
        # Inset past the card border/rounded corners before reading the rank.
        ix, iy = max(1, int(0.1 * w)), max(1, int(0.04 * h))
        upper = image.crop((ix, iy, w - ix, int(h * self.rank_fraction)))
        rank = self.rank_ocr.read_text(upper)
        ink = ~face & (arr.max(axis=2) - arr.min(axis=2) > 25) | \
            (~face & (arr.mean(axis=2) < 90))
        if not ink.any() or len(rank.text) != 1 or rank.text not in RANKS:
            return CardReading(None, True, 0.0)
        mean = arr[ink].mean(axis=0)
        dists = sorted((float(np.linalg.norm(mean - c)), s)
                       for s, c in self.suit_colors.items())
        (d1, suit), (d2, _) = dists[0], dists[1]
        margin = max(0.0, min(1.0, 1.0 - d1 / max(d2, 1e-9)))
        return CardReading(rank.text + suit, True,
                           float(rank.confidence * min(1.0, 2.0 * margin)))
