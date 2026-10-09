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


# -- real PokerNow cards ---------------------------------------------------------

_RANK_GLYPHS = {r: r for r in "23456789JQKA"} | {"T": "10"}
_SUIT_GLYPHS = {"s": "♠", "c": "♣", "h": "♥", "d": "♦"}
_RED, _BLACK = ("h", "d"), ("s", "c")


def _shape(mask: np.ndarray, size=(16, 24)):
    """Tight-cropped bitmap (resized) and aspect ratio of a binary glyph."""
    from PIL import Image

    ys, xs = np.nonzero(mask)
    t = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray((t * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0, t.shape[1] / t.shape[0]


def _render_shape(text: str, font, angle: float = 0.0):
    from PIL import Image, ImageDraw

    img = Image.new("L", (260, 180), 0)
    ImageDraw.Draw(img).text((70, 30), text, fill=255, font=font)
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR)
    return _shape(np.asarray(img) > 128)     # binarized at 50%, like the input


# PokerNow fans the hole cards, so each is tilted by a few degrees.
_TILTS = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)


def _shape_score(g, t) -> float:
    a, b = g[0] - g[0].mean(), t[0] - t[0].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    ncc = float((a * b).sum() / den) if den > 0 else 0.0
    return max(0.0, ncc) * float(np.exp(-1.5 * abs(np.log(g[1] / t[1]))))


def _margin(best: float, second: float) -> float:
    return float(min(1.0, 2.0 * (best - second) / max(best, 1e-9)))


class PokerNowCardRecognizer:
    """Face-up cards as PokerNow renders them (two-colour deck).

    * presence — a low-saturation card face that is bright *relative to the
      card itself* (PokerNow dims the hero's cards, e.g. to grey ~100);
    * ink — pixels darker than the face by more than half the strongest
      difference in the slot;
    * rank — the ink in the upper-left corner, matched against DejaVu Serif
      Bold glyphs ("10" for ``T``) rotated by up to +-12 degrees (the hole
      cards are fanned);
    * suit — colour picks red / black, then the largest ink blob below the
      rank is matched against the two suit shapes of that colour.

    Confidence is the weakest of the rank margin, the colour margin and the
    suit-shape margin: a match score, not a calibrated probability. Checked
    on one real frame only (J of spades, 7 of hearts; see
    ``tests/fixtures/pokernow``); other ranks and suits are untested on real
    PokerNow images.
    """

    def __init__(self, min_face_level: float = 80.0, min_face_fraction: float = 0.35,
                 ink_level: float = 0.5) -> None:
        from .text import dejavu_font

        self.min_face_level = min_face_level
        self.min_face_fraction = min_face_fraction
        self.ink_level = ink_level
        rank_font = dejavu_font("DejaVuSerif-Bold.ttf")(64)
        suit_font = dejavu_font("DejaVuSans.ttf")(64)
        self.rank_templates = {r: [_render_shape(g, rank_font, a) for a in _TILTS]
                               for r, g in _RANK_GLYPHS.items()}
        self.suit_templates = {s: [_render_shape(g, suit_font, a) for a in _TILTS]
                               for s, g in _SUIT_GLYPHS.items()}

    @staticmethod
    def _best(g, templates) -> float:
        return max(_shape_score(g, t) for t in templates)

    def recognize(self, image) -> CardReading:
        from scipy import ndimage

        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
        if arr.size == 0:
            return CardReading(None, False, 0.0)
        gray = arr.mean(axis=2)
        low_sat = arr.max(axis=2) - arr.min(axis=2) < 25
        level = float(np.percentile(gray[low_sat], 90)) if low_sat.any() else 0.0
        face = low_sat & (gray >= 0.75 * level)
        frac = float(face.mean())
        if level < self.min_face_level or frac < self.min_face_fraction:
            empty = 1.0 if level < self.min_face_level else 1.0 - frac / self.min_face_fraction
            return CardReading(None, False, float(min(1.0, empty + 0.5)))
        dev = level - gray
        ink = ~face & (dev > self.ink_level * dev.max())
        labels, _ = ndimage.label(ink, structure=np.ones((3, 3)))
        H, W = ink.shape
        rank_ids, suit = [], None
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            n = int((labels[sl] == i).sum())
            cy = (sl[0].start + sl[0].stop) / 2 / H
            cx = (sl[1].start + sl[1].stop) / 2 / W
            # Blobs on the left / top / bottom edge belong to the background
            # or the neighbouring card, not this card's symbols.
            if sl[1].start == 0 or sl[0].start == 0 or sl[0].stop == H or n < 4:
                continue
            if cy < 0.45 and cx < 0.6:
                rank_ids.append(i)
            elif 0.4 < cy < 0.95 and (suit is None or n > suit[0]):
                suit = (n, i)
        if not rank_ids or suit is None:
            return CardReading(None, True, 0.0)
        g = _shape(np.isin(labels, rank_ids))
        ranked = sorted(((self._best(g, t), r) for r, t in self.rank_templates.items()),
                        reverse=True)
        (r1, rank), (r2, _) = ranked[0], ranked[1]
        smask = labels == suit[1]
        mean = arr[smask].mean(axis=0)
        redness = (mean[0] - 0.5 * (mean[1] + mean[2])) / max(level, 1.0)
        pair = _RED if redness > 0.25 else _BLACK
        colour_conf = min(1.0, abs(redness - 0.25) / 0.2)
        gs = _shape(smask)
        (s1, s), (s2, _) = sorted(((self._best(gs, self.suit_templates[k]), k)
                                   for k in pair), reverse=True)
        conf = min(r1 * _margin(r1, r2), _margin(s1, s2), colour_conf)
        return CardReading(rank + s, True, float(max(0.0, conf)))
