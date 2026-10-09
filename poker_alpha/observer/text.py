"""OCR backends for numeric table text.

:class:`OCRBackend` is the pluggable interface. Two implementations:

* :class:`TemplateOCR` — pure NumPy glyph template matching. Templates are
  rendered from a font (default: Pillow's embedded font) or learned from
  labelled glyph crops of real screenshots via :meth:`TemplateOCR.from_samples`.
  It is reliable only when the on-screen font matches its templates; its
  confidence is the weakest glyph's normalized cross-correlation — a match
  score, **not** a calibrated probability.
* :class:`TesseractOCR` — wraps ``pytesseract``; raises :class:`OCRUnavailable`
  at construction if the module or the ``tesseract`` binary is missing. It is
  never silently replaced by a fake.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .errors import OCRUnavailable, require_pil

TEMPLATE_H = 24
TEMPLATE_W = 16
# Templates are rendered at full contrast (0 -> 255); threshold them at the
# same *relative* level the default contrast rule applies to typical light-on-
# felt UI text (~35% of the text/background difference).
TEMPLATE_THRESHOLD = 90.0


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float                 # weakest glyph match score in [0, 1]
    glyph_scores: Tuple[float, ...] = ()


class OCRBackend(Protocol):
    def read_text(self, image) -> OCRResult:
        ...


def _gray(image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _background(g: np.ndarray, mode: str = "border") -> float:
    if mode == "mode":        # the region's most common grey level
        hist, edges = np.histogram(g, bins=32, range=(0, 256))
        k = int(np.argmax(hist))
        return float(np.median(g[(g >= edges[k]) & (g <= edges[k + 1])]))
    return float(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))


def foreground_mask(image, contrast: float = 60.0, background: str = "border") -> np.ndarray:
    """Pixels that differ strongly from the background: the median of the
    region's border (default) or, with ``background="mode"``, the region's
    most common grey level (robust when the region's edge clips a
    neighbouring element, e.g. a bet pill inside a player plate)."""
    g = _gray(image)
    if g.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return np.abs(g - _background(g, background)) > contrast


def segment_glyphs(mask: np.ndarray, min_pixels: int = 2,
                   drop_edge_blobs: bool = False):
    """Split a single text line into glyph column spans.

    Glyphs are 8-connected components (so a comma tucked under a ``7`` is
    still its own glyph); components whose column ranges mostly overlap are
    merged. Returns ``(band_top, band_bottom, [(c0, c1), ...])`` sorted left
    to right, or ``None`` if the region holds no text. With
    ``drop_edge_blobs`` components touching the top or bottom edge are
    discarded: text is centred in its region, so such blobs are intruders
    (a neighbouring button, chip or card) rather than glyphs; so are blobs
    lying entirely above or below the tallest glyph.
    """
    from scipy import ndimage

    if mask.size == 0 or not mask.any():
        return None
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        rs, cs = sl
        if mask[rs, cs].sum() < min_pixels:
            continue
        if drop_edge_blobs and (rs.start == 0 or rs.stop == mask.shape[0]):
            continue
        if drop_edge_blobs and (cs.stop - cs.start) > 4 * (rs.stop - rs.start):
            continue  # a bar/underline (e.g. an action highlight), not a glyph
        boxes.append([cs.start, cs.stop, rs.start, rs.stop])
    if not boxes:
        return None
    if drop_edge_blobs:
        # One text line: blobs entirely above or below the tallest glyph are
        # fragments of neighbouring UI (a pill or plate edge), not text.
        tall = max(boxes, key=lambda b: b[3] - b[2])
        boxes = [b for b in boxes if b[2] < tall[3] and b[3] > tall[2]]
    boxes.sort()
    merged = [boxes[0]]
    for c0, c1, r0, r1 in boxes[1:]:
        m = merged[-1]
        overlap = min(m[1], c1) - max(m[0], c0)
        if overlap > 0.6 * min(c1 - c0, m[1] - m[0]):
            merged[-1] = [min(m[0], c0), max(m[1], c1), min(m[2], r0), max(m[3], r1)]
        else:
            merged.append([c0, c1, r0, r1])
    top = min(b[2] for b in merged)
    bottom = max(b[3] for b in merged)
    return top, bottom, [(b[0], b[1]) for b in merged]


@dataclass(frozen=True)
class GlyphFeatures:
    bitmap: np.ndarray     # tight-crop bitmap resized to the template grid
    aspect: float          # tight width / tight height
    height: float          # tight height / line cap height
    top: float             # (glyph top - cap top) / cap height
    pixels: int = 99       # tight height in pixels (tiny glyphs => shape unreliable)


def _tight(glyph: np.ndarray):
    rows = np.flatnonzero(glyph.any(axis=1))
    cols = np.flatnonzero(glyph.any(axis=0))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def glyph_features(glyph: np.ndarray, cap_top: int, cap_h: int) -> GlyphFeatures:
    """Features of one glyph given the line's cap band (top, height)."""
    from PIL import Image

    r0, r1, c0, c1 = _tight(glyph)
    tight = glyph[r0:r1, c0:c1]
    h, w = tight.shape
    img = Image.fromarray((tight * 255).astype(np.uint8))
    arr = np.asarray(img.resize((TEMPLATE_W, TEMPLATE_H), Image.BILINEAR),
                     dtype=np.float64) / 255.0
    return GlyphFeatures(arr, w / max(h, 1), h / max(cap_h, 1),
                         (r0 - cap_top) / max(cap_h, 1), int(h))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _similarity(g: GlyphFeatures, t: GlyphFeatures) -> float:
    shape = max(0.0, _ncc(g.bitmap, t.bitmap))
    if g.pixels < 6:
        # A few-pixel blob ('.', ',') has no usable shape at this scale;
        # identify it by size and position, capped below a clean match.
        shape = max(shape, 0.85)
    elif g.bitmap.std() == 0 or t.bitmap.std() == 0:  # solid blobs
        shape = 1.0 if g.bitmap.std() == t.bitmap.std() else 0.5
    return shape * float(np.exp(
        -1.5 * abs(np.log(max(g.aspect, 1e-3) / max(t.aspect, 1e-3)))
        - 3.0 * abs(g.height - t.height) - 3.0 * abs(g.top - t.top)))


def _cap_band(mask: np.ndarray, spans) -> Tuple[int, int]:
    """Top and height of the tallest glyph (a digit/capital) in the line."""
    best = (0, 1)
    for c0, c1 in spans:
        t = _tight(mask[:, c0:c1])
        if t is not None and t[1] - t[0] > best[1]:
            best = (t[0], t[1] - t[0])
    return best


class TemplateOCR:
    """Glyph-template OCR for a fixed, known UI font (see module docstring)."""

    def __init__(self, charset: str = "0123456789.,",
                 templates: Optional[Dict[str, "GlyphFeatures"]] = None,
                 font_factory: Optional[Callable[[int], object]] = None,
                 contrast: float = 60.0,
                 relative_contrast: Optional[float] = None,
                 confidence_ignores: str = "",
                 background: str = "border") -> None:
        require_pil()
        self.charset = charset
        self.contrast = contrast
        # If set, a pixel is text when it differs from the background by more
        # than this fraction of the region's strongest difference (and at
        # least ``contrast``): keeps small, bold, anti-aliased glyphs apart.
        self.relative_contrast = relative_contrast
        # Glyphs whose identity the caller resolves otherwise (separators by
        # number grammar, a leading sign) do not cap the confidence.
        self.confidence_ignores = confidence_ignores
        self.background = background
        self.templates = templates or self._render_templates(charset, font_factory,
                                                             contrast)

    @staticmethod
    def _render_templates(charset: str, font_factory=None, contrast: float = 60.0):
        from PIL import Image, ImageDraw, ImageFont

        font = (font_factory or (lambda s: ImageFont.load_default(size=s)))(48)
        out = {}
        for ch in charset:
            # Render between reference glyphs so the line band (cap height,
            # baseline, descender) matches how the glyph sits in real text.
            text = f"8{ch}8"
            img = Image.new("L", (240, 100), 0)
            ImageDraw.Draw(img).text((10, 10), text, fill=255, font=font)
            # Binarize exactly like input text (same contrast rule), so
            # anti-aliased fringes, and hence small-glyph geometry, match.
            mask = np.asarray(img, dtype=np.float64) > TEMPLATE_THRESHOLD
            _, _, spans = segment_glyphs(mask)
            if len(spans) != 3:
                raise RuntimeError(f"could not isolate template glyph {ch!r}")
            cap_top, cap_h = _cap_band(mask, [spans[0]])
            c0, c1 = spans[1]
            out[ch] = glyph_features(mask[:, c0:c1], cap_top, cap_h)
        return out

    @classmethod
    def from_samples(cls, samples: Dict[str, Sequence], contrast: float = 60.0
                     ) -> "TemplateOCR":
        """Learn templates from labelled crops of real screenshots.

        ``samples[ch]`` are images each showing the reference digit ``8``,
        then ``ch``, then ``8`` (as the UI renders them), so the glyph's
        height and baseline position relative to the digits are learned.
        """
        templates = {}
        for ch, images in samples.items():
            feats = []
            for im in images:
                mask = foreground_mask(im, contrast)
                _, _, spans = segment_glyphs(mask)
                if len(spans) != 3:
                    raise ValueError(f"sample for {ch!r} must show '8{ch}8'")
                cap_top, cap_h = _cap_band(mask, [spans[0]])
                c0, c1 = spans[1]
                feats.append(glyph_features(mask[:, c0:c1], cap_top, cap_h))
            templates[ch] = GlyphFeatures(
                np.mean([f.bitmap for f in feats], axis=0),
                float(np.mean([f.aspect for f in feats])),
                float(np.mean([f.height for f in feats])),
                float(np.mean([f.top for f in feats])))
        return cls("".join(templates), templates=templates, contrast=contrast)

    def classify(self, feats: GlyphFeatures) -> Tuple[str, float]:
        best, best_score = "?", -1.0
        for ch, tpl in self.templates.items():
            score = _similarity(feats, tpl)
            if score > best_score:
                best, best_score = ch, score
        return best, max(0.0, best_score)

    def _max_aspect(self) -> float:
        # Full-height glyphs only: small punctuation can be nearly square.
        tall = [t.aspect for t in self.templates.values() if t.height > 0.6]
        return max(tall or [t.aspect for t in self.templates.values()])

    def _split(self, mask: np.ndarray, span, cap_h: int, depth: int = 0):
        """Split touching glyphs at the lowest-ink column."""
        c0, c1 = span
        t = _tight(mask[:, c0:c1])
        if t is None:
            return []
        w, h = t[3] - t[2], max(cap_h, 1)
        if depth >= 4 or w / h <= 1.25 * self._max_aspect():
            return [span]
        ink = mask[:, c0:c1].sum(axis=0).astype(float)
        lo, hi = int(0.25 * (c1 - c0)), int(0.75 * (c1 - c0))
        if hi <= lo:
            return [span]
        cut = c0 + lo + int(np.argmin(ink[lo:hi]))
        return (self._split(mask, (c0, cut), cap_h, depth + 1)
                + self._split(mask, (cut, c1), cap_h, depth + 1))

    def _contrast(self, image) -> float:
        if self.relative_contrast is None:
            return self.contrast
        g = _gray(image)
        if g.size == 0:
            return self.contrast
        peak = float(np.abs(g - _background(g, self.background)).max())
        return max(self.contrast, self.relative_contrast * peak)

    def read_text(self, image) -> OCRResult:
        mask = foreground_mask(image, self._contrast(image), self.background)
        seg = segment_glyphs(mask, drop_edge_blobs=True)
        if seg is None:
            return OCRResult("", 1.0)
        top, bottom, spans = seg
        # Keep only the text line: blobs dropped by the segmentation (a name's
        # descender above, a pill or chip edge below) must not stretch glyphs.
        mask = mask[top:bottom]
        cap_top, cap_h = _cap_band(mask, spans)
        pieces = [p for sp in spans for p in self._split(mask, sp, cap_h)]
        chars, scores = [], []
        for c0, c1 in pieces:
            if _tight(mask[:, c0:c1]) is None:
                continue
            ch, sc = self.classify(glyph_features(mask[:, c0:c1], cap_top, cap_h))
            chars.append(ch)
            scores.append(sc)
        counted = [sc for ch, sc in zip(chars, scores) if ch not in self.confidence_ignores]
        counted = counted or scores
        return OCRResult("".join(chars), min(counted) if counted else 0.0,
                         tuple(scores))


def dejavu_font(name: str = "DejaVuSans-Bold.ttf") -> Callable[[int], object]:
    """Font factory for a DejaVu face bundled with matplotlib (a core
    dependency), so these templates need no system fonts."""
    import os

    import matplotlib
    from PIL import ImageFont

    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                        "ttf", name)
    return lambda size: ImageFont.truetype(path, size)


def pokernow_ocr(charset: str = "0123456789.,+") -> TemplateOCR:
    """Template OCR for PokerNow amounts (pot, stacks, bet pills).

    DejaVu Sans Bold is close to PokerNow's bold sans digits; the relative
    threshold keeps adjacent bold digits apart at small sizes. Separators
    (resolved by :func:`fix_separators`) and the bet pill's leading "+" do
    not cap the confidence. Chosen on one real heads-up frame
    (tests/fixtures/pokernow): a tuned setting, not an independently
    validated one.
    """
    return TemplateOCR(charset, font_factory=dejavu_font("DejaVuSans-Bold.ttf"),
                       relative_contrast=0.5, confidence_ignores=".,+",
                       background="mode")


class TesseractOCR:
    """``pytesseract`` backend (optional)."""

    def __init__(self, config: str = "--psm 7 -c tessedit_char_whitelist=0123456789.,") -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise OCRUnavailable("pytesseract is not installed") from exc
        if shutil.which("tesseract") is None:
            raise OCRUnavailable("the tesseract binary is not on PATH")
        self._tess = pytesseract
        self.config = config

    def read_text(self, image) -> OCRResult:
        data = self._tess.image_to_data(image, config=self.config,
                                        output_type=self._tess.Output.DICT)
        words = [(w, float(c)) for w, c in zip(data["text"], data["conf"])
                 if w.strip() and float(c) >= 0]
        if not words:
            return OCRResult("", 0.0)
        text = "".join(w for w, _ in words)
        return OCRResult(text, min(c for _, c in words) / 100.0)


def fix_separators(text: str) -> str:
    """Resolve '.' vs ',' in an amount by number grammar.

    Tiny punctuation glyphs are the least reliable OCR output, so for amounts
    the final separator is a decimal point if 1-2 digits follow it, and every
    other separator is a thousands comma — the same lexicon constraint
    general-purpose OCR engines apply.
    """
    seps = [i for i, ch in enumerate(text) if ch in ".,"]
    if not seps:
        return text
    chars = list(text)
    last = seps[-1]
    tail = len(text) - last - 1
    for i in seps:
        chars[i] = ","
    if 1 <= tail <= 2:
        chars[last] = "."
    return "".join(chars)


_AMOUNT = re.compile(r"^\d{1,3}(,\d{3})*(\.\d+)?$|^\d+(\.\d+)?$")


def parse_amount(text: str) -> Optional[float]:
    """Parse ``"1,234.5"``-style amounts; ``None`` if not a clean number."""
    t = text.strip()
    if not t or not _AMOUNT.match(t):
        return None
    return float(t.replace(",", ""))
