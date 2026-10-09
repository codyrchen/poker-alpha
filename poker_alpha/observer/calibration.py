"""Table calibration: where everything is, in table-normalized coordinates.

A calibration is versioned JSON (``pokeralpha.calibration/v1``) so a user can
calibrate once per layout (theme, seat count) and reuse it at any window size:
regions are relative to the table bounding box, which is either fixed in
pixels or detected each frame (``table_detector``): ``"felt_color"`` takes the
extent of pixels near ``felt_color``; ``"green_oval"`` takes the largest
connected blob with the felt's *hue* (robust to the shading gradient, logo
watermark and pot pill on PokerNow's felt).

Region names (``i`` = seat index):

==================  ========================================================
``pot``             pot amount text
``board_0..4``      community card slots
``hero_card_0..1``  hero hole cards
``seat{i}_name``    player name text (presence => seat occupied)
``seat{i}_stack``   stack text (or "ALL IN")
``seat{i}_bet``     current street bet text
``seat{i}_cards``   where card backs show while the player is in the hand
``seat{i}_dealer``  dealer-button spot
``seat{i}_active``  acting-player highlight
==================  ========================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .errors import CalibrationError
from .regions import Box, Region

CALIBRATION_FORMAT = "pokeralpha.calibration/v1"
RGB = Tuple[int, int, int]
TABLE_DETECTORS = ("felt_color", "green_oval")
CLIENTS = ("generic", "pokernow")


@dataclass
class TableCalibration:
    name: str
    num_seats: int
    hero_seat: int
    regions: Dict[str, Region]
    table_bbox: Optional[Box] = None          # None => detect felt each frame
    table_detector: str = "felt_color"        # or "green_oval" (see module doc)
    client: str = "generic"                   # "pokernow": real-PokerNow recognizers
    pot_includes_bets: bool = True            # PokerNow's pot excludes street bets
    felt_color: RGB = (31, 94, 61)
    felt_tolerance: int = 40
    text_color: RGB = (240, 240, 240)
    highlight_color: RGB = (250, 210, 60)
    button_color: RGB = (250, 250, 250)
    card_back_color: RGB = (170, 40, 50)
    suit_colors: Dict[str, RGB] = field(default_factory=lambda: {
        "s": (20, 20, 20), "h": (200, 30, 40), "d": (30, 90, 210),
        "c": (30, 150, 60)})

    def __post_init__(self) -> None:
        if not 2 <= self.num_seats <= 9:
            raise CalibrationError("2-9 seats")
        if self.table_detector not in TABLE_DETECTORS:
            raise CalibrationError(f"table_detector must be one of {TABLE_DETECTORS}")
        if self.client not in CLIENTS:
            raise CalibrationError(f"client must be one of {CLIENTS}")
        if not 0 <= self.hero_seat < self.num_seats:
            raise CalibrationError("hero seat out of range")
        required = {"pot"} | {f"board_{i}" for i in range(5)} | \
            {"hero_card_0", "hero_card_1"} | {
                f"seat{i}_{k}" for i in range(self.num_seats)
                for k in ("stack", "bet", "cards", "dealer", "active")}
        missing = required - set(self.regions)
        if missing:
            raise CalibrationError(f"calibration missing regions: {sorted(missing)[:6]}...")

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": CALIBRATION_FORMAT, "name": self.name,
            "num_seats": self.num_seats, "hero_seat": self.hero_seat,
            "table_bbox": list(self.table_bbox) if self.table_bbox else None,
            "table_detector": self.table_detector,
            "client": self.client,
            "pot_includes_bets": self.pot_includes_bets,
            "felt_color": list(self.felt_color),
            "felt_tolerance": self.felt_tolerance,
            "text_color": list(self.text_color),
            "highlight_color": list(self.highlight_color),
            "button_color": list(self.button_color),
            "card_back_color": list(self.card_back_color),
            "suit_colors": {k: list(v) for k, v in self.suit_colors.items()},
            "regions": {k: r.to_list() for k, r in sorted(self.regions.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableCalibration":
        if d.get("format") != CALIBRATION_FORMAT:
            raise CalibrationError(f"unsupported calibration format {d.get('format')!r}")
        return cls(
            name=d["name"], num_seats=int(d["num_seats"]),
            hero_seat=int(d["hero_seat"]),
            regions={k: Region(*v) for k, v in d["regions"].items()},
            table_bbox=tuple(d["table_bbox"]) if d.get("table_bbox") else None,
            table_detector=d.get("table_detector", "felt_color"),
            client=d.get("client", "generic"),
            pot_includes_bets=bool(d.get("pot_includes_bets", True)),
            felt_color=tuple(d["felt_color"]),
            felt_tolerance=int(d["felt_tolerance"]),
            text_color=tuple(d["text_color"]),
            highlight_color=tuple(d["highlight_color"]),
            button_color=tuple(d["button_color"]),
            card_back_color=tuple(d["card_back_color"]),
            suit_colors={k: tuple(v) for k, v in d["suit_colors"].items()})

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TableCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def color_mask(image, color: RGB, tolerance: int) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr - np.array(color, dtype=np.int16)).max(axis=2)
    return diff <= tolerance


def locate_table(image, calibration: TableCalibration,
                 min_fraction: float = 0.05) -> Box:
    """Table bounding box: the calibrated one, or the felt's extent.

    Felt detection takes the bounding box of pixels near ``felt_color``
    after discarding sparse rows/columns (robust to stray matching pixels).
    """
    if calibration.table_bbox is not None:
        return tuple(int(v) for v in calibration.table_bbox)
    if calibration.table_detector == "green_oval":
        return locate_hue_blob(image, calibration.felt_color)
    mask = color_mask(image, calibration.felt_color, calibration.felt_tolerance)
    if mask.mean() < min_fraction:
        raise CalibrationError("table felt not found; recalibrate felt_color "
                               "or set table_bbox")
    rows = np.flatnonzero(mask.mean(axis=1) > 0.05)
    cols = np.flatnonzero(mask.mean(axis=0) > 0.05)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _hsv(arr: np.ndarray):
    mx, mn = arr.max(axis=-1), arr.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    d = np.maximum(mx - mn, 1e-9)
    hue = np.where(mx == r, ((g - b) / d) % 6,
                   np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    return hue, sat, mx / 255.0


def locate_hue_blob(image, felt_color: RGB, hue_tol: float = 25.0,
                    min_sat: float = 0.35, min_val: float = 0.2,
                    min_fraction: float = 0.03) -> Box:
    """Bounding box of the ellipse fitted to the largest blob with the felt's hue.

    Unlike :func:`locate_table` with a colour tolerance, this ignores
    brightness changes across the felt (PokerNow's radial shading, the
    watermark logo, the pot pill) and stray matching pixels elsewhere on
    screen (buttons, badges), which are separate, smaller blobs.
    """
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"))
    if rgb.size == 0:
        raise CalibrationError("empty frame")
    # The felt is large: find it on a subsampled frame (~640 px on the long
    # side), then scale the fitted ellipse back (error ~ the step, 1-3 px).
    step = max(1, int(round(max(rgb.shape[:2]) / 640)))
    arr = rgb[::step, ::step].astype(np.float64)
    hue, sat, val = _hsv(arr)
    ref_hue, _, _ = _hsv(np.array([[felt_color]], dtype=np.float64))
    dh = np.abs((hue - float(ref_hue[0, 0]) + 180.0) % 360.0 - 180.0)
    mask = (dh <= hue_tol) & (sat >= min_sat) & (val >= min_val)
    if mask.mean() < min_fraction:
        raise CalibrationError("table felt not found (no large blob with the felt hue); "
                               "check felt_color or set table_bbox")
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_fraction * mask.size:
        raise CalibrationError("table felt not found (felt-hue blob too small)")
    ys, xs = np.nonzero(labels == k)
    # Ellipse fitted by moments (a filled ellipse has semi-axis = 2 sigma):
    # unlike the blob's raw extent it barely moves when a badge, chip or the
    # dealer button touches the felt edge and joins the blob.
    cx, cy = (xs.mean() + 0.5) * step, (ys.mean() + 0.5) * step
    ax, ay = 2.0 * xs.std() * step, 2.0 * ys.std() * step
    return (int(round(cx - ax)), int(round(cy - ay)),
            int(round(cx + ax)), int(round(cy + ay)))
