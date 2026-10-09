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

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .errors import CalibrationError
from .regions import Box, Region

CALIBRATION_FORMAT = "pokeralpha.calibration/v2"
CALIBRATION_V1 = "pokeralpha.calibration/v1"
SCHEMA_VERSION = 2
# Keys a v2 file may contain; anything else was written by an unknown
# (newer) version and is refused rather than silently ignored.
V2_KEYS = {"format", "schema_version", "name", "num_seats", "hero_seat", "table_bbox",
           "table_detector", "client", "pot_includes_bets", "felt_color", "felt_tolerance",
           "text_color", "highlight_color", "button_color", "card_back_color", "suit_colors",
           "regions", "region_semantics", "source", "created", "updated", "migrated_from",
           "checksum"}
REGION_SEMANTICS = ("x, y, w, h as fractions of the table box (located table in "
                    "captured-image pixels; see observer/geometry.py)")
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
    # provenance (not part of the geometry checksum)
    source: str = ""
    created: Optional[str] = None
    updated: Optional[str] = None
    migrated_from: Optional[str] = None

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

    def geometry_payload(self) -> dict:
        """Everything that changes what the observer reads (no provenance)."""
        return {
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
            "suit_colors": {k: list(v) for k, v in sorted(self.suit_colors.items())},
            "regions": {k: r.to_list() for k, r in sorted(self.regions.items())},
        }

    def geometry_checksum(self) -> str:
        blob = json.dumps(self.geometry_payload(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict:
        return {"format": CALIBRATION_FORMAT, "schema_version": SCHEMA_VERSION,
                "name": self.name, **self.geometry_payload(),
                "region_semantics": REGION_SEMANTICS, "source": self.source,
                "created": self.created, "updated": self.updated,
                "migrated_from": self.migrated_from,
                "checksum": self.geometry_checksum()}

    @classmethod
    def from_dict(cls, d: dict, verify: bool = True) -> "TableCalibration":
        """Load v2, or migrate v1 (same geometry semantics; documented v1
        defaults for keys v1 files may lack). Anything else is refused."""
        fmt = d.get("format")
        if fmt == CALIBRATION_V1:
            migrated = "v1"
        elif fmt == CALIBRATION_FORMAT:
            migrated = d.get("migrated_from")
            unknown = set(d) - V2_KEYS
            if unknown:
                raise CalibrationError(f"calibration has unknown fields {sorted(unknown)} "
                                       "(written by a newer PokerAlpha?)")
        elif isinstance(fmt, str) and fmt.startswith("pokeralpha.calibration/"):
            raise CalibrationError(f"calibration format {fmt!r} is newer than this "
                                   f"PokerAlpha understands ({CALIBRATION_FORMAT}); upgrade")
        else:
            raise CalibrationError(f"unsupported calibration format {fmt!r}")
        try:
            cal = cls(
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
                suit_colors={k: tuple(v) for k, v in d["suit_colors"].items()},
                source=str(d.get("source", "")), created=d.get("created"),
                updated=d.get("updated"), migrated_from=migrated)
        except KeyError as exc:
            raise CalibrationError(f"calibration is missing field {exc}") from exc
        if fmt == CALIBRATION_FORMAT and verify:
            want = d.get("checksum")
            if want is None:
                raise CalibrationError("v2 calibration without checksum")
            if want != cal.geometry_checksum():
                raise CalibrationError(
                    "calibration checksum mismatch: the geometry was changed outside "
                    "PokerAlpha or the file is damaged; re-save it from the UI, or load with "
                    "verify=False if the edit was intentional")
        return cal

    def save(self, path: Union[str, Path]) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.created = self.created or now
        self.updated = now
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Union[str, Path], verify: bool = True) -> "TableCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()), verify=verify)


def color_mask(image, color: RGB, tolerance: int) -> np.ndarray:
    # Per-channel np.maximum: same result as .max(axis=2), ~10x faster
    # (a reduction over a length-3 last axis is slow in NumPy).
    arr = np.asarray(image.convert("RGB"))
    if arr.size == 0:
        return np.zeros(arr.shape[:2], dtype=bool)
    diff = None
    for ch in range(3):
        d = np.abs(arr[..., ch].astype(np.int16) - int(color[ch]))
        diff = d if diff is None else np.maximum(diff, d)
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
        return _plausible(locate_hue_blob(image, calibration.felt_color))
    mask = color_mask(image, calibration.felt_color, calibration.felt_tolerance)
    if mask.mean() < min_fraction:
        raise CalibrationError("table felt not found; recalibrate felt_color "
                               "or set table_bbox" + NOT_FOUND_HINT)
    rows = np.flatnonzero(mask.mean(axis=1) > 0.05)
    cols = np.flatnonzero(mask.mean(axis=0) > 0.05)
    if len(rows) == 0 or len(cols) == 0:
        raise CalibrationError("table felt not found" + NOT_FOUND_HINT)
    H, W = mask.shape
    if cols[0] == 0 or rows[0] == 0 or cols[-1] == W - 1 or rows[-1] == H - 1:
        raise CalibrationError(CUT_OFF)
    return _plausible((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


NOT_FOUND_HINT = ("; check that the capture shows the poker table (not another tab, "
                  "a modal, or PokerAlpha's own window)")
TABLE_ASPECT = (1.0, 3.5)      # width / height of any plausible poker table
CUT_OFF = ("the table felt touches the edge of the capture (table cut off): widen the "
           "capture rectangle so the whole table and the player plates are inside")


def _plausible(box: Box) -> Box:
    """Reject 'tables' no poker client draws (a toolbar, a strip, a column):
    a broad shape check, not tuned to any client."""
    w, h = box[2] - box[0], box[3] - box[1]
    aspect = w / max(h, 1)
    if not TABLE_ASPECT[0] <= aspect <= TABLE_ASPECT[1]:
        raise CalibrationError(f"felt-coloured region {w}x{h} is not table-shaped "
                               f"(aspect {aspect:.1f})" + NOT_FOUND_HINT)
    return box


def _hsv(arr: np.ndarray):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)       # == arr.max(-1), much faster
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
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
                               "check felt_color or set table_bbox" + NOT_FOUND_HINT)
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_fraction * mask.size:
        raise CalibrationError("table felt not found (felt-hue blob too small)")
    ys, xs = np.nonzero(labels == k)
    if xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1] - 1 \
            or ys.max() == mask.shape[0] - 1:
        raise CalibrationError(CUT_OFF)
    # Ellipse fitted by moments (a filled ellipse has semi-axis = 2 sigma):
    # unlike the blob's raw extent it barely moves when a badge, chip or the
    # dealer button touches the felt edge and joins the blob.
    cx, cy = (xs.mean() + 0.5) * step, (ys.mean() + 0.5) * step
    ax, ay = 2.0 * xs.std() * step, 2.0 * ys.std() * step
    return (int(round(cx - ax)), int(round(cy - ay)),
            int(round(cx + ax)), int(round(cy + ay)))
