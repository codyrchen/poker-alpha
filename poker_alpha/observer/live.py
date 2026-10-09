"""Live screen observer session (read-only) and calibration/debug helpers.

    screen source -> PokerNowStyleAdapter -> StateTracker -> ObservedTableState

:class:`LiveObserverSession` owns one source, adapter and tracker and is
advanced one frame at a time by :meth:`LiveObserverSession.step` — the UI
calls it on a timer (no blocking loop). Everything here only *reads*
pixels: there is no mouse, keyboard, browser control or action execution,
and frames never leave the machine (``save_frame`` writes a local file
only when explicitly asked).

The capture rectangle ``(left, top, width, height)`` is relative to the
selected monitor, in the monitor's coordinate units (screen points on
macOS; Retina captures then come back at 2x pixel size, which does not
matter because calibration regions are table-normalized).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .calibration import TableCalibration
from .errors import CalibrationError, ObserverDependencyError
from .fusion import StateTracker
from .pokernow import FrameObservation, PokerNowStyleAdapter
from .regions import Region

Rect = Tuple[int, int, int, int]  # left, top, width, height

PERMISSION_HELP = (
    "Screen capture failed or returned a blank frame. On macOS grant Screen "
    "Recording permission to the app that runs Streamlit (System Settings -> "
    "Privacy & Security -> Screen Recording: Terminal / iTerm / your IDE), "
    "then quit and restart that app.")


# -- monitors and sources ----------------------------------------------------

def list_monitors() -> List[Dict[str, int]]:
    """mss monitors (index 1.. = physical displays; 0 = all combined)."""
    try:
        import mss
    except ImportError as exc:
        raise ObserverDependencyError(
            "live capture needs mss: pip install 'poker-alpha[vision]'") from exc
    with mss.mss() as sct:
        return [dict(index=i, left=m["left"], top=m["top"], width=m["width"],
                     height=m["height"]) for i, m in enumerate(sct.monitors)]


def absolute_box(monitor: Dict[str, int], rect: Optional[Rect]) -> Rect:
    """Capture box in global screen coordinates; ``rect`` is monitor-relative
    and clipped to the monitor; ``None`` (or zero size) = whole monitor."""
    ml, mt, mw, mh = monitor["left"], monitor["top"], monitor["width"], monitor["height"]
    if not rect or rect[2] <= 0 or rect[3] <= 0:
        return (ml, mt, mw, mh)
    left = max(0, min(int(rect[0]), mw - 1))
    top = max(0, min(int(rect[1]), mh - 1))
    w = max(1, min(int(rect[2]), mw - left))
    h = max(1, min(int(rect[3]), mh - top))
    return (ml + left, mt + top, w, h)


def mss_source_factory(monitor: Dict[str, int], rect: Optional[Rect]):
    from .source import MSSScreenSource

    return MSSScreenSource(absolute_box(monitor, rect))


@dataclass(frozen=True)
class CaptureSettings:
    monitor: Dict[str, int]
    rect: Optional[Rect] = None

    def key(self) -> tuple:
        return (tuple(sorted(self.monitor.items())), tuple(self.rect) if self.rect else None)


# -- session -----------------------------------------------------------------

@dataclass
class StepResult:
    ok: bool
    error: Optional[str] = None
    warning: Optional[str] = None
    events: Tuple[str, ...] = ()


def calibration_key(cal: TableCalibration) -> str:
    return json.dumps(cal.to_dict(), sort_keys=True)


def is_blank(image, threshold: float = 2.0) -> bool:
    """True for an (almost) uniform frame — what macOS returns without
    Screen Recording permission, or a covered/minimized window."""
    import numpy as np

    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return arr.size == 0 or float(arr.std()) < threshold


class LiveObserverSession:
    """One live observer: source + adapter + tracker, advanced per frame."""

    def __init__(self, source_factory: Callable = mss_source_factory) -> None:
        self.source_factory = source_factory
        self.source = None
        self.capture: Optional[CaptureSettings] = None
        self.calibration: Optional[TableCalibration] = None
        self.adapter: Optional[PokerNowStyleAdapter] = None
        self.tracker: Optional[StateTracker] = None
        self._tracker_key = None
        self.running = False
        self.frames = 0
        self.errors = 0
        self.last_frame = None
        self.last_observation: Optional[FrameObservation] = None
        self.last_error: Optional[str] = None
        self.last_warning: Optional[str] = None
        self.last_time: Optional[float] = None
        # per-step diagnostics (seconds) and the tracker events of the last step
        self.last_timings: Dict[str, float] = {}
        self.last_events: list = []

    # configuration; the tracker survives frames and is rebuilt only when
    # the calibration, blinds or seat layout change
    def configure(self, capture: CaptureSettings, calibration: TableCalibration,
                  small_blind: float, big_blind: float) -> None:
        if self.capture is None or capture.key() != self.capture.key():
            self.capture = capture
            self.source = None                    # recreated lazily
        tkey = (calibration_key(calibration), float(small_blind), float(big_blind))
        if tkey != self._tracker_key:
            self.calibration = calibration
            self.adapter = PokerNowStyleAdapter(calibration)
            self.tracker = StateTracker(calibration, small_blind, big_blind)
            self._tracker_key = tkey

    def reset_tracker(self) -> None:
        if self.calibration is not None:
            self.tracker = StateTracker(self.calibration, self.tracker.sb, self.tracker.bb)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def grab(self):
        """Capture one frame (no recognition). Raises on failure."""
        if self.source is None:
            m = self.capture.monitor if self.capture else {"left": 0, "top": 0, "width": 0, "height": 0}
            self.source = self.source_factory(m, self.capture.rect if self.capture else None)
        return self.source.capture()

    def step(self, now: Optional[float] = None, recognize: bool = True) -> StepResult:
        """Capture one frame, read it and feed the tracker. Never raises:
        failures are returned and kept in ``last_error``."""
        now = time.time() if now is None else now
        self.last_timings, self.last_events = {}, []
        t0 = time.perf_counter()
        try:
            img = self.grab()
        except Exception as exc:  # noqa: BLE001 - any capture failure is reported
            self.errors += 1
            self.source = None
            self.last_error = f"capture failed: {type(exc).__name__}: {exc}. {PERMISSION_HELP}"
            self.last_timings["capture"] = time.perf_counter() - t0
            return StepResult(False, self.last_error)
        self.last_timings["capture"] = time.perf_counter() - t0
        self.last_frame = img
        self.frames += 1
        self.last_time = now
        self.last_error = None
        self.last_warning = PERMISSION_HELP if is_blank(img) else None
        if not recognize or self.adapter is None:
            return StepResult(True, warning=self.last_warning)
        t1 = time.perf_counter()
        try:
            obs = self.adapter.read_frame(img, timestamp=now)
        except CalibrationError as exc:
            self.last_timings["recognition"] = time.perf_counter() - t1
            self.last_error = f"calibration: {exc}"
            return StepResult(False, self.last_error, self.last_warning)
        except Exception as exc:  # noqa: BLE001
            self.last_timings["recognition"] = time.perf_counter() - t1
            self.last_error = f"recognition failed: {type(exc).__name__}: {exc}"
            return StepResult(False, self.last_error, self.last_warning)
        t2 = time.perf_counter()
        self.last_timings["recognition"] = t2 - t1
        self.last_observation = obs
        events = self.tracker.update(obs)
        self.last_events = list(events)
        self.last_timings["tracker"] = time.perf_counter() - t2
        self.last_timings["total"] = time.perf_counter() - t0
        return StepResult(True, warning=self.last_warning,
                          events=tuple(f"{e.kind}" + (f" seat {e.seat}" if e.seat is not None else "")
                                       + (f" {e.amount:g}" if e.amount is not None else "")
                                       for e in events))

    def table_bbox(self):
        if self.last_observation is not None:
            return self.last_observation.table_bbox
        if self.last_frame is not None and self.calibration is not None:
            try:
                return self.adapter.locate_table(self.last_frame)
            except CalibrationError:
                return None
        return None


def save_frame(image, directory, meta: Optional[dict] = None) -> Path:
    """Write ``image`` as PNG (plus a small JSON sidecar) under ``directory``.
    Local only; called only on an explicit user request."""
    d = Path(directory).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    ignore = d / ".gitignore"
    if not ignore.exists():                    # captured frames never get committed
        ignore.write_text("*\n")
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    path = d / f"frame-{stamp}.png"
    k = 1
    while path.exists():                       # never overwrite an earlier capture
        path = d / f"frame-{stamp}-{k}.png"
        k += 1
    image.save(path)
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return path


# -- calibration editing -------------------------------------------------------

def region_kind(name: str) -> str:
    if name.startswith("hero_card"):
        return "hero cards"
    if name.startswith("board_"):
        return "board"
    if name == "pot":
        return "pot"
    for k in ("stack", "bet", "dealer"):
        if name.endswith("_" + k):
            return k
    return "seat"


KIND_COLORS = {"table": (255, 230, 0), "hero cards": (255, 0, 255), "board": (0, 230, 255),
               "pot": (255, 150, 0), "stack": (0, 220, 80), "bet": (255, 60, 60),
               "dealer": (255, 255, 255), "seat": (90, 140, 255)}


def draw_overlay(image, calibration: TableCalibration, bbox=None, readings: Optional[FrameObservation] = None,
                 labels: bool = True):
    """Copy of ``image`` with the table bounds and every calibrated region
    drawn on it (colour per kind), optionally labelled with the reading."""
    from PIL import ImageDraw

    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    if bbox is None:
        try:
            bbox = PokerNowStyleAdapter(calibration).locate_table(image)
        except CalibrationError:
            bbox = None
    W, H = out.size
    width = max(1, round(min(W, H) / 400))
    if bbox is None:
        d.text((8, 8), "table not located: set a fixed table bbox or felt colour", fill=(255, 60, 60))
        return out
    d.rectangle(bbox, outline=KIND_COLORS["table"], width=width * 2)
    for name, region in sorted(calibration.regions.items()):
        kind = region_kind(name)
        box = region.to_pixels(bbox)
        d.rectangle(box, outline=KIND_COLORS[kind], width=width)
        # label every region except the small seat highlight / card-back boxes
        if labels and not name.endswith(("_cards", "_active")):
            text = name.replace("seat", "s").replace("hero_card_", "hero").replace("board_", "b")
            if readings is not None:
                key = name.replace("_stack", ".stack").replace("_bet", ".bet")
                if key in readings.fields:
                    text += f"={readings.value(key)}"
            d.text((box[0] + 2, box[1] + 1), text, fill=KIND_COLORS[kind])
    return out


def transform_regions(cal: TableCalibration, dx: float = 0.0, dy: float = 0.0,
                      sx: float = 1.0, sy: float = 1.0) -> TableCalibration:
    """All regions shifted by (dx, dy) and scaled about the table centre —
    the coarse alignment step of the calibration workflow."""
    out = {}
    for name, r in cal.regions.items():
        cx, cy = r.x + r.w / 2 - 0.5, r.y + r.h / 2 - 0.5
        w, h = r.w * sx, r.h * sy
        nx, ny = 0.5 + cx * sx + dx - w / 2, 0.5 + cy * sy + dy - h / 2
        out[name] = Region(nx, ny, w, h)
    return replace(cal, regions=out)


def with_region(cal: TableCalibration, name: str, region: Region) -> TableCalibration:
    regions = dict(cal.regions)
    regions[name] = region
    return replace(cal, regions=regions)


# -- display rows ---------------------------------------------------------------

def raw_rows(obs: Optional[FrameObservation]) -> List[dict]:
    """Per-field raw recognition of the last frame (before fusion)."""
    if obs is None:
        return []
    return [{"field": k, "value": None if f.value is None else str(f.value),
             "confidence": round(float(f.confidence), 3)} for k, f in sorted(obs.fields.items())]


def fused_rows(tracker: StateTracker) -> List[dict]:
    """The tracker's stable state, field by field, with its confidence."""
    t = tracker.tracked()
    snap, conf = t.snapshot, t.field_confidence
    rows = [{"field": "hero cards", "value": " ".join(c or "?" for c in snap.hero_cards),
             "confidence": round(min(conf["hero_card_0"], conf["hero_card_1"]), 3)},
            {"field": "board", "value": " ".join(snap.board) or "-",
             "confidence": round(min([conf[f"board_{i}"] for i in range(len(snap.board))] or [1.0]), 3)},
            {"field": "pot", "value": "?" if snap.pot is None else f"{snap.pot:g}",
             "confidence": round(conf["pot"], 3)},
            {"field": "dealer seat", "value": "?" if snap.dealer is None else str(snap.dealer),
             "confidence": round(conf["dealer"], 3)},
            {"field": "actor seat", "value": "?" if snap.actor is None else str(snap.actor),
             "confidence": round(conf["actor"], 3)}]
    return rows


def seat_state_rows(tracker: StateTracker) -> List[dict]:
    t = tracker.tracked()
    snap, conf = t.snapshot, t.field_confidence
    rows = []
    for s in range(tracker.cal.num_seats):
        status = "empty" if not snap.occupied[s] else (
            "all-in" if snap.all_in[s] else ("in hand" if snap.in_hand[s] else "folded"))
        rows.append({"seat": s, "hero": s == tracker.cal.hero_seat, "status": status,
                     "stack": snap.stacks[s], "stack conf": round(conf[f"seat{s}.stack"], 3),
                     "bet": snap.bets[s], "bet conf": round(conf[f"seat{s}.bet"], 3),
                     "dealer": snap.dealer == s})
    return rows


@dataclass(frozen=True)
class CriticalCheck:
    ok: bool
    problems: Tuple[str, ...]
    critical_confidence: float


def critical_check(tracker: StateTracker, min_confidence: float = 0.6) -> CriticalCheck:
    """Gate before any decision: confirmed hero cards, pot, hero stack,
    dealer, critical confidence and rules-level validation."""
    from ..holdem import validate

    snap = tracker.snapshot()
    hero = tracker.cal.hero_seat
    problems = []
    if None in snap.hero_cards:
        problems.append("hero cards not confirmed")
    if snap.pot is None:
        problems.append("pot not read")
    if snap.stacks[hero] is None and not snap.all_in[hero]:
        problems.append("hero stack not read")
    if snap.dealer is None:
        problems.append("dealer button not found")
    cc = tracker.critical_confidence()
    if cc < min_confidence:
        problems.append(f"critical confidence {cc:.0%} below {min_confidence:.0%}")
    if not problems:
        for issue in validate(tracker.to_observed_state()):
            if issue.severity == "error":
                problems.append(f"state invalid: {issue.message}")
    return CriticalCheck(not problems, tuple(problems), cc)
