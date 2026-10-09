"""Observer test sessions: diagnostic, read-only recording of a live observer.

A test session watches the table exactly like live mode but its purpose is
to *collect data*: it keeps a small, meaningful subset of frames together
with everything the observer concluded about them, so the session can later
be replayed (:mod:`poker_alpha.observer.session_replay`), annotated and
exported as fixtures. Decisions are off by default.

Layout of one session (``~/pokeralpha_sessions/<YYYYmmdd_HHMMSS>/``)::

    session.json          metadata, policy, limits, notes, status, counters
    calibration.json      the calibration in force at the start
    calibrations/         later calibrations, by checksum (if changed mid-session)
    manifest.json         retained samples (also streamed to manifest.jsonl)
    frames/<id>.png       retained frames (PNG, lossless)
    observations/<id>.json  raw per-field readings + confidences + provenance
    observations/stream.jsonl  raw readings of every processed frame (no images)
    tracked_states/<id>.json  fused tracker state, critical confidence, validation
    diagnostics/<id>.json     timings, FPS, warnings, retention reasons
    events/events.jsonl   tracker events + session events (marks, notes, errors)
    annotations/          for later ground truth (pokeralpha.screenshot_annotation/v1)
    .gitignore            "*": the directory can never be committed by accident

Nothing is uploaded and nothing here touches the network: the recorder only
writes local files, only for retained samples, and stops at the configured
duration / frame / disk limits.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SESSION_FORMAT = "pokeralpha.observer_session/v1"
DEFAULT_SESSION_ROOT = "~/pokeralpha_sessions"

# Retention triggers. "manual" (Mark / Save buttons) is always available.
TRIGGERS = ("manual", "interval", "state_change", "warning", "low_confidence",
            "new_hand", "street")
TRIGGER_LABELS = {
    "manual": "marked / saved by hand",
    "interval": "every N seconds",
    "state_change": "fused table state changed",
    "warning": "new warning or error",
    "low_confidence": "critical confidence crossed the threshold",
    "new_hand": "new hand detected",
    "street": "street changed",
}
DEFAULT_TRIGGERS = ("state_change", "warning", "new_hand", "street")
POLICY_PRESETS: Dict[str, Tuple[str, ...]] = {
    "manual / marked only": (),
    "every N seconds": ("interval",),
    "state change": ("state_change",),
    "warning / error": ("warning",),
    "low-confidence transition": ("low_confidence",),
    "new hand": ("new_hand",),
    "street transition": ("street",),
    "hybrid (default)": DEFAULT_TRIGGERS,
}
SUBDIRS = ("frames", "observations", "tracked_states", "events", "diagnostics",
           "annotations", "calibrations")


@dataclass(frozen=True)
class SessionLimits:
    """Hard stops so an unattended session cannot fill the disk."""

    max_duration_s: float = 4 * 3600.0
    max_frames: int = 2000           # retained frames
    max_disk_mb: float = 2000.0      # bytes written by this session

    def __post_init__(self) -> None:
        if self.max_duration_s <= 0 or self.max_frames <= 0 or self.max_disk_mb <= 0:
            raise ValueError("session limits must be positive")


@dataclass(frozen=True)
class RetentionPolicy:
    triggers: Tuple[str, ...] = DEFAULT_TRIGGERS
    interval_s: float = 10.0
    low_confidence: float = 0.5

    def __post_init__(self) -> None:
        bad = set(self.triggers) - set(TRIGGERS)
        if bad:
            raise ValueError(f"unknown retention triggers {sorted(bad)}")
        if self.interval_s <= 0:
            raise ValueError("interval must be positive")


def calibration_checksum(calibration) -> str:
    """Geometry checksum (provenance such as save timestamps excluded)."""
    return calibration.geometry_checksum()


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except ImportError:  # pragma: no cover
        pass
    return str(v)


def _write_json(path: Path, obj) -> int:
    data = json.dumps(_jsonable(obj), indent=1, sort_keys=True).encode()
    path.write_bytes(data)
    return len(data)


def recognizer_provenance(adapter) -> Dict[str, object]:
    """Which recognizers produced the raw readings (for later comparison)."""
    if adapter is None:
        return {}

    def desc(obj):
        if obj is None:
            return None
        d = {"class": type(obj).__name__}
        for k in ("charset", "contrast", "relative_contrast", "background",
                  "confidence_ignores", "min_face_level", "min_face_fraction",
                  "ink_level", "face_threshold"):
            if hasattr(obj, k):
                d[k] = getattr(obj, k)
        return d
    return {"amount_ocr": desc(adapter.amount_ocr), "stack_ocr": desc(adapter.stack_ocr),
            "cards": desc(adapter.cards), "client": adapter.cal.client,
            "table_detector": adapter.cal.table_detector}


def git_commit(repo: Optional[Path] = None) -> Optional[str]:
    from ..utils.provenance import git_commit as _gc

    return _gc(repo)


def state_key(snapshot, hand_number: int) -> tuple:
    """Everything that defines 'the table changed' (timestamp excluded)."""
    return (hand_number, snapshot.pot, snapshot.board, snapshot.hero_cards,
            snapshot.dealer, snapshot.actor, snapshot.stacks, snapshot.bets,
            snapshot.in_hand, snapshot.occupied, snapshot.all_in)


def snapshot_dict(snapshot) -> dict:
    d = asdict(snapshot)
    d.pop("timestamp", None)
    return d


STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


@dataclass
class Sample:
    id: str
    frame_number: int
    reasons: Tuple[str, ...]
    wall_time: str
    monotonic: float
    files: Dict[str, str] = field(default_factory=dict)
    bytes: int = 0


class TestSessionRecorder:
    """Decides which frames to keep and writes them (see module docstring).

    Feed it after every :meth:`LiveObserverSession.step` via
    :meth:`on_step`; manual retention via :meth:`mark` / :meth:`save_current`.
    """

    __test__ = False      # not a pytest test class despite the name

    def __init__(self, root: os.PathLike = DEFAULT_SESSION_ROOT,
                 policy: Optional[RetentionPolicy] = None,
                 limits: Optional[SessionLimits] = None,
                 session_id: Optional[str] = None,
                 clock=time.monotonic, wall_clock=datetime.now) -> None:
        self.policy = policy or RetentionPolicy()
        self.limits = limits or SessionLimits()
        self._clock, self._wall = clock, wall_clock
        sid = session_id or self._wall().strftime("%Y%m%d_%H%M%S")
        base = Path(root).expanduser()
        path = base / sid
        k = 1
        while path.exists():                     # never overwrite a session
            path = base / f"{sid}_{k}"
            k += 1
        self.path = path
        self.id = path.name
        self.status = "created"
        self.samples: List[Sample] = []
        self.notes: List[dict] = []
        self.frames_seen = 0
        self.capture_errors = 0
        self.bytes_written = 0
        self.stop_reason: Optional[str] = None
        self.started_mono: Optional[float] = None
        self.started_wall: Optional[str] = None
        self.capture: Optional[dict] = None
        self.calibration_sum: Optional[str] = None
        self.calibration_name: Optional[str] = None
        self.decisions_enabled = False
        self.meta: Dict[str, object] = {}
        # trigger state
        self._last_key = None
        self._last_hand = None
        self._last_street = None
        self._last_warn: Tuple[str, ...] = ()
        self._last_low: Optional[bool] = None
        self._last_interval = None
        self._frame_times: List[float] = []
        self._pending_mark: Optional[str] = None
        self._tracker_ref = None

    # -- lifecycle ----------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.status in ("recording", "paused", "stopped")

    @property
    def recording(self) -> bool:
        return self.status == "recording"

    def start(self, calibration, monitor: Optional[dict] = None,
              rect: Optional[Sequence[int]] = None, decisions_enabled: bool = False,
              meta: Optional[dict] = None) -> Path:
        if self.status != "created":
            raise RuntimeError(f"session already {self.status}")
        self.path.mkdir(parents=True)
        (self.path / ".gitignore").write_text("*\n")
        for d in SUBDIRS:
            (self.path / d).mkdir()
        self.started_mono = self._clock()
        self.started_wall = self._wall().isoformat(timespec="seconds")
        self.capture = {"monitor": dict(monitor) if monitor else None,
                        "rect": list(rect) if rect else None}
        self.decisions_enabled = bool(decisions_enabled)
        self.meta = dict(meta or {})
        self._set_calibration(calibration, initial=True)
        self.status = "recording"
        self._event("session_start", detail=self.id)
        self._write_session()
        return self.path

    def pause(self) -> None:
        if self.status == "recording":
            self.status = "paused"
            self._event("pause")
            self._write_session()

    def resume(self) -> None:
        if self.status in ("paused", "stopped"):
            self.status = "recording"
            self._event("resume")
            self._write_session()

    def stop(self, reason: str = "stopped by user") -> None:
        """Stop capturing; notes / marks of the current frame stay possible."""
        if self.status in ("recording", "paused"):
            self.status = "stopped"
            self.stop_reason = reason
            self._event("stop", detail=reason)
            self._write_manifest()
            self._write_session()

    def finish(self) -> Path:
        if self.status == "created":
            raise RuntimeError("session was never started")
        if self.status != "finished":
            self._event("finish")
            self.status = "finished"
            self._write_manifest()
            self._write_session()
        return self.path

    # -- user actions -------------------------------------------------------------

    def add_note(self, text: str, frame_number: Optional[int] = None) -> None:
        text = text.strip()
        if not text or not self.active:
            return
        note = {"text": text, "frame_number": frame_number,
                "wall_time": self._wall().isoformat(timespec="seconds"),
                "monotonic": self._rel()}
        self.notes.append(note)
        self._event("note", detail=text, frame_number=frame_number)
        self._write_session()

    def mark(self, session, note: str = "") -> Optional[Sample]:
        """Keep the current frame because the user flagged it."""
        if note:
            self.add_note(note, session.frames)
        return self._retain(session, ("marked",), extra={"note": note} if note else None)

    def save_current(self, session) -> Optional[Sample]:
        return self._retain(session, ("manual_save",))

    def calibration_changed(self, calibration) -> None:
        if self.active and calibration_checksum(calibration) != self.calibration_sum:
            self._set_calibration(calibration, initial=False)
            self._write_session()

    # -- per-frame ------------------------------------------------------------------

    def on_step(self, session, result) -> Optional[Sample]:
        """Call after each observer step; returns the sample if one was kept.

        Steps taken while paused / stopped (e.g. "Capture one frame") are not
        retained but still streamed, so a replay sees every tracker update."""
        if not self.active:
            return None
        if session.tracker is not self._tracker_ref:
            self._tracker_ref = session.tracker
            self._stream_reset(session)
        if not self.recording:
            if session.last_frame is not None:
                self._stream(session, result)
            return None
        now = self._rel()
        if now > self.limits.max_duration_s:
            self._limit(f"max session duration {self.limits.max_duration_s:.0f} s reached")
            return None
        if not result.ok and (result.error or "").startswith("capture failed"):
            self.capture_errors += 1
            self._event("capture_error", detail=result.error)
            sig = ("capture_error",)
            if sig != self._last_warn:
                self._last_warn = sig
                self._write_session()
            return None
        self.frames_seen += 1
        self._frame_times.append(now)
        self._frame_times = self._frame_times[-20:]
        for e in session.last_events:
            self._event(e.kind, seat=e.seat, amount=e.amount, detail=e.detail,
                        frame_number=session.frames)
        self._stream(session, result)
        reasons = self._reasons(session, result, now)
        if not reasons:
            return None
        return self._retain(session, tuple(reasons), result=result)

    def _reasons(self, session, result, now: float) -> List[str]:
        trig = set(self.policy.triggers)
        reasons: List[str] = []
        tracker = session.tracker
        warn = self._warning_signature(session, result)
        if warn != self._last_warn:
            if warn and "warning" in trig:
                reasons.append("warning")
            self._last_warn = warn
        if "interval" in trig and (self._last_interval is None
                                   or now - self._last_interval >= self.policy.interval_s):
            reasons.append("interval")
            self._last_interval = now
        if tracker is None or session.last_observation is None or not result.ok:
            return reasons
        snap = tracker.snapshot()
        key = state_key(snap, tracker.hand_number)
        if key != self._last_key:
            if self._last_key is None:
                if trig - {"manual"}:       # automatic policies keep a baseline sample
                    reasons.append("first_frame")
            elif "state_change" in trig:
                reasons.append("state_change")
            self._last_key = key
        if tracker.hand_number != self._last_hand:
            if self._last_hand is not None and "new_hand" in trig:
                reasons.append("new_hand")
            self._last_hand = tracker.hand_number
        street = STREETS.get(len(snap.board), f"{len(snap.board)} board cards")
        if street != self._last_street:
            if self._last_street is not None and "street" in trig:
                reasons.append("street")
            self._last_street = street
        low = tracker.critical_confidence() < self.policy.low_confidence
        if self._last_low is not None and low != self._last_low and "low_confidence" in trig:
            reasons.append("low_confidence")
        self._last_low = low
        return reasons

    @staticmethod
    def _warning_signature(session, result) -> Tuple[str, ...]:
        sig = []
        if result.error:
            sig.append("error: " + result.error.split(".")[0])
        if result.warning:
            sig.append("warning: " + result.warning.split(".")[0])
        if session.tracker is not None:
            sig.extend(f"tracker: {f}" for f in session.tracker.flags[-5:])
        return tuple(sig)

    # -- writing -----------------------------------------------------------------------

    def _retain(self, session, reasons: Tuple[str, ...], result=None,
                extra: Optional[dict] = None) -> Optional[Sample]:
        if not self.active or session.last_frame is None:
            return None
        if len(self.samples) >= self.limits.max_frames:
            self._limit(f"max retained frames {self.limits.max_frames} reached")
            return None
        if self.bytes_written >= self.limits.max_disk_mb * 1e6:
            self._limit(f"max disk usage {self.limits.max_disk_mb:.0f} MB reached")
            return None
        n = session.frames
        if self.samples and self.samples[-1].frame_number == n:
            # Same frame retained twice (e.g. auto + Mark): add the reason only.
            s = self.samples[-1]
            s.reasons = tuple(dict.fromkeys(s.reasons + reasons))
            self._append_manifest(s)
            return s
        sid = f"{n:06d}"
        sample = Sample(sid, n, reasons, self._wall().isoformat(timespec="milliseconds"),
                        self._rel())
        written = 0
        fpath = self.path / "frames" / f"{sid}.png"
        session.last_frame.save(fpath, format="PNG")
        written += fpath.stat().st_size
        sample.files["frame"] = f"frames/{sid}.png"
        img_w, img_h = session.last_frame.size
        obs = session.last_observation if session.last_observation is not None and (
            session.last_observation.timestamp == session.last_time) else None
        common = {"sample": sid, "frame_number": n, "wall_time": sample.wall_time,
                  "monotonic": sample.monotonic, "calibration_name": self.calibration_name,
                  "calibration_checksum": self.calibration_sum}
        if obs is not None:
            written += _write_json(self.path / "observations" / f"{sid}.json", {
                **common, "table_bbox": list(obs.table_bbox),
                "fields": {k: {"value": f.value, "confidence": f.confidence,
                               "region": f.region} for k, f in sorted(obs.fields.items())},
                "provenance": recognizer_provenance(session.adapter)})
            sample.files["observation"] = f"observations/{sid}.json"
        tracker = session.tracker
        if tracker is not None:
            state = {**common, "hand_number": tracker.hand_number,
                     "critical_confidence": tracker.critical_confidence(),
                     "snapshot": snapshot_dict(tracker.snapshot()),
                     "field_confidence": tracker.tracked().field_confidence,
                     "tracker_warnings": list(tracker.flags[-20:])}
            try:
                from ..holdem.observed import validate

                st = tracker.to_observed_state()
                state["pot_total"] = st.pot_total
                state["validation"] = [{"severity": i.severity, "code": i.code,
                                        "message": i.message} for i in validate(st)]
            except Exception as exc:  # noqa: BLE001 - recorded, never raised
                state["validation"] = [{"severity": "error", "code": "state_build_failed",
                                        "message": f"{type(exc).__name__}: {exc}"}]
            written += _write_json(self.path / "tracked_states" / f"{sid}.json", state)
            sample.files["tracked_state"] = f"tracked_states/{sid}.json"
        rect = (self.capture or {}).get("rect")
        mon = (self.capture or {}).get("monitor") or {}
        geometry = None
        try:
            from .geometry import CaptureGeometry

            if mon and mon.get("width"):
                geometry = CaptureGeometry(mon, tuple(rect) if rect else None,
                                           (img_w, img_h)).summary()
        except (ValueError, KeyError, ZeroDivisionError):
            geometry = None
        diag = {**common, "reasons": list(reasons), "status": self.status,
                "source_monitor": mon, "capture_rect": rect,
                "captured_size": [img_w, img_h],
                "pixels_per_point": (list(geometry["pixels_per_point"])
                                     if geometry else None),
                "geometry": geometry,
                "table_bbox": list(obs.table_bbox) if obs is not None else None,
                "timings_s": dict(session.last_timings),
                "observer_fps": self.fps(),
                "decision_enabled": self.decisions_enabled,
                "step_error": getattr(result, "error", None) if result else session.last_error,
                "step_warning": getattr(result, "warning", None) if result else session.last_warning,
                "events": [{"kind": e.kind, "seat": e.seat, "amount": e.amount,
                            "detail": e.detail} for e in session.last_events]}
        if extra:
            diag.update(extra)
        written += _write_json(self.path / "diagnostics" / f"{sid}.json", diag)
        sample.files["diagnostics"] = f"diagnostics/{sid}.json"
        sample.bytes = written
        self.bytes_written += written
        self.samples.append(sample)
        self._append_manifest(sample)
        if len(self.samples) % 25 == 0:
            self._write_manifest()
            self._write_session()
        return sample

    def _stream(self, session, result) -> None:
        """Raw readings of *every* processed frame (no image): lets a replay
        rebuild the exact tracker sequence, not just the kept samples."""
        obs = session.last_observation
        if not result.ok or obs is None or obs.timestamp != session.last_time:
            row = {"n": session.frames, "ok": False, "error": result.error}
        else:
            row = {"n": session.frames, "ok": True, "t": obs.timestamp,
                   "bbox": list(obs.table_bbox),
                   "f": {k: [f.value, f.confidence, f.region]
                         for k, f in sorted(obs.fields.items())}}
        data = (json.dumps(_jsonable(row), separators=(",", ":")) + "\n").encode()
        with open(self.path / "observations" / "stream.jsonl", "ab") as fh:
            fh.write(data)
        self.bytes_written += len(data)

    def _stream_reset(self, session) -> None:
        """A new tracker (start, reset, recalibration, blinds change)."""
        tr = session.tracker
        row = {"n": session.frames, "reset": True,
               "sb": tr.sb if tr is not None else None,
               "bb": tr.bb if tr is not None else None,
               "calibration": (calibration_checksum(session.calibration)
                               if session.calibration is not None else None)}
        data = (json.dumps(_jsonable(row), separators=(",", ":")) + "\n").encode()
        with open(self.path / "observations" / "stream.jsonl", "ab") as fh:
            fh.write(data)
        self.bytes_written += len(data)
        if session.calibration is not None:
            self.calibration_changed(session.calibration)

    def _limit(self, reason: str) -> None:
        self.status = "stopped"
        self.stop_reason = "limit: " + reason
        self._event("limit", detail=reason)
        self._write_manifest()
        self._write_session()

    def _set_calibration(self, calibration, initial: bool) -> None:
        self.calibration_sum = calibration_checksum(calibration)
        self.calibration_name = calibration.name
        name = "calibration.json" if initial else f"calibrations/{self.calibration_sum[:12]}.json"
        self.bytes_written += _write_json(self.path / name, calibration.to_dict())
        if not initial:
            self._event("calibration_changed", detail=f"{calibration.name} "
                        f"{self.calibration_sum[:12]}")

    def _event(self, kind: str, **kw) -> None:
        if self.status == "created" and kind != "session_start":
            return
        row = {"kind": kind, "wall_time": self._wall().isoformat(timespec="milliseconds"),
               "monotonic": self._rel(), **{k: v for k, v in kw.items() if v is not None}}
        data = (json.dumps(_jsonable(row), sort_keys=True) + "\n").encode()
        with open(self.path / "events" / "events.jsonl", "ab") as fh:
            fh.write(data)
        self.bytes_written += len(data)

    def _append_manifest(self, s: Sample) -> None:
        data = (json.dumps(_jsonable(asdict(s)), sort_keys=True) + "\n").encode()
        with open(self.path / "manifest.jsonl", "ab") as fh:
            fh.write(data)
        self.bytes_written += len(data)

    def _write_manifest(self) -> None:
        self.bytes_written += _write_json(self.path / "manifest.json", {
            "format": SESSION_FORMAT, "session": self.id,
            "samples": [asdict(s) for s in self.samples]})

    def _write_session(self) -> None:
        self.bytes_written += _write_json(self.path / "session.json", self.summary())

    # -- info ---------------------------------------------------------------------------

    def _rel(self) -> float:
        return 0.0 if self.started_mono is None else self._clock() - self.started_mono

    def fps(self) -> Optional[float]:
        t = self._frame_times
        if len(t) < 2 or t[-1] <= t[0]:
            return None
        return (len(t) - 1) / (t[-1] - t[0])

    def summary(self) -> dict:
        return {
            "format": SESSION_FORMAT, "id": self.id, "status": self.status,
            "stop_reason": self.stop_reason, "started": self.started_wall,
            "elapsed_s": self._rel(), "capture": self.capture,
            "policy": asdict(self.policy), "limits": asdict(self.limits),
            "calibration_name": self.calibration_name,
            "calibration_checksum": self.calibration_sum,
            "decisions_enabled": self.decisions_enabled,
            "frames_seen": self.frames_seen, "frames_retained": len(self.samples),
            "capture_errors": self.capture_errors,
            "bytes_written": self.bytes_written, "notes": self.notes,
            "read_only": True, "uploads": "none (local files only)",
            "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                         "pokeralpha_commit": git_commit()},
            **({"meta": self.meta} if self.meta else {}),
        }
