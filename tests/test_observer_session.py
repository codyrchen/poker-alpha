"""Phase 41: observer test-session recorder (synthetic frames, fake screen).

No real PokerNow data is involved; the synthetic renderer provides a table
whose state changes on command.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from poker_alpha.observer.live import CaptureSettings, LiveObserverSession  # noqa: E402
from poker_alpha.observer.pokernow import default_layout  # noqa: E402
from poker_alpha.observer.session import (RetentionPolicy, SessionLimits,  # noqa: E402
                                          TestSessionRecorder, calibration_checksum)
from poker_alpha.observer.synthetic import (SyntheticSeat, SyntheticTable,  # noqa: E402
                                            render_table)

pytestmark = pytest.mark.vision
ROOT = Path(__file__).resolve().parents[1]
CAL = default_layout(2, 0)
MON = {"index": 1, "left": 0, "top": 0, "width": 1280, "height": 800}


def table(board=(), pot=1.5, dealer=1, bet1=0.0, hero=("As", "Kd")):
    return SyntheticTable(seats=[SyntheticSeat("h", 99.0), SyntheticSeat("v", 98.0, bet=bet1)],
                          dealer=dealer, hero_cards=hero, board=board, pot=pot, actor=0)


class Screen:
    """Plays a fixed list of frames; ``None`` = capture failure."""

    def __init__(self, frames):
        self.frames, self.i = list(frames), 0

    def capture(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        if f is None:
            raise PermissionError("screen recording denied")
        return f


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(tmp_path, frames, policy=None, limits=None, rect=None, clock=None, marks=()):
    scr = Screen(frames)
    s = LiveObserverSession(source_factory=lambda m, r: scr)
    s.configure(CaptureSettings(MON, rect), CAL, 0.5, 1.0)
    kw = {"clock": clock} if clock else {}
    rec = TestSessionRecorder(tmp_path / "sessions", policy, limits, **kw)
    rec.start(CAL, MON, rect)
    kept = []
    for i in range(len(frames)):
        if clock:
            clock.t = float(i)
        r = s.step(now=float(i))
        smp = rec.on_step(s, r)
        kept.append(smp.reasons if smp else None)
        if i in marks:
            rec.mark(s, f"mark at {i}")
    return rec, s, kept


def frames_of(*tables, repeat=3):
    out = []
    for t in tables:
        out += [render_table(t, CAL)] * repeat
    return out


def test_default_policy_keeps_baseline_state_changes_streets_and_new_hands(tmp_path):
    frames = frames_of(table(), table(board=("Qs", "Jh", "4c"), pot=3.0),
                       table(board=("Qs", "Jh", "4c", "2d"), pot=3.0),
                       table(dealer=0, hero=("7c", "7d")), repeat=5)
    rec, s, kept = run(tmp_path, frames)
    reasons = [r for k in kept if k for r in k]
    assert kept[0] == ("first_frame",)
    assert "street" in reasons and "new_hand" in reasons and "state_change" in reasons
    assert len(rec.samples) < len(frames)            # not every frame
    # fields confirm within 3 frames; the 4th and 5th identical frames add nothing
    settled = [k for i, k in enumerate(kept) if i % 5 in (3, 4)]
    assert all(k is None for k in settled)


def test_manual_only_policy_keeps_only_marked_frames(tmp_path):
    frames = frames_of(table(), table(board=("Qs", "Jh", "4c"), pot=3.0))
    rec, s, kept = run(tmp_path, frames, policy=RetentionPolicy(()), marks=(2,))
    assert all(k is None for k in kept)
    assert [smp.reasons for smp in rec.samples] == [("marked",)]
    assert rec.notes[0]["text"] == "mark at 2"


def test_interval_policy_uses_the_clock(tmp_path):
    frames = frames_of(table(), repeat=7)
    rec, s, kept = run(tmp_path, frames, policy=RetentionPolicy(("interval",), interval_s=3.0),
                       clock=Clock())
    assert [i for i, k in enumerate(kept) if k] == [0, 3, 6]


def test_warning_policy_keeps_transitions_not_every_warned_frame(tmp_path):
    blank = Image.new("RGB", (1280, 800), (0, 0, 0))
    frames = frames_of(table(), repeat=2) + [blank] * 4 + frames_of(table(), repeat=2)
    rec, s, kept = run(tmp_path, frames, policy=RetentionPolicy(("warning",)))
    idx = [i for i, k in enumerate(kept) if k and "warning" in k]
    assert idx and idx[0] == 2 and len(idx) <= 3      # into the warning, (out), not 4x


def test_low_confidence_transition(tmp_path):
    frames = frames_of(table(), repeat=4)
    rec, s, kept = run(tmp_path, frames, policy=RetentionPolicy(("low_confidence",),
                                                                low_confidence=0.5))
    # hero cards need 3 frames to confirm: critical confidence crosses 0.5 once
    assert kept[0] == ("first_frame",)
    assert sum(1 for k in kept if k and "low_confidence" in k) == 1


def test_capture_errors_are_logged_without_frames(tmp_path):
    frames = frames_of(table(), repeat=2) + [None, None]
    rec, s, kept = run(tmp_path, frames)
    assert rec.capture_errors == 2 and rec.frames_seen == 2
    events = [json.loads(x) for x in (rec.path / "events" / "events.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "capture_error" for e in events) == 2


@pytest.mark.parametrize("limits,expect", [
    (SessionLimits(max_frames=2), "max retained frames"),
    (SessionLimits(max_disk_mb=0.01), "max disk usage"),
    (SessionLimits(max_duration_s=2.5), "max session duration"),
])
def test_limits_stop_the_session(tmp_path, limits, expect):
    frames = frames_of(table(), table(board=("Qs", "Jh", "4c"), pot=3.0),
                       table(dealer=0, hero=("7c", "7d")))
    rec, s, kept = run(tmp_path, frames, policy=RetentionPolicy(("interval",), interval_s=1.0),
                       limits=limits, clock=Clock())
    assert rec.status == "stopped" and expect in rec.stop_reason
    n = len(rec.samples)
    assert n <= 3
    r = s.step(now=99.0)
    assert rec.on_step(s, r) is None and len(rec.samples) == n


def test_files_metadata_and_git_safety(tmp_path):
    frames = frames_of(table(), table(board=("Qs", "Jh", "4c"), pot=3.0))
    rect = (100, 50, 640, 400)
    rec, s, kept = run(tmp_path, frames, rect=rect)
    rec.add_note("first note")
    path = rec.finish()
    assert (path / ".gitignore").read_text().strip() == "*"
    for d in ("frames", "observations", "tracked_states", "events", "diagnostics",
              "annotations"):
        assert (path / d).is_dir()
    session = json.loads((path / "session.json").read_text())
    assert session["status"] == "finished" and session["read_only"] is True
    assert session["calibration_checksum"] == calibration_checksum(CAL)
    assert session["notes"][0]["text"] == "first note"
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest["samples"]) == len(rec.samples) >= 2
    smp = rec.samples[-1]
    for k in ("frame", "observation", "tracked_state", "diagnostics"):
        assert (path / smp.files[k]).exists()
    diag = json.loads((path / smp.files["diagnostics"]).read_text())
    assert diag["capture_rect"] == list(rect) and diag["captured_size"] == [1280, 800]
    assert diag["pixels_per_point"] == [2.0, 2.0]          # Retina-like 2x capture
    assert {"capture", "recognition", "tracker"} <= set(diag["timings_s"])
    assert diag["decision_enabled"] is False and diag["reasons"]
    obs = json.loads((path / smp.files["observation"]).read_text())
    assert obs["fields"]["hero_card_0"]["value"] == "As"
    assert obs["provenance"]["cards"]["class"] == "TemplateCardRecognizer"
    state = json.loads((path / smp.files["tracked_state"]).read_text())
    assert "critical_confidence" in state and "validation" in state
    assert Image.open(path / smp.files["frame"]).size == (1280, 800)
    # a second session with the same id never overwrites the first
    rec2 = TestSessionRecorder(tmp_path / "sessions", session_id=rec.id)
    assert rec2.path != rec.path


def test_mark_and_auto_retention_of_the_same_frame_merge(tmp_path):
    frames = frames_of(table(), repeat=1)
    rec, s, kept = run(tmp_path, frames, marks=(0,))
    assert len(rec.samples) == 1
    assert set(rec.samples[0].reasons) == {"first_frame", "marked"}


def test_pause_resume_stop_finish(tmp_path):
    frames = frames_of(table(), table(board=("Qs", "Jh", "4c"), pot=3.0))
    scr = Screen(frames)
    s = LiveObserverSession(source_factory=lambda m, r: scr)
    s.configure(CaptureSettings(MON, None), CAL, 0.5, 1.0)
    rec = TestSessionRecorder(tmp_path)
    rec.start(CAL, MON, None)
    rec.on_step(s, s.step(now=0.0))
    rec.pause()
    assert rec.on_step(s, s.step(now=1.0)) is None and rec.frames_seen == 1
    rec.resume()
    rec.on_step(s, s.step(now=2.0))
    assert rec.frames_seen == 2
    rec.stop()
    assert rec.on_step(s, s.step(now=3.0)) is None
    assert rec.save_current(s) is not None              # still possible after stop
    rec.finish()
    assert not rec.active and rec.status == "finished"
    with pytest.raises(RuntimeError):
        rec.start(CAL)


def test_streamlit_test_session_mode(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "poker_alpha" / "ui" / "app.py"), default_timeout=180)
    scr = Screen(frames_of(table(), table(board=("Qs", "Jh", "4c"), pot=3.0)))
    at.session_state["live_source_factory"] = lambda monitor, rect: scr
    at.session_state["live_monitor_lister"] = lambda: [MON, MON]
    at.run()
    at.sidebar.radio[0].set_value("Live screen").run()
    at.radio(key="live_mode").set_value("Observer Test Session (diagnostic)").run()
    assert not at.exception, at.exception
    assert at.checkbox(key="ts_compute").value is False        # decisions off by default
    at.text_input(key="ts_root").set_value(str(tmp_path)).run()
    [b for b in at.button if b.label == "Start session"][0].click().run()
    assert not at.exception, at.exception
    for _ in range(3):
        at.run()
    [b for b in at.button if b.label == "Mark current frame"][0].click().run()
    [b for b in at.button if b.label == "Finish session"][0].click().run()
    assert not at.exception, at.exception
    sessions = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(sessions) == 1
    session = json.loads((sessions[0] / "session.json").read_text())
    assert session["status"] == "finished" and session["frames_retained"] >= 2
    assert session["decisions_enabled"] is False
