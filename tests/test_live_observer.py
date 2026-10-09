"""Live screen observer: session, calibration overlay, decision gate and the
Streamlit "Live screen" mode — all with a mocked screen source, so CI needs
no monitor, no mss display and no screen-recording permission."""

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from poker_alpha.decision import DecisionConfig, recommend_action  # noqa: E402
from poker_alpha.observer.calibration import TableCalibration  # noqa: E402
from poker_alpha.observer.live import (CaptureSettings, LiveObserverSession,  # noqa: E402
                                       absolute_box, critical_check, draw_overlay,
                                       fused_rows, raw_rows, save_frame,
                                       seat_state_rows, transform_regions, with_region)
from poker_alpha.observer.regions import Region  # noqa: E402

pytestmark = pytest.mark.vision

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
MON = {"index": 1, "left": 0, "top": 0, "width": 1280, "height": 800}


class FakeScreen:
    """Stands in for MSSScreenSource: returns fixed frames, records calls."""

    def __init__(self, images, fail_first=0, exc=PermissionError("screen recording denied")):
        self.images, self.calls, self.fail_first, self.exc = list(images), 0, fail_first, exc

    def capture(self):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise self.exc
        return self.images[min(self.calls - 1, len(self.images) - 1)]


def table_img():
    return Image.open(FIX / "table.png").convert("RGB")


def cal():
    return TableCalibration.load(FIX / "table_calibration.json")


def session(screen, rect=None):
    made = []

    def factory(monitor, r):
        made.append((monitor, r))
        return screen
    s = LiveObserverSession(source_factory=factory)
    s.configure(CaptureSettings(MON, rect), cal(), 0.5, 1.0)
    return s, made


def test_absolute_box_is_monitor_relative_and_clipped():
    m = {"left": 1440, "top": 0, "width": 1920, "height": 1080}
    assert absolute_box(m, None) == (1440, 0, 1920, 1080)
    assert absolute_box(m, (0, 0, 0, 0)) == (1440, 0, 1920, 1080)
    assert absolute_box(m, (100, 50, 800, 600)) == (1540, 50, 800, 600)
    assert absolute_box(m, (1800, 1000, 800, 600)) == (3240, 1000, 120, 80)   # clipped


def test_tracker_persists_across_frames_and_state_is_recognized():
    screen = FakeScreen([table_img()])
    s, made = session(screen, (10, 20, 640, 400))
    tracker = s.tracker
    for i in range(3):
        res = s.step(now=float(i))
        assert res.ok, res.error
    assert s.tracker is tracker and s.frames == 3 and screen.calls == 3
    assert made == [(MON, (10, 20, 640, 400))]         # one source, reused
    fused = {r["field"]: r["value"] for r in fused_rows(s.tracker)}
    assert fused["hero cards"] == "As Ks" and fused["board"] == "Qs Js 4h"
    assert fused["pot"] == "13.5" and fused["dealer seat"] == "0"
    seats = seat_state_rows(s.tracker)
    assert seats[0]["hero"] and seats[0]["stack"] == 96.5 and seats[1]["status"] == "folded"
    raw = {r["field"]: r for r in raw_rows(s.last_observation)}
    assert raw["hero_card_0"]["value"] == "As" and 0 <= raw["pot"]["confidence"] <= 1


def test_same_settings_keep_tracker_changed_calibration_resets_it():
    s, _ = session(FakeScreen([table_img()]))
    t = s.tracker
    s.configure(CaptureSettings(MON, None), cal(), 0.5, 1.0)
    assert s.tracker is t
    c2 = cal()
    c2.table_bbox = (0, 0, 1000, 700)
    s.configure(CaptureSettings(MON, None), c2, 0.5, 1.0)
    assert s.tracker is not t


def test_decision_only_after_critical_validation_passes():
    s, _ = session(FakeScreen([table_img()]))
    check = critical_check(s.tracker, 0.5)
    assert not check.ok and "hero cards not confirmed" in check.problems
    s.step(now=0.0)                                    # one frame: cards not yet confirmed
    assert not critical_check(s.tracker, 0.5).ok
    for i in range(1, 3):
        s.step(now=float(i))
    check = critical_check(s.tracker, 0.5)
    assert check.ok, check.problems
    assert not critical_check(s.tracker, 0.99).ok      # threshold respected
    rep = recommend_action(s.tracker.to_observed_state(),
                           config=DecisionConfig(equity_simulations=200,
                                                 observer_confidence=check.critical_confidence))
    assert rep.recommended is not None


def test_capture_errors_are_reported_not_raised():
    screen = FakeScreen([table_img()], fail_first=2)
    s, made = session(screen)
    r = s.step(now=0.0)
    assert not r.ok and "Screen Recording" in r.error and s.errors == 1
    assert s.last_frame is None
    r = s.step(now=1.0)
    assert not r.ok and len(made) == 2                 # source recreated after a failure
    r = s.step(now=2.0)
    assert r.ok and s.last_error is None and s.frames == 1


def test_blank_frame_warns_about_permission():
    s, _ = session(FakeScreen([Image.new("RGB", (800, 500), (0, 0, 0))]))
    r = s.step(now=0.0)
    assert r.warning and "Screen Recording" in r.warning


def test_overlay_draws_all_region_kinds():
    img = table_img()
    c = cal()
    out = draw_overlay(img, c, readings=None)
    assert out.size == img.size and out.tobytes() != img.tobytes()
    assert img.getpixel((5, 5)) == table_img().getpixel((5, 5))   # input untouched
    # felt colour that is not on screen and no fixed box -> explained, not an exception
    c.felt_color = (255, 0, 255)
    out = draw_overlay(img, c)
    assert out.size == img.size


def test_calibration_edits():
    c = cal()
    same = transform_regions(c)
    assert all(abs(a - b) < 1e-12 for n in c.regions
               for a, b in zip(c.regions[n].to_list(), same.regions[n].to_list()))
    moved = transform_regions(c, dx=0.01, dy=-0.02)
    assert moved.regions["pot"].x == pytest.approx(c.regions["pot"].x + 0.01)
    assert moved.regions["pot"].y == pytest.approx(c.regions["pot"].y - 0.02)
    edited = with_region(c, "pot", Region(0.4, 0.5, 0.2, 0.05))
    assert edited.regions["pot"].w == 0.2 and c.regions["pot"].w != 0.2
    assert TableCalibration.from_dict(edited.to_dict()).regions["pot"].w == 0.2


def test_frames_are_saved_only_on_request(tmp_path):
    s, _ = session(FakeScreen([table_img()]))
    for i in range(3):
        s.step(now=float(i))
    assert list(tmp_path.iterdir()) == []
    p = save_frame(s.last_frame, tmp_path, {"note": "test"})
    assert p.exists() and p.suffix == ".png" and p.with_suffix(".json").exists()
    assert Image.open(p).size == s.last_frame.size


def test_live_modules_are_read_only():
    banned = ("pyautogui", "pynput", "selenium", "playwright", "webbrowser", "Quartz",
              "keyboard", "mouse", "click(", "submit")
    for f in ("poker_alpha/observer/live.py", "poker_alpha/ui/live_panel.py",
              "poker_alpha/observer/source.py"):
        text = (ROOT / f).read_text()
        for b in banned:
            assert f"import {b}" not in text and f"from {b}" not in text, (f, b)
        assert "requests" not in text and "urllib" not in text and "http" not in text.lower().replace(
            "https://", ""), f


# -- Streamlit "Live screen" mode, headless, mocked screen ---------------------

def _app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "poker_alpha" / "ui" / "app.py"), default_timeout=180)
    screen = FakeScreen([table_img()])
    at.session_state["live_source_factory"] = lambda monitor, rect: screen
    at.session_state["live_monitor_lister"] = lambda: [{"index": 0, "left": 0, "top": 0,
                                                        "width": 1280, "height": 800}, MON]
    at.run()
    at.sidebar.slider[0].set_value(200)
    at.sidebar.slider[1].set_value(0)
    at.sidebar.radio[0].set_value("Live screen").run()
    assert not at.exception, at.exception
    return at, screen


def _count(at, kind):
    n = 0
    stack = [at._tree]
    while stack:
        node = stack.pop()
        n += getattr(node, "type", None) == kind
        ch = getattr(node, "children", None)
        if isinstance(ch, dict):
            stack.extend(ch.values())
    return n


def _button(at, label):
    return [b for b in at.button if b.label == label][0]


def _texts(at):
    return " ".join(str(x.value) for x in list(at.markdown) + list(at.caption) + list(at.error)
                    + list(at.info) + list(at.subheader) + list(at.warning))


def test_streamlit_live_mode_calibration_flow_without_decisions():
    at, screen = _app()
    assert "No frame yet" in _texts(at)
    at.checkbox(key="live_compute").set_value(False).run()
    for _ in range(3):
        _button(at, "Capture one frame").click().run()
        assert not at.exception, at.exception
    assert screen.calls == 3
    txt = _texts(at)
    assert "Raw recognition" in txt and "Fused StateTracker state" in txt
    assert "Decision computation is disabled" in txt
    assert _count(at, "image") >= 2                # raw frame + calibration overlay
    assert "Suggested" not in txt


def test_streamlit_live_mode_decision_and_start_stop():
    at, screen = _app()
    _button(at, "Start Live Observer").click().run()
    assert not at.exception, at.exception
    for _ in range(2):          # each rerun advances the running observer by one frame
        at.run()
    assert screen.calls >= 3
    txt = _texts(at)
    assert "RUNNING" in txt
    assert "Suggested" in txt or "No recommendation" in txt, txt[:2000]
    _button(at, "Stop Live Observer").click().run()
    calls = screen.calls
    at.run()
    assert screen.calls == calls and "stopped" in _texts(at)


def test_streamlit_live_mode_blocks_decision_on_invalid_state():
    at, screen = _app()
    _button(at, "Capture one frame").click().run()       # one frame: hero cards unconfirmed
    txt = _texts(at)
    assert "No decision: critical state validation failed" in txt
    assert "Suggested" not in txt


def test_streamlit_calibration_editor(tmp_path):
    at, screen = _app()
    at.checkbox(key="live_compute").set_value(False).run()
    _button(at, "Capture one frame").click().run()
    before = at.session_state["live_cal"].regions["pot"].to_list()
    at.radio(key="live_bbox_mode").set_value("Fixed box").run()
    assert not at.exception, at.exception
    assert at.session_state["live_cal"].table_bbox is not None
    at.number_input(key="live_dx").set_value(0.01).run()
    _button(at, "Apply").click().run()
    assert not at.exception, at.exception
    after = at.session_state["live_cal"].regions["pot"].to_list()
    assert after[0] == pytest.approx(before[0] + 0.01)
    at.selectbox(key="live_region").set_value("pot").run()
    at.number_input(key="live_rw_pot").set_value(0.2).run()
    _button(at, "Update").click().run()
    assert at.session_state["live_cal"].regions["pot"].w == pytest.approx(0.2)
    out = tmp_path / "cal.json"
    at.text_input(key="live_cal_path").set_value(str(out)).run()
    _button(at, "Save calibration").click().run()
    assert TableCalibration.load(out).regions["pot"].w == pytest.approx(0.2)
    _button(at, "Load calibration").click().run()
    assert not at.exception, at.exception


def test_streamlit_pokernow_heads_up_preset():
    at, _ = _app()
    at.checkbox(key="live_compute").set_value(False).run()
    at.radio(key="live_preset").set_value("PokerNow Heads-Up").run()
    assert not at.exception, at.exception
    _button(at, "Use PokerNow Heads-Up layout").click().run()
    assert not at.exception, at.exception
    cal = at.session_state["live_cal"]
    assert (cal.client, cal.num_seats, cal.table_detector) == ("pokernow", 2, "green_oval")
    assert at.radio(key="live_bbox_mode").value == "PokerNow felt (hue)"
    at.radio(key="live_hu_side").set_value("left").run()
    _button(at, "Use PokerNow Heads-Up layout").click().run()
    left = at.session_state["live_cal"]
    assert left.regions["seat0_stack"].x < left.regions["seat1_stack"].x
    at.radio(key="live_preset").set_value("Generic layout").run()
    _button(at, "Use default layout").click().run()
    assert at.session_state["live_cal"].client == "generic"
    assert not at.exception, at.exception
