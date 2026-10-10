"""Phase 48: observer failure injection (synthetic frames; fake screen).

Required behaviour for every bad frame: never raise, never replace a
confirmed value with a misread one ("no hallucinated state"), keep the last
valid state, report an error / warning where one applies, and refuse a
decision (live_check) while the latest frame is unusable or contradicts the
confirmed critical state.
"""

import io

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("scipy")

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter  # noqa: E402

from observer_helpers import CAL, MON, Screen, table  # noqa: E402

from poker_alpha.observer.calibration import TableCalibration  # noqa: E402
from poker_alpha.observer.errors import CalibrationError, OCRUnavailable  # noqa: E402
from poker_alpha.observer.fusion import StateTracker  # noqa: E402
from poker_alpha.observer.live import (CaptureSettings, LiveObserverSession,  # noqa: E402
                                       absolute_box, critical_check, live_check)
from poker_alpha.observer.pokernow import (FieldReading, PokerNowStyleAdapter,  # noqa: E402
                                           pokernow_hu_layout)
from poker_alpha.observer.synthetic import render_table  # noqa: E402
from poker_alpha.observer.text import OCRResult  # noqa: E402

pytestmark = pytest.mark.vision

T = table(board=("Qs", "Jh", "4c"), pot=3.0, bet1=2.0, stacks=(99.0, 96.0))
GOOD = render_table(T, CAL)
CRITICAL = ("hero_card_0", "hero_card_1", "board_0", "board_1", "board_2", "pot",
            "seat0.stack", "seat1.stack", "seat1.bet", "dealer")


def _jpeg(im, q):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _box(im, box, col=(20, 20, 20)):
    im = im.copy()
    ImageDraw.Draw(im).rectangle(box, fill=col)
    return im


def _tint(im, r, b):
    a = np.asarray(im).astype(float)
    a[..., 0] *= r
    a[..., 2] *= b
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _modal(im):
    a = Image.fromarray((np.asarray(im).astype(float) * 0.35).astype(np.uint8))
    ImageDraw.Draw(a).rectangle((440, 250, 840, 550), fill=(245, 245, 245))
    return a


def _self_capture():
    page = Image.new("RGB", (1280, 800), (250, 250, 250))
    ImageDraw.Draw(page).text((40, 40), "PokerAlpha decision support", fill=(0, 0, 0))
    page.paste(render_table(table(pot=77.0), CAL).resize((320, 200)), (900, 560))
    return page


def _moved(dx, dy):
    page = Image.new("RGB", (1600, 1000), (40, 40, 48))
    page.paste(GOOD, (dx, dy))
    return page


def _session(frames, cal=CAL):
    scr = Screen(frames)
    s = LiveObserverSession(source_factory=lambda m, r: scr)
    s.configure(CaptureSettings(MON, None), cal, 0.5, 1.0)
    return s


def _established(bad, n_bad=3):
    s = _session([GOOD] * 3 + list(bad) * n_bad if isinstance(bad, list) else
                 [GOOD] * 3 + [bad] * n_bad)
    for i in range(3):
        s.step(now=float(i))
    assert critical_check(s.tracker, 0.5).ok
    before = {k: s.tracker._stable(k) for k in CRITICAL}
    results = [s.step(now=3.0 + i) for i in range(n_bad)]
    after = {k: s.tracker._stable(k) for k in CRITICAL}
    return s, before, after, results


# frames on which the table cannot be found: error, state kept, no decision
UNUSABLE = {
    "black frame": Image.new("RGB", (1280, 800), (0, 0, 0)),
    "white frame": Image.new("RGB", (1280, 800), (255, 255, 255)),
    "blank browser": _box(Image.new("RGB", (1280, 800), (245, 245, 245)), (0, 0, 1280, 70),
                          (200, 200, 200)),
    "PokerAlpha captures itself": _self_capture(),
    "modal over table": _modal(GOOD),
    "tab switched away": _box(Image.new("RGB", (1280, 800), (255, 255, 255)),
                              (0, 0, 1280, 70), (60, 60, 60)),
    "table not found": Image.new("RGB", (1280, 800), (90, 30, 120)),
}


@pytest.mark.parametrize("name", list(UNUSABLE))
def test_unusable_frames_keep_state_and_refuse_decisions(name):
    s, before, after, results = _established(UNUSABLE[name])
    assert after == before
    assert all(not r.ok for r in results) and "table" in results[-1].error
    check = live_check(s, 0.5)
    assert not check.ok and check.problems[0].startswith("stale")
    assert s.bad_streak == 3


# frames that are readable but degraded / partly hidden: never a changed
# confirmed value; a decision only when the frame agrees with the state
DEGRADED = {
    "50% scale": render_table(T, CAL, (640, 400)),
    "80% scale": render_table(T, CAL, (1024, 640)),
    "125% scale": render_table(T, CAL, (1600, 1000)),
    "150% scale": render_table(T, CAL, (1920, 1200)),
    "Retina 2x capture": render_table(T, CAL, (2560, 1600)),
    "browser moved": _moved(250, 120),
    "monitor dimensions change": render_table(T, CAL, (1440, 900)),
    "blur 1px": GOOD.filter(ImageFilter.GaussianBlur(1)),
    "blur 2px": GOOD.filter(ImageFilter.GaussianBlur(2)),
    "JPEG q30": _jpeg(GOOD, 30),
    "JPEG q10": _jpeg(GOOD, 10),
    "brightness +40%": ImageEnhance.Brightness(GOOD).enhance(1.4),
    "brightness -40%": ImageEnhance.Brightness(GOOD).enhance(0.6),
    "contrast 50%": ImageEnhance.Contrast(GOOD).enhance(0.5),
    "warm colour temperature": _tint(GOOD, 1.15, 0.85),
    "cool colour temperature": _tint(GOOD, 0.85, 1.15),
    "partial occlusion (hero area)": _box(GOOD, (520, 560, 760, 720)),
    "partial occlusion (board)": _box(GOOD, (420, 280, 640, 420), (240, 240, 240)),
}


@pytest.mark.parametrize("name", list(DEGRADED))
def test_degraded_frames_never_change_confirmed_state(name):
    s, before, after, results = _established(DEGRADED[name])
    assert all(r.ok for r in results)
    assert after == before, {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    check = live_check(s, 0.5)
    if check.ok:                     # allowed only when the frame agrees with the state
        obs = s.last_observation
        assert obs.value("hero_card_0") == "As" and obs.value("pot") == 3.0


def test_clean_rescaled_frames_still_allow_decisions():
    for name in ("80% scale", "125% scale", "browser moved", "Retina 2x capture"):
        s, before, after, _ = _established(DEGRADED[name])
        assert live_check(s, 0.5).ok, name


def test_browser_resized_mid_session_keeps_reading():
    frames = [GOOD] * 3 + [render_table(T, CAL, (1000, 625))] * 3 + [GOOD] * 2
    s = _session(frames)
    for i in range(8):
        assert s.step(now=float(i)).ok
    assert s.tracker._stable("pot") == 3.0 and live_check(s, 0.5).ok


def test_capture_rectangle_outside_monitor_is_an_error_not_a_1px_strip():
    m = {"left": 0, "top": 0, "width": 1280, "height": 800}
    with pytest.raises(ValueError, match="outside the monitor"):
        absolute_box(m, (1400, 10, 300, 300))
    with pytest.raises(ValueError, match="outside the monitor"):
        absolute_box(m, (-5, 10, 300, 300))
    from poker_alpha.observer.live import mss_source_factory  # noqa: F401 (import only)

    def factory(mon, rect):
        absolute_box(mon, rect)
        return Screen([GOOD])
    s = LiveObserverSession(source_factory=factory)
    s.configure(CaptureSettings(m, (1400, 10, 300, 300)), CAL, 0.5, 1.0)
    r = s.step()
    assert not r.ok and "outside the monitor" in r.error
    assert not live_check(s, 0.5).ok


def test_zero_width_crop_means_whole_monitor():
    m = {"left": 100, "top": 0, "width": 1280, "height": 800}
    assert absolute_box(m, (10, 10, 0, 300)) == (100, 0, 1280, 800)


def test_monitor_disconnected_then_reconnected():
    frames = [GOOD] * 3 + [None, None] + [GOOD] * 2
    s = _session(frames)
    for i in range(3):
        s.step(now=float(i))
    before = {k: s.tracker._stable(k) for k in CRITICAL}
    for i in range(2):
        r = s.step(now=3.0 + i)
        assert not r.ok and "capture failed" in r.error
        assert s.source is None                       # recreated on the next try
    assert {k: s.tracker._stable(k) for k in CRITICAL} == before
    assert not live_check(s, 0.5).ok
    for i in range(2):
        assert s.step(now=5.0 + i).ok
    assert live_check(s, 0.5).ok and s.bad_streak == 0


def test_monitor_settings_change_rebuilds_source_only():
    s = _session([GOOD] * 4)
    for i in range(3):
        s.step(now=float(i))
    tracker = s.tracker
    s.configure(CaptureSettings(dict(MON, width=1440, height=900), None), CAL, 0.5, 1.0)
    assert s.source is None and s.tracker is tracker      # same table state


def test_malformed_and_missing_calibration(tmp_path):
    good = CAL.to_dict()
    with pytest.raises(CalibrationError, match="format"):
        TableCalibration.from_dict(dict(good, format="nope"))
    bad = dict(good, regions={k: v for k, v in good["regions"].items() if k != "pot"})
    with pytest.raises(CalibrationError, match="missing regions"):
        TableCalibration.from_dict(bad)
    with pytest.raises(CalibrationError, match="table_detector"):
        TableCalibration.from_dict(dict(good, table_detector="magic"))
    with pytest.raises((ValueError, TypeError)):
        TableCalibration.from_dict(dict(good, regions=dict(good["regions"], pot=[0, 0, -1, 1])))
    with pytest.raises(FileNotFoundError):
        TableCalibration.load(tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(ValueError):
        TableCalibration.load(tmp_path / "broken.json")


def test_ocr_and_tesseract_unavailable(monkeypatch):
    import builtins
    import sys

    from poker_alpha.observer.text import TesseractOCR

    monkeypatch.setitem(sys.modules, "pytesseract", None)
    real_import = builtins.__import__

    def no_tess(name, *a, **k):
        if name == "pytesseract":
            raise ImportError("no pytesseract")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_tess)
    with pytest.raises(OCRUnavailable):
        TesseractOCR()
    # the default adapter never needs Tesseract
    assert PokerNowStyleAdapter(CAL).read_frame(GOOD).value("pot") == 3.0


class _GarbageOCR:
    def read_text(self, image):
        return OCRResult("4?..,1", 0.95)


def test_malformed_ocr_output_is_unreadable_not_a_number():
    ad = PokerNowStyleAdapter(CAL, amount_ocr=_GarbageOCR(), stack_ocr=_GarbageOCR())
    obs = ad.read_frame(GOOD)
    assert obs.value("pot") is None and obs.confidence("pot") == 0.0
    tr = StateTracker(CAL, 0.5, 1.0)
    for _ in range(3):
        tr.update(obs)
    assert tr._stable("pot") is None and not critical_check(tr, 0.5).ok


def _obs(**fields):
    base = PokerNowStyleAdapter(CAL).read_frame(GOOD)
    for k, v in fields.items():
        base.fields[k] = FieldReading(v[0], v[1], k)
    return base


def _tracker_established():
    tr = StateTracker(CAL, 0.5, 1.0)
    for _ in range(3):
        tr.update(_obs())
    return tr


def test_card_crop_blank_and_hero_cards_disappearing_transiently():
    tr = _tracker_established()
    for _ in range(2):                                 # hero cards unreadable for 2 frames
        tr.update(_obs(hero_card_0=(None, 1.0), hero_card_1=(None, 1.0)))
    assert tr._stable("hero_card_0") == "As" and tr._stable("hero_card_1") == "Kd"
    assert any("held hero_card_0" in f for f in tr.flags)
    tr.update(_obs())
    assert tr._stable("hero_card_0") == "As"


def test_impossible_duplicate_cards_refuse_decision():
    tr = StateTracker(CAL, 0.5, 1.0)
    for _ in range(3):
        tr.update(_obs(board_0=("As", 0.95)))           # board card equals a hero card
    check = critical_check(tr, 0.3)
    assert not check.ok and any("duplicate" in p.lower() or "invalid" in p for p in check.problems)


def test_impossible_board_transition_is_rejected():
    tr = _tracker_established()
    for _ in range(4):
        tr.update(_obs(board_1=("Kh", 0.99)))
    assert tr._stable("board_1") == "Jh"
    assert any("rejected board_1 change" in f for f in tr.flags)


def test_stack_increase_mid_hand_is_held():
    tr = _tracker_established()
    for _ in range(2):
        tr.update(_obs(**{"seat1.stack": (150.0, 0.95)}))
    assert tr._stable("seat1.stack") == 96.0
    assert any("held stack increase" in f for f in tr.flags)


def test_dealer_jump_mid_hand_is_rejected():
    tr = _tracker_established()
    for _ in range(3):
        tr.update(_obs(dealer=(0, 1.0)))
    assert tr._stable("dealer") == 1 and tr.hand_number == 0
    assert any("rejected dealer move" in f for f in tr.flags)


def test_bet_cannot_shrink_without_collection():
    tr = _tracker_established()
    for _ in range(2):
        tr.update(_obs(**{"seat1.bet": (0.0, 0.85)}))
    assert tr._stable("seat1.bet") == 2.0
    # a real collection: pot rises (bets swept in) -> the drop is accepted
    for _ in range(2):
        tr.update(_obs(**{"seat1.bet": (0.0, 0.85), "pot": (5.0, 0.95)}))
    assert tr._stable("seat1.bet") == 0.0 and tr._stable("pot") == 5.0


def test_misread_hero_card_is_held_but_a_real_new_hand_updates_cards():
    tr = _tracker_established()
    for _ in range(3):
        tr.update(_obs(hero_card_1=("Ks", 0.75)))       # occluded suit
    assert tr._stable("hero_card_1") == "Kd"
    new = render_table(table(dealer=0, hero=("7c", "7d")), CAL)
    ad = PokerNowStyleAdapter(CAL)
    for _ in range(4):
        tr.update(ad.read_frame(new))
    assert tr.hand_number == 1 and tr._stable("hero_card_0") == "7c"


def test_tiny_table_warns_or_fails_with_self_capture_hint():
    from dataclasses import replace

    page = Image.new("RGB", (2560, 1600), (250, 250, 250))
    page.paste(GOOD.resize((320, 200)), (100, 100))
    r = _session([page]).step()                    # felt below the detector minimum
    assert not r.ok and "PokerAlpha's own window" in r.error
    page = Image.new("RGB", (2560, 1600), (250, 250, 250))
    page.paste(GOOD.resize((640, 400)), (100, 100))
    r = _session([page], replace(CAL, table_detector="green_oval")).step()
    assert r.ok and "is PokerAlpha capturing its own window" in r.warning


def test_pokernow_preset_unusable_frame_fails_safe():
    cal = pokernow_hu_layout()
    s = _session([Image.new("RGB", (1200, 700), (30, 30, 30))], cal)
    r = s.step()
    assert not r.ok and "felt" in r.error and not live_check(s, 0.3).ok


def test_tracker_history_is_bounded():
    tr = StateTracker(CAL, 0.5, 1.0)
    tr.EVENT_KEEP, tr.FLAG_KEEP = 10, 5
    for i in range(200):
        tr.flags.append(f"f{i}")
        tr.events.append(i)
        tr.update(_obs())
    assert len(tr.flags) <= 2 * tr.FLAG_KEEP + 5 and len(tr.events) <= 2 * tr.EVENT_KEEP + 5
    assert tr.flag_total >= 200                      # monotonic despite trimming
    assert "f199" in tr.flags                         # the newest entries are kept
    assert tr._stable("pot") == 3.0                      # trimming never touches state
