"""Real PokerNow heads-up frame: layout geometry, recognition and the live
path, against the human ground truth in tests/fixtures/pokernow.

The frame is a real PokerNow table (cropped to the table; the chat preview and
player names are painted over). It is also the frame the heads-up layout was
aligned on and the PokerNow recognizers were tuned on, so these are
regression tests, not independent accuracy measurements.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("scipy")

from PIL import Image  # noqa: E402

from poker_alpha.holdem.observed import validate  # noqa: E402
from poker_alpha.observer.calibration import TableCalibration, locate_table  # noqa: E402
from poker_alpha.observer.cards import PokerNowCardRecognizer  # noqa: E402
from poker_alpha.observer.fusion import StateTracker  # noqa: E402
from poker_alpha.observer.live import (CaptureSettings, LiveObserverSession,  # noqa: E402
                                       critical_check, draw_overlay)
from poker_alpha.observer.pokernow import (PokerNowStyleAdapter, default_layout,  # noqa: E402
                                           pokernow_hu_layout)

pytestmark = pytest.mark.vision

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "pokernow"
FRAME = FIX / "raw" / "hu_preflop_0001.png"
TRUTH = json.loads((FIX / "annotations" / "hu_preflop_0001.json").read_text())
REF_TABLE = (4, 9, 631, 312)      # the felt ellipse box in the reference frame


def frame(scale: float = 1.0):
    img = Image.open(FRAME).convert("RGB")
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


def expected_fields():
    seats = {s["seat"]: s for s in TRUTH["seats"]}
    return {"hero_card_0": TRUTH["hero_cards"][0], "hero_card_1": TRUTH["hero_cards"][1],
            "seat0.stack": seats[0]["stack"], "seat1.stack": seats[1]["stack"],
            "seat0.bet": seats[0]["bet"], "seat1.bet": seats[1]["bet"],
            "pot": TRUTH["pot"], "dealer": TRUTH["dealer_seat"],
            "seat0.occupied": True, "seat1.occupied": True, "seat1.in_hand": True}


# -- layout / geometry ------------------------------------------------------------

def test_hu_layout_seats_hero_bottom_right_opponent_bottom_left():
    cal = pokernow_hu_layout()
    assert (cal.num_seats, cal.hero_seat, cal.client) == (2, 0, "pokernow")
    r = cal.regions
    assert r["seat0_stack"].x > r["seat1_stack"].x                  # hero right of opponent
    for s in (0, 1):                                                 # both along the bottom
        assert r[f"seat{s}_stack"].y > 1.0 and r[f"seat{s}_cards"].y > 0.9
    assert r["hero_card_0"].x < r["hero_card_1"].x < r["seat0_stack"].x
    left = pokernow_hu_layout("left").regions
    assert left["seat0_stack"].x == pytest.approx(r["seat1_stack"].x)
    assert left["seat1_stack"].x == pytest.approx(r["seat0_stack"].x)
    with pytest.raises(ValueError):
        pokernow_hu_layout("top")


def test_calibration_json_round_trip_and_old_files_still_load():
    cal = pokernow_hu_layout()
    back = TableCalibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert (back.table_detector, back.client, back.pot_includes_bets) == \
        ("green_oval", "pokernow", False)
    assert back.regions == cal.regions
    stored = TableCalibration.load(FIX / "calibration.json")
    assert stored.regions.keys() == cal.regions.keys()
    old = TableCalibration.load(ROOT / "tests" / "fixtures" / "table_calibration.json")
    assert (old.table_detector, old.client, old.pot_includes_bets) == \
        ("felt_color", "generic", True)


def test_felt_ellipse_locator_is_stable_under_crop_offset_and_scale():
    cal = pokernow_hu_layout()
    assert locate_table(frame(), cal) == REF_TABLE
    canvas = Image.new("RGB", (1400, 900), (24, 24, 24))      # browser chrome etc.
    canvas.paste(frame(), (300, 200))
    l, t, r, b = locate_table(canvas, cal)
    assert max(abs(l - 304), abs(t - 209), abs(r - 931), abs(b - 512)) <= 2
    l, t, r, b = locate_table(frame(2.0), cal)
    assert max(abs(l - 8), abs(t - 18), abs(r - 1262), abs(b - 624)) <= 3


def test_overlay_regions_land_on_the_measured_elements():
    """Overlay regression: key region boxes in reference-frame pixels."""
    img, cal = frame(), pokernow_hu_layout()
    bbox = locate_table(img, cal)
    px = {k: r.to_pixels(bbox) for k, r in cal.regions.items()}
    expected = {"pot": (262, 51, 367, 78), "seat1_dealer": (176, 281, 202, 302),
                "seat1_bet": (198, 362, 250, 379), "seat0_stack": (435, 348, 487, 365),
                "hero_card_0": (334, 320, 377, 380), "hero_card_1": (380, 319, 428, 379),
                "seat1_cards": (107, 321, 182, 376)}
    for k, box in expected.items():
        assert max(abs(a - b) for a, b in zip(px[k], box)) <= 1, (k, px[k])
    arr = np.asarray(img, dtype=int)

    def centre(k):
        l, t, r, b = px[k]
        return arr[(t + b) // 2, (l + r) // 2]
    assert centre("seat1_cards")[0] > centre("seat1_cards")[1] + 50     # red card back
    assert min(centre("seat1_dealer")) > 200                             # white D button
    assert centre("pot")[1] > centre("pot")[0] + 40                      # on the felt
    out = draw_overlay(img, cal, bbox)
    assert out.size == img.size and out.tobytes() != img.tobytes()


def test_generic_two_seat_oval_does_not_fit_pokernow_heads_up():
    """The bug that motivated the preset: the generic 2-seat oval puts seat 1
    at the top of the table, where PokerNow HU has nobody."""
    g = default_layout(2, 0)
    assert g.regions["seat1_stack"].y < 0.5
    hu = pokernow_hu_layout()
    assert hu.regions["seat1_stack"].y > 1.0


# -- recognition --------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 0.8, 1.5, 2.0, 3.0])
def test_real_frame_fields_match_ground_truth(scale):
    obs = PokerNowStyleAdapter(pokernow_hu_layout()).read_frame(frame(scale))
    for k, v in expected_fields().items():
        assert obs.value(k) == v, (scale, k, obs.value(k))
    for i in range(5):
        assert obs.value(f"board_{i}") is None                # preflop: no board
    for k in ("hero_card_0", "hero_card_1", "seat0.stack", "seat1.stack", "seat1.bet", "pot"):
        assert obs.confidence(k) >= 0.25, (scale, k, obs.confidence(k))


def test_pokernow_card_recognizer_rejects_non_cards():
    img, cal = frame(), pokernow_hu_layout()
    bbox = locate_table(img, cal)
    rec = PokerNowCardRecognizer()
    for name in ("seat1_cards", "board_0", "pot", "seat0_stack", "seat1_dealer"):
        r = rec.recognize(img.crop(cal.regions[name].to_pixels(bbox)))
        assert r.card is None, name
    assert rec.recognize(Image.new("RGB", (40, 60), (30, 30, 30))).present is False


def test_live_path_fuses_and_validates_real_frame():
    img = frame(2.0)                                           # ~ a Retina capture

    class Screen:
        def capture(self):
            return img
    s = LiveObserverSession(source_factory=lambda m, r: Screen())
    s.configure(CaptureSettings({"left": 0, "top": 0, "width": img.width,
                                 "height": img.height}, None),
                pokernow_hu_layout(), 0.25, 0.5)
    for i in range(3):
        assert s.step(now=float(i)).ok
    st = s.tracker.to_observed_state()
    assert st.hero_cards is not None and st.dealer == 1
    # PokerNow's centre pot leaves out bets in front of players: 0.00 + 1.00.
    assert st.pot_total == pytest.approx(1.0)
    assert not [p for p in validate(st) if p.severity == "error"]
    check = critical_check(s.tracker, 0.3)
    assert check.ok, check.problems


def test_pot_includes_bets_flag_only_changes_clients_that_need_it():
    obs = PokerNowStyleAdapter(pokernow_hu_layout()).read_frame(frame())
    for flag, pot in ((False, 1.0), (True, 0.0)):
        cal = pokernow_hu_layout()
        cal.pot_includes_bets = flag
        tr = StateTracker(cal, 0.25, 0.5)
        for _ in range(3):
            tr.update(obs)
        assert tr.to_observed_state().pot_total == pytest.approx(pot)
