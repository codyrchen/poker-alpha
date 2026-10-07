"""Phase 17: PokerNow-style adapter, smoothing and event inference.

All images here are SYNTHETIC (rendered by poker_alpha.observer.synthetic).
Exact PokerNow visual accuracy is not validated without representative
screenshots; these tests pin the architecture and the synthetic pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from poker_alpha.observer.calibration import TableCalibration  # noqa: E402
from poker_alpha.observer.evaluation import evaluate  # noqa: E402
from poker_alpha.observer.pokernow import (FieldReading,  # noqa: E402
                                           PokerNowStyleAdapter,
                                           default_layout)
from poker_alpha.observer.synthetic import (SyntheticSeat,  # noqa: E402
                                            SyntheticTable, random_table,
                                            render_table)
from poker_alpha.observer.tracker import (FieldTracker,  # noqa: E402
                                          TableSnapshot, infer_events)

pytestmark = pytest.mark.vision
FIX = Path(__file__).resolve().parent / "fixtures"


def test_committed_fixture_matches_ground_truth():
    cal = TableCalibration.load(FIX / "table_calibration.json")
    truth = json.loads((FIX / "table.json").read_text())["table"]
    obs = PokerNowStyleAdapter(cal).read_frame(Image.open(FIX / "table.png").convert("RGB"))
    assert (obs.value("hero_card_0"), obs.value("hero_card_1")) == tuple(truth["hero_cards"])
    assert [obs.value(f"board_{i}") for i in range(3)] == truth["board"]
    assert obs.value("board_3") is None
    assert obs.value("pot") == truth["pot"]
    for s, seat in enumerate(truth["seats"]):
        assert obs.value(f"seat{s}.stack") == seat["stack"]
        assert obs.value(f"seat{s}.bet") == seat["bet"]
    assert obs.value("dealer") == truth["dealer"]
    assert obs.value("actor") == truth["actor"]


@pytest.mark.parametrize("n,size,noise", [(6, (1280, 800), 0), (3, (1280, 800), 0),
                                          (6, (1600, 1000), 6), (9, (1280, 800), 0)])
def test_synthetic_accuracy(n, size, noise):
    rng = np.random.default_rng(n)
    cal = default_layout(n, 0)
    frames = [(render_table(t := random_table(rng, n), cal, size, noise=noise,
                            seed=i), t) for i in range(12)]
    rep = evaluate(PokerNowStyleAdapter(cal), frames)
    assert rep.card_accuracy == 1.0
    assert rep.discrete_accuracy >= 0.99
    assert rep.numeric_accuracy >= 0.97
    if rep.mean_conf_wrong == rep.mean_conf_wrong:  # any errors at all
        assert rep.mean_conf_wrong < rep.mean_conf_correct


def test_resolution_independent_readings():
    cal = default_layout(6, 0)
    t = random_table(np.random.default_rng(7), 6)
    ad = PokerNowStyleAdapter(cal)
    a = ad.read_frame(render_table(t, cal, (1280, 800)))
    b = ad.read_frame(render_table(t, cal, (1920, 1200)))
    for name in a.fields:
        assert a.value(name) == b.value(name), name


def r(value, conf, ts=0.0):
    return FieldReading(value, conf, "x", ts)


def test_single_bad_frame_does_not_overwrite_stable_value():
    f = FieldTracker("pot", confirm_frames=2, high_confidence=0.9)
    f.update(r(13.5, 0.8))
    assert not f.has_value                 # needs confirmation
    f.update(r(13.5, 0.8))
    assert f.stable == 13.5
    f.update(r(18.5, 0.6))                 # one misread
    assert f.stable == 13.5
    f.update(r(13.5, 0.7))
    f.update(r(22.0, 0.6))
    f.update(r(22.0, 0.6))                 # repeated -> accepted
    assert f.stable == 22.0
    f.update(r(99.0, 0.95))                # high confidence -> immediate
    assert f.stable == 99.0
    f.update(r(1.0, 0.1))                  # below min confidence: ignored
    assert f.stable == 99.0 and f.rejected == 1


def test_always_confirm_and_pinning():
    f = FieldTracker("hero_cards", confirm_frames=3, always_confirm=True)
    for _ in range(2):
        f.update(r(("As", "Ks"), 0.99))
    assert not f.has_value
    f.update(r(("As", "Ks"), 0.99))
    assert f.stable == ("As", "Ks")
    f.pin(("Ah", "Kh"))
    f.update(r(("As", "Ks"), 0.99))
    assert f.stable == ("Ah", "Kh")
    f.unpin()
    f.update(r(("As", "Ks"), 0.99))
    assert f.stable == ("Ah", "Kh")        # still needs 3 agreeing frames


def snap(**kw):
    base = dict(pot=10.0, board=(), hero_cards=("As", "Ks"), dealer=0, actor=1,
                stacks=(100.0, 100.0, 100.0), bets=(0.0, 0.0, 0.0),
                in_hand=(True, True, True), occupied=(True,) * 3,
                all_in=(False,) * 3)
    base.update(kw)
    return TableSnapshot(**base)


def kinds(events):
    return [(e.kind, e.seat) for e in events]


def test_event_inference():
    a = snap()
    b = snap(stacks=(100.0, 94.0, 100.0), bets=(0.0, 6.0, 0.0))
    assert ("bet", 1) in kinds(infer_events(a, b))
    assert ("stack_decrease", 1) in kinds(infer_events(a, b))
    c = snap(stacks=(100.0, 94.0, 100.0), in_hand=(True, True, False),
             bets=(0.0, 6.0, 0.0))
    assert ("fold", 2) in kinds(infer_events(b, c))
    d = snap(pot=16.0, board=("Qs", "Js", "4h"), stacks=(100.0, 94.0, 100.0),
             in_hand=(True, True, False))
    ev = kinds(infer_events(c, d))
    assert ("board", None) in ev and ("bets_collected", None) in ev
    e = snap(dealer=1)
    assert kinds(infer_events(d, e)) == [("new_hand", None)]
    f = snap(stacks=(100.0, 110.0, 100.0))
    assert ("stack_increase", 1) in kinds(infer_events(a, f))
