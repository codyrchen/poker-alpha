"""Phase 25: the annotated-screenshot validation command works end to end.

The images used here are SYNTHETIC stand-ins written to a temp dir purely to
exercise the command. tests/fixtures/pokernow holds the real (cropped)
PokerNow frames; see its README.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from poker_alpha.observer.pokernow import default_layout  # noqa: E402
from poker_alpha.observer.synthetic import (SyntheticSeat,  # noqa: E402
                                            SyntheticTable, render_table)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.vision


def run(fixture_dir, out):
    return subprocess.run([sys.executable, "experiments/observer_validation.py",
                           "--fixture-dir", str(fixture_dir), "--out", str(out)],
                          capture_output=True, text=True, check=True, cwd=ROOT).stdout


def test_real_fixture_dir_holds_only_annotated_real_frames(tmp_path):
    """tests/fixtures/pokernow holds real PokerNow frames only (cropped to the
    table), each with a human annotation; the command scores them and says
    that so few frames — also used for tuning — are not validation."""
    fdir = ROOT / "tests/fixtures/pokernow"
    raw = sorted(p.stem for p in (fdir / "raw").glob("*.png"))
    anns = sorted(p.stem for p in (fdir / "annotations").glob("*.json"))
    assert raw == anns == ["hu_preflop_0001"]
    out = run(fdir, tmp_path / "r.json")
    assert "1 real fixtures found" in out
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["metrics"]["screenshots"] == 1
    assert "not a statistically meaningful" in res["note"]
    assert "not independent validation" in res["note"]


def test_empty_fixture_dir_reports_nothing_measured(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "annotations").mkdir()
    out = run(tmp_path, tmp_path / "r.json")
    assert "0 real fixtures found" in out
    assert "real PokerNow accuracy: NOT MEASURED" in out
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["metrics"]["screenshots"] == 0 and "unvalidated" in res["note"]


def test_command_scores_annotated_screenshots(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "annotations").mkdir()
    cal = default_layout(6, 3)
    tables = [
        SyntheticTable(seats=[SyntheticSeat("a", 97.5, bet=2.5), SyntheticSeat(stack=None),
                              SyntheticSeat("b", 50.0, in_hand=False),
                              SyntheticSeat("hero", 100.0), SyntheticSeat("c", 88.0, bet=2.5),
                              SyntheticSeat("d", 120.0, in_hand=False)],
                       dealer=3, hero_cards=("As", "Kd"), board=("Qs", "Jh", "4c"),
                       pot=13.5, actor=3),
        SyntheticTable(seats=[SyntheticSeat("a", 40.0), SyntheticSeat("x", 60.0),
                              SyntheticSeat(stack=None), SyntheticSeat("hero", 70.0, bet=1.0),
                              SyntheticSeat("c", 80.0), SyntheticSeat(stack=None)],
                       dealer=0, hero_cards=("7c", "7d"), board=(), pot=1.5, actor=4),
    ]
    for i, t in enumerate(tables):
        render_table(t, cal).save(tmp_path / "raw" / f"shot{i}.png")
        ann = {"num_seats": 6, "hero_seat": 3, "dealer_seat": t.dealer,
               "hero_cards": list(t.hero_cards), "board": list(t.board), "pot": t.pot,
               "seats": [{"seat": s, "stack": seat.stack, "bet": seat.bet,
                          "active": seat.in_hand}
                         for s, seat in enumerate(t.seats) if seat.stack is not None]}
        if i == 1:
            ann["pot"] = 2.0          # deliberately wrong truth -> must be counted
        (tmp_path / "annotations" / f"shot{i}.json").write_text(json.dumps(ann))
    out = run(tmp_path, tmp_path / "res.json")
    m = json.loads((tmp_path / "res.json").read_text())["metrics"]
    assert m["screenshots"] == 2
    assert m["hero_cards"]["accuracy"] == 1.0 and m["board_cards"]["accuracy"] == 1.0
    assert m["stack"]["accuracy"] == 1.0 and m["stack"]["mae"] == 0.0
    assert m["bet"]["accuracy"] == 1.0
    assert m["pot"]["correct"] == 1 and m["pot"]["mae"] == pytest.approx(0.25)
    assert m["dealer"]["accuracy"] == 1.0 and m["seat_occupancy"]["accuracy"] == 1.0
    assert m["full_state"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert "pot" in out and "full_state" in out
