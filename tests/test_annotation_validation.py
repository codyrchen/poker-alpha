"""Phase 38A/B: annotation validation and the real-fixture scoring harness.

The harness self-test uses the SYNTHETIC fixture image in a temporary
directory; it proves the pipeline works and says nothing about real
PokerNow accuracy."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from poker_alpha.observer.annotations import load_fixture_dir, validate_annotation

FIX = Path(__file__).parent / "fixtures"
GOOD = {"format": "pokeralpha.screenshot_annotation/v1", "num_seats": 6, "hero_seat": 0,
        "dealer_seat": 0, "hero_cards": ["As", "Ks"], "board": ["Qs", "Js", "4h"], "pot": 13.5,
        "street": "flop",
        "seats": [{"seat": 0, "stack": 96.5, "bet": 0.0, "active": True, "name": "hero"},
                  {"seat": 2, "stack": 90.0, "bet": 4.5, "active": True},
                  {"seat": 1, "stack": 99.5, "bet": 0.0, "active": False}]}


def test_valid_annotation_passes():
    assert validate_annotation(GOOD) == []


@pytest.mark.parametrize("patch,needle", [
    ({"hero_cards": ["As", "As"]}, "duplicate"),
    ({"board": ["Qs", "Js"]}, "board must have"),
    ({"street": "turn"}, "inconsistent"),
    ({"hero_seat": 7}, "hero_seat out of range"),
    ({"hero_cards": ["Xx", "Ks"]}, "bad card"),
    ({"pot": -1}, "pot"),
    ({"hero_seat": 4}, "hero seat not listed"),
    ({"seats": [{"seat": 0, "stack": -5, "bet": 0}]}, "non-negative"),
    ({"seats": [{"seat": 0, "stack": 10, "all_in": True}]}, "all_in with non-zero stack"),
])
def test_invalid_annotations_are_caught(patch, needle):
    d = dict(GOOD, **patch)
    errs = validate_annotation(d)
    assert any(needle in e for e in errs), errs


def test_harness_self_test_on_synthetic_image(tmp_path):
    pytest.importorskip("PIL")
    sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
    from observer_validation import fixture_mode

    (tmp_path / "raw").mkdir()
    (tmp_path / "annotations").mkdir()
    shutil.copy(FIX / "table.png", tmp_path / "raw" / "synthetic_0.png")
    shutil.copy(FIX / "table_calibration.json", tmp_path / "calibration.json")
    truth = json.loads((FIX / "table.json").read_text())["table"]
    ann = {"num_seats": len(truth["seats"]), "hero_seat": 0, "dealer_seat": truth["dealer"],
           "hero_cards": truth["hero_cards"], "board": truth["board"], "pot": truth["pot"],
           "street": "flop",
           "seats": [{"seat": i, "stack": s["stack"], "bet": s["bet"], "active": s["in_hand"]}
                     for i, s in enumerate(truth["seats"])]}
    assert validate_annotation(ann) == []
    (tmp_path / "annotations" / "synthetic_0.json").write_text(json.dumps(ann))
    assert len(load_fixture_dir(tmp_path)) == 1
    res = fixture_mode(tmp_path, tmp_path / "out.json")
    m = res["metrics"]
    assert m["screenshots"] == 1
    for k in ("hero_cards", "board_cards", "street", "stack", "bet", "pot", "dealer",
              "seat_occupancy", "full_state", "hero_cards_exact_pair", "board_exact",
              "confidence_calibration"):
        assert k in m
    assert m["hero_cards"]["accuracy"] == 1.0 and m["board_exact"]["accuracy"] == 1.0
