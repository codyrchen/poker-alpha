"""Strategy explorer, what-if analysis and trainer (Study Mode engine)."""

from __future__ import annotations

import pytest

from poker_alpha.analysis.explorer import (PREFLOP_NODES, hand_classes,
                                           inspect_state, lookup_cell,
                                           matrix_stats, preflop_matrix,
                                           representative_combo)
from poker_alpha.analysis.trainer import (SPOT_LIBRARY, TrainerStats,
                                          answer_options, evaluate_answer,
                                          run_spot)
from poker_alpha.analysis.whatif import (WHATIF_VARIABLES, WhatIfError,
                                         apply_whatif, compare_whatif)
from poker_alpha.decision import DecisionConfig
from poker_alpha.pipeline import load_solver
from poker_alpha.solver_config import RELEASE_STRATEGY

CFG = DecisionConfig(equity_simulations=150, rollout_simulations=150)


@pytest.fixture(scope="module")
def solver():
    return load_solver(RELEASE_STRATEGY)


# -- explorer -----------------------------------------------------------------

def test_hand_classes_grid():
    grid = hand_classes()
    assert len(grid) == 13 and all(len(r) == 13 for r in grid)
    flat = [c for row in grid for c in row]
    assert len(set(flat)) == 169
    assert grid[0][0] == "AA" and grid[0][1] == "AKs" and grid[1][0] == "AKo"
    assert grid[12][12] == "22"
    assert representative_combo("AA") == "As Ah"
    assert representative_combo("AKs") == "As Ks"
    assert representative_combo("AKo") == "As Kh"


def test_preflop_matrix_from_artifact(solver):
    m = preflop_matrix(solver, "BTN first action")
    stats = matrix_stats(m)
    assert sum(stats.values()) == 169
    # the release gate is strict preflop; statuses must be the real ones
    assert set(stats) <= {"ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                          "UNSUPPORTED"}
    for row in m:
        for c in row:
            if c.status in ("ACCEPTED", "LOW CONFIDENCE"):
                assert c.mix and abs(sum(p for _, p in c.mix) - 1.0) < 1e-6
                assert c.dominant == c.mix[0][0]
            else:
                assert c.dominant is None      # no numbers without trust
                assert c.reasons or c.notes


def test_all_nodes_queryable(solver):
    for node in PREFLOP_NODES:
        cell = lookup_cell(solver, "72o", node)
        assert cell.status in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                               "UNSUPPORTED")


def test_inspect_state_validates(solver):
    from poker_alpha.ui.play import DEMO_SPOTS

    good = DEMO_SPOTS["River, checked to hero — solver accepted"]
    r = inspect_state(solver, good)
    assert r.status == "ACCEPTED" and r.mix
    dup = {**good, "hero_cards": "Qs Qs"}
    assert inspect_state(solver, dup).status == "UNSUPPORTED"
    board_dup = {**good, "hero_cards": "Qs 2c"}
    assert inspect_state(solver, board_dup).status == "UNSUPPORTED"


# -- what-if ------------------------------------------------------------------

BASE = {
    "num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
    "big_blind": 1.0, "hero_cards": "9h 9c", "board": "Qs 7d 2c Th 3h",
    "pot": 7.5, "actor": 0,
    "seats": [{"stack": 97.5, "committed": 2.5, "bet": 0.0},
              {"stack": 95.0, "committed": 5.0, "bet": 2.5}],
    "actions": [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
                {"street": 0, "seat": 1, "kind": "call", "amount": 1.5},
                {"street": 1, "seat": 1, "kind": "check"},
                {"street": 1, "seat": 0, "kind": "check"},
                {"street": 2, "seat": 1, "kind": "check"},
                {"street": 2, "seat": 0, "kind": "check"},
                {"street": 3, "seat": 1, "kind": "bet", "amount": 2.5}],
}


def test_whatif_rejects_impossible_states():
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "hero_cards", "Qs Qd")      # duplicate with board
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "hero_cards", "Xx Yy")      # not cards
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "pot", -4)                  # negative
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "pot", 1.0)                 # below chips bet
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "villain_bet", -1)
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "villain_bet", 2000)        # beyond stack
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "effective_stack", 0)
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "board_card", "5d")         # river already dealt
    with pytest.raises(WhatIfError):
        apply_whatif(BASE, "unknown_var", 1)


def test_whatif_changes_one_variable():
    v = apply_whatif(BASE, "villain_bet", 5.0)
    assert v["seats"][1]["bet"] == 5.0
    assert v["pot"] == pytest.approx(10.0)
    assert v["seats"][1]["stack"] == pytest.approx(92.5)
    assert v["actions"][-1]["amount"] == 5.0
    assert BASE["seats"][1]["bet"] == 2.5              # base untouched
    h = apply_whatif(BASE, "hero_cards", "Ah Ad")
    assert h["hero_cards"] == "Ah Ad" and h["pot"] == BASE["pot"]


def test_whatif_board_card_street_transition():
    turn_state = {**BASE, "board": "Qs 7d 2c", "pot": 5.0,
                  "seats": [{"stack": 97.5, "committed": 2.5, "bet": 0.0},
                            {"stack": 97.5, "committed": 2.5, "bet": 0.0}],
                  "actions": BASE["actions"][:4]}
    v = apply_whatif(turn_state, "board_card", "Kh")
    assert v["board"].endswith("Kh") and len(v["board"].split()) == 4
    with pytest.raises(WhatIfError):
        apply_whatif(turn_state, "board_card", "Qs")   # duplicate
    betting = {**turn_state,
               "seats": [{"stack": 97.5, "committed": 2.5, "bet": 0.0},
                         {"stack": 95.0, "committed": 5.0, "bet": 2.5}],
               "pot": 7.5}
    with pytest.raises(WhatIfError):
        apply_whatif(betting, "board_card", "Kh")      # bets outstanding


def test_compare_whatif_runs_engine(solver):
    variants, rejected = compare_whatif(BASE, "villain_bet", [3.75, -1],
                                        config=CFG, solver=solver)
    assert len(variants) == 2 and variants[0].value == "base"
    assert "-1" in rejected
    for v in variants:
        assert v.solver_state in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                                  "OFF")
        assert v.strategy_text()
    assert set(WHATIF_VARIABLES) == {"villain_bet", "pot", "effective_stack",
                                     "hero_cards", "board_card"}


# -- trainer ------------------------------------------------------------------

def test_spot_library_covers_required_states(solver):
    """Library labels must match what the engine ACTUALLY does."""
    from poker_alpha.analysis.alignment import solver_state

    covers = set()
    for spot in SPOT_LIBRARY:
        rep = run_spot(spot, config=CFG, solver=solver)
        st = solver_state(rep)
        if "solver_accepted" in spot.covers:
            assert st == "ACCEPTED", spot.name
        if "solver_low_confidence" in spot.covers:
            assert st == "LOW CONFIDENCE", spot.name
        if "gate_rejected" in spot.covers:
            assert st == "REJECTED", spot.name
        if "rollout_fallback" in spot.covers:
            assert "rollout" in rep.method.lower(), spot.name
        if "off_tree" in spot.covers:
            info = rep.details["solver"]
            assert "OFF_TREE_TRANSLATION" in info.get("reasons", ()), spot.name
        covers |= set(spot.covers)
    assert {"preflop", "flop", "turn", "river", "solver_accepted",
            "solver_low_confidence", "rollout_fallback",
            "off_tree"} <= covers


def test_trainer_grading_language(solver):
    accepted = next(s for s in SPOT_LIBRARY if "solver_accepted" in s.covers)
    rep = run_spot(accepted, config=CFG, solver=solver)
    opts = answer_options(rep)
    assert rep.recommended in opts
    fb = evaluate_answer(rep, rep.recommended)
    assert fb.grade in ("COMMON", "MIXED", "RARE")
    assert "solver mix" in fb.headline or "Mixed option" in fb.headline
    assert "wrong" not in fb.headline.lower()          # never binary
    low = next(s for s in SPOT_LIBRARY if "solver_low_confidence" in s.covers)
    rep2 = run_spot(low, config=CFG, solver=solver)
    fb2 = evaluate_answer(rep2, answer_options(rep2)[0])
    assert fb2.grade == "UNCERTAIN"
    assert fb2.headline == "PokerAlpha uncertain — review only"
    roll = next(s for s in SPOT_LIBRARY if "rollout_fallback" in s.covers)
    rep3 = run_spot(roll, config=CFG, solver=solver)
    fb3 = evaluate_answer(rep3, answer_options(rep3)[-1])
    assert fb3.grade in ("REGRET", "UNSCORED")
    if fb3.grade == "REGRET":
        assert fb3.estimated_regret_bb is not None
        assert "money lost" not in fb3.headline


def test_trainer_stats_local(solver):
    stats = TrainerStats()
    spot = SPOT_LIBRARY[0]
    rep = run_spot(spot, config=CFG, solver=solver)
    fb = evaluate_answer(rep, answer_options(rep)[0])
    stats.record(spot, fb)
    s = stats.summary()
    assert s["spots_attempted"] == 1
    assert s["by_street"].get("river") == 1
    assert set(s) == {"spots_attempted", "alignment", "by_street",
                      "eligible_regret_n", "avg_estimated_regret_bb",
                      "uncertain_or_unscored"}
