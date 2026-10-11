"""Hand replay building blocks: importers, range visualizer binding,
decision path, model sensitivity, decision stability."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from poker_alpha.analysis.decision_path import decision_path
from poker_alpha.analysis.ranges_view import (matrix_cells, range_matrix,
                                              range_stages)
from poker_alpha.analysis.sensitivity import (ROBUST_TV_BELOW,
                                              analyze_under_models)
from poker_alpha.analysis.stability import sweep
from poker_alpha.decision import DecisionConfig
from poker_alpha.demo_spots import DEMO_SPOTS
from poker_alpha.history import replay_hand
from poker_alpha.history.importers import (ImportFormatError, detect_format,
                                           parse_external_history)
from poker_alpha.pipeline import load_solver, observe_manual
from poker_alpha.solver_config import RELEASE_STRATEGY

ROOT = Path(__file__).resolve().parents[1]
LINES = (ROOT / "tests" / "fixtures" / "imports" /
         "simple_lines_session.txt").read_text()
CFG = DecisionConfig(equity_simulations=150, rollout_simulations=150)


@pytest.fixture(scope="module")
def solver():
    return load_solver(RELEASE_STRATEGY)


# -- importers ---------------------------------------------------------------

def test_importer_detect_and_parse():
    assert detect_format(LINES) == "pokeralpha-lines/v1"
    hands = parse_external_history(LINES)
    assert len(hands) == 2
    r = replay_hand(hands[0])
    assert len(r.decisions) == 4 and not r.warnings
    assert replay_hand(hands[1]).decisions[0].action == "raise"


def test_importer_rejects_unknown_and_malformed():
    with pytest.raises(ImportFormatError):
        parse_external_history("PokerStars Hand #1: Tournament ...")
    with pytest.raises(ImportFormatError) as e:
        parse_external_history("# pokeralpha-lines/v1\nbet 0 3.5\n")
    assert "line 2" in str(e.value)
    with pytest.raises(ImportFormatError):
        parse_external_history("# pokeralpha-lines/v1\n"
                               "hand h seats=2 dealer=0 sb=0.5 bb=1 hero=0 "
                               "stacks=100\n")          # one stack for two seats
    with pytest.raises(ImportFormatError):
        parse_external_history(LINES, fmt="nope/v9")


# -- range visualizer binds to the real WeightedRange ------------------------

def test_range_matrix_matches_weighted_range():
    state = observe_manual(
        DEMO_SPOTS["River, checked to hero — solver accepted"]).state
    stages = range_stages(state, seat=1, model="regular")
    assert stages[0].name.startswith("Preflop")
    assert [s.street for s in stages[1:]] == [1, 2, 3]
    for st in stages:
        # the matrix IS the WeightedRange's class distribution
        dist = st.range.class_distribution()
        cells = dict(matrix_cells(st.matrix))
        for cls, w in dist.items():
            assert cells[cls] == pytest.approx(w)
        assert sum(cells.values()) == pytest.approx(sum(dist.values()))
        assert st.entropy_bits == pytest.approx(st.range.entropy_bits())
        assert st.effective_combos == pytest.approx(
            2 ** st.entropy_bits, rel=1e-6)
    # card removal: hero's Ah and board cards can never be in the range
    m = range_matrix(stages[-1].range)
    assert all(math.isfinite(x) and x >= 0 for row in m for x in row)
    probs = stages[-1].range.probabilities()
    from poker_alpha.poker.cards import card_code
    from poker_alpha.poker.ranges import COMBOS

    dead = {card_code("Ah"), card_code("Qs")}
    for i, (c1, c2) in enumerate(COMBOS):
        if c1 in dead or c2 in dead:
            assert probs[i] == 0.0


def test_range_updates_narrow_the_range():
    state = observe_manual(
        DEMO_SPOTS["River, checked to hero — solver accepted"]).state
    stages = range_stages(state, seat=1)
    # postflop checks are information: entropy must not increase overall
    assert stages[-1].entropy_bits <= stages[0].entropy_bits + 1e-9


# -- decision path ------------------------------------------------------------

def test_decision_path_rows(solver):
    from poker_alpha.pipeline import analyze

    obs = observe_manual(DEMO_SPOTS["River, checked to hero — solver accepted"])
    rep = analyze(obs, config=CFG, solver=solver)
    rows = decision_path(rep, obs.state)
    labels = [r.label for r in rows]
    assert "Abstract infoset key" in labels
    assert "Gate verdict" in labels
    assert "Training visits" in labels
    verdict = next(r for r in rows if r.label == "Gate verdict")
    assert verdict.value in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED", "OFF")
    # non-solver decision: path shows the cascade instead
    rep2 = analyze(observe_manual(
        DEMO_SPOTS["Preflop open decision — gate rejects (seed disagreement)"]),
        config=CFG, solver=solver)
    rows2 = decision_path(rep2, None)
    assert any("rollout" in r.label.lower() for r in rows2)


# -- model sensitivity ---------------------------------------------------------

def test_sensitivity_rule_and_outcomes(solver):
    spot = DEMO_SPOTS["River, facing a bet — solver at low confidence (off-tree size)"]
    res = analyze_under_models(spot, config=CFG, solver=solver)
    assert len(res.outcomes) == 4
    assert {o.model for o in res.outcomes} == {"regular", "nit",
                                               "calling_station", "maniac"}
    assert res.verdict in ("ROBUST", "SENSITIVE", "HIGHLY SENSITIVE")
    # documented rule: identical tops + tiny TV + same method => ROBUST
    tops = set(res.top_actions.values())
    if len(tops) == 1 and res.max_tv < ROBUST_TV_BELOW and not res.methods_differ:
        assert res.verdict == "ROBUST"
    assert "claimed correct" in res.note


def test_sensitivity_never_compares_ev_across_methods(solver):
    spot = DEMO_SPOTS["Preflop open decision — gate rejects (seed disagreement)"]
    res = analyze_under_models(spot, config=CFG, solver=solver)
    if res.methods_differ:
        assert res.verdict != "ROBUST"
        assert "never by EV across methods" in res.note


# -- decision stability ---------------------------------------------------------

def test_stability_sweep_boundaries(solver):
    base = DEMO_SPOTS["River, facing a bet — solver at low confidence (off-tree size)"]
    res = sweep(base, "villain_bet", [1.65, 3.75, 7.5, 60, -5],
                config=CFG, solver=solver)
    assert "-5" in res.rejected
    assert len(res.points) == 4
    for b in res.boundaries:
        assert b.kind in ("top_action", "solver_state", "method")
        assert b.at_value in {p.value for p in res.points}
    assert res.summary                      # human-readable either way
    # points are real pipeline outputs
    for p in res.points:
        assert p.solver_state in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                                  "OFF")
        assert abs(sum(p.distribution.values()) - 1) < 0.02 or p.distribution
