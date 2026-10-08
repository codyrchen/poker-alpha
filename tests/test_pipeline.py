"""Phase 31: one normalized path, decision-source priority and rejection
reasons, uncertainty kept separate."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.decision import DecisionConfig
from poker_alpha.decision.strategy import REJECTION_CODES, LookupMiss
from poker_alpha.history import load_hands
from poker_alpha.pipeline import (analyze, load_solver, observe_hand_history,
                                  observe_manual, observe_screenshot,
                                  observe_simulation)
from poker_alpha.solver_config import PRIMARY_CONFIG
from poker_alpha.solvers.serialize import save_checkpoint
from poker_alpha.solvers.strategy_artifact import export_solver

FIX = Path(__file__).parent / "fixtures"
FAST = DecisionConfig(equity_simulations=300)
UNCERTAINTY_KEYS = {"observation", "range_estimation", "sampling", "abstraction",
                    "response_model"}


def hu(hero_cards="As Ks", stack=100.0):
    return {"num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
            "big_blind": 1.0, "hero_cards": hero_cards, "board": "", "actor": 0,
            "seats": [{"stack": stack - 0.5, "bet": 0.5, "committed": 0.5},
                      {"stack": stack - 1.0, "bet": 1.0, "committed": 1.0}],
            "pot": 1.5, "actions": []}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    s = PRIMARY_CONFIG.build_solver(seed=0)
    s.train(60)
    d = tmp_path_factory.mktemp("art")
    return (export_solver(s, d / "a.npz", meta={"seed": 0}),
            save_checkpoint(s, d / "c.npz"), s)


def visited_hand(solver):
    """A BTN-unopened hand whose abstract infoset was visited in training."""
    from poker_alpha.solvers.holdem_analysis import spot_state
    from poker_alpha.validation.canonical_matrix import PREFLOP_HANDS

    game = solver.game
    for hole in PREFLOP_HANDS.values():
        key = game.infoset_key(spot_state(game, "BTN", hole))
        n = solver.infosets.get(key)
        if n is not None and n.strategy_sum.sum() > 0:
            return " ".join(hole)
    raise AssertionError("no visited preflop spot")


def _cascade(rep):
    return {c["source"]: c for c in rep.details["source_cascade"]}


def test_every_source_reaches_the_same_report_type():
    from poker_alpha.holdem_demo import CHIP, build_hand

    obs = [observe_manual(hu()),
           observe_hand_history(load_hands(FIX / "hands" / "sample.json")[0])]
    st = build_hand()
    obs.append(observe_simulation(st, st.actor, chip_unit=CHIP))
    pytest.importorskip("PIL")
    obs.append(observe_screenshot(FIX / "table.png"))
    for o in obs:
        rep = analyze(o, FAST)
        assert rep.details["input_source"] == o.source
        assert set(rep.uncertainty) == UNCERTAINTY_KEYS
        assert rep.details["source_cascade"][0]["source"] == "solver"
    assert obs[-1].observer_confidence is not None
    assert "screen recognition" in analyze(obs[-1], FAST).uncertainty["observation"]


def test_solver_used_from_artifact_and_checkpoint(artifact):
    art, ckpt, solver = artifact
    for path in (art, ckpt):
        prov = load_solver(path, min_visits=0.0)
        assert not isinstance(prov, LookupMiss)
        rep = analyze(observe_manual(hu(visited_hand(solver))), FAST, solver=prov)
        assert rep.method == "solver", rep.details["source_cascade"]
        c = _cascade(rep)
        assert c["solver"]["status"] == "used"
        assert c["heuristic fallback"]["status"] == "not needed"
        assert "IMPERFECT-RECALL" in rep.mix_meaning
        assert "visits" in rep.uncertainty["abstraction"]


def test_rejection_reasons_are_coded(artifact, tmp_path):
    art, ckpt, solver = artifact
    other = dataclasses.replace(PRIMARY_CONFIG, raise_cap=2)
    for path in (art, ckpt):
        miss = load_solver(path, config=other)
        assert isinstance(miss, LookupMiss) and miss.code == "config_mismatch"
        rep = analyze(observe_manual(hu()), FAST, solver=miss)
        assert _cascade(rep)["solver"]["code"] == "config_mismatch"
        assert rep.method == "heuristic fallback"
    (tmp_path / "junk.npz").write_bytes(b"junk")
    assert load_solver(tmp_path / "junk.npz").code == "incompatible_checkpoint"
    assert load_solver(tmp_path / "missing.npz").code == "incompatible_checkpoint"

    prov = load_solver(art, min_visits=1e9)
    rep = analyze(observe_manual(hu(visited_hand(solver))), FAST, solver=prov)
    assert _cascade(rep)["solver"]["code"] == "insufficient_visits"
    rep = analyze(observe_manual(hu(stack=40)), FAST, solver=prov)
    assert _cascade(rep)["solver"]["code"] == "out_of_abstraction"
    empty = load_solver(art, min_visits=0.0)
    empty.strategy = {}
    rep = analyze(observe_manual(hu()), FAST, solver=empty)
    assert _cascade(rep)["solver"]["code"] == "unvisited"
    assert set(REJECTION_CODES) == {"config_mismatch", "incompatible_checkpoint",
                                    "out_of_abstraction", "unvisited", "insufficient_visits"}


def test_priority_rollout_before_heuristic():
    rep = analyze(observe_manual(hu()), dataclasses.replace(FAST, rollout_simulations=150))
    c = _cascade(rep)
    assert rep.method == "Monte Carlo rollout"
    assert c["solver"]["status"] == "not configured"
    assert c["Monte Carlo rollout"]["status"] == "used"
    assert "rollout EV standard error" in rep.uncertainty["sampling"]
    rep = analyze(observe_manual(hu()), FAST)
    assert rep.method == "heuristic fallback"
    assert _cascade(rep)["Monte Carlo rollout"]["status"] == "not run"


def test_report_format_shows_cascade_and_uncertainty():
    text = analyze(observe_manual(hu()), FAST).format()
    assert "Decision sources" in text and "Uncertainty by source" in text
    assert "GTO" not in text


def test_observation_does_not_mutate_input():
    d = hu()
    before = repr(d)
    observe_manual(d)
    assert repr(d) == before
    assert np.isfinite(analyze(observe_manual(d), FAST).pot_bb)


def test_solver_rollout_conflict_is_flagged():
    from poker_alpha.decision import SolverStrategyProvider
    from poker_alpha.games import HoldemGame

    # A deliberately bad "strategy": fold AKs preflop when folding is legal.
    game = HoldemGame()
    prov = SolverStrategyProvider(game, {"0|50,51||": {"f": 1.0}}, {"0|50,51||": 500})
    rep = analyze(observe_manual(hu()), dataclasses.replace(FAST, rollout_simulations=200),
                  solver=prov)
    assert rep.method == "solver" and rep.recommended == "fold"
    assert rep.confidence == "low"
    assert any("solver and rollouts disagree" in w for w in rep.warnings)
    assert _cascade(rep)["consistency check"]["status"] == "conflict"
