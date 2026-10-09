"""Phase 37: solver-use gate, confidence table, provenance in reports and UI."""

import math
from pathlib import Path

import pytest

from poker_alpha.decision import DecisionConfig
from poker_alpha.decision.solver_gate import (ACCEPT, LOW, REJECT, ConfidenceTable,
                                              GateThresholds, KeyStats, gate)
from poker_alpha.pipeline import analyze, load_solver, observe_manual
from poker_alpha.solver_config import PRIMARY_CONFIG
from poker_alpha.solvers.holdem_analysis import spot_state
from poker_alpha.ui.view import solver_signal_rows, solver_status, source_rows, uncertainty_rows

ROOT = Path(__file__).parents[1]
TH = GateThresholds(reject_visits_below=20, low_visits_below=100, reject_seed_disagreement=1.0,
                    low_seed_disagreement=0.5, reject_movement=0.8, low_movement=0.3)


def test_gate_decisions():
    assert gate(None, TH).status == REJECT and gate(None, TH).reasons == ("UNSEEN_STATE",)
    assert gate(KeyStats(500, 0.05, 0.1, 0.1), TH).status == ACCEPT
    d = gate(KeyStats(50, 0.05, 0.1, 0.1), TH)
    assert d.status == LOW and d.reasons == ("LOW_VISIT_COUNT",)
    d = gate(KeyStats(5, 0.05, 0.1, 0.1), TH)
    assert d.status == REJECT and "LOW_VISIT_COUNT" in d.reasons
    d = gate(KeyStats(500, 0.05, 1.3, 0.1), TH)
    assert d.status == REJECT and d.reasons[0] == "HIGH_SEED_DISAGREEMENT"
    d = gate(KeyStats(500, 0.4, 0.1, 0.1), TH)
    assert d.status == LOW and d.reasons == ("UNSTABLE_ACROSS_CHECKPOINTS",)
    d = gate(KeyStats(500, 0.05, 0.1, 0.9), TH)
    assert d.status == LOW and d.reasons == ("HIGH_COLLISION_DISPERSION",)
    d = gate(KeyStats(500), TH)
    assert d.status == LOW and d.reasons == ("NO_STABILITY_DATA",)
    assert gate(KeyStats(500, 0.0, 0.0, 0.0), TH, pathological=True).reasons == ("KNOWN_PATHOLOGICAL_BUCKET",)


def test_confidence_table_roundtrip(tmp_path):
    t = ConfidenceTable("sig", {"a": KeyStats(10, 0.1, math.nan, 0.2), "b": KeyStats(3)}, ("b",), {"x": 1})
    p = t.save(tmp_path / "c.npz")
    u = ConfidenceTable.load(p, "sig")
    assert u.get("a").visits == pytest.approx(10) and math.isnan(u.get("a").seed_disagreement)
    assert "b" in u.pathological and u.meta == {"x": 1}
    with pytest.raises(ValueError):
        ConfidenceTable.load(p, "other")


def _hu(cards):
    return {"num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
            "hero_cards": cards, "board": "", "actor": 0,
            "seats": [{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                      {"stack": 99.0, "bet": 1.0, "committed": 1.0}], "pot": 1.5, "actions": []}


@pytest.fixture(scope="module")
def provider():
    p = load_solver(ROOT / "results" / "strategy" / "holdem_v1_seed0.npz")
    assert not hasattr(p, "code"), p
    assert p.confidence is not None, "committed confidence table must load"
    return p


def test_committed_strategy_is_gated(provider):
    """A noisy preflop spot is rejected and the cascade falls back."""
    game = provider.game
    by_status = {}
    for cards in ("Ah Qd", "7c 2h", "As Ah", "Kd Kh", "9c 8c", "Ts 4d", "Jh Jd", "5s 5h"):
        key = game.infoset_key(spot_state(game, "BTN", cards.split(), (), ("",),
                                          villain_hole=("2c", "3d")))
        d = gate(provider.confidence.get(key), provider.thresholds,
                 key in provider.confidence.pathological)
        by_status.setdefault(d.status, []).append(cards)
        rep = analyze(observe_manual(_hu(cards)), DecisionConfig(equity_simulations=200), solver=provider)
        info = rep.details["solver"]
        if d.status == REJECT:
            assert info["used"] is False and rep.method == "heuristic fallback"
            assert "Solver strategy not used" in solver_status(rep)
            assert set(d.reasons) <= set(info["reasons"])
        else:
            assert info["used"] is True and info["confidence"] == d.status
            if d.status == LOW:
                assert rep.confidence == "low"
    assert REJECT in by_status, by_status


def test_report_and_ui_expose_provenance(provider):
    rep = analyze(observe_manual(_hu("Ah Qd")), DecisionConfig(equity_simulations=200), solver=provider)
    rows = source_rows(rep)
    assert rows[0]["source"] == "solver"
    assert {r["source of uncertainty"] for r in uncertainty_rows(rep)} >= {"abstraction", "sampling"}
    if rep.details["solver"]["used"]:
        assert any(r["signal"] == "seed_disagreement" for r in solver_signal_rows(rep))


def test_calibrated_thresholds_load():
    th = GateThresholds.calibrated()
    assert th.reject_visits_below <= th.low_visits_below
    assert th.low_seed_disagreement <= th.reject_seed_disagreement


def test_illegal_abstract_sizes_never_recommended(provider):
    """v1 offers a 1.66 BB open ('raise33'); the lookup drops it."""
    rep = analyze(observe_manual(_hu("7c 2h")), DecisionConfig(equity_simulations=200), solver=provider)
    labels = [c.label for c in rep.candidates]
    if rep.details["solver"]["used"]:
        assert "raise_33" not in labels
        assert sum(c.probability for c in rep.candidates) == pytest.approx(1.0)


def test_illegal_size_mass_is_removed_deterministically():
    from poker_alpha.decision import SolverStrategyProvider

    game = PRIMARY_CONFIG.build_game()
    key = game.infoset_key(spot_state(game, "BTN", ("Ah", "Qd"), (), ("",), villain_hole=("2c", "3d")))
    prov = SolverStrategyProvider(game, {key: {"f": 0.0, "c": 0.2, "b33": 0.5, "b75": 0.3, "b150": 0.0,
                                              "a": 0.0}}, {key: 1000.0})
    rep = analyze(observe_manual(_hu("Ah Qd")), DecisionConfig(equity_simulations=200), solver=prov)
    assert rep.method == "solver"
    mix = {c.label: c.probability for c in rep.candidates}
    assert "raise_33" not in mix
    assert mix["raise_75"] == pytest.approx(0.6) and mix["call"] == pytest.approx(0.4)
    assert rep.details["solver"]["illegal_size_mass_removed"] == pytest.approx(0.5)


def test_config_inferred_from_artifact_signature(tmp_path):
    from poker_alpha.solver_config import V2_CONFIG
    from poker_alpha.solvers.strategy_artifact import export_solver

    s = V2_CONFIG.build_solver(seed=1)
    s.train(5)
    path = export_solver(s, tmp_path / "v2.npz", meta={"seed": 1})
    prov = load_solver(path)
    assert not hasattr(prov, "code") and prov.game.signature() == V2_CONFIG.build_game().signature()
    miss = load_solver(path, config=PRIMARY_CONFIG)
    assert miss.code == "CONFIG_MISMATCH"


def test_release_strategy_is_v2_gated_and_legal():
    from poker_alpha.solver_config import RELEASE_CONFIG, RELEASE_STRATEGY
    from poker_alpha.solvers.strategy_artifact import load_artifact

    path = ROOT / RELEASE_STRATEGY
    art = load_artifact(path, RELEASE_CONFIG.build_game())
    assert art.config_signature == RELEASE_CONFIG.signature() and art.meta["iterations"] >= 100000
    prov = load_solver(path)
    assert prov.confidence is not None and prov.confidence.config_signature == RELEASE_CONFIG.signature()
    for cards in ("Ah Qd", "As Ah", "7c 2h", "Kd Kh"):
        rep = analyze(observe_manual(_hu(cards)), DecisionConfig(equity_simulations=200), solver=prov)
        info = rep.details["solver"]
        assert info["confidence"] in ("SOLVER_ACCEPT", "SOLVER_LOW_CONFIDENCE", "rejected")
        if info["used"]:
            for c in rep.candidates:
                if c.kind in ("raise", "bet"):
                    assert c.amount_to >= 2.0 - 1e-9      # legal NLHE open: at least a min-raise
            assert "illegal_size_mass_removed" not in info


def test_gate_v2_downgrade_only_lowers():
    from poker_alpha.decision.solver_gate import REASONS, downgrade

    assert {"OFF_TREE_TRANSLATION", "ILLEGAL_SIZE_MASS"} <= set(REASONS)
    acc = {"status": ACCEPT, "reasons": [], "signals": {}}
    d = downgrade(acc, "OFF_TREE_TRANSLATION")
    assert d["status"] == LOW and d["reasons"] == ["OFF_TREE_TRANSLATION"]
    assert acc["status"] == ACCEPT and acc["reasons"] == []          # input untouched
    rej = {"status": REJECT, "reasons": ["HIGH_SEED_DISAGREEMENT"], "signals": {}}
    assert downgrade(rej, "OFF_TREE_TRANSLATION")["status"] == REJECT
    low = downgrade(downgrade(acc, "ILLEGAL_SIZE_MASS"), "ILLEGAL_SIZE_MASS")
    assert low["reasons"] == ["ILLEGAL_SIZE_MASS"]


def test_large_illegal_size_mass_lowers_confidence():
    from poker_alpha.decision import SolverStrategyProvider

    game = PRIMARY_CONFIG.build_game()
    key = game.infoset_key(spot_state(game, "BTN", ("Ah", "Qd"), (), ("",), villain_hole=("2c", "3d")))
    prov = SolverStrategyProvider(game, {key: {"f": 0.0, "c": 0.2, "b33": 0.5, "b75": 0.3, "b150": 0.0,
                                              "a": 0.0}}, {key: 1000.0})
    rep = analyze(observe_manual(_hu("Ah Qd")), DecisionConfig(equity_simulations=200), solver=prov)
    assert rep.method == "solver" and rep.confidence == "low"
    assert rep.details["solver"]["confidence"] == LOW
    assert "ILLEGAL_SIZE_MASS" in rep.details["solver"]["reasons"]
    small = SolverStrategyProvider(game, {key: {"f": 0.0, "c": 0.5, "b33": 0.1, "b75": 0.4,
                                                "b150": 0.0, "a": 0.0}}, {key: 1000.0})
    rep = analyze(observe_manual(_hu("Ah Qd")), DecisionConfig(equity_simulations=200), solver=small)
    assert "ILLEGAL_SIZE_MASS" not in rep.details["solver"]["reasons"]
