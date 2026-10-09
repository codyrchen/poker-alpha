"""Decision pipeline fault injection (Phase 68).

Every bad input or failing component must either refuse with a coded reason
or use a documented, reported fallback: never a silent recommendation.
"""

import numpy as np
import pytest

from poker_alpha.decision import DecisionConfig, recommend_action
from poker_alpha.decision import recommend as recmod
from poker_alpha.decision.strategy import LookupMiss
from poker_alpha.pipeline import analyze, load_solver, observe_manual
from poker_alpha.poker.ranges import WeightedRange

FAST = DecisionConfig(equity_simulations=200, rollout_simulations=120, seed=3)


def hu(**kw):
    d = {"num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
         "big_blind": 1.0, "hero_cards": "As Ks", "board": "", "actor": 0,
         "seats": [{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                   {"stack": 99.0, "bet": 1.0, "committed": 1.0}],
         "pot": 1.5, "actions": []}
    d.update(kw)
    return d


def state(**kw):
    return observe_manual(hu(**kw)).state


def cascade(r):
    return r.details["source_cascade"]


def refused(r, code):
    assert r.recommended is None and r.method == "none"
    assert r.recommended_mix == {}
    assert code in r.details["refusal_codes"], r.details["refusal_codes"]


def test_baseline_recommends():
    r = recommend_action(state(), config=FAST)
    assert r.recommended is not None
    assert r.details["assumptions"] == []


def test_invalid_card_rejected_at_input():
    with pytest.raises(ValueError, match="Xx"):
        state(hero_cards="Xx Ks")


@pytest.mark.parametrize("key", ["dealer", "hero_seat"])
def test_unknown_dealer_or_hero_seat_is_a_clear_error(key):
    with pytest.raises(ValueError, match="unknown"):
        state(**{key: None})
    with pytest.raises(ValueError, match="seat number"):
        state(**{key: "button"})


def test_duplicate_cards_refused():
    refused(recommend_action(state(board="As 7d 2c"), config=FAST), "duplicate_cards")


def test_impossible_pot_refused():
    refused(recommend_action(state(pot=0.2), config=FAST), "pot_below_bets")


def test_missing_hero_stack_refused():
    r = recommend_action(state(seats=[{"stack": None, "bet": 0.5, "committed": 0.5},
                                      {"stack": 99.0, "bet": 1.0, "committed": 1.0}]),
                         config=FAST)
    refused(r, "HERO_STACK_UNKNOWN")
    assert any("hero stack unknown" in w for w in r.warnings)


def test_missing_opponent_stack_is_a_reported_assumption():
    r = recommend_action(state(seats=[{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                                      {"stack": None, "bet": 1.0, "committed": 1.0}]),
                         config=FAST)
    assert "OPPONENT_STACK_UNKNOWN" in r.details["assumptions"]
    assert r.confidence == "low"
    assert any("assumed to cover the hero" in w for w in r.warnings)


def test_unknown_actor_is_a_reported_assumption():
    r = recommend_action(state(actor=None), config=FAST)
    assert "ACTOR_UNKNOWN" in r.details["assumptions"]
    assert r.confidence == "low"


def test_not_hero_turn_refused():
    refused(recommend_action(state(actor=1), config=FAST), "NOT_HERO_TURN")


def test_hero_folded_refused():
    st = state(actor=None, seats=[{"stack": 99.5, "bet": 0.5, "committed": 0.5, "folded": True},
                                  {"stack": 99.0, "bet": 1.0, "committed": 1.0}])
    refused(recommend_action(st, config=FAST), "HERO_FOLDED")


def test_unknown_hero_cards_refused():
    refused(recommend_action(state(hero_cards=""), config=FAST), "HERO_CARDS_UNKNOWN")


def test_partial_range_is_used_as_given():
    one = WeightedRange.from_combos([(("Ah", "Kh"), 1.0)])
    r = recommend_action(state(), opponent_ranges={1: one}, config=FAST)
    assert r.recommended is not None
    assert r.opponent_ranges[0].live_combos == 1


def test_empty_range_gives_no_recommendation_when_facing_a_bet():
    empty = WeightedRange.from_combos([(("As", "Ks"), 1.0)])  # removed by hero cards
    r = recommend_action(state(), opponent_ranges={1: empty}, config=FAST)
    assert r.recommended is None and r.method == "none"
    assert any(c.get("code") == "RANGE_EMPTY" for c in cascade(r))
    assert any(c.get("code") == "NO_EQUITY" for c in cascade(r))
    assert r.opponent_ranges[0].live_combos == 0


def test_equity_failure_reported(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("sampler exploded")
    monkeypatch.setattr(recmod, "multiway_equity", boom)
    r = recommend_action(state(), config=FAST)
    assert r.hero_equity is None
    assert any(c.get("code") == "EQUITY_ERROR" for c in cascade(r))
    assert any("equity estimate failed" in w for w in r.warnings)


def test_equity_and_rollout_failure_facing_bet_refuses(monkeypatch):
    import poker_alpha.decision.rollout as ro

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(recmod, "multiway_equity", boom)
    monkeypatch.setattr(ro, "rollout_action_evs", boom)
    r = recommend_action(state(), config=FAST)
    assert r.recommended is None and r.method == "none"
    codes = [c.get("code") for c in cascade(r)]
    assert "EQUITY_ERROR" in codes and "ROLLOUT_ERROR" in codes and "NO_EQUITY" in codes


def test_rollout_failure_falls_back_to_heuristic(monkeypatch):
    import poker_alpha.decision.rollout as ro

    def boom(*a, **k):
        raise ro.RolloutError("could not sample collision-free opponent hands")
    monkeypatch.setattr(ro, "rollout_action_evs", boom)
    r = recommend_action(state(), config=FAST)
    assert r.method == "heuristic fallback"
    assert r.recommended in ("call", "fold")
    assert r.confidence == "low"
    assert any(c.get("code") == "ROLLOUT_ERROR" for c in cascade(r))


class _BrokenSolver:
    description = "broken"

    def lookup(self, obs):
        raise KeyError("infoset table corrupted")


class _RejectingSolver:
    description = "rejecting"

    def lookup(self, obs):
        return LookupMiss("spot outside the abstraction", "OUTSIDE_ABSTRACTION")


def test_solver_exception_becomes_coded_rejection():
    r = recommend_action(state(), config=DecisionConfig(
        equity_simulations=200, rollout_simulations=0, solver=_BrokenSolver()))
    sol = [c for c in cascade(r) if c["source"] == "solver"]
    assert sol and sol[0]["status"] == "rejected" and sol[0]["code"] == "SOLVER_ERROR"
    assert r.details["solver"]["used"] is False
    assert r.method == "heuristic fallback"


def test_solver_rejection_reported():
    r = recommend_action(state(), config=DecisionConfig(
        equity_simulations=200, rollout_simulations=0, solver=_RejectingSolver()))
    sol = [c for c in cascade(r) if c["source"] == "solver"][0]
    assert sol["code"] == "OUTSIDE_ABSTRACTION"
    assert any("solver strategy not used" in w for w in r.warnings)


def test_solver_unavailable_and_corrupt_artifact(tmp_path):
    missing = load_solver(tmp_path / "nope.npz")
    assert isinstance(missing, LookupMiss) and missing.code == "INCOMPATIBLE_CHECKPOINT"
    bad = tmp_path / "corrupt.npz"
    bad.write_bytes(b"not a zip file at all")
    corrupt = load_solver(bad)
    assert isinstance(corrupt, LookupMiss) and corrupt.code == "INCOMPATIBLE_CHECKPOINT"
    np.savez(tmp_path / "wrong.npz", format=np.array("pokeralpha-strategy-v2"))
    wrong = load_solver(tmp_path / "wrong.npz")
    assert isinstance(wrong, LookupMiss)
    r = analyze(observe_manual(hu()), config=FAST, solver=corrupt)
    sol = [c for c in cascade(r) if c["source"] == "solver"][0]
    assert sol["status"] == "rejected" and sol["code"] == "INCOMPATIBLE_CHECKPOINT"
    assert r.recommended is not None and r.method != "solver"
