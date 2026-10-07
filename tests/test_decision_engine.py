"""Phase 13: decision engine, method hierarchy and provenance."""

import pytest

from poker_alpha.decision import (DecisionConfig, SolverStrategyProvider,
                                  recommend_action)
from poker_alpha.decision.strategy import LookupMiss
from poker_alpha.games import HoldemGame
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.poker.ranges import WeightedRange

FAST = DecisionConfig(equity_simulations=600, seed=1)


def six_max_flop(**kw):
    d = {
        "num_seats": 6, "hero_seat": 0, "dealer": 0,
        "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "As Ks", "board": "Qs Js 4h", "pot": 13.5, "actor": 0,
        "seats": [
            {"stack": 96.5}, {"stack": 99.5, "folded": True},
            {"stack": 90.0, "bet": 0}, {"stack": 100, "folded": True},
            {"stack": 100, "folded": True}, {"stack": 100, "folded": True},
        ],
        "actions": [
            {"street": 0, "seat": 3, "kind": "fold"},
            {"street": 0, "seat": 4, "kind": "fold"},
            {"street": 0, "seat": 5, "kind": "fold"},
            {"street": 0, "seat": 0, "kind": "raise", "amount": 3.5},
            {"street": 0, "seat": 1, "kind": "fold"},
            {"street": 0, "seat": 2, "kind": "call", "amount": 2.5},
            {"street": 1, "seat": 2, "kind": "check"},
        ],
    }
    d.update(kw)
    return ManualStateAdapter.from_dict(d)


def test_heuristic_report_structure_and_provenance():
    rep = recommend_action(six_max_flop(), config=FAST)
    assert rep.method == "heuristic fallback"
    assert rep.hero_equity is not None and 0.4 < rep.hero_equity < 0.9
    assert rep.pot_odds is None and rep.to_call_bb == 0
    assert rep.spr == pytest.approx(90 / 13.5)
    labels = [c.label for c in rep.candidates]
    assert labels[0] == "check" and "fold" not in labels and "all_in" in labels
    assert rep.recommended == "check"
    assert all(c.source == "heuristic fallback" for c in rep.candidates)
    bet = rep.candidate("bet_75")
    assert bet.ev_bb is None and "rollout" in bet.note
    assert rep.opponent_ranges[0].line == "call_open"
    assert rep.opponent_ranges[0].position == "BB"
    assert "GTO" not in rep.format()


def test_pot_odds_heuristic_call_vs_fold():
    obs = ManualStateAdapter.from_dict(_dict_with_bet(10.0))
    rep = recommend_action(obs, config=FAST)
    assert rep.pot_odds == pytest.approx(10 / 33.5)
    call = rep.candidate("call")
    assert call.ev_bb == pytest.approx(rep.hero_equity * 33.5 - 10, abs=1e-6)
    assert rep.recommended == ("call" if rep.hero_equity >= rep.pot_odds else "fold")


def _dict_with_bet(bet):
    from poker_alpha.holdem import observed_to_dict
    d = observed_to_dict(six_max_flop())
    d["seats"][2]["bet"] = bet
    d["pot"] = 13.5 + bet
    d["actions"] = d["actions"][:-1] + [
        {"street": 1, "seat": 2, "kind": "bet", "amount": bet}]
    return d


def test_explicit_ranges_override_inference():
    nuts = {2: WeightedRange.from_string("TT")}  # BB holds a set or straight-ish
    rep = recommend_action(six_max_flop(), opponent_ranges=nuts, config=FAST)
    assert rep.opponent_ranges[0].line == "given"
    assert rep.opponent_ranges[0].live_combos == 6


def test_invalid_state_returns_no_recommendation():
    bad = six_max_flop(board="As Js 4h")      # collides with hero
    rep = recommend_action(bad, config=FAST)
    assert rep.recommended is None and rep.confidence == "low"
    assert any("board" in w for w in rep.warnings)
    not_turn = six_max_flop(actor=2)
    assert recommend_action(not_turn, config=FAST).recommended is None


def hu_state(stack=100.0, hero_cards="As Ks", actions=(), board="",
             street=None):
    d = {
        "num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": hero_cards, "board": board,
        "actor": 0, "seats": [
            {"stack": stack - 0.5, "bet": 0.5, "committed": 0.5},
            {"stack": stack - 1.0, "bet": 1.0, "committed": 1.0}],
        "pot": 1.5, "actions": list(actions),
    }
    if street:
        d["street"] = street
    return ManualStateAdapter.from_dict(d)


def test_solver_method_a_used_for_matching_heads_up_spot():
    game = HoldemGame()
    obs = hu_state()
    provider = SolverStrategyProvider(game, {"0|50,51||": {
        "f": 0.0, "c": 0.1, "b50": 0.0, "b100": 0.7, "b200": 0.2, "a": 0.0}},
        visits={"0|50,51||": 500})
    rep = recommend_action(obs, config=DecisionConfig(
        equity_simulations=300, solver=provider))
    assert rep.method == "solver"
    assert rep.recommended == "raise_100"
    assert rep.recommended_mix["raise_100"] == pytest.approx(0.7)
    assert rep.candidate("raise_100").amount_to == pytest.approx(3.0)
    assert any("abstracted heads-up" in w for w in rep.warnings)


def test_solver_refuses_outside_abstraction():
    game = HoldemGame()
    provider = SolverStrategyProvider(game, {}, {})
    assert isinstance(provider.lookup(hu_state(stack=40)), LookupMiss)
    assert "never visited" in provider.lookup(hu_state()).reason
    assert "heads-up" in provider.lookup(six_max_flop()).reason
    low = SolverStrategyProvider(game, {"0|50,51||": {"c": 1.0}},
                                 {"0|50,51||": 3}, min_visits=20)
    assert "visited only" in low.lookup(hu_state()).reason
    rep = recommend_action(hu_state(stack=40), config=DecisionConfig(
        equity_simulations=200, solver=provider))
    assert rep.method == "heuristic fallback"
    assert any("solver strategy not used" in w for w in rep.warnings)


def test_solver_interpolates_off_tree_sizes():
    game = HoldemGame()
    # BTN raised to 2.5 (abstract sizes are 2.0 / 3.0 / 5.0), BB to act.
    obs = ManualStateAdapter.from_dict({
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": "Qh Qd", "actor": 1, "pot": 3.5,
        "seats": [{"stack": 97.5, "bet": 2.5, "committed": 2.5},
                  {"stack": 99.0, "bet": 1.0, "committed": 1.0}],
        "actions": [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5}]})
    state, exact = SolverStrategyProvider(game, {}, {})._to_state(obs)
    # 2.5 is nearer 3.0 (b100) than 2.0 (b50) in log space.
    assert not exact and state.streets == ("b100",)
    provider = SolverStrategyProvider(game, {"1|23,36||b100": {"c": 1.0}})
    look = provider.lookup(obs)
    assert not look.exact
    rep = recommend_action(obs, config=DecisionConfig(equity_simulations=200,
                                                      solver=provider))
    assert rep.method == "interpolated abstraction" and rep.confidence == "low"
