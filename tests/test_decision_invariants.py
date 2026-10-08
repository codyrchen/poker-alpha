"""Phase 25: decision-quality invariants against exact formulas.

Not a GTO test: these catch recommendations that are mathematically wrong
under the engine's own stated model. Opponent responses are pinned with a
fixed-probability model so rollout EVs have closed forms.
"""

import numpy as np
import pytest

from poker_alpha.decision import DecisionConfig, recommend_action
from poker_alpha.decision.rollout import RolloutCandidate, rollout_action_evs
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.poker import card_code
from poker_alpha.poker.multiway import exact_equity_enumeration, multiway_equity
from poker_alpha.poker.ranges import WeightedRange

C = card_code


class FixedResponse:
    """Opponent model with strength-independent response probabilities."""

    name = "fixed"

    def __init__(self, fold=0.0, raise_=0.0, bet=0.0):
        self.fold, self.raise_, self.bet = fold, raise_, bet

    def probabilities(self, strength, facing_bet, size_pot_fraction=0.5):
        n = len(np.atleast_1d(strength))
        if not facing_bet:
            return {"bet": np.full(n, self.bet), "check": np.full(n, 1 - self.bet)}
        return {"raise": np.full(n, self.raise_), "fold": np.full(n, self.fold),
                "call": np.full(n, 1 - self.fold - self.raise_)}

    def likelihood(self, action, strength, facing_bet, size_pot_fraction=0.5):
        return np.ones(len(strength))


def combo(a, b):
    return WeightedRange.from_combos([((C(a), C(b)), 1.0)])


def state(seats, pot, board, hero_cards, hero_seat, dealer=0):
    return ManualStateAdapter.from_dict({
        "num_seats": len(seats), "hero_seat": hero_seat, "dealer": dealer,
        "small_blind": 0.5, "big_blind": 1.0, "hero_cards": hero_cards,
        "board": board, "pot": pot, "actor": hero_seat, "seats": seats})


def hu(hero_cards, board, *, pot, opp_bet=0.0, opp_stack=100.0, opp_all_in=False,
       hero_stack=100.0):
    # seat 0 = villain (button), seat 1 = hero (BB, acts first postflop).
    return state([{"stack": opp_stack, "bet": opp_bet, "all_in": opp_all_in},
                  {"stack": hero_stack, "bet": 0.0}], pot, board, hero_cards, 1)


BOARD = "2c 7d 9s Th 3h"


def evs(obs, hero, ranges, models, cands, sims=400, seed=0):
    res = rollout_action_evs(obs, [C(x) for x in hero.split()], ranges, models,
                             [RolloutCandidate(*c) for c in cands],
                             simulations=sims, seed=seed)
    return res


# -- river ----------------------------------------------------------------------

def test_1_nuts_never_folds():
    # Hero holds the nut straight flush; check is free, and facing a bet.
    rep = recommend_action(hu("Jh Qh", "Th 9h 8h 2c 3d", pot=10.0),
                           opponent_ranges={0: WeightedRange.uniform()},
                           config=DecisionConfig(equity_simulations=300,
                                                 rollout_simulations=200))
    assert rep.hero_equity == 1.0 and rep.recommended != "fold"
    assert "fold" not in [c.label for c in rep.candidates]
    rep = recommend_action(hu("Jh Qh", "Th 9h 8h 2c 3d", pot=30.0, opp_bet=20.0),
                           opponent_ranges={0: WeightedRange.uniform()},
                           config=DecisionConfig(equity_simulations=300,
                                                 rollout_simulations=200))
    assert rep.recommended != "fold"
    assert rep.candidate("call").ev_bb > rep.candidate("fold").ev_bb


def test_2_zero_equity_call_costs_exactly_the_call():
    obs = hu("4c 5d", BOARD, pot=60.0, opp_bet=50.0, opp_stack=0.0, opp_all_in=True)
    res = evs(obs, "4c 5d", {0: combo("As", "Ad")}, {0: FixedResponse()},
              [("fold", "fold", 0.0), ("call", "call", 50.0)])
    assert res.ev("call").ev_bb == pytest.approx(-50.0)
    assert res.ev("call").se_bb == pytest.approx(0.0)
    assert res.ev("fold").ev_bb == 0.0


@pytest.mark.parametrize("bet,pot,expect", [(100.0, 110.0, "fold"), (20.0, 120.0, "call")])
def test_3_4_pot_odds_threshold_under_fixed_showdown(bet, pot, expect):
    # Villain all-in; KK beats exactly one of three combos: equity 1/3.
    villain = WeightedRange.from_combos([((C("Ac"), C("Ad")), 1), ((C("Ah"), C("As")), 1),
                                         ((C("4c"), C("4d")), 1)])
    obs = hu("Kc Kd", BOARD, pot=pot, opp_bet=bet, opp_stack=0.0, opp_all_in=True)
    share = exact_equity_enumeration([C("Kc"), C("Kd")], [C(x) for x in BOARD.split()],
                                     [villain])
    assert share == pytest.approx(1 / 3)
    exact_call = share * (pot + bet) - bet
    res = evs(obs, "Kc Kd", {0: villain}, {0: FixedResponse()},
              [("fold", "fold", 0.0), ("call", "call", bet)], sims=3000)
    e = res.ev("call")
    assert abs(e.ev_bb - exact_call) < 4 * e.se_bb + 1e-9
    winner = "call" if exact_call > 0 else "fold"
    assert winner == expect
    best = max(res.evs, key=lambda x: x.ev_bb).label
    assert best == expect


# -- betting ----------------------------------------------------------------------

def test_5_always_fold_bet_wins_exactly_the_pot():
    obs = hu("4c 5d", BOARD, pot=12.0)
    cands = [("bet_33", "bet", 4.0), ("bet_100", "bet", 12.0), ("all_in", "all_in", 100.0)]
    res = evs(obs, "4c 5d", {0: combo("As", "Ad")}, {0: FixedResponse(fold=1.0)}, cands)
    for e in res.evs:
        assert e.ev_bb == pytest.approx(12.0)   # the pot, independent of size
        assert e.se_bb == pytest.approx(0.0)


@pytest.mark.parametrize("hero,villain,win", [("Kc Kd", ("4c", "4d"), True),
                                              ("Kc Kd", ("As", "Ad"), False)])
def test_6_always_call_bet_matches_showdown_geometry(hero, villain, win):
    obs = hu(hero, BOARD, pot=12.0)
    cands = [("check", "check", 0.0), ("bet_50", "bet", 6.0), ("bet_100", "bet", 12.0)]
    res = evs(obs, hero, {0: combo(*villain)}, {0: FixedResponse(fold=0.0)}, cands)
    for label, _, size in cands:
        expected = (12.0 + size) if win else -size
        assert res.ev(label).ev_bb == pytest.approx(expected)


def test_7_bigger_bets_not_better_when_math_says_worse():
    # Villain always calls; hero has 1/3 showdown share. EV(X) = s*P + (2s-1)X,
    # strictly decreasing in X. Paired differences must agree exactly with CRN.
    villain = WeightedRange.from_combos([((C("Ac"), C("Ad")), 1), ((C("Ah"), C("As")), 1),
                                         ((C("4c"), C("4d")), 1)])
    obs = hu("Kc Kd", BOARD, pot=12.0)
    sizes = [("check", "check", 0.0), ("bet_33", "bet", 4.0), ("bet_75", "bet", 9.0),
             ("bet_100", "bet", 12.0), ("all_in", "all_in", 100.0)]
    res = evs(obs, "Kc Kd", {0: villain}, {0: FixedResponse(fold=0.0)}, sizes, sims=2000)
    means = [res.ev(l).ev_bb for l, _, _ in sizes]
    assert means == sorted(means, reverse=True)
    # Exact per-sample identity: EV(X2) - EV(X1) = (2 s_hat - 1)(X2 - X1).
    s_hat = (res.ev("check").ev_bb) / 12.0
    for (l1, _, x1), (l2, _, x2) in zip(sizes, sizes[1:]):
        d, _ = res.paired_difference(l2, l1)
        assert d == pytest.approx((2 * s_hat - 1) * (x2 - x1), abs=1e-9)
    best = recommend_action(obs, opponent_ranges={0: villain},
                            opponent_models={0: FixedResponse(fold=0.0)},
                            config=DecisionConfig(equity_simulations=500,
                                                  rollout_simulations=1000))
    assert best.recommended == "check"


# -- multiway ----------------------------------------------------------------------

def three_way(pot, bets=(0.0, 0.0, 0.0), stacks=(100.0, 100.0, 100.0)):
    # seat 0 = button, seat 1 = hero (SB, acts first postflop), seat 2 = BB.
    return state([{"stack": stacks[0], "bet": bets[0]},
                  {"stack": stacks[1], "bet": bets[1]},
                  {"stack": stacks[2], "bet": bets[2]}],
                 pot, BOARD, "Kc Kd", hero_seat=1)


def test_8_no_double_counting_of_pot_shares():
    # Hero (KK) ties with seat 0 (KhKs), beats seat 2: hero owns half the pot.
    obs = three_way(30.0)
    ranges = {0: combo("Kh", "Ks"), 2: combo("4c", "4d")}
    eq = multiway_equity([C("Kc"), C("Kd")], [C(x) for x in BOARD.split()],
                         list(ranges.values()), simulations=200, seed=1)
    assert eq.expected_share == pytest.approx(0.5)
    assert eq.expected_share + sum(eq.opponent_shares) == pytest.approx(1.0)
    res = evs(obs, "Kc Kd", ranges, {0: FixedResponse(), 2: FixedResponse()},
              [("check", "check", 0.0), ("bet_100", "bet", 30.0)])
    assert res.ev("check").ev_bb == pytest.approx(15.0)
    # Both call 30: pot 120 split two ways -> 60 back for 30 invested.
    assert res.ev("bet_100").ev_bb == pytest.approx(60.0 - 30.0)


def test_9_folded_players_cannot_win():
    # Seat 0 holds the nuts but folds; seat 2 (worse than hero) calls.
    obs = three_way(30.0)
    ranges = {0: combo("Jh", "8h"), 2: combo("4c", "4d")}   # 0 has a straight
    models = {0: FixedResponse(fold=1.0), 2: FixedResponse(fold=0.0)}
    res = evs(obs, "Kc Kd", ranges, models, [("bet_50", "bet", 15.0)])
    assert res.ev("bet_50").ev_bb == pytest.approx(30.0 + 15.0)


def test_10_side_pot_eligibility_in_rollouts():
    # Seat 0: 10 behind with the nuts; seat 2: deep, worse than hero.
    # Hero bets 50: seat 0 calls all-in for 10, seat 2 calls 50.
    # Main (pot 30 + 3x10) -> seat 0; side 2x40 = 80 -> hero. EV = 80 - 50.
    obs = three_way(30.0, stacks=(10.0, 100.0, 100.0))
    ranges = {0: combo("Jh", "8h"), 2: combo("4c", "4d")}
    models = {0: FixedResponse(fold=0.0), 2: FixedResponse(fold=0.0)}
    res = evs(obs, "Kc Kd", ranges, models, [("bet", "bet", 50.0)])
    assert res.ev("bet").ev_bb == pytest.approx(80.0 - 50.0)
