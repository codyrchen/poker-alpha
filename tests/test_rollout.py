"""Phase 14: action EV rollouts with common random numbers."""

import pytest

from poker_alpha.decision import DecisionConfig, recommend_action
from poker_alpha.decision.rollout import (RolloutCandidate,
                                          rollout_action_evs)
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.opponent import ARCHETYPE_MODELS
from poker_alpha.poker import card_code
from poker_alpha.poker.multiway import exact_equity_enumeration
from poker_alpha.poker.ranges import WeightedRange

REG = ARCHETYPE_MODELS["regular"]


def hu(hero_cards, board, *, hero_bet=0.0, opp_bet=0.0, pot, hero_stack=100.0,
       opp_stack=100.0, opp_all_in=False):
    return ManualStateAdapter.from_dict({
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": hero_cards, "board": board,
        "pot": pot, "actor": 1,
        "seats": [{"stack": opp_stack, "bet": opp_bet, "all_in": opp_all_in},
                  {"stack": hero_stack, "bet": hero_bet}]})


def C(*cards):
    return [card_code(c) for c in cards]


def test_fold_is_zero_and_results_are_seeded():
    obs = hu("Ah Kh", "2c 7d 9s", opp_bet=5.0, pot=15.0)
    rng_ranges = {0: WeightedRange.from_string("random")}
    cands = [RolloutCandidate("fold", "fold", 0.0),
             RolloutCandidate("call", "call", 5.0),
             RolloutCandidate("raise_100", "raise", 30.0)]
    a = rollout_action_evs(obs, C("Ah", "Kh"), rng_ranges, {0: REG}, cands,
                           simulations=400, seed=3)
    b = rollout_action_evs(obs, C("Ah", "Kh"), rng_ranges, {0: REG}, cands,
                           simulations=400, seed=3)
    assert a.ev("fold").ev_bb == 0.0 and a.ev("fold").se_bb == 0.0
    assert a.evs == b.evs
    assert sum(a.best_probability.values()) == pytest.approx(1.0)


def test_common_random_numbers_reduce_difference_variance():
    obs = hu("Ah Kh", "2c 7d 9s", pot=10.0)
    cands = [RolloutCandidate("check", "check", 0.0),
             RolloutCandidate("bet_75", "bet", 7.5)]
    res = rollout_action_evs(obs, C("Ah", "Kh"), {0: WeightedRange.uniform()},
                             {0: REG}, cands, simulations=1500, seed=1)
    _, paired_se = res.paired_difference("bet_75", "check")
    indep_se = (res.ev("bet_75").se_bb ** 2 + res.ev("check").se_bb ** 2) ** 0.5
    assert paired_se < 0.8 * indep_se


def test_call_vs_all_in_matches_closed_form():
    # Opponent is all-in: no responses, no future betting -> EV is exact
    # equity arithmetic, which the rollout must reproduce within noise.
    board = "2c 7d 9s Th 3c"
    obs = hu("Jc Jd", board, opp_bet=40.0, pot=45.0, opp_stack=0.0,
             opp_all_in=True)
    villain = WeightedRange.from_string("TT,99,AK,QQ")
    res = rollout_action_evs(obs, C("Jc", "Jd"), {0: villain}, {0: REG},
                             [RolloutCandidate("call", "call", 40.0)],
                             simulations=3000, seed=2)
    share = exact_equity_enumeration(C("Jc", "Jd"), C(*board.split()),
                                     [villain])
    closed = share * (45 + 40) - 40
    e = res.ev("call")
    assert abs(e.ev_bb - closed) < 4 * e.se_bb + 1e-6


def report(obs, ranges, sims=1500):
    return recommend_action(obs, opponent_ranges=ranges, config=DecisionConfig(
        equity_simulations=800, rollout_simulations=sims, seed=5))


def test_obvious_fold_zero_equity_bluff_catcher():
    # River, hero has 7-high, villain's range is all straights or better.
    obs = hu("7c 2d", "Ah Kh Qd Js 4c", opp_bet=50.0, pot=60.0)
    rep = report(obs, {0: WeightedRange.from_string("T9s,T9o,TT")})
    assert rep.hero_equity == 0.0
    assert rep.recommended == "fold"
    assert rep.candidate("call").ev_bb == pytest.approx(-50.0)
    assert rep.method == "Monte Carlo rollout"


def test_nut_hand_prefers_betting_to_checking():
    obs = hu("Th 9h", "Ah Kh Qh Js 4c", pot=20.0)          # royal flush
    rep = report(obs, {0: WeightedRange.from_string("AK,AQ,KQ,AJ,QJ")})
    assert rep.hero_equity == 1.0
    assert rep.recommended != "check"
    assert rep.candidate(rep.recommended).ev_bb > rep.candidate("check").ev_bb


def test_free_check_has_no_fold_option():
    obs = hu("7c 2d", "Ah Kh Qd Js 4c", pot=20.0)
    rep = report(obs, {0: WeightedRange.uniform()}, sims=300)
    labels = [c.label for c in rep.candidates]
    assert "fold" not in labels and labels[0] == "check"
    assert rep.candidate("check").ev_bb >= 0.0


def test_forced_all_in_call_menu():
    obs = hu("As Ad", "", opp_bet=100.0, hero_bet=1.0, pot=101.0,
             hero_stack=19.0, opp_stack=0.0, opp_all_in=True)
    rep = report(obs, {0: WeightedRange.from_string("random")}, sims=500)
    assert [c.label for c in rep.candidates] == ["fold", "call"]
    assert rep.recommended == "call"
    assert rep.candidate("call").ev_bb > 0
