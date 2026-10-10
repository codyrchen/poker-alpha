"""Rollout options: opponent response models (Phases 59-60) and depth (Phase 61).

The defaults (response="behavior", depth="street") are unchanged and pinned
by tests/test_golden_e2e.py; these tests cover the opt-in options.
"""

import math

import pytest

from poker_alpha.decision.rollout import (DEPTHS, RESPONSE_MODELS, RolloutCandidate,
                                          RolloutError, rollout_action_evs)
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.opponent import ARCHETYPE_MODELS
from poker_alpha.poker import card_code
from poker_alpha.poker.ranges import WeightedRange

REG = ARCHETYPE_MODELS["regular"]
STATION = ARCHETYPE_MODELS["calling_station"]


def hu(hero_cards, board, *, opp_bet=0.0, pot, dealer=0, hero_stack=100.0, opp_stack=100.0):
    return ManualStateAdapter.from_dict({
        "num_seats": 2, "hero_seat": 1, "dealer": dealer, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": hero_cards, "board": board, "pot": pot, "actor": 1,
        "seats": [{"stack": opp_stack, "bet": opp_bet}, {"stack": hero_stack, "bet": 0.0}]})


def C(*cards):
    return [card_code(c) for c in cards]


CANDS = [RolloutCandidate("fold", "fold", 0.0), RolloutCandidate("check", "check", 0.0),
         RolloutCandidate("bet_75", "bet", 7.5)]


def run(obs, hero, rng, **kw):
    return rollout_action_evs(obs, C(*hero.split()), {0: rng}, {0: kw.pop("model", REG)},
                              CANDS, simulations=kw.pop("sims", 600), seed=kw.pop("seed", 4), **kw)


def test_option_validation():
    obs = hu("Ah Kh", "2c 7d 9s", pot=10.0)
    assert "street" in DEPTHS and "showdown" in DEPTHS
    assert set(RESPONSE_MODELS) == {"behavior", "mdf_range", "mdf_calibrated"}
    with pytest.raises(ValueError):
        run(obs, "Ah Kh", WeightedRange.uniform(), depth="deep")
    with pytest.raises(ValueError):
        run(obs, "Ah Kh", WeightedRange.uniform(), response="nash")
    with pytest.raises(ValueError):
        run(obs, "Ah Kh", WeightedRange.uniform(), response="mdf_calibrated")


def test_showdown_depth_rejects_multiway():
    obs = ManualStateAdapter.from_dict({
        "num_seats": 3, "hero_seat": 2, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "Ah Kh", "board": "2c 7d 9s", "pot": 10.0, "actor": 2,
        "seats": [{"stack": 100.0}, {"stack": 100.0}, {"stack": 100.0}]})
    with pytest.raises(RolloutError):
        rollout_action_evs(obs, C("Ah", "Kh"), {0: WeightedRange.uniform(),
                                                1: WeightedRange.uniform()},
                           {0: REG, 1: REG}, CANDS, simulations=50, seed=1, depth="showdown")


def test_showdown_equals_street_on_the_river():
    obs = hu("Ah Kh", "2c 7d 9s Th 3c", pot=10.0)
    a = run(obs, "Ah Kh", WeightedRange.uniform(), depth="street")
    b = run(obs, "Ah Kh", WeightedRange.uniform(), depth="showdown")
    assert a.evs == b.evs


def test_showdown_is_seeded_and_bounded():
    obs = hu("Qs Qd", "Qh 7c 2d", pot=10.0)
    a = run(obs, "Qs Qd", WeightedRange.uniform(), depth="showdown")
    b = run(obs, "Qs Qd", WeightedRange.uniform(), depth="showdown")
    assert a.evs == b.evs
    assert a.ev("fold").ev_bb == 0.0
    for e in a.evs:
        assert math.isfinite(e.ev_bb) and -100.0 <= e.ev_bb <= 110.0


def test_showdown_credits_later_value_against_a_calling_station():
    # Top set on a dry flop: checking now loses nothing in "showdown" depth,
    # because the hero bets later streets and a calling station pays; the
    # one-street rollout checks the hand down and misses that value.
    obs = hu("Qs Qd", "Qh 7c 2d", pot=10.0, dealer=1)
    rng = WeightedRange.from_string("22+,A2s+,K2s+,Q2s+,J7s+,A2o+,K9o+,QTo+")
    street = run(obs, "Qs Qd", rng, model=STATION, depth="street", sims=1500)
    deep = run(obs, "Qs Qd", rng, model=STATION, depth="showdown", sims=1500)
    gain = deep.ev("check").ev_bb - street.ev("check").ev_bb
    se = math.hypot(deep.ev("check").se_bb, street.ev("check").se_bb)
    assert gain > 3 * se and gain > 2.0


def test_mdf_range_defends_by_range_percentile():
    # Against MDF responses, a pot-size bet is called by exactly the top half
    # of the range: a pure bluff with no equity loses ~ (0.5 * -bet) + 0.5 * pot.
    obs = hu("2h 3h", "Ac Kd Qs Jh 9c", pot=10.0)
    cands = [RolloutCandidate("check", "check", 0.0), RolloutCandidate("bet_100", "bet", 10.0)]
    res = rollout_action_evs(obs, C("2h", "3h"), {0: WeightedRange.from_string("22+,A2+,K2+")},
                             {0: REG}, cands, simulations=4000, seed=9, response="mdf_range")
    e = res.ev("bet_100")
    assert e.ev_bb == pytest.approx(0.5 * 10.0 - 0.5 * 10.0, abs=4 * e.se_bb + 0.3)
