"""Phase 12: multiway equity with joint card constraints and side pots."""

import numpy as np
import pytest

from poker_alpha.poker import card_code, estimate_equity
from poker_alpha.poker.multiway import (SamplingError,
                                        exact_equity_enumeration,
                                        multiway_equity)
from poker_alpha.poker.ranges import WeightedRange


def C(*cards):
    return [card_code(c) for c in cards]


def test_heads_up_matches_existing_engine():
    a = multiway_equity(C("As", "Ad"), (), [None], simulations=4000, seed=1)
    b = estimate_equity(["As", "Ad"], simulations=4000, seed=2)
    assert abs(a.expected_share - b.equity) < 4 * (a.std_error + b.std_error)
    assert 0.82 < a.expected_share < 0.88
    assert a.win + a.tie <= 1.0 and a.runtime_s > 0


@pytest.mark.parametrize("method", ["rejection", "importance"])
def test_matches_exact_enumeration_with_overlapping_ranges(method):
    hero, board = C("Jc", "Jd"), C("2c", "7d", "9h", "Ts", "3s")
    r1 = WeightedRange.from_string("AA,KK,JTs")
    r2 = WeightedRange.from_string("AK,QQ,T9s:0.5")
    exact = exact_equity_enumeration(hero, board, [r1, r2])
    est = multiway_equity(hero, board, [r1, r2], simulations=6000,
                          method=method, seed=3)
    assert abs(est.expected_share - exact) < 4 * est.std_error + 1e-3
    if method == "importance":
        assert est.effective_samples < est.simulations


def test_naive_sequential_sampling_would_be_biased():
    """Why opponents are not simply sampled one after another.

    r1 = {AsAh, 6c6d}, r2 = {AsAc, 7c7d}; hero JcJh beats only 66/77.
    Exact joint: (AA,AA) collides, so (AA,77), (66,AA), (66,77) are equally
    likely and hero wins 1/3. Naive sequential sampling picks opp1 50/50 and
    gives hero 0.5 * 0.5 = 0.25.
    """
    hero, board = C("Jc", "Jh"), C("Kd", "Qd", "8h", "3c", "2s")
    r1 = WeightedRange.from_combos([(C("As", "Ah"), 1), (C("6c", "6d"), 1)])
    r2 = WeightedRange.from_combos([(C("As", "Ac"), 1), (C("7c", "7d"), 1)])
    exact = exact_equity_enumeration(hero, board, [r1, r2])
    assert exact == pytest.approx(1 / 3)
    for method in ("rejection", "importance"):
        est = multiway_equity(hero, board, [r1, r2], simulations=3000,
                              method=method, seed=4)
        assert abs(est.expected_share - exact) < 4 * est.std_error + 1e-3
        assert abs(est.expected_share - 0.25) > 4 * est.std_error


def test_side_pot_aware_share():
    # Hero has the nuts (royal) but is all-in short for 10; opponents 50 each.
    hero, board = C("As", "Ks"), C("Qs", "Js", "Ts", "2d", "3c")
    res = multiway_equity(hero, board, [None, None], simulations=200,
                          contributions=[10, 50, 50], seed=5)
    assert res.expected_share == pytest.approx(30 / 110)
    assert res.win == 1.0
    assert sum(res.opponent_shares) == pytest.approx(80 / 110)


def test_nine_players_no_collisions_and_seeded():
    a = multiway_equity(C("Ah", "Kh"), C("2c", "7d", "9s"), [None] * 8,
                        simulations=300, seed=7)
    b = multiway_equity(C("Ah", "Kh"), C("2c", "7d", "9s"), [None] * 8,
                        simulations=300, seed=7)
    assert a.expected_share == b.expected_share
    assert a.expected_share + sum(a.opponent_shares) == pytest.approx(1.0)
    assert 0.0 < a.expected_share < 0.5


def test_incompatible_ranges_raise():
    hero = C("As", "2d")
    aa = WeightedRange.from_string("AA")
    for method in ("rejection", "importance"):
        with pytest.raises(SamplingError):
            multiway_equity(hero, (), [aa, aa], simulations=5, method=method,
                            seed=0, min_acceptance=0.01)


def test_input_validation():
    with pytest.raises(ValueError):
        multiway_equity(C("As", "As"), (), [None])
    with pytest.raises(ValueError):
        multiway_equity(C("As", "Ks"), (), [None] * 9)
    with pytest.raises(ValueError):
        multiway_equity(C("As", "Ks"), (), [None], contributions=[1])
