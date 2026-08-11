"""Tests for the Monte Carlo equity engine."""

import pytest

from poker_alpha.poker import estimate_equity


def test_probabilities_sum_to_one():
    r = estimate_equity(["As", "Ah"], simulations=2000, seed=0)
    assert r.win + r.tie + r.lose == pytest.approx(1.0)
    assert r.equity == pytest.approx(r.win + r.tie / 2)


def test_deterministic_given_seed():
    a = estimate_equity(["As", "Kd"], simulations=3000, seed=42)
    b = estimate_equity(["As", "Kd"], simulations=3000, seed=42)
    assert (a.win, a.tie, a.lose) == (b.win, b.tie, b.lose)


def test_aces_preflop_vs_random_near_85_percent():
    r = estimate_equity(["As", "Ah"], simulations=20_000, seed=42)
    assert r.equity == pytest.approx(0.852, abs=0.01)


def test_seven_deuce_is_weak():
    r = estimate_equity(["7d", "2c"], simulations=20_000, seed=42)
    assert r.equity == pytest.approx(0.346, abs=0.015)


def test_nearly_unbeatable_hand():
    # Hero holds a royal flush after the river: only a chop... no, royal is nuts.
    r = estimate_equity(["As", "Ks"],
                        board=["Qs", "Js", "Ts", "2d", "3c"],
                        simulations=2000, seed=0)
    assert r.win == pytest.approx(1.0)
    assert r.lose == 0.0


def test_board_plays_identical_showdown():
    # Royal flush on the board: every showdown is a tie.
    r = estimate_equity(["2c", "3d"],
                        board=["As", "Ks", "Qs", "Js", "Ts"],
                        simulations=1000, seed=0)
    assert r.tie == pytest.approx(1.0)
    assert r.equity == pytest.approx(0.5)


def test_drawing_hand_on_flop():
    # Nut flush draw vs a made hand region: equity should be materially
    # between 0 and 1, not degenerate.
    r = estimate_equity(["As", "Ks"], board=["Qs", "7s", "2d"],
                        simulations=10_000, seed=42)
    assert 0.5 < r.equity < 0.85  # strong draw + overcards vs random


def test_weighted_range_shifts_equity():
    # Versus a range of only aces/kings, AKo does much worse than vs random.
    strong_range = [
        (("Ac", "Ad"), 1.0), (("Ac", "Ah"), 1.0), (("Ad", "Ah"), 1.0),
        (("Kc", "Kd"), 1.0), (("Kc", "Kh"), 1.0), (("Kd", "Kh"), 1.0),
    ]
    vs_range = estimate_equity(["As", "Ks"], opponent_range=strong_range,
                               simulations=8000, seed=42)
    vs_random = estimate_equity(["As", "Ks"], simulations=8000, seed=42)
    assert vs_range.equity < vs_random.equity - 0.15


def test_blocked_combos_are_dropped():
    # Hero holds Ac: a range listing combos with Ac keeps only live ones.
    rng = [(("Ac", "Ad"), 1.0), (("Kc", "Kd"), 1.0)]
    r = estimate_equity(["Ac", "2d"], opponent_range=rng,
                        simulations=1000, seed=0)
    # Only KK is live; equity of A2o vs KK is poor.
    assert r.equity < 0.4


def test_fully_blocked_range_raises():
    with pytest.raises(ValueError):
        estimate_equity(["Ac", "Ad"],
                        opponent_range=[(("Ac", "Kd"), 1.0)],
                        simulations=100, seed=0)


def test_input_validation():
    with pytest.raises(ValueError):
        estimate_equity(["As"], simulations=10)
    with pytest.raises(ValueError):
        estimate_equity(["As", "As"], simulations=10)
    with pytest.raises(ValueError):
        estimate_equity(["As", "Kd"], board=["As", "2c", "3c"], simulations=10)
