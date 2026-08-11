"""Tests for external-sampling MCCFR."""

import numpy as np
import pytest

from poker_alpha.games import KuhnPoker
from poker_alpha.solvers import MCCFRSolver, exploitability, expected_value

KUHN_NASH_VALUE = -1.0 / 18.0


def test_mccfr_converges_approximately_to_kuhn_value():
    solver = MCCFRSolver(KuhnPoker(), seed=42)
    solver.train(50_000)
    game = KuhnPoker()
    avg = solver.average_strategy()
    # Sampling noise: looser tolerance than exact CFR.
    assert expected_value(game, avg) == pytest.approx(KUHN_NASH_VALUE, abs=5e-3)
    assert exploitability(game, avg) < 2e-2


def test_mccfr_is_deterministic_given_seed():
    a = MCCFRSolver(KuhnPoker(), seed=7)
    a.train(2_000)
    b = MCCFRSolver(KuhnPoker(), seed=7)
    b.train(2_000)
    assert a.average_strategy() == b.average_strategy()


def test_mccfr_seeds_differ():
    a = MCCFRSolver(KuhnPoker(), seed=1)
    a.train(2_000)
    b = MCCFRSolver(KuhnPoker(), seed=2)
    b.train(2_000)
    assert a.average_strategy() != b.average_strategy()


def test_mccfr_average_strategy_normalized():
    solver = MCCFRSolver(KuhnPoker(), seed=0)
    solver.train(1_000)
    for probs in solver.average_strategy().values():
        assert sum(probs.values()) == pytest.approx(1.0)


def test_mccfr_respects_dominant_actions():
    solver = MCCFRSolver(KuhnPoker(), seed=42)
    solver.train(50_000)
    avg = solver.average_strategy()
    assert avg["2b"]["b"] > 0.95   # King never folds to a bet
    assert avg["0b"]["p"] > 0.95   # Jack never calls a bet
