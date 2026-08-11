"""Tests for CFR+: clipping, averaging, convergence, and speedup vs CFR."""

import numpy as np
import pytest

from poker_alpha.games import KuhnPoker
from poker_alpha.solvers import (
    CFRPlusSolver,
    CFRSolver,
    exploitability,
    expected_value,
)

KUHN_NASH_VALUE = -1.0 / 18.0


def test_regrets_are_never_negative():
    solver = CFRPlusSolver(KuhnPoker())
    solver.train(500)
    for node in solver.infosets.values():
        assert np.all(node.regret_sum >= 0.0)


def test_vanilla_cfr_accumulates_negative_regret():
    """Sanity check that the clipping test above is actually discriminating."""
    solver = CFRSolver(KuhnPoker())
    solver.train(500)
    assert any(np.any(node.regret_sum < 0.0)
               for node in solver.infosets.values())


def test_cfr_plus_converges_to_kuhn_value():
    solver = CFRPlusSolver(KuhnPoker())
    solver.train(5000)
    game = KuhnPoker()
    avg = solver.average_strategy()
    assert expected_value(game, avg) == pytest.approx(KUHN_NASH_VALUE, abs=2e-3)
    assert exploitability(game, avg) < 5e-3


def test_cfr_plus_converges_faster_than_cfr():
    """At an equal, modest iteration budget CFR+ should be closer to Nash."""
    game = KuhnPoker()
    iters = 2000
    cfr = CFRSolver(game)
    cfr.train(iters)
    plus = CFRPlusSolver(game)
    plus.train(iters)
    expl_cfr = exploitability(game, cfr.average_strategy())
    expl_plus = exploitability(game, plus.average_strategy())
    assert expl_plus < expl_cfr


def test_average_strategy_normalized():
    solver = CFRPlusSolver(KuhnPoker())
    solver.train(200)
    for probs in solver.average_strategy().values():
        assert sum(probs.values()) == pytest.approx(1.0)
