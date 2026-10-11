"""Phase II: prioritized-sampling MCCFR accounting and unbiasedness checks."""

import numpy as np
import pytest

from poker_alpha.games.leduc import LeducPoker
from poker_alpha.research.active_compute import CountingGame, PrioritizedMCCFR, outcome_infosets
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.evaluation import exploitability


def test_uniform_prioritized_is_bit_identical_to_mccfr():
    a_game, b_game = CountingGame(LeducPoker()), CountingGame(LeducPoker())
    a = MCCFRSolver(a_game, seed=4)
    b = PrioritizedMCCFR(b_game, seed=4, signal="uniform", exact_game=b_game.inner)
    a.train(300)
    b.train(300)
    assert a_game.nodes == b_game.nodes
    assert set(a.infosets) == set(b.infosets)
    for k in a.infosets:
        assert np.array_equal(a.infosets[k].regret_sum, b.infosets[k].regret_sum)
        assert np.array_equal(a.infosets[k].strategy_sum, b.infosets[k].strategy_sum)


def test_outcome_infosets_cover_every_key():
    g = LeducPoker()
    per = outcome_infosets(g)
    assert len(per) == 30
    s = MCCFRSolver(g, seed=0)
    s.train(2000)
    allkeys = set().union(*map(set, per))
    assert set(s.infosets) <= allkeys


@pytest.mark.parametrize("signal", ["positive_regret", "scarcity", "entropy", "random_fixed"])
def test_prioritized_signals_remain_consistent_estimators(signal):
    g = LeducPoker()
    s = PrioritizedMCCFR(g, seed=1, signal=signal, eps=0.5, refresh=25)
    s.train(3000)
    early = exploitability(g, s.average_strategy())
    s.train(27000)
    late = exploitability(g, s.average_strategy())
    assert late < early and late < 0.2
    assert s.max_weight <= 1.0 / (1.0 - 0.5) + 1e-9
    assert np.isclose(s.q.sum(), 1.0)


def test_rejects_unknown_signal():
    with pytest.raises(ValueError):
        PrioritizedMCCFR(LeducPoker(), signal="magic")
