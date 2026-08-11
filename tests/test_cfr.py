"""Tests for regret matching, CFR convergence, and exploitability."""

import numpy as np
import pytest

from poker_alpha.games import KuhnPoker
from poker_alpha.solvers import (
    CFRSolver,
    best_response_value,
    exploitability,
    expected_value,
    regret_matching,
)
from poker_alpha.solvers.evaluation import Strategy

KUHN_NASH_VALUE = -1.0 / 18.0


def test_regret_matching_normalizes_positive_regret():
    s = regret_matching(np.array([3.0, 1.0, 0.0]))
    assert s == pytest.approx([0.75, 0.25, 0.0])
    assert s.sum() == pytest.approx(1.0)


def test_regret_matching_uniform_when_no_positive_regret():
    assert regret_matching(np.zeros(3)) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert regret_matching(np.array([-2.0, -1.0])) == pytest.approx([0.5, 0.5])


def test_average_strategy_sums_to_one():
    solver = CFRSolver(KuhnPoker())
    solver.train(200)
    for probs in solver.average_strategy().values():
        assert sum(probs.values()) == pytest.approx(1.0)


def _uniform_strategy(game) -> Strategy:
    strat: Strategy = {}

    def walk(state):
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for _, s in game.chance_outcomes(state):
                walk(s)
            return
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        strat[key] = {a: 1.0 / len(actions) for a in actions}
        for a in actions:
            walk(game.next_state(state, a))

    walk(game.root())
    return strat


def test_cfr_converges_to_kuhn_game_value():
    solver = CFRSolver(KuhnPoker())
    solver.train(20000)
    value = expected_value(KuhnPoker(), solver.average_strategy())
    assert value == pytest.approx(KUHN_NASH_VALUE, abs=2e-3)


def test_cfr_reduces_exploitability():
    game = KuhnPoker()
    uniform_expl = exploitability(game, _uniform_strategy(game))
    solver = CFRSolver(game)
    solver.train(20000)
    trained_expl = exploitability(game, solver.average_strategy())
    assert uniform_expl > 0.1
    assert trained_expl < 0.01
    assert trained_expl < uniform_expl


def test_exploitability_is_nonnegative_and_zero_at_equilibrium():
    game = KuhnPoker()
    solver = CFRSolver(game)
    solver.train(30000)
    expl = exploitability(game, solver.average_strategy())
    assert expl >= 0.0
    assert expl < 5e-3


def test_learned_strategy_respects_dominant_actions():
    """A converged Kuhn strategy never folds the King nor calls the Jack."""
    solver = CFRSolver(KuhnPoker())
    solver.train(20000)
    avg = solver.average_strategy()
    # King (2) facing a bet always calls.
    assert avg["2b"]["b"] > 0.99
    assert avg["2pb"]["b"] > 0.99
    # Jack (0) facing a bet always folds.
    assert avg["0b"]["p"] > 0.99
    assert avg["0pb"]["p"] > 0.99
    # Queen (1) never bets as the opening action (checks first).
    assert avg["1"]["p"] > 0.99


def test_best_response_recovers_game_value_against_equilibrium():
    game = KuhnPoker()
    solver = CFRSolver(game)
    solver.train(30000)
    avg = solver.average_strategy()
    # At (near) equilibrium, best-responding wins only ~the game value.
    br0 = best_response_value(game, avg, br_player=0)
    br1 = best_response_value(game, avg, br_player=1)
    assert br0 == pytest.approx(KUHN_NASH_VALUE, abs=5e-3)
    assert br1 == pytest.approx(-KUHN_NASH_VALUE, abs=5e-3)


def test_best_response_beats_a_weak_strategy():
    """Against the always-check/fold strategy, a best responder wins a lot."""
    game = KuhnPoker()
    weak = _uniform_strategy(game)
    for key in weak:
        weak[key] = {"p": 1.0, "b": 0.0}  # always pass/fold
    # Player 0 best-responding should do strictly better than the game value.
    br0 = best_response_value(game, weak, br_player=0)
    assert br0 > 0.1
