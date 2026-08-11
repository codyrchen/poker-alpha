"""Shared fixtures: a solved Leduc equilibrium is expensive, build it once."""

import pytest

from poker_alpha.games import LeducPoker
from poker_alpha.solvers import CFRPlusSolver


@pytest.fixture(scope="session")
def leduc_game():
    return LeducPoker()


@pytest.fixture(scope="session")
def leduc_equilibrium(leduc_game):
    solver = CFRPlusSolver(leduc_game)
    solver.train(300)
    return solver.average_strategy()
