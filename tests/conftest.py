"""Shared fixtures: a solved Leduc equilibrium is expensive, build it once.

Tests marked ``@pytest.mark.slow`` run by default and are skipped when the
environment variable ``POKERALPHA_SKIP_SLOW=1`` is set (fast inner loop).
"""

import os

import pytest

from poker_alpha.games import LeducPoker
from poker_alpha.solvers import CFRPlusSolver


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: expensive regression test (skip with "
                   "POKERALPHA_SKIP_SLOW=1)")
    config.addinivalue_line(
        "markers", "vision: needs the optional [vision] dependencies")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("POKERALPHA_SKIP_SLOW", "") not in ("1", "true", "yes"):
        return
    skip = pytest.mark.skip(reason="POKERALPHA_SKIP_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def leduc_game():
    return LeducPoker()


@pytest.fixture(scope="session")
def leduc_equilibrium(leduc_game):
    solver = CFRPlusSolver(leduc_game)
    solver.train(300)
    return solver.average_strategy()
