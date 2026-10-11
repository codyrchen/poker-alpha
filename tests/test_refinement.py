"""Phase II: adaptive abstraction refinement machinery."""

import sys
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.games.reduced_holdem import RiverSubgame
from poker_alpha.research.refinement import RiverRefinementProblem, choose_split, refine
from poker_alpha.research.river_solver import from_subgame

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from phase33_reduced_games import SUBGAMES, percentile_range  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    name, board, b0, b1 = SUBGAMES[3]
    g = RiverSubgame(board, percentile_range(board, b0, 12), percentile_range(board, b1, 12),
                     pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2, name=name)
    return RiverRefinementProblem(from_subgame(g), iterations=150)


def test_br_gain_signal_decomposes_exploitability(problem):
    cells = problem.uniform([3, 3])
    sv = problem.solve(cells)
    total = sum(float(x.sum()) for x in sv.signals["br_gain"])
    assert total == pytest.approx(2 * sv.exploitability, rel=1e-9)
    for sig in ("shared_action_regret", "positive_regret", "equity_spread"):
        for p in (0, 1):
            assert sv.signals[sig][p].shape == (3,)
            assert np.all(sv.signals[sig][p] >= -1e-12)


def test_split_keeps_strength_contiguity_and_adds_one_cell(problem):
    cells = problem.uniform([2, 2])
    new = problem.split(cells, 0, 1)
    assert new[0].max() == 2 and new[1].max() == 1
    assert sorted(np.bincount(new[0])) == sorted([6, 3, 3])
    order = problem.order[0]
    # members of each cell are contiguous in strength order
    labels = new[0][order]
    changes = int((np.diff(labels) != 0).sum())
    assert changes == 2
    assert problem.splittable(cells, 0, 0) and not problem.splittable([np.arange(12), np.arange(12)], 0, 0)


def test_choose_split_prefers_largest_signal(problem):
    cells = problem.uniform([3, 3])
    sv = problem.solve(cells)
    p, c = choose_split(problem, cells, sv, "br_gain", None)
    best = max((float(sv.signals["br_gain"][q][k]), q, k) for q in (0, 1) for k in range(3))
    assert (p, c) == (best[1], best[2])


def test_refine_reduces_error_and_matches_capacity(problem):
    cells = problem.uniform([2, 2])
    r = refine(problem, cells, splits=3, signal="br_gain")
    assert len(r["trace"]) == 3
    assert r["final"].cells_per_player[0] + r["final"].cells_per_player[1] == 7
    base = problem.solve(cells).exploitability
    assert r["final"].exploitability < base
    matched = problem.solve(problem.uniform(r["final"].cells_per_player))
    assert matched.num_infosets == r["final"].num_infosets


def test_random_refinement_is_seeded(problem):
    cells = problem.uniform([2, 2])
    a = refine(problem, cells, splits=2, signal="random", seed=7)
    b = refine(problem, cells, splits=2, signal="random", seed=7)
    assert [(t["player"], t["cell"]) for t in a["trace"]] == [(t["player"], t["cell"]) for t in b["trace"]]
