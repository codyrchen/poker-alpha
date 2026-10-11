"""Phase II H4/H13: translation mappings and nested resolving after off-tree bets."""

import sys
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.research.translation import evaluate_mappings, evaluate_nested_resolving, map_size

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from phase2_adaptive_abstraction import sample_range  # noqa: E402
from phase33_reduced_games import SUBGAMES  # noqa: E402


def test_map_size_rules():
    assert map_size(0.2, (0.5, 1.0), "nearest") == [(0.5, 1.0)]
    assert map_size(3.0, (0.5, 1.0), "pseudo_harmonic") == [(1.0, 1.0)]
    assert map_size(0.6, (0.5, 1.0), "nearest") == [(0.5, 1.0)]
    assert map_size(0.75, (0.5, 1.0), "nearest_log") == [(1.0, 1.0)]       # log(1.5) > log(4/3)
    ph = dict(map_size(0.75, (0.5, 1.0), "pseudo_harmonic"))
    assert ph[0.5] == pytest.approx(0.25 * 1.5 / (0.5 * 1.75)) and sum(ph.values()) == pytest.approx(1.0)
    assert map_size(1.0, (0.5, 1.0), "pseudo_harmonic") == [(1.0, 1.0)]


@pytest.fixture(scope="module")
def ranges():
    name, board, b0, b1 = SUBGAMES[0]
    rng = np.random.default_rng(1)
    return board, sample_range(board, b0, 12, rng), sample_range(board, b1, 12, rng)


def test_translation_never_beats_the_restricted_equilibrium(ranges):
    board, r0, r1 = ranges
    res = evaluate_mappings(board, r0, r1, (0.5, 1.0), (0.25, 0.5, 0.75, 1.0, 1.5), iters=300)
    for seat, r in res.items():
        for m in ("nearest", "nearest_log", "pseudo_harmonic"):
            assert r[m] >= r["restricted_menu_equilibrium"] - 0.02, (seat, m)


def test_nested_resolving_runs_and_unsafe_is_worse(ranges):
    board, r0, r1 = ranges
    r = evaluate_nested_resolving(board, r0, r1, (0.33, 0.75, 1.5), (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                                  iters=300, resolve_iters=100)
    assert r["resolve_roots"] > 0
    assert r["resolve:unsafe"] > r["translate:pseudo_harmonic"]
    assert np.isfinite(r["resolve:safe"])
