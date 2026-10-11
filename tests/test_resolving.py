"""Phase II: decision-time subgame resolving (safe gadget vs unsafe)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.research.resolving import SubgameResolver, opponent_br_value, reach_at, resolve_all
from poker_alpha.research.river_solver import RiverRangeSolver, equal_count_cells

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from phase2_adaptive_abstraction import sample_range  # noqa: E402
from phase33_reduced_games import SUBGAMES  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    name, board, b0, b1 = SUBGAMES[0]
    rng = np.random.default_rng(7000)
    r0, r1 = sample_range(board, b0, 20, rng), sample_range(board, b1, 20, rng)

    def mk():
        return RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    bp = mk()
    bp.set_cells([equal_count_cells(bp, 0, 3), equal_count_cells(bp, 1, 3)])
    bp.train(300)
    return mk, bp.lifted(bp.average_strategy())


def _combined(mk, blue, roots, mode, iters):
    T = mk().tree
    comb = [None if x is None else x.copy() for x in blue]
    for r in roots:
        res = SubgameResolver(mk(), blue, r, mode)
        for _ in range(iters):
            res.iterate()
        part = res.combined()
        for i in res.nodes:
            if T.nodes[i].player == res.resolver:
                comb[i] = part[i]
    return comb


def test_safe_resolving_is_never_worse_than_blueprint(setup):
    mk, blue = setup
    ev = mk()
    roots = list(ev.tree.nodes[0].children)
    base = opponent_br_value(ev, blue, 1)
    safe = opponent_br_value(ev, _combined(mk, blue, roots, "safe", 2000), 1)
    assert safe <= base + 1e-4


def test_unsafe_resolving_can_be_exploited_more(setup):
    mk, blue = setup
    ev = mk()
    roots = list(ev.tree.nodes[0].children)
    base = opponent_br_value(ev, blue, 1)
    unsafe = opponent_br_value(ev, _combined(mk, blue, roots, "unsafe", 500), 1)
    assert unsafe > base   # the documented failure mode of unsafe subgame solving


def test_gadget_terminate_values_upper_bound_profile_values(setup):
    mk, blue = setup
    s = mk()
    res = SubgameResolver(s, blue, s.tree.nodes[0].children[0], "safe")
    assert np.all(res.t >= res.t_profile - 1e-12)


def test_reach_and_validation(setup):
    mk, blue = setup
    s = mk()
    r = reach_at(s, blue, s.tree.nodes[0].children[1])
    assert np.allclose(r[1], 1.0) and np.all(r[0] <= 1.0)
    with pytest.raises(ValueError):
        SubgameResolver(mk(), blue, 0, "reckless")
    term = next(i for i, n in enumerate(s.tree.nodes) if n.player < 0)
    with pytest.raises(ValueError):
        SubgameResolver(mk(), blue, term, "safe")
    out = resolve_all(mk, blue, list(s.tree.nodes[0].children), "safe", 20)
    assert len(out["iterations"]) == len(s.tree.nodes[0].children)
