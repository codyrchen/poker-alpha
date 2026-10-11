"""Phase II: vectorized river solver parity with the generic tree solver."""

import sys
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.games.reduced_holdem import RiverSubgame
from poker_alpha.poker.cards import card_str
from poker_alpha.research.river_solver import (RiverRangeSolver, equal_count_cells,
                                               from_subgame, strength_order)
from poker_alpha.solvers.cfr_plus import CFRPlusSolver
from poker_alpha.solvers.evaluation import (best_response_value, expected_value,
                                            exploitability)

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from phase33_reduced_games import SUBGAMES, percentile_range  # noqa: E402


@pytest.fixture(scope="module")
def subgame():
    name, board, b0, b1 = SUBGAMES[1]
    r0 = percentile_range(board, b0, 7)
    r1 = percentile_range(board, b1, 7)
    return RiverSubgame(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2, name=name)


def _lifted_as_dict(v, strategy):
    lifted = v.lifted(strategy)
    out = {}
    for i in v.tree.decision_nodes:
        node = v.tree.nodes[i]
        p = node.player
        for hi, h in enumerate(v.hands[p]):
            key = f"{p}|{card_str(h[0])}{card_str(h[1])}|{'.'.join(node.history)}"
            out[key] = {a: float(lifted[i][hi, ai]) for ai, a in enumerate(node.actions)}
    return out


def test_cfr_plus_parity_with_generic_solver(subgame):
    g = subgame
    s = CFRPlusSolver(g)
    s.train(25)
    ref = s.average_strategy()
    v = from_subgame(g)
    res = v.solve(25)
    assert v.num_infosets() == len(s.infosets)
    assert res.exploitability == pytest.approx(exploitability(g, ref), abs=1e-10)
    assert res.ev_p0 == pytest.approx(expected_value(g, ref), abs=1e-10)
    mine = _lifted_as_dict(v, res.strategy)
    assert set(mine) == set(ref)
    for k in ref:
        for a in ref[k]:
            assert mine[k][a] == pytest.approx(ref[k][a], abs=1e-12)


def test_best_response_parity_on_abstract_strategy(subgame):
    g = subgame
    v = from_subgame(g)
    cells = [equal_count_cells(v, 0, 3), equal_count_cells(v, 1, 2)]
    v.set_cells(cells)
    assert v.num_infosets() < from_subgame(g).num_infosets()
    res = v.solve(30)
    lifted = _lifted_as_dict(v, res.strategy)
    # the lifted strategy evaluated by the generic exact evaluators
    assert res.exploitability == pytest.approx(exploitability(g, lifted), abs=1e-9)
    br0 = best_response_value(g, lifted, 0)
    br1 = best_response_value(g, lifted, 1)
    ev = expected_value(g, lifted)
    assert res.br_gain[0] == pytest.approx(br0 - ev, abs=1e-9)
    assert res.br_gain[1] == pytest.approx(br1 + ev, abs=1e-9)
    # hands in the same cell play the same policy
    for i in v.tree.decision_nodes:
        p = v.tree.nodes[i].player
        for c in range(v.K[p]):
            members = np.flatnonzero(cells[p] == c)
            rows = v.lifted(res.strategy)[i][members]
            assert np.allclose(rows, rows[0])


def test_raw_cells_match_default_and_strength_order(subgame):
    v = from_subgame(g := subgame)
    w = from_subgame(g, cells=[np.arange(v.n[0]), np.arange(v.n[1])])
    v.train(5)
    w.train(5)
    for i in v.tree.decision_nodes:
        assert np.allclose(v.regret[i], w.regret[i])
    order = strength_order(v, 0)
    vals = v.values[0][order]
    assert np.all(np.diff(vals) >= 0)
    cells = equal_count_cells(v, 0, 4)
    assert cells.max() == 3 and len(set(np.bincount(cells))) <= 2


def test_abstraction_never_beats_raw_floor_materially(subgame):
    g = subgame
    raw = from_subgame(g).solve(300)
    v = from_subgame(g)
    v.set_cells([equal_count_cells(v, 0, 2), equal_count_cells(v, 1, 2)])
    coarse = v.solve(300)
    assert raw.exploitability < 0.02
    assert coarse.exploitability > raw.exploitability


def test_anytime_interface_counts_iterations(subgame):
    v = from_subgame(subgame)
    n = v.train_for_seconds(0.05, min_iterations=3)
    assert n >= 3 and v.iterations == n


def test_invalid_inputs():
    with pytest.raises(ValueError):
        RiverRangeSolver(["Ks", "7d", "2c", "Qh", "4s"], [((0, 1), 1.0)], [((2, 3), 1.0)],
                         algorithm="bogus")
    with pytest.raises(ValueError):
        # Ks is on the board
        RiverRangeSolver(["Ks", "7d", "2c", "Qh", "4s"], [((50, 1), 1.0)], [((2, 3), 1.0)])
