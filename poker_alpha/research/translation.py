"""Exact evaluation of action-translation mappings (Phase II hypothesis H4).

A strategy is trained in an abstract river game whose bet menu is ``coarse``
(for both players). The real game lets the opponent use a ``fine`` menu. When
the opponent bets a size x that is not in the coarse menu, the strategy must
*translate* x to coarse sizes and play its strategy from the translated node.
Our own actions stay on the coarse menu.

Mappings (x, A < x < B as pot fractions, A/B neighbouring coarse sizes):
  ``nearest``          deterministic, nearest in pot fraction
  ``nearest_log``      deterministic, nearest in log(pot fraction) (what the
                       release's off-tree matcher does)
  ``pseudo_harmonic``  randomized: P(A) = (B - x)(1 + A) / ((B - A)(1 + x))
                       (Ganzfried & Sandholm, IJCAI 2013)
Sizes below the smallest / above the largest coarse size map to that size.

The translated strategy is a valid behaviour strategy in the fine game (at
each of our nodes, a mixture of the coarse strategies at the mapped nodes
weighted by mapping probabilities). Its exact worst case is the opponent's
best-response value in the fine game, computed with the vectorized solver.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .river_solver import RiverRangeSolver

MAPPINGS = ("nearest", "nearest_log", "pseudo_harmonic")


def _frac(tok: str) -> float:
    return int(tok[1:]) / 100.0


def map_size(x: float, coarse: Sequence[float], mapping: str) -> List[Tuple[float, float]]:
    """[(coarse size, probability)]"""
    cs = sorted(coarse)
    if x <= cs[0]:
        return [(cs[0], 1.0)]
    if x >= cs[-1]:
        return [(cs[-1], 1.0)]
    for a, b in zip(cs, cs[1:]):
        if a <= x <= b:
            if abs(x - a) < 1e-12:
                return [(a, 1.0)]
            if abs(x - b) < 1e-12:
                return [(b, 1.0)]
            if mapping == "nearest":
                return [(a, 1.0)] if x - a <= b - x else [(b, 1.0)]
            if mapping == "nearest_log":
                return [(a, 1.0)] if math.log(x / a) <= math.log(b / x) else [(b, 1.0)]
            if mapping == "pseudo_harmonic":
                pa = (b - x) * (1 + a) / ((b - a) * (1 + x))
                return [(a, pa), (b, 1 - pa)]
            raise ValueError(mapping)
    raise AssertionError


def translated_strategy(fine: RiverRangeSolver, coarse: RiverRangeSolver, coarse_lifted, us: int,
                        coarse_sizes: Sequence[float], mapping: str):
    """Per-hand strategy arrays on the FINE tree for player ``us``."""
    ctree = coarse.tree
    index = {n.history: i for i, n in enumerate(ctree.nodes)}
    out = [None] * len(fine.tree.nodes)
    for i in fine.tree.decision_nodes_of(us):
        node = fine.tree.nodes[i]
        # all mapped coarse histories with probabilities
        options = []
        for tok in node.history:
            if tok.startswith("b") and _frac(tok) not in coarse_sizes:
                options.append([(f"b{int(round(s * 100))}", p) for s, p in map_size(_frac(tok), coarse_sizes, mapping)])
            else:
                options.append([(tok, 1.0)])
        pol = np.zeros((fine.n[us], len(node.actions)))
        total = 0.0
        for combo in itertools.product(*options):
            hist = tuple(t for t, _ in combo)
            w = float(np.prod([p for _, p in combo]))
            j = index.get(hist)
            if j is None or ctree.nodes[j].player != us:
                continue                       # mapped history not reachable in the coarse tree
            cnode = ctree.nodes[j]
            cpol = coarse_lifted[j]
            for ai, a in enumerate(cnode.actions):
                if a in node.actions:
                    pol[:, node.actions.index(a)] += w * cpol[:, ai]
                else:                          # coarse action illegal here (stack/pot differ): treat as all-in
                    tgt = node.actions.index("a") if "a" in node.actions else node.actions.index("c")
                    pol[:, tgt] += w * cpol[:, ai]
            total += w
        if total <= 0:
            pol[:] = 0.0
            pol[:, node.actions.index("c")] = 1.0   # unreachable mapping: passive default (counted by caller)
        else:
            pol /= total
        out[i] = pol
    return out


def evaluate_mappings(board, r0, r1, coarse_sizes, fine_sizes, iters=1500, pot=10.0, stack=20.0, raise_cap=2):
    """Exact opponent BR value (worst case for us, lower is better) for every
    mapping and both seats, plus references."""
    res = {}
    fine_eq = RiverRangeSolver(board, r0, r1, pot=pot, stack=stack, bets=fine_sizes, raise_cap=raise_cap)
    fine_eq.train(iters)
    fl = fine_eq.lifted(fine_eq.average_strategy())
    coarse = RiverRangeSolver(board, r0, r1, pot=pot, stack=stack, bets=coarse_sizes, raise_cap=raise_cap)
    coarse.train(iters)
    cl = coarse.lifted(coarse.average_strategy())
    for us in (0, 1):
        opp = 1 - us
        # fine game where WE may only use coarse sizes (opponent keeps fine)
        menus = (coarse_sizes, fine_sizes) if us == 0 else (fine_sizes, coarse_sizes)
        ev = RiverRangeSolver(board, r0, r1, pot=pot, stack=stack, bets=menus[0], raise_cap=raise_cap, bets_p1=menus[1])
        # reference A: solve that game exactly (our best strategy restricted to the coarse menu, no translation)
        ref = RiverRangeSolver(board, r0, r1, pot=pot, stack=stack, bets=menus[0], raise_cap=raise_cap, bets_p1=menus[1])
        ref.train(iters)
        rl = ref.lifted(ref.average_strategy())
        r = {"restricted_menu_equilibrium": float(ev.best_response_values(rl, opp).sum())}
        for m in MAPPINGS:
            tl = translated_strategy(ev, coarse, cl, us, coarse_sizes, m)
            full = [x if x is not None else (rl[i] if rl[i] is not None else None) for i, x in enumerate(tl)]
            r[m] = float(ev.best_response_values(full, opp).sum())
        # coarse world: opponent ALSO restricted to coarse (no translation needed)
        cw = RiverRangeSolver(board, r0, r1, pot=pot, stack=stack, bets=coarse_sizes, raise_cap=raise_cap)
        r["coarse_world_no_offtree"] = float(cw.best_response_values(cl, opp).sum())
        res[f"us={us}"] = r
    return res
