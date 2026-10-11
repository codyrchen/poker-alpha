"""Decision-time subgame resolving on exact river subgames (Phase II, Program 6).

RESEARCH / SANDBOX ONLY. Nothing here connects to any poker client.

Setting
-------
A *blueprint* strategy for the whole river subgame is precomputed in a coarse
card abstraction (``K`` strength cells per player, see
:mod:`poker_alpha.research.river_solver`). At decision time, for a subtree
rooted at a decision node ``r`` of the *resolver* player, PokerAlpha re-solves
the subtree at full hand resolution (one cell per hand) within a time budget
and replaces the blueprint there.

Two resolving constructions:

``unsafe``
    Both players' ranges at ``r`` are the blueprint reach probabilities; the
    subtree is solved as a game in its own right (Ganzfried & Sandholm 2015
    call this "unsafe" subgame solving). If the opponent deviates from the
    blueprint before ``r``, the resolved strategy can be *more* exploitable
    than the blueprint.

``safe`` (re-solve gadget, Burch, Johanson & Bowling, AAAI 2014, "Solving
Imperfect Information Games Using Decomposition")
    The opponent ``o`` enters the subtree through a gadget: for each of their
    hands ``h`` they may *terminate* and receive their blueprint
    counterfactual value ``t(h)`` at ``r``, or *follow* into the subtree. The
    resolver's range is the blueprint reach. At the gadget's equilibrium no
    opponent hand gains counterfactual value relative to the blueprint, so -
    to the extent the re-solve converges - the opponent's best response
    against the combined strategy does not exceed its best response against
    the blueprint (the CFR-D safety argument). With a finite budget the
    guarantee holds only approximately; this module *measures* it rather
    than assuming it.

Metric
------
The quality of the resolver's strategy is the value of the opponent's exact
best response against it in the raw (unabstracted) game, in chips. It is
compared with the same quantity for the blueprint and for the raw
equilibrium (a lower bound, up to solver error). Lower is better. Only the
resolver's strategy is changed, so the metric is a property of one player's
strategy and is semantically compatible across all arms.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .river_solver import RiverRangeSolver


def reach_at(solver: RiverRangeSolver, lifted, target: int) -> List[np.ndarray]:
    """Per-hand reach (own action probabilities only) of both players at node
    ``target`` under per-hand strategies ``lifted``."""
    path = []
    T = solver.tree

    def find(i, acc):
        if i == target:
            path.extend(acc)
            return True
        node = T.nodes[i]
        for a, c in enumerate(node.children):
            if find(c, acc + [(i, a)]):
                return True
        return False
    if not find(0, []):
        raise ValueError("target node not in tree")
    reach = [np.ones(solver.n[0]), np.ones(solver.n[1])]
    for i, a in path:
        p = T.nodes[i].player
        reach[p] = reach[p] * lifted[i][:, a]
    return reach


def subtree_nodes(solver: RiverRangeSolver, root: int) -> List[int]:
    out, stack = [], [root]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(solver.tree.nodes[i].children)
    return out


class SubgameResolver:
    """Re-solve the subtree under ``root`` at full hand resolution."""

    def __init__(self, raw: RiverRangeSolver, blueprint_lifted, root: int, mode: str = "safe") -> None:
        if mode not in ("safe", "unsafe"):
            raise ValueError("mode must be 'safe' or 'unsafe'")
        node = raw.tree.nodes[root]
        if node.player < 0:
            raise ValueError("root must be a decision node")
        if raw.K != raw.n:
            raise ValueError("resolver needs a raw (one cell per hand) solver")
        self.s = raw
        self.s._reset()
        self.root = root
        self.mode = mode
        self.resolver = node.player
        self.o = 1 - self.resolver
        self.blueprint = blueprint_lifted
        self.reach = reach_at(raw, blueprint_lifted, root)
        self.nodes = subtree_nodes(raw, root)
        # opponent's blueprint counterfactual values at the root (terminate payoffs)
        self.t = raw._profile_pass(root, self.o, self.reach[self.o], self.reach[self.resolver], blueprint_lifted)
        self.gadget_regret = np.zeros((raw.n[self.o], 2))   # [terminate, follow]
        self.iterations = 0

    def _gadget_follow(self) -> np.ndarray:
        return RiverRangeSolver._match(self.gadget_regret)[:, 1]

    def iterate(self) -> None:
        s = self.s
        weight = float(self.iterations + 1)
        for u in (0, 1):
            s._pending.clear()
            r_res = self.reach[self.resolver]
            if self.mode == "unsafe":
                r_o = self.reach[self.o]
            else:
                r_o = self._gadget_follow()
            reach_u = r_res if u == self.resolver else r_o
            reach_other = r_o if u == self.resolver else r_res
            v = s._pass(self.root, u, reach_u, reach_other, True, weight)
            for i, delta in s._pending.items():
                s.regret[i] = np.maximum(s.regret[i] + delta, 0.0)
            if self.mode == "safe" and u == self.o:
                # gadget: per-hand choice between blueprint cfv and following
                g = RiverRangeSolver._match(self.gadget_regret)
                vals = np.stack([self.t, v], axis=1)
                node_v = (g * vals).sum(axis=1)
                self.gadget_regret = np.maximum(self.gadget_regret + (vals - node_v[:, None]), 0.0)
        self.iterations += 1

    def solve_for_ms(self, budget_ms: float, min_iterations: int = 1) -> int:
        """Anytime interface: iterate until ``budget_ms`` elapse."""
        t0 = time.perf_counter()
        n = 0
        while n < min_iterations or (time.perf_counter() - t0) * 1000.0 < budget_ms:
            self.iterate()
            n += 1
        return n

    def combined(self) -> List[Optional[np.ndarray]]:
        """Blueprint with the resolver's subtree strategy replaced."""
        avg = self.s.average_strategy()
        out = [None if x is None else x.copy() for x in self.blueprint]
        for i in self.nodes:
            node = self.s.tree.nodes[i]
            if node.player == self.resolver:
                ss = self.s.strategy_sum[i]
                tot = ss.sum(axis=1, keepdims=True)
                # hands with zero blueprint reach keep the blueprint action
                out[i] = np.where(tot > 0, avg[i], self.blueprint[i])
        return out


def opponent_br_value(raw: RiverRangeSolver, lifted, resolver: int) -> float:
    """Exact value (chips) of the opponent's best response against the
    resolver's strategy in ``lifted`` (only the resolver's nodes are read)."""
    return float(raw.best_response_values(lifted, 1 - resolver).sum())


def resolve_all(raw_factory, blueprint_lifted, roots: Sequence[int], mode: str, budget_ms: float) -> Dict:
    """Resolve several disjoint subtrees (same resolver), splitting the
    budget equally; returns the combined strategy and iteration counts."""
    combined = [None if x is None else x.copy() for x in blueprint_lifted]
    iters = []
    for r in roots:
        res = SubgameResolver(raw_factory(), blueprint_lifted, r, mode)
        iters.append(res.solve_for_ms(budget_ms / len(roots)))
        part = res.combined()
        for i in res.nodes:
            if res.s.tree.nodes[i].player == res.resolver:
                combined[i] = part[i]
    return {"lifted": combined, "iterations": iters}
