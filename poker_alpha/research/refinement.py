"""Adaptive abstraction refinement (Phase II, Program 1).

Research question: can *measured* strategic error decide where additional
abstraction capacity should go, and does that beat spending the same
capacity uniformly or at random?

The machinery is generic over a :class:`RefinementProblem`: anything that
can (a) solve an abstract game for a given partition of each player's
private information into cells, (b) report exact error of the lifted
strategy, (c) report per-cell *failure signals*, and (d) split a cell.
The river range solver provides the concrete problem used in the Phase II
study (:class:`RiverRefinementProblem`).

Failure signals (all exact, per player and cell, in chips unless noted)
-----------------------------------------------------------------------
``br_gain``
    Sum over member hands of (best-response cf value - profile cf value)
    of the lifted strategy. The per-player sum of this signal over all
    cells is exactly that player's best-response gain, so the two players'
    sums add to twice the exploitability: the signal *decomposes* the
    measured error by cell. It is the "exact local exploitability" signal.
``shared_action_regret``
    One-step cost of sharing a pure action: at each of the player's nodes,
    sum over member hands of ``max_a V(h, a) - V(h, a*)`` where ``V`` are the
    hands' cf action values under the profile and ``a*`` is the single
    action that is best for the cell as a whole. Zero iff all member hands
    agree on the locally best action.
``equity_spread``
    Max minus min static equity (vs the opponent's prior range) among
    member hands. Structural; needs no solve.
``positive_regret``
    Sum of the solver's final cumulative positive regret over the cell's
    information sets (CFR+ clipped regret). Solver-internal.
``random``
    Uniform random cell (control).

Literature: refining abstractions using regret-like signals is the idea of
Brown & Sandholm, "Simultaneous Abstraction and Equilibrium Finding in
Games" (IJCAI 2015) and of potential-aware/imperfect-recall abstraction
work (Ganzfried & Sandholm 2014); exact-error-guided splitting on a small
exact domain, with capacity-matched uniform and random controls, is what
this module measures. Nothing here is claimed as novel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .river_solver import RiverRangeSolver, equal_count_cells, strength_order

SIGNALS = ("br_gain", "shared_action_regret", "equity_spread", "positive_regret", "random")


@dataclass
class RefinementSolve:
    exploitability: float
    num_infosets: int
    cells_per_player: List[int]
    seconds: float
    iterations: int
    signals: Dict[str, List[np.ndarray]] = field(default_factory=dict)  # name -> [per-player (K_p,)]
    extra: Dict[str, object] = field(default_factory=dict)


class RefinementProblem(Protocol):
    def solve(self, cells: Sequence[np.ndarray]) -> RefinementSolve: ...
    def split(self, cells: Sequence[np.ndarray], player: int, cell: int) -> List[np.ndarray]: ...
    def splittable(self, cells: Sequence[np.ndarray], player: int, cell: int) -> bool: ...
    def uniform(self, cells_per_player: Sequence[int]) -> List[np.ndarray]: ...


class RiverRefinementProblem:
    """River subgame with per-player strength-ordered hand partitions."""

    def __init__(self, solver: RiverRangeSolver, iterations: int) -> None:
        self.solver = solver
        self.iterations = int(iterations)
        self.order = [strength_order(solver, p) for p in (0, 1)]
        self.rank_in_order = [np.empty(solver.n[p], dtype=np.int64) for p in (0, 1)]
        for p in (0, 1):
            self.rank_in_order[p][self.order[p]] = np.arange(solver.n[p])
        # static equity vs the opponent's prior range
        self.equity = []
        for p in (0, 1):
            Pm = solver.P[p]
            Sm = solver.S[p]
            tot = Pm.sum(axis=1)
            eq = np.where(tot > 0, (0.5 * Pm.sum(axis=1) + 0.5 * Sm.sum(axis=1)) / np.where(tot > 0, tot, 1), 0.5)
            self.equity.append(eq)

    def uniform(self, cells_per_player: Sequence[int]) -> List[np.ndarray]:
        return [equal_count_cells(self.solver, p, int(cells_per_player[p])) for p in (0, 1)]

    def splittable(self, cells, player, cell) -> bool:
        return int((cells[player] == cell).sum()) >= 2

    def split(self, cells, player, cell) -> List[np.ndarray]:
        new = [np.array(c, copy=True) for c in cells]
        members = np.flatnonzero(new[player] == cell)
        members = members[np.argsort(self.rank_in_order[player][members])]
        half = len(members) // 2
        new[player][members[half:]] = int(new[player].max()) + 1
        return new

    def solve(self, cells) -> RefinementSolve:
        s = self.solver
        s.set_cells(cells)
        t0 = time.perf_counter()
        s.train(self.iterations)
        strat = s.average_strategy()
        ev = s.evaluate(strat)
        secs = time.perf_counter() - t0
        lifted = s.lifted(strat)
        K = list(s.K)
        signals: Dict[str, List[np.ndarray]] = {}
        # br_gain per cell
        signals["br_gain"] = [np.bincount(s.cells[p], weights=ev["per_hand_br_gain"][p], minlength=K[p])
                              for p in (0, 1)]
        # shared-action one-step regret and positive regret
        sar = [np.zeros(K[p]) for p in (0, 1)]
        posr = [np.zeros(K[p]) for p in (0, 1)]
        for p in (0, 1):
            values: Dict[int, np.ndarray] = {}
            s._profile_pass(0, p, np.ones(s.n[p]), np.ones(s.n[1 - p]), lifted, values_out=values)
            for i, vals in values.items():          # vals: (A, n_p) cf action values
                best_per_hand = vals.max(axis=0)
                cell_sum = np.stack([np.bincount(s.cells[p], weights=vals[a], minlength=K[p])
                                     for a in range(vals.shape[0])])   # (A, K)
                a_star = cell_sum.argmax(axis=0)                        # (K,)
                v_star = vals[a_star[s.cells[p]], np.arange(s.n[p])]
                sar[p] += np.bincount(s.cells[p], weights=best_per_hand - v_star, minlength=K[p])
            for i in s.tree.decision_nodes_of(p):
                posr[p] += np.maximum(s.regret[i], 0.0).sum(axis=1)
        signals["shared_action_regret"] = sar
        signals["positive_regret"] = posr
        spread = []
        for p in (0, 1):
            mx = np.full(K[p], -np.inf)
            mn = np.full(K[p], np.inf)
            np.maximum.at(mx, s.cells[p], self.equity[p])
            np.minimum.at(mn, s.cells[p], self.equity[p])
            spread.append(mx - mn)
        signals["equity_spread"] = spread
        return RefinementSolve(exploitability=float(ev["exploitability"]), num_infosets=s.num_infosets(),
                               cells_per_player=K, seconds=secs, iterations=s.iterations,
                               signals=signals, extra={"ev_p0": ev["ev_p0"], "br_gain": ev["br_gain"],
                                                       "node_visits": s.node_visits})


# --------------------------------------------------------------------------
# Refinement policies
# --------------------------------------------------------------------------

def choose_split(problem: RefinementProblem, cells, solve: RefinementSolve, signal: str,
                 rng: Optional[np.random.Generator]) -> Optional[tuple]:
    """(player, cell) with the largest signal among splittable cells."""
    cands = []
    for p in (0, 1):
        K = solve.cells_per_player[p]
        for c in range(K):
            if problem.splittable(cells, p, c):
                if signal == "random":
                    score = float(rng.random())
                else:
                    score = float(solve.signals[signal][p][c])
                cands.append((score, p, c))
    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    return cands[0][1], cands[0][2]


def refine(problem: RefinementProblem, cells, splits: int, signal: str,
           seed: int = 0, oracle: bool = False) -> Dict[str, object]:
    """Greedy sequential refinement: ``splits`` rounds of (solve, pick, split).

    ``oracle=True`` tries every splittable cell each round and keeps the one
    whose re-solve gives the lowest exact exploitability (an upper bound on
    what one-step greedy selection could achieve; expensive)."""
    rng = np.random.default_rng(seed)
    cells = [np.array(c, copy=True) for c in cells]
    trace = []
    cur = problem.solve(cells)
    t_start = time.perf_counter()
    total_solve_seconds = cur.seconds
    for r in range(splits):
        if oracle:
            best = None
            for p in (0, 1):
                for c in range(cur.cells_per_player[p]):
                    if not problem.splittable(cells, p, c):
                        continue
                    cand = problem.split(cells, p, c)
                    sv = problem.solve(cand)
                    total_solve_seconds += sv.seconds
                    if best is None or sv.exploitability < best[0]:
                        best = (sv.exploitability, p, c, cand, sv)
            if best is None:
                break
            _, p, c, cells, nxt = best
        else:
            pick = choose_split(problem, cells, cur, signal, rng)
            if pick is None:
                break
            p, c = pick
            cells = problem.split(cells, p, c)
            nxt = problem.solve(cells)
            total_solve_seconds += nxt.seconds
        trace.append({"round": r + 1, "player": int(p), "cell": int(c),
                      "exploitability_before": cur.exploitability, "exploitability_after": nxt.exploitability,
                      "num_infosets": nxt.num_infosets, "cells_per_player": nxt.cells_per_player})
        cur = nxt
    return {"cells": cells, "final": cur, "trace": trace,
            "wall_seconds": time.perf_counter() - t_start, "solve_seconds": total_solve_seconds}
