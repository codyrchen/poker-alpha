"""Vectorized exact range-vs-range river solver with card-bucket abstraction.

Why this exists
---------------
The generic tree solvers (:class:`~poker_alpha.solvers.cfr_plus.CFRPlusSolver`
over :class:`~poker_alpha.games.reduced_holdem.RiverSubgame`) enumerate every
``(hand0, hand1)`` deal as a separate history and take ~3 minutes per exact
solve of a 14x14-combo river subgame. Controlled abstraction research needs
thousands of exact solves (seeds x boards x partitions x controls), so this
module solves the *same game* with hand-indexed vectors: one betting tree,
strategies and reaches as arrays over each player's range, showdown and fold
payoffs as range-vs-range matrices.

Semantics are identical to ``RiverSubgame`` (same tree, same chip utilities,
same CFR+ update order, same averaging) and are pinned by a parity test
against the generic solver.

Abstraction
-----------
Each player's hands are partitioned into *cells* (``cells[p][h]`` = cell id
of hand ``h``). An abstract information set is ``(betting node, cell)``:
regrets and average-strategy sums are accumulated per cell (summed over the
member hands' counterfactual regrets, exactly as the generic solver sums
over the histories of an information set), and every member hand plays the
cell's strategy. With one cell per hand the solver is the raw (perfect
information set) game. Because the river has a single street and no further
chance, a partition of the private hand is a perfect-recall abstraction, so
CFR's guarantees hold in the abstract game; the *abstraction error* is
measured by lifting the abstract strategy to raw information sets (identity:
each hand plays its cell's policy) and computing exact exploitability in the
raw game.

Exploitability here is the average best-response gain over both players,
in chips (the subgame's pot/stack units), exactly as
:func:`poker_alpha.solvers.evaluation.exploitability`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..poker.cards import card_str, codes
from ..poker.evaluator import evaluate_best_codes

Hand = Tuple[int, int]


# --------------------------------------------------------------------------
# Betting tree
# --------------------------------------------------------------------------

@dataclass
class Node:
    history: Tuple[str, ...]
    player: int                      # -1 for terminal
    actions: List[str] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    terminal: Optional[str] = None   # "fold" | "showdown"
    contrib: Tuple[float, float] = (0.0, 0.0)
    folder: int = -1


class RiverTree:
    """Betting tree of ``RiverSubgame`` (OOP = player 0 acts first)."""

    def __init__(self, pot: float, stack: float, bets: Sequence[float] = (0.75,),
                 raise_cap: int = 2, oop_only_bets: bool = False,
                 bets_p1: Optional[Sequence[float]] = None) -> None:
        self.pot, self.stack = float(pot), float(stack)
        self.bets = tuple(float(b) for b in bets)
        # optional per-player menus (action-abstraction studies); None = shared
        self.menus = (self.bets, self.bets if bets_p1 is None else tuple(float(b) for b in bets_p1))
        self.raise_cap = int(raise_cap)
        self.oop_only_bets = oop_only_bets
        self.nodes: List[Node] = []
        self._build(())

    def _tok(self, frac: float) -> str:
        return f"b{int(round(frac * 100))}"

    def _state(self, history):
        c = [0.0, 0.0]
        level, p, n = 0.0, 0, 0
        for a in history:
            if a == "c":
                c[p] = level
            elif a == "a":
                level = self.stack
                c[p] = level
                n += 1
            elif a.startswith("b"):
                frac = int(a[1:]) / 100.0
                owe = level - c[p]
                pot_now = self.pot + c[0] + c[1]
                level = c[p] + owe + frac * (pot_now + owe)
                c[p] = level
                n += 1
            p = 1 - p
        return c, level, p, n

    def _is_terminal(self, h):
        if not h:
            return False
        if h[-1] == "f":
            return True
        return h[-1] == "c" and len(h) >= 2

    def _legal(self, h):
        c, level, p, n = self._state(h)
        facing = level > c[p] + 1e-9
        acts = (["f"] if facing else []) + ["c"]
        if self.oop_only_bets and p == 1:
            return acts
        if n < self.raise_cap and level < self.stack - 1e-9:
            pot_now = self.pot + c[0] + c[1]
            owe = level - c[p]
            for frac in self.menus[p]:
                to = c[p] + owe + frac * (pot_now + owe)
                if to < self.stack - 1e-9:
                    acts.append(self._tok(frac))
            acts.append("a")
        return acts

    def _build(self, h) -> int:
        idx = len(self.nodes)
        c, level, p, n = self._state(h)
        if self._is_terminal(h):
            node = Node(h, -1, contrib=(c[0], c[1]))
            node.terminal = "fold" if h[-1] == "f" else "showdown"
            node.folder = 1 - p if h[-1] == "f" else -1
            self.nodes.append(node)
            return idx
        node = Node(h, p, actions=self._legal(h), contrib=(c[0], c[1]))
        self.nodes.append(node)
        for a in node.actions:
            node.children.append(self._build(h + (a,)))
        return idx

    @property
    def decision_nodes(self) -> List[int]:
        return [i for i, n in enumerate(self.nodes) if n.player >= 0]

    def decision_nodes_of(self, player: int) -> List[int]:
        return [i for i, n in enumerate(self.nodes) if n.player == player]

    def num_infosets(self, cells_per_player: Sequence[int]) -> int:
        return sum(cells_per_player[n.player] for n in self.nodes if n.player >= 0)


# --------------------------------------------------------------------------
# Range-vs-range solver
# --------------------------------------------------------------------------

@dataclass
class SolveResult:
    iterations: int
    seconds: float
    strategy: List[Optional[np.ndarray]]  # per node: (K_p, A) average strategy, None for terminals
    exploitability: float                 # raw-game exploitability of the lifted strategy (chips)
    br_gain: Tuple[float, float]          # per-player best-response gain (br_p - profile value_p)
    ev_p0: float
    cf_regret: List[Optional[np.ndarray]]   # final cumulative (clipped) regret per node (K_p, A)
    node_visits: int                      # vectorized node evaluations (traversals x nodes)


class RiverRangeSolver:
    """Exact CFR / CFR+ over a river subgame with per-player hand partitions."""

    def __init__(self, board: Sequence, range0: Sequence[Tuple[Hand, float]],
                 range1: Sequence[Tuple[Hand, float]], pot: float = 10.0, stack: float = 20.0,
                 bets: Sequence[float] = (0.75,), raise_cap: int = 2, oop_only_bets: bool = False,
                 cells: Optional[Sequence[Optional[np.ndarray]]] = None,
                 algorithm: str = "cfr+", bets_p1: Optional[Sequence[float]] = None) -> None:
        self.board = tuple(codes(board))
        self.tree = RiverTree(pot, stack, bets, raise_cap, oop_only_bets, bets_p1)
        self.hands = [[tuple(sorted(h)) for h, _ in range0], [tuple(sorted(h)) for h, _ in range1]]
        self.weights = [np.array([w for _, w in range0], dtype=np.float64),
                        np.array([w for _, w in range1], dtype=np.float64)]
        self.n = [len(self.hands[0]), len(self.hands[1])]
        if algorithm not in ("cfr+", "cfr"):
            raise ValueError("algorithm must be 'cfr+' or 'cfr'")
        self.algorithm = algorithm
        bset = set(self.board)
        for p in (0, 1):
            for h in self.hands[p]:
                if h[0] in bset or h[1] in bset:
                    raise ValueError(f"hand {card_str(h[0])}{card_str(h[1])} collides with the board")
        # compatibility mask and showdown sign, oriented (p0 x p1)
        B = np.ones((self.n[0], self.n[1]), dtype=np.float64)
        for i, h0 in enumerate(self.hands[0]):
            s0 = set(h0)
            for j, h1 in enumerate(self.hands[1]):
                if s0 & set(h1):
                    B[i, j] = 0.0
        self.values = [np.array([self._rank(h) for h in self.hands[p]]) for p in (0, 1)]
        v0 = self.values[0][:, None]
        v1 = self.values[1][None, :]
        S = np.where(v0 > v1, 1.0, np.where(v0 < v1, -1.0, 0.0)) * B
        tot = float((B * self.weights[0][:, None] * self.weights[1][None, :]).sum())
        if tot <= 0:
            raise ValueError("no compatible deals")
        self.total_weight = tot
        # chance matrices from each player's perspective: P_p[h, h'] = P(deal h, h')
        P = B * self.weights[0][:, None] * self.weights[1][None, :] / tot
        self.P = [P, P.T.copy()]
        self.S = [S * P, (-S * P).T.copy()]   # showdown sign x chance prob, from p's view
        self.deal_count = [B.sum(axis=1), B.sum(axis=0)]  # # compatible opponent deals per hand
        self.cells = [None, None]
        self.K = [0, 0]
        self.set_cells(cells)
        self._reset()

    # -- hand ranking ---------------------------------------------------
    _rank_cache: Dict[Tuple, int] = {}

    def _rank(self, hand: Hand) -> int:
        """Totally ordered strength as an int (category / tiebreakers)."""
        key = (self.board, hand)
        v = self._rank_cache.get(key)
        if v is None:
            hv = evaluate_best_codes(list(hand) + list(self.board))
            # encode tuple lexicographically into an int (base 16, up to 6 slots)
            v = 0
            for x in tuple(hv) + (0,) * (6 - len(hv)):
                v = v * 16 + int(x)
            self._rank_cache[key] = v
        return v

    # -- abstraction ----------------------------------------------------
    def set_cells(self, cells: Optional[Sequence[Optional[np.ndarray]]]) -> None:
        for p in (0, 1):
            c = None if cells is None else cells[p]
            if c is None:
                c = np.arange(self.n[p])
            c = np.asarray(c, dtype=np.int64)
            if c.shape != (self.n[p],):
                raise ValueError("cells must have one entry per hand")
            self.cells[p] = c
            self.K[p] = int(c.max()) + 1 if len(c) else 0
        self._reset()

    def num_infosets(self) -> int:
        return self.tree.num_infosets(self.K)

    def _reset(self) -> None:
        T = self.tree
        self.regret = [None] * len(T.nodes)
        self.strategy_sum = [None] * len(T.nodes)
        for i in T.decision_nodes:
            n = T.nodes[i]
            self.regret[i] = np.zeros((self.K[n.player], len(n.actions)))
            self.strategy_sum[i] = np.zeros((self.K[n.player], len(n.actions)))
        self.iterations = 0
        self.node_visits = 0
        self._pending: Dict[int, np.ndarray] = {}

    # -- strategies -----------------------------------------------------
    @staticmethod
    def _match(regret: np.ndarray) -> np.ndarray:
        pos = np.maximum(regret, 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        out = np.where(tot > 0.0, pos / np.where(tot > 0.0, tot, 1.0), 1.0 / regret.shape[1])
        return out

    def current_strategy(self, i: int) -> np.ndarray:
        return self._match(self.regret[i])

    def average_strategy(self) -> List[Optional[np.ndarray]]:
        out: List[Optional[np.ndarray]] = [None] * len(self.tree.nodes)
        for i in self.tree.decision_nodes:
            s = self.strategy_sum[i]
            tot = s.sum(axis=1, keepdims=True)
            out[i] = np.where(tot > 0.0, s / np.where(tot > 0.0, tot, 1.0), 1.0 / s.shape[1])
        return out

    def lifted(self, strategy: Sequence[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """Per-hand strategy arrays (n_p, A) for each node."""
        out: List[Optional[np.ndarray]] = [None] * len(self.tree.nodes)
        for i in self.tree.decision_nodes:
            p = self.tree.nodes[i].player
            out[i] = strategy[i][self.cells[p]]
        return out

    # -- terminals ------------------------------------------------------
    def _terminal_value(self, i: int, u: int, reach_o: np.ndarray) -> np.ndarray:
        """Counterfactual value vector (n_u,) for player ``u`` at terminal ``i``:
        sum over opponent hands of chance prob x opponent reach x utility."""
        node = self.tree.nodes[i]
        half = self.tree.pot / 2.0
        c0, c1 = node.contrib
        if node.terminal == "fold":
            # utility for player 0
            u0 = -(half + c0) if node.folder == 0 else (half + c1)
            uu = u0 if u == 0 else -u0
            return uu * (self.P[u] @ reach_o)
        # showdown: contributions equal; winner takes half + opponent's contrib
        amount = half + (c1 if u == 0 else c0)
        return amount * (self.S[u] @ reach_o)

    # -- CFR traversals -------------------------------------------------
    def _pass(self, i: int, u: int, reach_u: np.ndarray, reach_o: np.ndarray,
              update: bool, weight: float) -> np.ndarray:
        self.node_visits += 1
        node = self.tree.nodes[i]
        if node.player < 0:
            return self._terminal_value(i, u, reach_o)
        p = node.player
        sigma = self.current_strategy(i)          # (K_p, A)
        sig_h = sigma[self.cells[p]]              # (n_p, A)
        if p == u:
            vals = np.empty((len(node.actions), self.n[u]))
            for a, child in enumerate(node.children):
                vals[a] = self._pass(child, u, reach_u * sig_h[:, a], reach_o, update, weight)
            node_val = (sig_h.T * vals).sum(axis=0)
            if update:
                delta = vals - node_val[None, :]       # (A, n_u) per-hand cf regret
                agg = np.zeros((self.K[u], len(node.actions)))
                for a in range(len(node.actions)):
                    agg[:, a] = np.bincount(self.cells[u], weights=delta[a], minlength=self.K[u])
                pend = self._pending.get(i)
                if pend is None:
                    self._pending[i] = agg
                else:
                    pend += agg
                own = np.bincount(self.cells[u], weights=reach_u * self.deal_count[u],
                                  minlength=self.K[u])
                self.strategy_sum[i] += weight * own[:, None] * sigma
            return node_val
        v = np.zeros(self.n[u])
        for a, child in enumerate(node.children):
            v += self._pass(child, u, reach_u, reach_o * sig_h[:, a], update, weight)
        return v

    def _traverse_for(self, u: int) -> None:
        self._pending.clear()
        weight = float(self.iterations + 1) if self.algorithm == "cfr+" else 1.0
        self._pass(0, u, np.ones(self.n[u]), np.ones(self.n[1 - u]), True, weight)
        for i, delta in self._pending.items():
            if self.algorithm == "cfr+":
                self.regret[i] = np.maximum(self.regret[i] + delta, 0.0)
            else:
                self.regret[i] = self.regret[i] + delta

    def iterate(self) -> None:
        if self.algorithm == "cfr+":
            self._traverse_for(0)
            self._traverse_for(1)
        else:
            # vanilla CFR: simultaneous update from the same strategy profile
            saved = [None if r is None else r.copy() for r in self.regret]
            self._pending.clear()
            self._pass(0, 0, np.ones(self.n[0]), np.ones(self.n[1]), True, 1.0)
            pend0 = dict(self._pending)
            self.regret = saved
            self._pending = {}
            self._pass(0, 1, np.ones(self.n[1]), np.ones(self.n[0]), True, 1.0)
            for i, d in self._pending.items():
                self.regret[i] = self.regret[i] + d
            for i, d in pend0.items():
                self.regret[i] = self.regret[i] + d
        self.iterations += 1

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iterate()

    def train_for_seconds(self, seconds: float, min_iterations: int = 1) -> int:
        """Anytime interface: iterate until ``seconds`` elapse. Returns the
        number of iterations performed."""
        t0 = time.perf_counter()
        done = 0
        while done < min_iterations or time.perf_counter() - t0 < seconds:
            self.iterate()
            done += 1
        return done

    # -- evaluation -----------------------------------------------------
    def _br_pass(self, i: int, u: int, reach_o: np.ndarray, lifted, policy_out=None,
                 values_out=None) -> np.ndarray:
        """Best-response cf value vector (n_u,) for player u vs lifted opp strategy."""
        node = self.tree.nodes[i]
        if node.player < 0:
            return self._terminal_value(i, u, reach_o)
        if node.player == u:
            vals = np.stack([self._br_pass(c, u, reach_o, lifted, policy_out, values_out)
                             for c in node.children])   # (A, n_u)
            if values_out is not None:
                values_out[i] = vals
            if policy_out is not None:
                policy_out[i] = vals.argmax(axis=0)
            return vals.max(axis=0)
        sig = lifted[i]
        v = np.zeros(self.n[u])
        for a, c in enumerate(node.children):
            v += self._br_pass(c, u, reach_o * sig[:, a], lifted, policy_out, values_out)
        return v

    def _profile_pass(self, i: int, u: int, reach_u: np.ndarray, reach_o: np.ndarray, lifted,
                      values_out=None) -> np.ndarray:
        """Per-hand cf value (n_u,) of following the profile (both players)."""
        node = self.tree.nodes[i]
        if node.player < 0:
            return self._terminal_value(i, u, reach_o)
        sig = lifted[i]
        if node.player == u:
            vals = np.stack([self._profile_pass(c, u, reach_u * sig[:, a], reach_o, lifted, values_out)
                             for a, c in enumerate(node.children)])
            if values_out is not None:
                values_out[i] = vals
            return (sig.T * vals).sum(axis=0)
        v = np.zeros(self.n[u])
        for a, c in enumerate(node.children):
            v += self._profile_pass(c, u, reach_u, reach_o * sig[:, a], lifted, values_out)
        return v

    def best_response_values(self, lifted, u: int) -> np.ndarray:
        """Per-hand best-response cf value for player u (chance-weighted, sums
        to the BR value)."""
        return self._br_pass(0, u, np.ones(self.n[1 - u]), lifted)

    def profile_values(self, lifted, u: int) -> np.ndarray:
        return self._profile_pass(0, u, np.ones(self.n[u]), np.ones(self.n[1 - u]), lifted)

    def evaluate(self, strategy=None) -> Dict[str, object]:
        """Exact raw-game exploitability / EV of a (cell) strategy."""
        if strategy is None:
            strategy = self.average_strategy()
        lifted = self.lifted(strategy)
        br = [self.best_response_values(lifted, u) for u in (0, 1)]
        pv = [self.profile_values(lifted, u) for u in (0, 1)]
        br_val = [float(b.sum()) for b in br]
        ev = [float(v.sum()) for v in pv]
        gain = (br_val[0] - ev[0], br_val[1] - ev[1])
        return {"exploitability": (br_val[0] + br_val[1]) / 2.0, "ev_p0": ev[0],
                "br_values": br_val, "br_gain": gain,
                "per_hand_br_gain": [br[u] - pv[u] for u in (0, 1)]}

    def solve(self, iterations: int) -> SolveResult:
        t0 = time.perf_counter()
        self.train(iterations)
        secs = time.perf_counter() - t0
        strat = self.average_strategy()
        ev = self.evaluate(strat)
        return SolveResult(iterations=self.iterations, seconds=secs, strategy=strat,
                           exploitability=ev["exploitability"], br_gain=ev["br_gain"],
                           ev_p0=ev["ev_p0"], cf_regret=[None if r is None else r.copy() for r in self.regret],
                           node_visits=self.node_visits)


# --------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------

def from_subgame(game, cells=None, algorithm: str = "cfr+") -> RiverRangeSolver:
    """Build a vectorized solver for a :class:`RiverSubgame` instance."""
    return RiverRangeSolver([card_str(c) for c in game.board], game.r0, game.r1, pot=game.pot,
                            stack=game.stack, bets=game.bets, raise_cap=game.raise_cap,
                            oop_only_bets=game.oop_only_bets, cells=cells, algorithm=algorithm)


def strength_order(solver: RiverRangeSolver, player: int) -> np.ndarray:
    """Hand indices of ``player`` sorted by increasing strength (ties by index)."""
    v = solver.values[player]
    return np.lexsort((np.arange(len(v)), v))


def equal_count_cells(solver: RiverRangeSolver, player: int, k: int) -> np.ndarray:
    """Partition ``player``'s hands into ``k`` equal-count strength buckets."""
    order = strength_order(solver, player)
    n = len(order)
    k = max(1, min(k, n))
    cells = np.empty(n, dtype=np.int64)
    bounds = np.linspace(0, n, k + 1).round().astype(int)
    for c in range(k):
        cells[order[bounds[c]:bounds[c + 1]]] = c
    return cells
