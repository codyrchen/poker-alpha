"""Scalable diagnostics for heads-up Hold'em MCCFR runs.

Exact exploitability needs a full traversal of the Hold'em chance tree and is
infeasible here, so it is **not** reported. Instead this module provides
convergence *proxies* — none of which is a bound on exploitability:

* :func:`strategy_l1_change` — visit-weighted average L1 distance between the
  average strategies of two checkpoints (over common information sets).
* :func:`top_infoset_stability` — the same, restricted to the K most visited
  information sets, where the strategy is best estimated.
* :func:`cross_play` — seeded Monte Carlo head-to-head between two strategies,
  both seat orders on identical deals (common random numbers).
* :func:`solver_metrics` — size, memory estimate, entropy, visit histogram.

:func:`describe_strategy_at` renders the policy at a human-readable spot
(e.g. "BTN, 100 BB, AKs, preflop unopened").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..games.holdem import HoldemGame, HoldemState
from ..poker.cards import codes
from .cfr import CFRSolver

Strategy = Mapping[str, Mapping[str, float]]


def infoset_visits(solver: CFRSolver) -> Dict[str, float]:
    """Visit proxy per infoset.

    External-sampling MCCFR adds the current strategy (summing to 1) to
    ``strategy_sum`` each time the infoset is reached as the non-updating
    player, so ``strategy_sum.sum()`` counts those visits exactly.
    """
    return {k: float(n.strategy_sum.sum()) for k, n in solver.infosets.items()}


def _entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


@dataclass(frozen=True)
class SolverMetrics:
    iterations: int
    infosets: int
    memory_bytes_estimate: int
    mean_entropy_bits: float          # visit-weighted, of the average strategy
    visits_total: float
    visits_quantiles: Tuple[float, float, float]  # median, p90, max
    singleton_fraction: float         # infosets visited at most once


def solver_metrics(solver: CFRSolver) -> SolverMetrics:
    """Size/entropy/visit diagnostics. Memory is an *estimate*: NumPy array
    payloads + key and action string bytes + a fixed per-object overhead."""
    visits = infoset_visits(solver)
    mem = 0
    ent_num = ent_den = 0.0
    for key, node in solver.infosets.items():
        mem += node.regret_sum.nbytes + node.strategy_sum.nbytes
        mem += len(key.encode("utf-8")) + sum(len(a) for a in node.actions)
        mem += 2 * 112 + 64 + 56  # two small ndarrays, InfoSet, dict slot
        v = visits[key]
        if v > 0:
            ent_num += v * _entropy(node.average_strategy())
            ent_den += v
    arr = np.array(list(visits.values()) or [0.0])
    return SolverMetrics(
        iterations=solver.iterations,
        infosets=len(solver.infosets),
        memory_bytes_estimate=int(mem),
        mean_entropy_bits=ent_num / ent_den if ent_den else float("nan"),
        visits_total=float(arr.sum()),
        visits_quantiles=(float(np.median(arr)), float(np.quantile(arr, 0.9)),
                          float(arr.max())),
        singleton_fraction=float(np.mean(arr <= 1.0)),
    )


def strategy_l1_change(old: Strategy, new: Strategy,
                       weights: Optional[Mapping[str, float]] = None) -> float:
    """Weighted mean L1 distance over infosets present in both strategies."""
    num = den = 0.0
    for key, p_new in new.items():
        p_old = old.get(key)
        if p_old is None:
            continue
        w = 1.0 if weights is None else float(weights.get(key, 0.0))
        if w <= 0:
            continue
        d = sum(abs(p_new.get(a, 0.0) - p_old.get(a, 0.0))
                for a in set(p_new) | set(p_old))
        num += w * d
        den += w
    return num / den if den else float("nan")


def top_infoset_stability(old: Strategy, new: Strategy,
                          visits: Mapping[str, float], k: int = 100) -> float:
    """Unweighted mean L1 change over the ``k`` most visited infosets."""
    top = sorted(visits, key=lambda key: (-visits[key], key))[:k]
    sub_new = {key: new[key] for key in top if key in new}
    return strategy_l1_change(old, sub_new)


def _policy(game: HoldemGame, strategy: Strategy, state: HoldemState):
    acts = game.legal_actions(state)
    probs = strategy.get(game.infoset_key(state))
    if probs is None:
        return acts, np.full(len(acts), 1.0 / len(acts))
    p = np.array([probs.get(a, 0.0) for a in acts], dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return acts, np.full(len(acts), 1.0 / len(acts))
    return acts, p / total


def play_hand(game: HoldemGame, strat0: Strategy, strat1: Strategy,
              deal_rng: np.random.Generator,
              action_rng: np.random.Generator) -> float:
    """Play one hand; return player 0's chip result. Unvisited infosets play
    uniformly (regret matching's default)."""
    state = game.deal(deal_rng)
    while not game.is_terminal(state):
        if game.is_chance(state):
            state = game.sample_chance(state, deal_rng)
            continue
        strat = strat0 if game.current_player(state) == 0 else strat1
        acts, p = _policy(game, strat, state)
        state = game.next_state(state, acts[int(action_rng.choice(len(acts), p=p))])
    return float(game.utility(state))


@dataclass(frozen=True)
class CrossPlayResult:
    mean_bb_per_hand: float   # strategy A's average result, both seats
    std_error: float
    hands: int


def cross_play(game: HoldemGame, strat_a: Strategy, strat_b: Strategy,
               hands: int, seed: int) -> CrossPlayResult:
    """A vs B, each deal played twice with seats swapped (variance reduction).

    Deals are identical across the two seatings (common random numbers);
    action sampling uses an independent seeded stream.
    """
    results = []
    root = np.random.SeedSequence(seed)
    deal_ss, act_ss = root.spawn(2)
    act_rng = np.random.default_rng(act_ss)
    deal_seeds = np.random.default_rng(deal_ss).integers(0, 2**63, size=hands)
    for s in deal_seeds:
        a_first = play_hand(game, strat_a, strat_b,
                            np.random.default_rng(int(s)), act_rng)
        a_second = -play_hand(game, strat_b, strat_a,
                              np.random.default_rng(int(s)), act_rng)
        results.append(0.5 * (a_first + a_second))
    arr = np.array(results)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return CrossPlayResult(float(arr.mean()), se, hands)


# -- human-readable lookup -----------------------------------------------------

_POSITIONS = {"BTN": 0, "SB": 0, "BB": 1}


def action_label(game: HoldemGame, state: HoldemState, action: str) -> str:
    """Human label for an internal Hold'em action token."""
    if action == "f":
        return "fold"
    if action == "a":
        return "all-in"
    street_paid, _, to_act, _ = game._replay(state)
    owe = street_paid[1 - to_act] - street_paid[to_act]
    if action == "c":
        return "call" if owe > 1e-9 else "check"
    return game.size_label(action, owe > 1e-9, state.street)


@dataclass(frozen=True)
class SpotReport:
    infoset_key: str
    visits: float
    found: bool
    actions: List[Tuple[str, str, float]]  # (token, label, probability)

    def format(self) -> str:
        lines = [f"infoset {self.infoset_key}  visits={self.visits:g}"
                 + ("" if self.found else "  (UNVISITED: uniform default)")]
        for _, label, p in self.actions:
            lines.append(f"  {label:<10} {p:6.1%}")
        return "\n".join(lines)


def spot_state(game: HoldemGame, position: str, hole: Sequence,
               board: Sequence = (), streets: Sequence[str] = ("",),
               villain_hole: Sequence = ("2c", "3d")) -> HoldemState:
    """Build a state where ``position`` (BTN/SB=player 0, BB=player 1) holds
    ``hole``. The villain's cards only fill the slot; the hero's information
    set never contains them."""
    seat = _POSITIONS[position.upper()]
    h, v = tuple(codes(hole)), tuple(codes(villain_hole))
    holes = (h, v) if seat == 0 else (v, h)
    state = HoldemState(holes=holes, board=tuple(codes(board)),
                        streets=tuple(streets), contrib=(0.0, 0.0))
    _, total, to_act, _ = game._replay(state)
    state = HoldemState(holes=holes, board=state.board, streets=state.streets,
                        contrib=(total[0], total[1]))
    if to_act != seat:
        raise ValueError(f"{position} is not to act after history {streets}")
    return state


def describe_strategy_at(game: HoldemGame, strategy: Strategy,
                         visits: Mapping[str, float], position: str,
                         hole: Sequence, board: Sequence = (),
                         streets: Sequence[str] = ("",)) -> SpotReport:
    state = spot_state(game, position, hole, board, streets)
    key = game.infoset_key(state)
    acts, p = _policy(game, strategy, state)
    return SpotReport(
        infoset_key=key,
        visits=float(visits.get(key, 0.0)),
        found=key in strategy,
        actions=[(a, action_label(game, state, a), float(x))
                 for a, x in zip(acts, p)],
    )
