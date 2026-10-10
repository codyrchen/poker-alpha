"""Duplicate cross-play with seat swap between abstract HU strategies (Phase 29).

Each deal is played twice with the same cards (hole cards and full board are
fixed by a per-deal seed): once with policy A in seat 0 (button) and once
with A in seat 1. Action sampling uses its own random stream. The per-deal
result for A is the sum of the two hands, which cancels most card luck.

This measures head-to-head results *inside the abstract game* only. It is
not an exploitability measurement and not evidence about real opponents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

Policy = Callable[[object, object, List[str]], np.ndarray]   # (game, state, legal) -> probs


def table_policy(lookup, fallback: str = "uniform") -> Policy:
    """Policy from a ``lookup(key) -> (probs by token, visits) | None``;
    unvisited infosets fall back to uniform over legal actions."""
    stats = {"hits": 0, "misses": 0}

    def pol(game, state, legal):
        hit = lookup(game.infoset_key(state))
        if hit is None:
            stats["misses"] += 1
            return np.full(len(legal), 1.0 / len(legal))
        stats["hits"] += 1
        p = np.array([hit[0].get(a, 0.0) for a in legal], dtype=float)
        s = p.sum()
        return p / s if s > 0 else np.full(len(legal), 1.0 / len(legal))

    pol.stats = stats  # type: ignore[attr-defined]
    return pol


def uniform_policy(game, state, legal):
    return np.full(len(legal), 1.0 / len(legal))


def calling_station(game, state, legal):
    p = np.zeros(len(legal))
    p[legal.index("c")] = 1.0
    return p


def _deal(game, rng) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, ...]]:
    cards = rng.choice(52, size=9, replace=False)
    holes = ((int(cards[0]), int(cards[1])), (int(cards[2]), int(cards[3])))
    return holes, tuple(int(c) for c in cards[4:9])


def play_hand(game, seat_policies, holes, board, rng) -> float:
    """Player-0 utility of one hand with a fixed deal."""
    from dataclasses import replace

    s = replace(game.root(), holes=holes)
    n_board = {1: 3, 2: 4, 3: 5}
    while not game.is_terminal(s):
        if game.is_chance(s):
            s = replace(s, board=board[:n_board[s.street + 1]], streets=s.streets + ("",))
            continue
        legal = game.legal_actions(s)
        p = seat_policies[game.current_player(s)](game, s, legal)
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        a = legal[int(np.searchsorted(cdf, rng.random(), side="right"))]
        s = game.next_state(s, a)
    return float(game.utility(s))


@dataclass
class MatchResult:
    deals: int
    mean_bb_per_hand: float          # for A, averaged over both seats
    stderr_bb_per_hand: float
    bb_per_100: float
    ci95_bb_per_100: Tuple[float, float]
    a_as_button_bb_per_hand: float
    a_as_bb_bb_per_hand: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def duplicate_match(game, policy_a: Policy, policy_b: Policy, deals: int,
                    seed: int) -> MatchResult:
    deal_rng = np.random.default_rng(seed)
    act_rng = np.random.default_rng(seed + 1_000_003)
    per, btn, bb = [], [], []
    for _ in range(deals):
        holes, board = _deal(game, deal_rng)
        u_btn = play_hand(game, (policy_a, policy_b), holes, board, act_rng)
        u_bb = -play_hand(game, (policy_b, policy_a), holes, board, act_rng)
        btn.append(u_btn)
        bb.append(u_bb)
        per.append((u_btn + u_bb) / 2)
    per = np.array(per)
    m = float(per.mean())
    se = float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan")
    return MatchResult(deals, m, se, 100 * m, (100 * (m - 1.96 * se), 100 * (m + 1.96 * se)),
                       float(np.mean(btn)), float(np.mean(bb)))
