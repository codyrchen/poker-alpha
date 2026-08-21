"""Monte Carlo showdown-equity estimation.

Estimates the probability that a hero hand wins/ties/loses at showdown against
an opponent range, by repeatedly sampling an opponent hand and the remaining
board, then evaluating both 7-card hands.

Equity is the pot fraction the hero expects at showdown:
``equity = P(win) + P(tie) / 2``.

The opponent range may be:

* ``None`` — uniform random over all hands not blocked by known cards, or
* a sequence of ``(hand, weight)`` pairs, where a hand is two cards; weights
  need not be normalized. Combos blocked by the hero's cards or the board are
  dropped (and their weight with them), matching how ranges collapse in play.

Sampling is driven by a seeded generator, so results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cards import NUM_CARDS, codes
from .evaluator import evaluate_best_codes


@dataclass(frozen=True)
class EquityResult:
    """Monte Carlo equity estimate with its sampling error."""

    win: float
    tie: float
    lose: float
    equity: float
    simulations: int

    @property
    def std_error(self) -> float:
        """Standard error of the equity estimate (per-sample outcomes in [0,1])."""
        # Per-sample equity outcome x in {0, 0.5, 1}; Var(x) <= 0.25.
        p = self.equity
        return float(np.sqrt(max(p * (1 - p), 1e-12) / self.simulations))


def estimate_equity(
    hero_hand: Iterable,
    board: Iterable = (),
    opponent_range: Optional[Sequence[Tuple[Iterable, float]]] = None,
    simulations: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> EquityResult:
    """Estimate hero's showdown equity by Monte Carlo simulation.

    Parameters
    ----------
    hero_hand:
        Two hero hole cards (codes, ``Card`` objects, or strings like "As").
    board:
        0-5 known community cards.
    opponent_range:
        ``None`` for a uniform random opponent, or ``(hand, weight)`` pairs.
    simulations:
        Number of Monte Carlo samples.
    rng, seed:
        Pass a generator, or a seed to create one; default is fresh entropy.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    hero = codes(hero_hand)
    board_known = codes(board)
    if len(hero) != 2:
        raise ValueError("hero_hand must be exactly two cards")
    if len(board_known) > 5:
        raise ValueError("board cannot exceed five cards")
    known = hero + board_known
    if len(set(known)) != len(known):
        raise ValueError("duplicate cards among hero hand and board")

    dead = set(known)
    remaining = np.array([c for c in range(NUM_CARDS) if c not in dead],
                         dtype=np.int64)
    need = 5 - len(board_known)

    # Materialize the opponent range as arrays for fast weighted sampling.
    if opponent_range is not None:
        combos: List[Tuple[int, int]] = []
        weights: List[float] = []
        for hand, weight in opponent_range:
            a, b = codes(hand)
            if a in dead or b in dead or a == b:
                continue  # blocked combo: drop it and its weight
            if weight < 0:
                raise ValueError("range weights must be non-negative")
            if weight > 0:
                combos.append((a, b))
                weights.append(float(weight))
        if not combos:
            raise ValueError("opponent range has no live combos")
        w = np.array(weights)
        w = w / w.sum()
        combo_idx_dist = (combos, w)
    else:
        combo_idx_dist = None

    wins = ties = 0
    for _ in range(simulations):
        if combo_idx_dist is None:
            # Uniform opponent + board completion in one draw.
            draw = rng.choice(len(remaining), size=2 + need, replace=False)
            picked = remaining[draw]
            opp = [int(picked[0]), int(picked[1])]
            runout = [int(c) for c in picked[2:]]
        else:
            combos, w = combo_idx_dist
            oi = int(rng.choice(len(combos), p=w))
            opp = list(combos[oi])
            # Complete the board from cards not blocked by the sampled combo.
            pool = remaining[(remaining != opp[0]) & (remaining != opp[1])]
            draw = rng.choice(len(pool), size=need, replace=False)
            runout = [int(pool[i]) for i in draw]

        full_board = board_known + runout
        # Codes are already validated ints here, so use the fast path.
        hv = evaluate_best_codes(hero + full_board)
        ov = evaluate_best_codes(opp + full_board)
        if hv > ov:
            wins += 1
        elif hv == ov:
            ties += 1

    losses = simulations - wins - ties
    win_p = wins / simulations
    tie_p = ties / simulations
    return EquityResult(
        win=win_p,
        tie=tie_p,
        lose=losses / simulations,
        equity=win_p + tie_p / 2.0,
        simulations=simulations,
    )
