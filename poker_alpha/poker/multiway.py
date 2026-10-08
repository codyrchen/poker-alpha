"""Multiway (hero vs 1-8 opponent ranges) Monte Carlo equity.

Joint card constraints
----------------------
Opponent hands must not share cards with each other, the hero or the board.
The correct joint distribution is ``P(h_1..h_n) ∝ Π_i w_i(h_i) · 1[no
collisions]``. Sampling opponents one after another from their *blocker-
adjusted* ranges does **not** produce this distribution (earlier opponents
get "first pick"), so it is not used naively. Two correct methods:

* ``"rejection"`` (default) — draw every opponent independently from its
  range and reject the whole draw on any collision. Accepted draws are exact
  i.i.d. samples from the joint distribution. If the acceptance rate is very
  low (narrow, overlapping ranges) a :class:`SamplingError` is raised rather
  than looping forever.
* ``"importance"`` — sequential sampling, each opponent from its range with
  already-used cards removed, weighted by the product of the remaining range
  masses ``Π_i Z_i``. This is self-normalized importance sampling of the same
  joint distribution: consistent, never stalls, but with an effective sample
  size below the raw count (reported).

Side pots
---------
With ``contributions`` (hero first, then opponents), each runout's chips are
split with the engine's side-pot rules, so a short all-in player can only win
what they are eligible for. ``expected_share`` is then hero's expected
fraction of all chips in the pot. Without it, everyone is assumed to have
contributed equally and the share is the classic multiway equity.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..holdem.pots import award_pots, build_pots
from .cards import NUM_CARDS, codes
from .evaluator import evaluate_best_codes
from .ranges import (CARD_MASK, COMBOS, NUM_COMBOS, WeightedRange,
                     blocked_mask, combo_cdf, draw_combo)


class SamplingError(RuntimeError):
    """Joint opponent hands could not be sampled (ranges incompatible)."""


@dataclass(frozen=True)
class MultiwayEquityResult:
    win: float                 # hero strictly best (or sole winner of every pot)
    tie: float                 # hero shares the best hand
    expected_share: float      # hero's expected fraction of the pot
    std_error: float           # of expected_share
    simulations: int           # accepted (or weighted) samples
    effective_samples: float   # == simulations for rejection sampling
    runtime_s: float
    method: str
    opponent_shares: tuple     # expected pot fraction per opponent
    acceptance_rate: float = 1.0

    @property
    def equity(self) -> float:
        return self.expected_share


def _range_probs(r: Optional[WeightedRange], dead_mask: np.ndarray) -> np.ndarray:
    w = np.ones(NUM_COMBOS) if r is None else r.weights.copy()
    w[dead_mask] = 0.0
    if w.sum() <= 0:
        raise SamplingError("an opponent range has no live combos")
    return w


def multiway_equity(hero: Sequence, board: Sequence = (),
                    opponent_ranges: Sequence[Optional[WeightedRange]] = (None,),
                    simulations: int = 10_000,
                    contributions: Optional[Sequence[int]] = None,
                    method: str = "rejection",
                    rng: Optional[np.random.Generator] = None,
                    seed: Optional[int] = None,
                    min_acceptance: float = 1e-3) -> MultiwayEquityResult:
    """Hero's equity against ``len(opponent_ranges)`` opponents (1-8).

    ``None`` as a range means "any two cards". See the module docstring for
    the sampling methods and side-pot handling.
    """
    start = time.perf_counter()
    if rng is None:
        rng = np.random.default_rng(seed)
    h = codes(hero)
    b = codes(board)
    n_opp = len(opponent_ranges)
    if len(h) != 2:
        raise ValueError("hero needs exactly two cards")
    if not 1 <= n_opp <= 8:
        raise ValueError("1-8 opponents supported (2-9 players)")
    if len(b) > 5 or len(set(h + b)) != len(h) + len(b):
        raise ValueError("invalid or duplicate hero/board cards")
    if contributions is not None:
        if len(contributions) != n_opp + 1 or any(int(c) != c or c < 0
                                                   for c in contributions):
            raise ValueError("contributions: one non-negative int per player")
    if method not in ("rejection", "importance"):
        raise ValueError("method must be 'rejection' or 'importance'")

    dead = blocked_mask(h + b)
    weights = [_range_probs(r, dead) for r in opponent_ranges]
    probs = [w / w.sum() for w in weights]
    cdfs = [combo_cdf(p) for p in probs]
    known = set(h + b)
    need = 5 - len(b)
    players = n_opp + 1
    contrib = list(contributions) if contributions is not None else [1] * players
    total = float(sum(contrib))
    seat_order = list(range(players))

    shares = np.zeros((simulations, players))
    sample_w = np.ones(simulations)
    wins = ties = 0.0
    attempts = 0
    accepted = 0
    max_attempts = int(math.ceil(simulations / min_acceptance))
    while accepted < simulations:
        attempts += 1
        if attempts > max_attempts:
            raise SamplingError(
                f"acceptance rate below {min_acceptance:g}: ranges are (nearly) "
                f"incompatible" + ("; try method='importance'"
                                   if method == "rejection" else ""))
        if method == "rejection":
            idx = [draw_combo(rng, c) for c in cdfs]
            cards = [c for i in idx for c in COMBOS[i]]
            if len(set(cards)) != len(cards):
                continue
            weight = 1.0
        else:
            used = np.zeros(NUM_COMBOS, dtype=bool)
            idx, weight = [], 1.0
            for w in weights:
                ww = w.copy()
                ww[used] = 0.0
                z = ww.sum()
                if z <= 0:
                    weight = 0.0
                    break
                i = int(rng.choice(NUM_COMBOS, p=ww / z))
                weight *= z / w.sum()
                idx.append(i)
                used |= CARD_MASK[COMBOS[i, 0]] | CARD_MASK[COMBOS[i, 1]]
            if weight == 0.0:
                continue
            cards = [c for i in idx for c in COMBOS[i]]
        holes = [h] + [[int(COMBOS[i, 0]), int(COMBOS[i, 1])] for i in idx]
        dead_now = known | set(int(c) for c in cards)
        live = [c for c in range(NUM_CARDS) if c not in dead_now]
        run = [live[j] for j in rng.choice(len(live), size=need, replace=False)] \
            if need else []
        full = b + run
        values = [evaluate_best_codes(hole + full) for hole in holes]
        pots = build_pots(contrib, [False] * players)
        won = award_pots(pots, lambda s: values[s], seat_order)
        row = np.array([won.get(s, 0) / total for s in range(players)])
        best = max(values)
        n_best = sum(1 for v in values if v == best)
        if values[0] == best:
            if n_best == 1:
                wins += weight
            else:
                ties += weight
        shares[accepted] = row
        sample_w[accepted] = weight
        accepted += 1

    wsum = sample_w.sum()
    mean = (shares * sample_w[:, None]).sum(axis=0) / wsum
    hero_s = shares[:, 0]
    if method == "rejection":
        ess = float(simulations)
        se = float(hero_s.std(ddof=1) / math.sqrt(simulations)) \
            if simulations > 1 else float("nan")
    else:
        ess = float(wsum ** 2 / (sample_w ** 2).sum())
        var = float((sample_w ** 2 * (hero_s - mean[0]) ** 2).sum() / wsum ** 2)
        se = math.sqrt(var)
    return MultiwayEquityResult(
        win=float(wins / wsum), tie=float(ties / wsum),
        expected_share=float(mean[0]), std_error=se,
        simulations=simulations, effective_samples=ess,
        runtime_s=time.perf_counter() - start, method=method,
        opponent_shares=tuple(float(x) for x in mean[1:]),
        acceptance_rate=accepted / attempts if attempts else 1.0)


def exact_equity_enumeration(hero: Sequence, board: Sequence,
                             opponent_ranges: Sequence[WeightedRange],
                             contributions: Optional[Sequence[int]] = None) -> float:
    """Exact expected pot share by full enumeration (river only; testing aid)."""
    h, b = codes(hero), codes(board)
    if len(b) != 5:
        raise ValueError("exact enumeration requires a complete board")
    n = len(opponent_ranges)
    contrib = list(contributions) if contributions is not None else [1] * (n + 1)
    total = float(sum(contrib))
    dead = blocked_mask(h + b)
    lists = []
    for r in opponent_ranges:
        w = r.weights.copy()
        w[dead] = 0.0
        lists.append([(i, w[i]) for i in np.flatnonzero(w)])
    num = den = 0.0

    def rec(k: int, used: set, chosen: List[int], weight: float) -> None:
        nonlocal num, den
        if k == n:
            holes = [h] + [[int(COMBOS[i, 0]), int(COMBOS[i, 1])] for i in chosen]
            values = [evaluate_best_codes(x + b) for x in holes]
            won = award_pots(build_pots(contrib, [False] * (n + 1)),
                             lambda s: values[s], list(range(n + 1)))
            num += weight * won.get(0, 0) / total
            den += weight
            return
        for i, w in lists[k]:
            a, c = int(COMBOS[i, 0]), int(COMBOS[i, 1])
            if a in used or c in used:
                continue
            rec(k + 1, used | {a, c}, chosen + [i], weight * w)

    rec(0, set(), [], 1.0)
    if den == 0:
        raise SamplingError("no collision-free combination of opponent hands")
    return num / den
