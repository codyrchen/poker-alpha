"""Kelly criterion: growth-optimal sizing and why it differs from EV-max.

For a binary bet paying ``b:1`` with win probability ``p`` (lose probability
``q = 1 − p``), the log-growth-optimal fraction of bankroll to stake is

    f* = (b p − q) / b

Betting more than f* raises expected *dollars* per bet (which is maximized by
staking everything, always) while *lowering* the long-run growth rate and
raising the risk of ruin — the cleanest illustration that maximizing expected
value and maximizing expected log-wealth are different objectives. Fractional
Kelly (half, quarter) trades growth for drawdown reduction, and is what
practitioners actually use because ``p`` is never known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def kelly_fraction(p: float, b: float) -> float:
    """Growth-optimal stake fraction for a ``b``:1 payoff with win prob ``p``.

    Clamped below at 0 — a negative-edge bet's optimal stake is zero.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability, got {p}")
    if b <= 0:
        raise ValueError(f"payout ratio b must be positive, got {b}")
    q = 1.0 - p
    return max((b * p - q) / b, 0.0)


def expected_log_growth(f: float, p: float, b: float) -> float:
    """Per-bet expected log-growth at stake fraction ``f``."""
    if f >= 1.0:
        return float("-inf")  # a single loss is ruin
    q = 1.0 - p
    if f <= 0.0:
        return 0.0
    return p * np.log(1.0 + b * f) + q * np.log(1.0 - f)


@dataclass(frozen=True)
class BankrollPaths:
    """Monte Carlo bankroll trajectories and their summary statistics."""

    final: np.ndarray            # terminal bankrolls, one per path
    ruin_probability: float      # fraction of paths that hit <= ruin_level
    expected_final: float
    median_final: float
    max_drawdown_median: float

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.final, q))


def simulate_kelly_paths(
    p: float,
    b: float,
    fraction_of_kelly: float,
    bets: int,
    paths: int = 2000,
    starting_bankroll: float = 1.0,
    ruin_level: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> BankrollPaths:
    """Simulate repeated fractional-Kelly betting.

    ``fraction_of_kelly`` scales f*: 1.0 = full Kelly, 0.5 = half Kelly, etc.
    Ruin means the bankroll falls to ``ruin_level`` × start or below (with
    fractional staking the bankroll never hits exactly zero, so a threshold
    stands in for practical ruin).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    f = kelly_fraction(p, b) * fraction_of_kelly
    f = min(f, 0.999)

    bank = np.full(paths, float(starting_bankroll))
    peak = bank.copy()
    max_dd = np.zeros(paths)
    ruined = np.zeros(paths, dtype=bool)
    for _ in range(bets):
        wins = rng.random(paths) < p
        stake = bank * f
        bank = bank + np.where(wins, b * stake, -stake)
        peak = np.maximum(peak, bank)
        max_dd = np.maximum(max_dd, (peak - bank) / peak)
        ruined |= bank <= ruin_level * starting_bankroll
    return BankrollPaths(
        final=bank,
        ruin_probability=float(np.mean(ruined)),
        expected_final=float(np.mean(bank)),
        median_final=float(np.median(bank)),
        max_drawdown_median=float(np.median(max_dd)),
    )
