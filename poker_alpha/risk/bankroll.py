"""Bankroll risk-of-ruin estimation for a fixed-stakes win-rate profile.

Given a measured win rate and standard deviation (both per 100 hands, the
units poker results are quoted in), simulate bankroll trajectories at fixed
stakes and report the ruin probability, terminal-bankroll percentiles, and the
maximum-drawdown distribution. Per-100-hand results are approximated as
normal, which the CLT justifies at that aggregation level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BankrollSimResult:
    ruin_probability: float
    median_final: float
    percentiles: Dict[int, float]      # {5: ..., 25: ..., 50: ..., 75: ..., 95: ...}
    max_drawdowns: np.ndarray          # per-path max drawdown (bb)
    final: np.ndarray                  # per-path terminal bankroll (bb)


def simulate_bankroll(
    starting_bankroll: float,
    expected_bb_per_100: float,
    std_bb_per_100: float,
    hands: int,
    simulations: int = 5000,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> BankrollSimResult:
    """Simulate fixed-stakes play; ruin = bankroll touching zero.

    All quantities are in big blinds. A ruined path stays ruined (no
    reloading), matching the standard risk-of-ruin definition.
    """
    if std_bb_per_100 < 0:
        raise ValueError("std must be non-negative")
    if rng is None:
        rng = np.random.default_rng(seed)
    blocks = max(hands // 100, 1)

    bank = np.full(simulations, float(starting_bankroll))
    alive = np.ones(simulations, dtype=bool)
    peak = bank.copy()
    max_dd = np.zeros(simulations)
    for _ in range(blocks):
        step = rng.normal(expected_bb_per_100, std_bb_per_100,
                          size=simulations)
        bank = np.where(alive, bank + step, bank)
        newly_ruined = alive & (bank <= 0.0)
        bank[newly_ruined] = 0.0
        alive &= ~newly_ruined
        peak = np.maximum(peak, bank)
        max_dd = np.maximum(max_dd, peak - bank)

    return BankrollSimResult(
        ruin_probability=float(np.mean(~alive)),
        median_final=float(np.median(bank)),
        percentiles={q: float(np.percentile(bank, q))
                     for q in (5, 25, 50, 75, 95)},
        max_drawdowns=max_dd,
        final=bank,
    )
