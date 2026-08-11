"""Risk and performance metrics for per-hand P&L series.

Poker winnings are treated *like* a strategy P&L series so the standard risk
vocabulary applies — with the explicit caveat that poker hands are (unlike
most financial returns) close to i.i.d., carry no time value, and the
Sharpe-style ratio here is an analogy for signal-to-noise, not a claim that
chips are an asset class.

All functions take a 1-D array of per-hand results in big blinds (Leduc: use
chips; the ante is the natural unit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def mean_return(pnl: Sequence[float]) -> float:
    return float(np.mean(pnl))


def variance(pnl: Sequence[float]) -> float:
    x = np.asarray(pnl, dtype=float)
    return float(np.var(x, ddof=1)) if x.size > 1 else 0.0


def std_dev(pnl: Sequence[float]) -> float:
    return float(np.sqrt(variance(pnl)))


def downside_deviation(pnl: Sequence[float], threshold: float = 0.0) -> float:
    """Root-mean-square of shortfalls below ``threshold`` (all samples in the
    denominator, the standard Sortino convention)."""
    x = np.asarray(pnl, dtype=float)
    short = np.minimum(x - threshold, 0.0)
    return float(np.sqrt(np.mean(short**2)))


def sharpe_like(pnl: Sequence[float]) -> float:
    """Mean over standard deviation, per hand. An analogy, not a Sharpe:
    there is no risk-free rate and no time aggregation."""
    sd = std_dev(pnl)
    return float(np.mean(pnl) / sd) if sd > 0 else float("nan")


def bb_per_100(pnl: Sequence[float]) -> float:
    return 100.0 * float(np.mean(pnl))


def win_rate(pnl: Sequence[float]) -> float:
    """Fraction of hands with strictly positive result."""
    x = np.asarray(pnl, dtype=float)
    return float(np.mean(x > 0))


def max_drawdown(pnl: Sequence[float]) -> float:
    """Largest peak-to-trough fall of the cumulative P&L (≥ 0)."""
    cum = np.cumsum(np.asarray(pnl, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum, initial=0.0))


def value_at_risk(pnl: Sequence[float], alpha: float = 0.05) -> float:
    """Per-hand VaR at level ``alpha``: loss size at the alpha-quantile,
    reported as a positive number."""
    x = np.asarray(pnl, dtype=float)
    return float(-np.quantile(x, alpha))


def conditional_value_at_risk(pnl: Sequence[float],
                              alpha: float = 0.05) -> float:
    """Expected loss given the loss exceeds VaR (positive number)."""
    x = np.asarray(pnl, dtype=float)
    cutoff = np.quantile(x, alpha)
    tail = x[x <= cutoff]
    return float(-np.mean(tail)) if tail.size else 0.0


def bootstrap_ci(
    pnl: Sequence[float],
    statistic=np.mean,
    level: float = 0.95,
    n_boot: int = 5000,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval for ``statistic`` of ``pnl``.

    The default (mean per hand) is what win-rate claims should carry: poker
    P&L is noisy enough that an unqualified "+X bb/100" is close to
    meaningless without one.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    x = np.asarray(pnl, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("empty P&L series")
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = statistic(x[idx], axis=1)
    lo = (1.0 - level) / 2.0
    return (float(np.quantile(stats, lo)),
            float(np.quantile(stats, 1.0 - lo)))


@dataclass(frozen=True)
class RiskReport:
    """Bundle of headline numbers for one P&L series."""

    hands: int
    mean: float
    std: float
    bb_per_100: float
    ci_low_bb100: float
    ci_high_bb100: float
    sharpe_like: float
    downside_dev: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float


def risk_report(pnl: Sequence[float], seed: int = 0,
                n_boot: int = 5000) -> RiskReport:
    """Compute the full report, with a bootstrap CI on bb/100."""
    lo, hi = bootstrap_ci(pnl, n_boot=n_boot, seed=seed)
    return RiskReport(
        hands=len(pnl),
        mean=mean_return(pnl),
        std=std_dev(pnl),
        bb_per_100=bb_per_100(pnl),
        ci_low_bb100=100.0 * lo,
        ci_high_bb100=100.0 * hi,
        sharpe_like=sharpe_like(pnl),
        downside_dev=downside_deviation(pnl),
        max_drawdown=max_drawdown(pnl),
        var_95=value_at_risk(pnl),
        cvar_95=conditional_value_at_risk(pnl),
        win_rate=win_rate(pnl),
    )
