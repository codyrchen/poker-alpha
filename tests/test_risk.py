"""Tests for risk metrics, Kelly sizing, and bankroll simulation."""

import numpy as np
import pytest

from poker_alpha.risk.bankroll import simulate_bankroll
from poker_alpha.risk.kelly import (
    expected_log_growth,
    kelly_fraction,
    simulate_kelly_paths,
)
from poker_alpha.risk.metrics import (
    bb_per_100,
    bootstrap_ci,
    conditional_value_at_risk,
    downside_deviation,
    max_drawdown,
    risk_report,
    sharpe_like,
    value_at_risk,
    win_rate,
)


# -- metrics -----------------------------------------------------------------

def test_max_drawdown_simple_series():
    # Cumulative: 1, 3, 2, 0, 4 -> worst peak-to-trough is 3 -> 0 = 3.
    assert max_drawdown([1, 2, -1, -2, 4]) == pytest.approx(3.0)


def test_max_drawdown_monotone_up_is_zero():
    assert max_drawdown([1.0, 2.0, 0.5]) >= 0
    assert max_drawdown([1, 1, 1]) == 0.0


def test_var_and_cvar_ordering():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 10_000)
    var = value_at_risk(pnl, 0.05)
    cvar = conditional_value_at_risk(pnl, 0.05)
    assert cvar > var > 0  # tail mean exceeds the tail threshold


def test_downside_deviation_ignores_gains():
    assert downside_deviation([1.0, 2.0, 3.0]) == 0.0
    assert downside_deviation([-1.0, 1.0]) == pytest.approx(np.sqrt(0.5))


def test_bb_per_100_and_win_rate():
    pnl = [1.0, -1.0, 1.0, 1.0]
    assert bb_per_100(pnl) == pytest.approx(50.0)
    assert win_rate(pnl) == pytest.approx(0.75)


def test_sharpe_like_scale_invariance():
    pnl = np.array([1.0, -0.5, 2.0, -1.0])
    assert sharpe_like(pnl * 3) == pytest.approx(sharpe_like(pnl))


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.05, 1.0, 2000)
    lo1, hi1 = bootstrap_ci(pnl, seed=42)
    lo2, hi2 = bootstrap_ci(pnl, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < np.mean(pnl) < hi1


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(2)
    small = rng.normal(0, 1, 100)
    large = rng.normal(0, 1, 10_000)
    lo_s, hi_s = bootstrap_ci(small, seed=0)
    lo_l, hi_l = bootstrap_ci(large, seed=0)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_risk_report_is_consistent():
    rng = np.random.default_rng(3)
    pnl = rng.normal(0.02, 1.0, 5000)
    report = risk_report(pnl, seed=0, n_boot=1000)
    assert report.hands == 5000
    assert report.ci_low_bb100 < report.bb_per_100 < report.ci_high_bb100
    assert report.max_drawdown >= 0
    assert 0 <= report.win_rate <= 1


# -- Kelly -------------------------------------------------------------------

def test_kelly_fraction_textbook_case():
    # p=0.6, even money: f* = (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)


def test_kelly_negative_edge_stakes_zero():
    assert kelly_fraction(0.4, 1.0) == 0.0


def test_kelly_maximizes_log_growth():
    p, b = 0.6, 1.0
    f_star = kelly_fraction(p, b)
    g_star = expected_log_growth(f_star, p, b)
    for f in (f_star / 2, f_star * 1.5, f_star * 2.5):
        assert g_star >= expected_log_growth(f, p, b)


def test_overbetting_kelly_raises_ruin_and_lowers_median():
    common = dict(p=0.55, b=1.0, bets=400, paths=3000, seed=7)
    full = simulate_kelly_paths(fraction_of_kelly=1.0, **common)
    over = simulate_kelly_paths(fraction_of_kelly=3.0, **common)
    # Over-betting: markedly worse median outcome and deeper drawdowns --
    # even though per-bet expected dollars are higher.
    assert over.median_final < full.median_final
    assert over.max_drawdown_median > full.max_drawdown_median


def test_half_kelly_cuts_drawdowns():
    common = dict(p=0.55, b=1.0, bets=400, paths=3000, seed=7)
    full = simulate_kelly_paths(fraction_of_kelly=1.0, **common)
    half = simulate_kelly_paths(fraction_of_kelly=0.5, **common)
    assert half.max_drawdown_median < full.max_drawdown_median


# -- bankroll ----------------------------------------------------------------

def test_bankroll_bigger_roll_less_ruin():
    kw = dict(expected_bb_per_100=2.0, std_bb_per_100=80.0,
              hands=50_000, simulations=3000, seed=11)
    small = simulate_bankroll(starting_bankroll=500, **kw)
    large = simulate_bankroll(starting_bankroll=5000, **kw)
    assert large.ruin_probability < small.ruin_probability
    assert small.ruin_probability > 0.05  # 500bb at 80bb/100 sigma is risky


def test_bankroll_losing_player_ruins():
    r = simulate_bankroll(starting_bankroll=1000, expected_bb_per_100=-5.0,
                          std_bb_per_100=80.0, hands=100_000,
                          simulations=1000, seed=1)
    assert r.ruin_probability > 0.9


def test_bankroll_percentiles_ordered_and_ruin_sticks():
    r = simulate_bankroll(starting_bankroll=1000, expected_bb_per_100=2.0,
                          std_bb_per_100=80.0, hands=20_000,
                          simulations=2000, seed=3)
    p = r.percentiles
    assert p[5] <= p[25] <= p[50] <= p[75] <= p[95]
    assert float(np.min(r.final)) >= 0.0  # no negative bankrolls (no reload)
