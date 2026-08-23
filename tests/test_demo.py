"""Tests for the demo's committed-result loaders and formatting helpers.

The demo prints numbers a reader is expected to trust, so the logic that
derives them from CSVs is tested here rather than eyeballed. The final tests
run against the *real* committed results, so they also catch a results file
being regenerated into a shape the demo can no longer read.
"""

import pytest

from poker_alpha.demo import (
    DATA_DIR,
    _read_csv,
    error_ratio_from_speedup,
    fmt_rate,
    high_confidence_fraction,
    identification_checkpoints,
    recovery_delays_by_repeat,
    sims_within_budget,
    speedup_row,
    sustained_recovery_delay,
)


def _row(repeat, hand, belief, correct, regime="after"):
    return {"repeat": str(repeat), "hand": str(hand), "regime": regime,
            "belief": belief, "correct": str(correct)}


# -- sustained_recovery_delay ------------------------------------------------

def test_recovery_delay_finds_first_sustained_hand():
    rows = [_row(0, 2020, "recency", False),
            _row(0, 2040, "recency", True),
            _row(0, 2060, "recency", True)]
    assert sustained_recovery_delay(rows, "recency", 2000) == 40


def test_recovery_delay_ignores_transient_correct_guess():
    """A lucky correct guess that later reverts is not recovery."""
    rows = [_row(0, 2020, "recency", True),    # transient
            _row(0, 2040, "recency", False),
            _row(0, 2060, "recency", True),
            _row(0, 2080, "recency", True)]
    assert sustained_recovery_delay(rows, "recency", 2000) == 60


def test_recovery_delay_none_when_never_correct():
    rows = [_row(0, h, "baseline", False) for h in (2020, 2040, 2060)]
    assert sustained_recovery_delay(rows, "baseline", 2000) is None


def test_recovery_delay_none_when_last_checkpoint_wrong():
    rows = [_row(0, 2020, "recency", True), _row(0, 2040, "recency", False)]
    assert sustained_recovery_delay(rows, "recency", 2000) is None


def test_recovery_delay_ignores_other_belief_and_pre_switch_rows():
    rows = [_row(0, 1000, "recency", True, regime="before"),
            _row(0, 2020, "baseline", True),
            _row(0, 2040, "recency", True)]
    assert sustained_recovery_delay(rows, "recency", 2000) == 40


def test_recovery_delay_is_order_independent():
    rows = [_row(0, 2060, "recency", True),
            _row(0, 2020, "recency", False),
            _row(0, 2040, "recency", True)]
    assert sustained_recovery_delay(rows, "recency", 2000) == 40


def test_recovery_delays_by_repeat_preserves_repeat_order():
    rows = ([_row(0, 2020, "recency", True)]
            + [_row(1, 2020, "recency", False), _row(1, 2040, "recency", True)])
    assert recovery_delays_by_repeat(rows, "recency", 2000) == [20, 40]


# -- benchmark lookups -------------------------------------------------------

def test_speedup_row_reads_fields():
    rows = [{"workload": "w", "throughput_before": "100",
             "throughput_after": "250", "speedup": "2.5", "unit": "sims"}]
    s = speedup_row(rows, "w")
    assert (s["before"], s["after"], s["speedup"], s["unit"]) == \
        (100.0, 250.0, 2.5, "sims")


def test_speedup_row_raises_on_unknown_workload():
    with pytest.raises(KeyError):
        speedup_row([], "nope")


def test_sims_within_budget_rounds():
    assert sims_within_budget(135_370.0, 0.010) == 1354
    assert sims_within_budget(17_108.0, 0.010) == 171


def test_error_ratio_is_sqrt_of_speedup():
    """N x throughput buys sqrt(N) tighter error bars, not N."""
    assert error_ratio_from_speedup(4.0) == pytest.approx(2.0)
    assert error_ratio_from_speedup(7.91) == pytest.approx(2.81, abs=0.01)


# -- formatting --------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (37_678.7, "37.7k"), (414_955.0, "415.0k"), (1_500_000.0, "1.5M"), (45.06, "45"),
])
def test_fmt_rate(value, expected):
    assert fmt_rate(value) == expected


def test_identification_checkpoints_are_increasing_and_end_at_total():
    cps = identification_checkpoints(200)
    assert cps == sorted(cps) and cps[-1] == 200
    assert all(c <= 200 for c in cps) and len(set(cps)) == len(cps)


def test_identification_checkpoints_handles_small_runs():
    assert identification_checkpoints(3) == [3]


# -- integration with the real committed results -----------------------------

def test_missing_results_file_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiments/"):
        _read_csv(tmp_path / "absent.csv")


def test_committed_regime_change_reproduces_documented_recovery():
    rows = _read_csv(DATA_DIR / "regime_change_switching.csv")
    baseline = recovery_delays_by_repeat(rows, "baseline")
    recency = recovery_delays_by_repeat(rows, "recency")
    assert all(d is None for d in baseline)        # never recovers
    assert all(d is not None for d in recency)     # recovers every repeat
    assert 380 <= min(recency) and max(recency) <= 500


def test_committed_benchmarks_reproduce_documented_speedups():
    rows = _read_csv(DATA_DIR / "performance_comparison.csv")
    assert speedup_row(rows, "hand_eval_7card")["speedup"] == pytest.approx(11.01, abs=0.01)
    flop = speedup_row(rows, "equity_flop_uniform")
    assert flop["speedup"] == pytest.approx(7.91, abs=0.01)
    assert sims_within_budget(flop["before"], 0.010) == 171
    assert sims_within_budget(flop["after"], 0.010) == 1354
    assert error_ratio_from_speedup(flop["speedup"]) == pytest.approx(2.8, abs=0.05)


# -- confidence aggregation --------------------------------------------------

def _conf_row(repeat, hand, belief, confidence, regime="after"):
    return {"repeat": str(repeat), "hand": str(hand), "regime": regime,
            "belief": belief, "confidence": str(confidence)}


def test_high_confidence_fraction_averages_per_checkpoint_first():
    """Repeats are averaged within a checkpoint before thresholding, so one
    outlying repeat cannot flip a checkpoint on its own."""
    rows = [_conf_row(0, 2020, "b", 1.00), _conf_row(1, 2020, "b", 0.90),
            _conf_row(0, 2040, "b", 1.00), _conf_row(1, 2040, "b", 1.00)]
    # hand 2020 mean = 0.95 (not > 0.99); hand 2040 mean = 1.00 -> 1 of 2.
    assert high_confidence_fraction(rows, "b", 0.99) == pytest.approx(0.5)


def test_high_confidence_fraction_ignores_pre_switch_and_other_beliefs():
    rows = [_conf_row(0, 1000, "b", 0.10, regime="before"),
            _conf_row(0, 2020, "other", 0.10),
            _conf_row(0, 2020, "b", 1.00)]
    assert high_confidence_fraction(rows, "b", 0.99) == pytest.approx(1.0)


def test_high_confidence_fraction_empty_is_zero():
    assert high_confidence_fraction([], "b") == 0.0


def test_committed_baseline_confidence_matches_documented_96_percent():
    rows = _read_csv(DATA_DIR / "regime_change_switching.csv")
    assert high_confidence_fraction(rows, "baseline", 0.99) == pytest.approx(0.96, abs=0.005)
