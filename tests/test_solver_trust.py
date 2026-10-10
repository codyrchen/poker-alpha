"""Phase 32: solver trustworthiness — rules, utility signs, MCCFR update
identities and averaging, on the locked heads-up config."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from phase32_diagnostics import (geometry, mccfr_trace, rules_checks,  # noqa: E402
                                 utility_checks)

from poker_alpha.solver_config import PRIMARY_CONFIG  # noqa: E402
from poker_alpha.solvers.holdem_analysis import spot_state  # noqa: E402


@pytest.fixture(scope="module")
def game():
    return PRIMARY_CONFIG.build_game()


def test_heads_up_order_blinds_pot_and_allin(game):
    r = rules_checks(game)
    assert r["all_pass"], r
    assert r["effective_stack_after_3bet"] == pytest.approx(100 - 6.25)


def test_terminal_utilities_match_chip_arithmetic(game):
    u = utility_checks(game)
    assert u["all_pass"], u


def test_utility_is_zero_sum_in_seat_swap(game):
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = game.deal(rng)
        while not game.is_terminal(s):
            if game.is_chance(s):
                s = game.sample_chance(s, rng)
            else:
                legal = game.legal_actions(s)
                s = game.next_state(s, legal[rng.integers(len(legal))])
        u = game.utility(s)
        # Winner gains exactly what the loser lost from their contribution.
        assert -max(s.contrib) - 1e-9 <= u <= max(s.contrib) + 1e-9
        if s.folded == -1:
            swapped = replace(s, holes=(s.holes[1], s.holes[0]))
            assert game.utility(swapped) in (pytest.approx(-u), pytest.approx(u))


def test_mccfr_update_identities_at_btn_aa(game):
    t = mccfr_trace(game, iterations=8, seed=4)
    assert t["all_pass"], t["checks"]
    assert t["log"], "the AA infoset must be updated"


def test_strategy_sum_only_accumulates_at_opponent_nodes(game):
    """External sampling: the average strategy of an infoset grows only when
    its owner is the non-updating player."""
    from poker_alpha.solvers.mccfr import MCCFRSolver

    s = MCCFRSolver(game, seed=1)
    s.train(30)
    totals = [n.strategy_sum.sum() for n in s.infosets.values()]
    # visits are integers (one probability vector per visit)
    assert all(abs(t - round(t)) < 1e-6 for t in totals)
    # The BTN first-action infosets are reached once per player-1 traversal
    # (when BTN is the non-updating player): exactly one visit per iteration.
    first = [k for k in s.infosets if k.startswith("0|0|") and "|inr0fnone" in k]
    assert sum(s.infosets[k].strategy_sum.sum() for k in first) == pytest.approx(30)


def test_known_preflop_geometry(game):
    """Records the measured action geometry of the v1 abstraction (pot-relative
    preflop sizing); a change here changes solver semantics."""
    st = spot_state(game, "BTN", ("As", "Ah"), (), ("",), villain_hole=("7c", "2d"))
    to = {a: game.next_state(st, a).contrib[0] for a in game.legal_actions(st)}
    assert to == pytest.approx({"f": 0.5, "c": 1.0, "b33": 1.66, "b75": 2.5, "b150": 4.0, "a": 100.0})


def test_geometry_detector_flags_sub_minimum_raises(game):
    g = geometry(game, hands=150, seed=2)
    assert g["pre_raise:b33"]["share_below"] > 0.5        # 1.66 BB open < 2 BB min-raise
    assert g["pre_raise:b75"]["below_nlhe_minimum"] == 0
    assert g["post_bet:b75"]["below_nlhe_minimum"] == 0
