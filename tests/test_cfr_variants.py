"""Phase II: discounted / linear CFR and outcome-sampling MCCFR."""

import math

import numpy as np
import pytest

from poker_alpha.games.kuhn import KuhnPoker
from poker_alpha.games.leduc import LeducPoker
from poker_alpha.research.cfr_variants import DiscountedCFRSolver, OutcomeSamplingMCCFR
from poker_alpha.solvers.cfr_plus import CFRPlusSolver
from poker_alpha.solvers.evaluation import exploitability


def test_discounted_with_cfr_plus_parameters_equals_cfr_plus():
    g = LeducPoker()
    a = CFRPlusSolver(g)
    b = DiscountedCFRSolver(g, alpha=math.inf, beta=-math.inf, gamma=1.0)
    a.train(12)
    b.train(12)
    sa, sb = a.average_strategy(), b.average_strategy()
    assert set(sa) == set(sb)
    for k in sa:
        for act in sa[k]:
            assert sb[k][act] == pytest.approx(sa[k][act], abs=1e-12)
    assert exploitability(g, sb) == pytest.approx(exploitability(g, sa), abs=1e-12)


@pytest.mark.parametrize("preset", ["dcfr", "lcfr"])
def test_discounted_presets_converge_on_kuhn(preset):
    g = KuhnPoker()
    s = DiscountedCFRSolver.preset(g, preset)
    s.train(50)
    early = exploitability(g, s.average_strategy())
    s.train(450)
    late = exploitability(g, s.average_strategy())
    assert late < early and late < 0.01


def test_lcfr_matches_linear_weighting_identity():
    # LCFR: every cumulative quantity after T iterations is proportional to
    # sum_t t * x_t; the discount factors must realise exactly t/(t+1).
    assert DiscountedCFRSolver._factor(3.0, 1.0) == pytest.approx(0.75)
    assert DiscountedCFRSolver._factor(3.0, math.inf) == 1.0
    assert DiscountedCFRSolver._factor(3.0, -math.inf) == 0.0


def test_outcome_sampling_converges_on_kuhn_and_leduc():
    for game, iters, bound in ((KuhnPoker(), 40000, 0.06), (LeducPoker(), 60000, 1.3)):
        s = OutcomeSamplingMCCFR(game, seed=1)
        s.train(iters // 10)
        early = exploitability(game, s.average_strategy())
        s.train(iters - iters // 10)
        late = exploitability(game, s.average_strategy())
        assert late < early and late < bound, (type(game).__name__, early, late)


def test_outcome_sampling_is_seeded():
    a = OutcomeSamplingMCCFR(KuhnPoker(), seed=3)
    b = OutcomeSamplingMCCFR(KuhnPoker(), seed=3)
    a.train(200)
    b.train(200)
    for k in a.infosets:
        assert np.allclose(a.infosets[k].regret_sum, b.infosets[k].regret_sum)
