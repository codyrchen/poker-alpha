"""Tests for archetypes, Bayesian modeling, beliefs, exploitation, matches."""

import numpy as np
import pytest

from poker_alpha.opponent import (
    ARCHETYPES,
    ArchetypeBelief,
    BetaPosterior,
    OpponentConfig,
    adaptive_strategy,
    blend,
    cap_lambda_by_exploitability,
    confidence_lambda,
    deviation_magnitude,
    exploitative_strategy,
    leduc_hand_likelihood,
    leduc_is_weak,
    match_summary,
    simulate_match,
    tilt_strategy,
)
from poker_alpha.solvers import exploitability
from poker_alpha.solvers.evaluation import profile_value


@pytest.fixture(scope="module")
def archetype_strategies(leduc_equilibrium):
    return {name: tilt_strategy(leduc_equilibrium, cfg, leduc_is_weak)
            for name, cfg in ARCHETYPES.items()}


# -- archetypes --------------------------------------------------------------

def test_balanced_tilt_is_identity(leduc_equilibrium):
    balanced = tilt_strategy(leduc_equilibrium, ARCHETYPES["balanced"],
                             leduc_is_weak)
    for key, probs in leduc_equilibrium.items():
        for action, p in probs.items():
            assert balanced[key][action] == pytest.approx(p)


def test_tilted_strategies_are_normalized(archetype_strategies):
    for strategy in archetype_strategies.values():
        for probs in strategy.values():
            assert sum(probs.values()) == pytest.approx(1.0)


def test_nit_folds_more_than_equilibrium(leduc_equilibrium,
                                         archetype_strategies):
    nit = archetype_strategies["nit"]
    fold_spots = [k for k, p in leduc_equilibrium.items()
                  if 0 < p.get("f", 0) < 1]
    assert fold_spots
    more = sum(nit[k]["f"] > leduc_equilibrium[k]["f"] for k in fold_spots)
    assert more == len(fold_spots)


def test_maniac_raises_more_than_equilibrium(leduc_equilibrium,
                                             archetype_strategies):
    maniac = archetype_strategies["maniac"]
    raise_spots = [k for k, p in leduc_equilibrium.items()
                   if 0 < p.get("r", 0) < 1]
    assert raise_spots
    more = sum(maniac[k]["r"] > leduc_equilibrium[k]["r"]
               for k in raise_spots)
    assert more == len(raise_spots)


def test_negative_multiplier_rejected():
    with pytest.raises(ValueError):
        OpponentConfig(fold_multiplier=-0.1)


def test_archetypes_are_more_exploitable_than_equilibrium(
        leduc_game, leduc_equilibrium, archetype_strategies):
    eq_expl = exploitability(leduc_game, leduc_equilibrium)
    for name, strategy in archetype_strategies.items():
        if name == "balanced":
            continue
        assert exploitability(leduc_game, strategy) > 10 * eq_expl


# -- Beta-Bernoulli model ----------------------------------------------------

def test_beta_posterior_updates_and_mean():
    post = BetaPosterior()
    post.update_counts(successes=3, failures=1)
    assert post.mean() == pytest.approx(4 / 6)  # Beta(4, 2)
    assert post.observations == 4


def test_uncertainty_shrinks_with_evidence():
    small = BetaPosterior()
    small.update_counts(3, 1)          # 3 of 4
    large = BetaPosterior()
    large.update_counts(300, 100)      # 300 of 400
    # Same empirical rate (0.75); the uniform prior shrinks the small sample
    # (4/6 ~ 0.67) but barely moves the large one (301/402 ~ 0.749).
    assert small.mean() == pytest.approx(0.667, abs=0.01)
    assert large.mean() == pytest.approx(0.749, abs=0.01)
    assert large.variance() < small.variance() / 10
    lo_s, hi_s = small.confidence_interval()
    lo_l, hi_l = large.confidence_interval()
    assert (hi_l - lo_l) < (hi_s - lo_s) / 3


def test_confidence_interval_brackets_mean():
    post = BetaPosterior()
    post.update_counts(30, 10)
    lo, hi = post.confidence_interval(0.95)
    assert lo < post.mean() < hi
    assert 0.0 <= lo and hi <= 1.0


# -- beliefs -----------------------------------------------------------------

def test_likelihood_favors_true_type(archetype_strategies):
    from poker_alpha.opponent import HandObservation

    # An opponent who raised twice preflop looks more like a maniac.
    obs = HandObservation(hero_rank=1, public_rank=None,
                          opp_decisions=(("?", "r", "r"),),
                          revealed_opp_rank=None)
    like_maniac = leduc_hand_likelihood(archetype_strategies["maniac"], obs)
    like_nit = leduc_hand_likelihood(archetype_strategies["nit"], obs)
    assert like_maniac > like_nit


def test_posterior_identifies_the_maniac(leduc_game, leduc_equilibrium,
                                         archetype_strategies):
    rng = np.random.default_rng(7)
    belief = ArchetypeBelief(candidates=archetype_strategies)
    simulate_match(leduc_game, leduc_equilibrium,
                   archetype_strategies["maniac"], hands=400, rng=rng,
                   on_hand=lambda r: belief.update(r.observation))
    assert belief.map_estimate() == "maniac"
    assert belief.posterior()["maniac"] > 0.9
    assert belief.confidence() > 0.7


def test_posterior_starts_uniform(archetype_strategies):
    belief = ArchetypeBelief(candidates=archetype_strategies)
    post = belief.posterior()
    for p in post.values():
        assert p == pytest.approx(1 / len(archetype_strategies))
    assert belief.confidence() == pytest.approx(0.0, abs=1e-9)


def test_mixture_strategy_is_normalized(archetype_strategies):
    belief = ArchetypeBelief(candidates=archetype_strategies)
    for probs in belief.mixture_strategy().values():
        assert sum(probs.values()) == pytest.approx(1.0)


# -- decay (recency-aware belief) --------------------------------------------

def test_decay_one_matches_baseline_exactly(archetype_strategies):
    from poker_alpha.opponent import HandObservation

    obs = HandObservation(hero_rank=1, public_rank=None,
                          opp_decisions=(("?", "r", "r"), ("0", "r/r", "c")),
                          revealed_opp_rank=None)
    baseline = ArchetypeBelief(candidates=archetype_strategies)
    explicit_one = ArchetypeBelief(candidates=archetype_strategies, decay=1.0)
    for _ in range(5):
        baseline.update(obs)
        explicit_one.update(obs)
    assert baseline.log_likelihood == pytest.approx(explicit_one.log_likelihood)


def test_invalid_decay_rejected(archetype_strategies):
    with pytest.raises(ValueError):
        ArchetypeBelief(candidates=archetype_strategies, decay=0.0)
    with pytest.raises(ValueError):
        ArchetypeBelief(candidates=archetype_strategies, decay=1.1)


def test_decay_discounts_old_evidence(leduc_game, leduc_equilibrium,
                                      archetype_strategies):
    """A decaying belief that first sees strong maniac evidence, then a long
    run of nit evidence, moves on to favor nit much more than a decay=1.0
    belief given the identical hands, which keeps weighing in the stale
    maniac evidence forever."""
    maniac_obs_rng = np.random.default_rng(1)
    nit_obs_rng = np.random.default_rng(2)

    baseline = ArchetypeBelief(candidates=archetype_strategies, decay=1.0)
    recency = ArchetypeBelief(candidates=archetype_strategies, decay=0.98)

    simulate_match(leduc_game, leduc_equilibrium, archetype_strategies["maniac"],
                   hands=300, rng=maniac_obs_rng,
                   on_hand=lambda r: (baseline.update(r.observation),
                                      recency.update(r.observation)))
    assert baseline.map_estimate() == "maniac"
    assert recency.map_estimate() == "maniac"

    simulate_match(leduc_game, leduc_equilibrium, archetype_strategies["nit"],
                   hands=300, rng=nit_obs_rng,
                   on_hand=lambda r: (baseline.update(r.observation),
                                      recency.update(r.observation)))
    assert recency.map_estimate() == "nit"
    assert baseline.map_estimate() != "nit"
    assert recency.posterior()["nit"] > baseline.posterior()["nit"]


# -- exploitation ------------------------------------------------------------

def test_blend_endpoints(leduc_equilibrium, archetype_strategies):
    exploit = archetype_strategies["maniac"]  # any distinct strategy works
    at_zero = blend(leduc_equilibrium, exploit, 0.0)
    at_one = blend(leduc_equilibrium, exploit, 1.0)
    for key in leduc_equilibrium:
        for a, p in leduc_equilibrium[key].items():
            assert at_zero[key][a] == pytest.approx(p)
            assert at_one[key][a] == pytest.approx(exploit[key].get(a, 0.0))


def test_deviation_zero_for_identical(leduc_equilibrium):
    assert deviation_magnitude(leduc_equilibrium, leduc_equilibrium) == \
        pytest.approx(0.0)


def test_confidence_lambda_properties():
    assert confidence_lambda(0.0, 0.5) == 0.0     # no confidence -> no exploit
    assert confidence_lambda(1.0, 0.0) == 0.0     # no deviation -> no exploit
    lam_small = confidence_lambda(1.0, 0.05)
    lam_big = confidence_lambda(1.0, 0.5)
    assert 0 < lam_small < lam_big < 1.0


def test_best_response_beats_equilibrium_against_maniac(
        leduc_game, leduc_equilibrium, archetype_strategies):
    maniac = archetype_strategies["maniac"]

    def seat_avg_ev(strategy):
        return 0.5 * (profile_value(leduc_game, strategy, maniac)
                      - profile_value(leduc_game, maniac, strategy))

    ev_eq = seat_avg_ev(leduc_equilibrium)
    exploit = exploitative_strategy(leduc_game, maniac, leduc_equilibrium)
    ev_exploit = seat_avg_ev(exploit)
    assert ev_exploit > ev_eq + 0.5  # over half a chip per hand better


def test_exploiter_is_itself_exploitable(leduc_game, leduc_equilibrium,
                                         archetype_strategies):
    exploit = exploitative_strategy(leduc_game, archetype_strategies["maniac"],
                                    leduc_equilibrium)
    assert exploitability(leduc_game, exploit) > \
        100 * exploitability(leduc_game, leduc_equilibrium)


def test_guardrail_caps_exploitability(leduc_game, leduc_equilibrium,
                                       archetype_strategies):
    maniac = archetype_strategies["maniac"]
    exploit = exploitative_strategy(leduc_game, maniac, leduc_equilibrium)
    eps = 0.05
    lam = cap_lambda_by_exploitability(leduc_game, leduc_equilibrium, exploit,
                                       lam=1.0, max_exploitability=eps)
    assert 0.0 < lam < 1.0
    capped = blend(leduc_equilibrium, exploit, lam)
    assert exploitability(leduc_game, capped) <= eps + 1e-6


def test_adaptive_strategy_respects_budget(leduc_game, leduc_equilibrium,
                                           archetype_strategies):
    decision = adaptive_strategy(
        leduc_game, leduc_equilibrium, archetype_strategies["maniac"],
        belief_confidence=1.0, max_exploitability=0.05)
    assert decision.lam_applied <= decision.lam_requested
    assert exploitability(leduc_game, decision.strategy) <= 0.05 + 1e-6


# -- match simulation --------------------------------------------------------

def test_match_is_deterministic_given_seed(leduc_game, leduc_equilibrium):
    def run():
        rng = np.random.default_rng(11)
        recs = simulate_match(leduc_game, leduc_equilibrium,
                              leduc_equilibrium, hands=50, rng=rng)
        return [r.hero_utility for r in recs]

    assert run() == run()


def test_seats_alternate(leduc_game, leduc_equilibrium):
    rng = np.random.default_rng(0)
    recs = simulate_match(leduc_game, leduc_equilibrium, leduc_equilibrium,
                          hands=4, rng=rng)
    assert [r.hero_seat for r in recs] == [0, 1, 0, 1]


def test_no_hidden_information_without_showdown(leduc_game,
                                                leduc_equilibrium):
    rng = np.random.default_rng(3)
    recs = simulate_match(leduc_game, leduc_equilibrium, leduc_equilibrium,
                          hands=200, rng=rng)
    for r in recs:
        if not r.went_to_showdown:
            assert r.observation.revealed_opp_rank is None
        else:
            assert r.observation.revealed_opp_rank is not None


def test_match_summary_math(leduc_game, leduc_equilibrium):
    rng = np.random.default_rng(5)
    recs = simulate_match(leduc_game, leduc_equilibrium, leduc_equilibrium,
                          hands=100, rng=rng)
    s = match_summary(recs)
    assert s["hands"] == 100
    assert s["total_chips"] == pytest.approx(
        sum(r.hero_utility for r in recs))
    assert s["chips_per_100"] == pytest.approx(s["total_chips"], abs=1e-9)
