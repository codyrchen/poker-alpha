"""Session analysis: strict regret semantics, alignment, categories,
aggregation, patterns and exports."""

from __future__ import annotations

import csv
import io
import json

import pytest

from dataclasses import replace

from poker_alpha.analysis import (categorize, confidence_report,
                                  detect_patterns, regret_eligibility,
                                  review_decision, review_hands,
                                  review_to_csv, review_to_json,
                                  review_to_markdown, solver_state,
                                  strategy_alignment, summarize)
from poker_alpha.analysis.alignment import COMMON_AT, RARE_BELOW
from poker_alpha.analysis.review import HIGH_REGRET_BB
from poker_alpha.decision import DecisionConfig
from poker_alpha.decision.report import CandidateAction, DecisionReport
from poker_alpha.history import load_hands

SOLVER_OFF = {"used": False, "confidence": "not configured", "reasons": []}
SOLVER_OK = {"used": True, "confidence": "SOLVER_ACCEPT", "reasons": []}
SOLVER_LOW = {"used": True, "confidence": "SOLVER_LOW_CONFIDENCE",
              "reasons": ["STREET_ABSTRACTION_ERROR"]}
SOLVER_REJ = {"used": False, "confidence": "rejected",
              "reasons": ["HIGH_SEED_DISAGREEMENT"]}


def _cand(label, kind, prob=None, ev=None, se=None, to=0.0,
          source="Monte Carlo rollout"):
    return CandidateAction(label=label, kind=kind, amount_to=to, added=0.0,
                           probability=prob, ev_bb=ev, ev_se_bb=se,
                           source=source)


def _report(candidates, recommended, method="Monte Carlo rollout",
            confidence="medium", solver=None, mix=None):
    return DecisionReport(
        state_summary="s", hero_equity=0.6, hero_equity_se=0.01, pot_bb=10.0,
        to_call_bb=0.0, pot_odds=None, spr=4.0, effective_stack_bb=40.0,
        opponent_ranges=(), candidates=tuple(candidates),
        recommended=recommended,
        recommended_mix=mix or {c.label: c.probability
                                for c in candidates if c.probability},
        mix_meaning="m", method=method, confidence=confidence, warnings=(),
        details={"solver": solver or SOLVER_OFF, "source_cascade": []},
        uncertainty={})


# -- regret eligibility ------------------------------------------------------

def test_regret_valid_only_for_same_method_rollout_evs():
    rep = _report([_cand("call", "call", ev=1.0, se=0.1),
                   _cand("fold", "fold", ev=0.0, se=0.0)], "call")
    r = regret_eligibility(rep, "fold")
    assert r.eligible and r.estimated_regret == pytest.approx(1.0)
    # no paired record on this synthetic report -> no uncertainty is
    # invented (independent-SE combination would be wrong under CRN)
    assert r.uncertainty is None
    assert r.uncertainty_source == "unavailable"
    assert "rollout" in r.source.lower()
    assert "ESTIMATED" in r.display().upper() or "estimated" in r.display()


def test_regret_uses_paired_se_not_independent_combination():
    """Common random numbers make candidate EVs strongly correlated: the
    paired-difference SE is the valid uncertainty and is much smaller than
    sqrt(se_a^2 + se_b^2). The gate must read the paired record."""
    rep = _report([_cand("call", "call", ev=1.0, se=0.10),
                   _cand("fold", "fold", ev=0.0, se=0.08)], "call")
    details = dict(rep.details)
    details["rollout_paired_differences"] = {"call|fold": [0.97, 0.03]}
    rep = replace(rep, details=details)
    r = regret_eligibility(rep, "fold")
    assert r.eligible
    naive = (0.10 ** 2 + 0.08 ** 2) ** 0.5
    assert r.uncertainty == pytest.approx(0.03)
    assert abs(r.uncertainty - naive) > 0.09          # genuinely different
    assert r.estimated_regret == pytest.approx(0.97)  # paired mean, not diff of means
    assert "paired" in r.uncertainty_source
    # reversed key is negated correctly
    rep2 = replace(rep, details={**details,
                                 "rollout_paired_differences":
                                 {"fold|call": [-0.97, 0.03]}})
    r2 = regret_eligibility(rep2, "fold")
    assert r2.estimated_regret == pytest.approx(0.97)


def test_action_mapping_quality():
    from poker_alpha.analysis.mapping import map_action

    rep = _report([_cand("check", "check"),
                   _cand("bet_75", "bet", ev=1.0, to=3.75),
                   _cand("bet_150", "bet", ev=0.8, to=7.5),
                   _cand("all_in", "all_in", ev=0.2, to=97.5)], "bet_75")
    assert map_action(rep, "check", None).quality == "EXACT"
    assert map_action(rep, "fold", None).quality == "UNMAPPED"
    m = map_action(rep, "bet", 3.75)
    assert m.quality == "EXACT" and m.label == "bet_75"
    off = map_action(rep, "bet", 3.0)                 # 60%-pot-ish, off-grid
    assert off.quality == "TRANSLATED" and off.label == "bet_75"
    assert off.distance == pytest.approx(0.75)
    assert "translated" in off.note()
    assert map_action(rep, "all_in", 97.5).quality == "EXACT"
    assert map_action(rep, "raise", None).quality == "UNMAPPED"


def test_no_false_regret_from_nearest_size_mapping():
    from poker_alpha.analysis.mapping import map_action

    rep = _report([_cand("check", "check", ev=0.0, se=0.01),
                   _cand("bet_75", "bet", ev=1.0, se=0.02, to=3.75)],
                  "bet_75")
    m = map_action(rep, "bet", 3.0)                   # real 60%-pot bet
    r = regret_eligibility(rep, m.label, m)
    assert not r.eligible
    assert r.reason == "ACTUAL_SIZE_NOT_EVALUATED"
    assert "exact sizing" in r.display()
    # with an EXACT mapping the same report scores normally
    ok = map_action(rep, "bet", 3.75)
    assert regret_eligibility(rep, ok.label, ok).eligible


def test_regret_never_from_solver_frequencies():
    # solver method with attached rollout EV estimates: NOT eligible
    rep = _report([_cand("check", "check", prob=0.8, ev=0.5, source="solver"),
                   _cand("bet_75", "bet", prob=0.2, ev=1.4, source="solver")],
                  "check", method="solver", solver=SOLVER_OK)
    r = regret_eligibility(rep, "bet_75")
    assert not r.eligible and r.reason == "EV_NOT_DECISION_BASIS"
    assert "unavailable" in r.display()


def test_regret_requires_mapped_action_and_evs():
    rep = _report([_cand("call", "call", ev=1.0), _cand("fold", "fold", ev=0.0)],
                  "call")
    assert regret_eligibility(rep, None).reason == "ACTION_NOT_COMPARABLE"
    rep2 = _report([_cand("call", "call", ev=1.0), _cand("fold", "fold")], "call")
    assert regret_eligibility(rep2, "fold").reason == "MISSING_EV"
    refusal = _report([], None, method="none", confidence="low")
    assert regret_eligibility(refusal, "call").reason == "NO_RECOMMENDATION"


def test_regret_rejects_mixed_methodology():
    rep = _report([_cand("call", "call", ev=1.0, source="rollout A"),
                   _cand("fold", "fold", ev=0.0, source="heuristic B")], "call")
    assert regret_eligibility(rep, "fold").reason == "MIXED_METHODOLOGY"


def test_heuristic_method_not_eligible():
    rep = _report([_cand("call", "call", ev=1.0), _cand("fold", "fold", ev=0.0)],
                  "call", method="heuristic fallback")
    assert regret_eligibility(rep, "fold").reason == "HEURISTIC_DECISION"


# -- alignment ---------------------------------------------------------------

def test_alignment_categories_and_surprise():
    rep = _report([_cand("check", "check", prob=0.64, source="solver"),
                   _cand("bet_75", "bet", prob=0.30, source="solver"),
                   _cand("bet_150", "bet", prob=0.06, source="solver"),
                   _cand("fold", "fold", prob=0.0, source="solver")],
                  "check", method="solver", solver=SOLVER_OK)
    a = strategy_alignment(rep, "check")
    assert a.category == "COMMON" and a.frequency == pytest.approx(0.64)
    assert a.highest_label == "check"
    assert strategy_alignment(rep, "bet_75").category == "COMMON"  # 30% >= 25%
    assert strategy_alignment(rep, "bet_150").category == "MIXED"  # 6% in [5,25)
    assert strategy_alignment(rep, "fold").category == "NOT_IN_MIX"
    b = strategy_alignment(rep, "bet_150")
    assert b.surprise_bits == pytest.approx(-__import__("math").log2(0.06))
    assert "64%" in a.display()


def test_alignment_rare_band_and_uncertain():
    rep = _report([_cand("check", "check", prob=0.97, source="solver"),
                   _cand("bet_75", "bet", prob=0.03, source="solver")],
                  "check", method="solver", solver=SOLVER_OK)
    assert strategy_alignment(rep, "bet_75").category == "RARE"
    low = _report([_cand("check", "check", prob=0.97, source="solver")],
                  "check", method="solver", solver=SOLVER_LOW)
    assert strategy_alignment(low, "check").category == "UNCERTAIN"
    roll = _report([_cand("call", "call", ev=0.5)], "call")
    assert strategy_alignment(roll, "call").category == "NOT_SOLVER"
    # thresholds documented and used
    assert COMMON_AT == 0.25 and RARE_BELOW == 0.05


def test_solver_state_mapping():
    mk = lambda info, method: solver_state(  # noqa: E731
        _report([_cand("check", "check", prob=1.0)], "check",
                method=method, solver=info))
    assert mk(SOLVER_OK, "solver") == "ACCEPTED"
    assert mk(SOLVER_LOW, "solver") == "LOW CONFIDENCE"
    assert mk(SOLVER_REJ, "Monte Carlo rollout") == "REJECTED"
    assert mk(SOLVER_OFF, "Monte Carlo rollout") == "OFF"


# -- categories --------------------------------------------------------------

def test_categorize_thresholds():
    rep = _report([_cand("call", "call", ev=1.0, se=0.05),
                   _cand("fold", "fold", ev=0.0, se=0.05)], "call")
    reg = regret_eligibility(rep, "fold")
    assert reg.estimated_regret >= HIGH_REGRET_BB
    al = strategy_alignment(rep, "fold")
    assert categorize(reg, al, "OFF", rep.method) == "HIGH-PRIORITY EV REVIEW"
    # below the bb threshold -> not high priority
    rep2 = _report([_cand("call", "call", ev=0.3, se=0.01),
                    _cand("fold", "fold", ev=0.0, se=0.01)], "call")
    reg2 = regret_eligibility(rep2, "fold")
    assert categorize(reg2, al, "OFF", rep2.method) == "OK"
    # rare action in an ACCEPTED mix -> strategy review
    srep = _report([_cand("check", "check", prob=0.98, source="solver"),
                    _cand("bet_75", "bet", prob=0.02, source="solver")],
                   "check", method="solver", solver=SOLVER_OK)
    sreg = regret_eligibility(srep, "bet_75")
    sal = strategy_alignment(srep, "bet_75")
    assert categorize(sreg, sal, "ACCEPTED", "solver") == "STRATEGY REVIEW"
    # gate rejected -> uncertain
    assert categorize(sreg, sal, "REJECTED", "Monte Carlo rollout") == "UNCERTAIN"
    # refusal -> unscored
    assert categorize(sreg, sal, "OFF", "none") == "UNSCORED"


# -- end-to-end review on the bundled fixtures -------------------------------

@pytest.fixture(scope="module")
def demo_review():
    hands = load_hands("poker_alpha/data/demo_session.json")
    from poker_alpha.pipeline import load_solver
    from poker_alpha.solver_config import RELEASE_STRATEGY

    solver = load_solver(RELEASE_STRATEGY)
    cfg = DecisionConfig(equity_simulations=200, rollout_simulations=200)
    return review_hands(hands, config=cfg, solver=solver)


def test_review_bundled_session(demo_review):
    rev = demo_review
    assert rev.hands == 8 and len(rev.records) >= 20
    states = {r.solver_state for r in rev.records}
    assert "ACCEPTED" in states and "REJECTED" in states
    for r in rev.records:
        # regret must never be "eligible" on a solver-method decision
        if r.source_kind == "solver":
            assert not r.regret.eligible
            assert r.regret.reason == "EV_NOT_DECISION_BASIS"
        if r.regret.eligible:
            assert r.source_kind == "rollout"
        assert r.category in ("HIGH-PRIORITY EV REVIEW", "STRATEGY REVIEW",
                              "UNCERTAIN", "UNSCORED", "OK")


def test_summary_and_exports(demo_review):
    rev = demo_review
    s = summarize(rev.records, rev.hands, rev.warnings)
    assert s.decisions == len(rev.records)
    assert (s.solver_accepted + s.solver_low_confidence + s.solver_rejected
            + s.solver_off) == s.decisions
    assert s.regret_eligible <= s.decisions
    if s.regret_eligible == 0:
        assert s.avg_estimated_regret_bb is None
    if s.decisions < 20:
        assert any("low sample" in w for w in s.warnings)
    pats = detect_patterns(rev.records)
    j = json.loads(review_to_json(rev, s, pats))
    assert j["format"] == "pokeralpha.session_review/v1"
    assert "ESTIMATED DECISION REGRET" in j["semantics"]["estimated_regret"]
    rows = list(csv.DictReader(io.StringIO(review_to_csv(rev))))
    assert len(rows) == s.decisions
    assert {"regret_eligible", "solver_state", "category"} <= set(rows[0])
    md = review_to_markdown(rev, s, pats)
    assert "ESTIMATED decision regret" in md and "not money lost" in md
    cr = confidence_report(rev.records, recognition_confidences=[0.9, 0.95])
    assert cr["recognition"]["mean"] == pytest.approx(0.925)
    assert "solver" not in cr["recognition"]["note"].split()[0]


def test_patterns_require_min_samples():
    hands = load_hands("poker_alpha/data/demo_session.json")
    from poker_alpha.pipeline import load_solver
    from poker_alpha.solver_config import RELEASE_STRATEGY

    # one hand only -> nothing reaches MIN_PATTERN
    rev = review_hands(hands[:1],
                       config=DecisionConfig(equity_simulations=100,
                                             rollout_simulations=100),
                       solver=load_solver(RELEASE_STRATEGY))
    assert detect_patterns(rev.records) == [] or all(
        p.count >= 3 for p in detect_patterns(rev.records))


def test_pattern_language_grading():
    from poker_alpha.analysis.patterns import _grade

    assert _grade(3) == "Potential pattern"
    assert _grade(5) == "Repeated deviation"


def test_review_mapping_and_paired_invariants(demo_review):
    """End-to-end: no TRANSLATED mapping ever carries eligible regret, and
    every eligible regret's uncertainty is paired (or identically zero)."""
    for r in demo_review.records:
        assert r.mapping_quality in ("EXACT", "TRANSLATED", "UNMAPPED")
        if r.mapping_quality == "TRANSLATED":
            assert not r.regret.eligible
            assert r.regret.reason in ("ACTUAL_SIZE_NOT_EVALUATED",
                                       "EV_NOT_DECISION_BASIS")
        if r.regret.eligible:
            assert r.regret.uncertainty_source in (
                "paired (common random numbers)", "identical action")
    # the exact-size injection really happens for off-menu aggression
    labels = {r.actual_label for r in demo_review.records}
    assert "actual_size" in labels
