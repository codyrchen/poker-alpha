"""Strict EV-regret semantics for session review.

A difference between two EV numbers is only evidence about a decision when
both numbers were produced by the same methodology AND that methodology is
what actually produced the recommendation. Solver frequencies are NOT EVs:
when ``DecisionReport.method`` is a solver lookup, any ``ev_bb`` on the
candidates is a separately computed rollout *estimate* ("used for EVs only"
in the source cascade) and must never be read as what the MCCFR strategy
optimised.

Uncertainty uses PAIRED differences, never marginal SEs. The rollout
evaluates every candidate on common random numbers (one shared sample
tape), so Var[EV(a) - EV(b)] is the variance of the per-sample differences
— typically far below ``se_a² + se_b²``, which would treat the estimates
as independent when they are strongly positively correlated. The engine
preserves the paired mean/SE for every candidate pair in
``details["rollout_paired_differences"]`` whenever the rollout is the
decision basis; this module reads it and NEVER reconstructs an
independent-SE approximation. If the paired value is unavailable the
regret point estimate may still be shown (when otherwise valid) with
``uncertainty=None``.

Two more honesty rules enforced here:

* the real action must map onto an analysed candidate EXACTLY
  (:mod:`.mapping`); a nearest-size translation is NOT the action the
  user took, so it yields ``ACTUAL_SIZE_NOT_EVALUATED`` instead of a
  fabricated regret;
* the product phrase is **ESTIMATED DECISION REGRET** (bb under the
  rollout model), not "money lost".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..decision.report import DecisionReport
from .mapping import EXACT, TRANSLATED, UNMAPPED, ActionMapping

# Why a decision has no valid EV regret (code -> human sentence).
INELIGIBLE_REASONS = {
    "NO_RECOMMENDATION": "the engine refused to analyse this state",
    "EV_NOT_DECISION_BASIS": (
        "the recommendation came from solver frequencies; candidate EVs are "
        "separate rollout estimates, not what produced the strategy"),
    "HEURISTIC_DECISION": (
        "the recommendation is a heuristic fallback; its EVs are not a "
        "comparable decision basis"),
    "ACTION_NOT_COMPARABLE": (
        "the action actually taken does not map onto any analysed candidate"),
    "ACTUAL_SIZE_NOT_EVALUATED": (
        "the exact sizing taken was not evaluated — the nearest analysed "
        "size is a different action, and its EV is not the EV of yours"),
    "MISSING_EV": "no EV was measured for the compared actions",
    "MIXED_METHODOLOGY": (
        "the compared EVs come from different estimation methods"),
}


@dataclass(frozen=True)
class RegretEligibility:
    """Result of the regret gate for one decision."""

    eligible: bool
    reason: str = ""                       # code from INELIGIBLE_REASONS, "" when eligible
    best_ev: Optional[float] = None        # EV (bb) of the engine's choice
    actual_ev: Optional[float] = None      # EV (bb) of the action actually taken
    estimated_regret: Optional[float] = None   # paired mean of best - actual (bb)
    uncertainty: Optional[float] = None    # paired-difference SE (shared samples)
    uncertainty_source: str = ""           # "paired (common random numbers)" | "unavailable"
    source: str = ""                       # methodology both EVs come from

    @property
    def reason_text(self) -> str:
        return INELIGIBLE_REASONS.get(self.reason, self.reason)

    def display(self) -> str:
        if not self.eligible:
            return f"EV regret unavailable — {self.reason_text}"
        r = f"{self.estimated_regret:+.2f} BB"
        if self.uncertainty is not None:
            r += f" ±{self.uncertainty:.2f}"
        return f"estimated decision regret {r}"


def paired_difference(report: DecisionReport, a: str,
                      b: str) -> Optional[Tuple[float, float]]:
    """Paired mean/SE of EV(a) - EV(b) from the report's shared-sample
    rollout, or None when the engine did not preserve it."""
    diffs = (report.details or {}).get("rollout_paired_differences")
    if not diffs:
        return None
    if f"{a}|{b}" in diffs:
        m, s = diffs[f"{a}|{b}"]
        return float(m), float(s)
    if f"{b}|{a}" in diffs:
        m, s = diffs[f"{b}|{a}"]
        return -float(m), float(s)
    return None


def _ev_is_decision_basis(method: str) -> Optional[str]:
    """Ineligibility code for the method, or None when EVs are the basis."""
    m = method.lower()
    if "solver" in m or "abstraction" in m:
        return "EV_NOT_DECISION_BASIS"
    if "rollout" in m:
        return None
    if m in ("", "none"):
        return "NO_RECOMMENDATION"
    return "HEURISTIC_DECISION"


def regret_eligibility(report: DecisionReport,
                       actual_label: Optional[str],
                       mapping: Optional[ActionMapping] = None,
                       ) -> RegretEligibility:
    """Whether ``best EV - actual EV`` is a valid estimate for this decision.

    Valid only when (1) the EVs are the actual decision basis — the report's
    method is the Monte Carlo rollout; (2) the hero's real action was itself
    EVALUATED (an EXACT mapping, never a nearest-size translation); (3) both
    candidates carry an EV from the same methodology. ``mapping``, when
    given, is the honest record of how ``actual_label`` was obtained.
    """
    if report.recommended is None or not report.candidates:
        return RegretEligibility(False, "NO_RECOMMENDATION")
    code = _ev_is_decision_basis(report.method)
    if code is not None:
        return RegretEligibility(False, code)
    if mapping is not None:
        if mapping.quality == UNMAPPED or mapping.label is None:
            return RegretEligibility(False, "ACTION_NOT_COMPARABLE")
        if mapping.quality == TRANSLATED:
            return RegretEligibility(False, "ACTUAL_SIZE_NOT_EVALUATED")
        actual_label = mapping.label
    by_label = {c.label: c for c in report.candidates}
    best = by_label.get(report.recommended)
    actual = by_label.get(actual_label) if actual_label else None
    if best is None:
        return RegretEligibility(False, "MISSING_EV")
    if actual is None:
        return RegretEligibility(False, "ACTION_NOT_COMPARABLE")
    if best.ev_bb is None or actual.ev_bb is None:
        return RegretEligibility(False, "MISSING_EV")
    if best.source != actual.source:
        return RegretEligibility(False, "MIXED_METHODOLOGY")
    if best.label == actual.label:
        # the action taken IS the engine's choice: the paired difference is
        # identically zero on every shared sample
        return RegretEligibility(
            True, "", best_ev=best.ev_bb, actual_ev=actual.ev_bb,
            estimated_regret=0.0, uncertainty=0.0,
            uncertainty_source="identical action", source=best.source)
    pd = paired_difference(report, best.label, actual.label)
    if pd is not None:
        mean, se = pd
        return RegretEligibility(
            True, "", best_ev=best.ev_bb, actual_ev=actual.ev_bb,
            estimated_regret=mean, uncertainty=se,
            uncertainty_source="paired (common random numbers)",
            source=best.source)
    # No paired record (e.g. a report produced before the engine preserved
    # them): the point estimate stands, but no uncertainty is invented —
    # sqrt(se_a² + se_b²) would be wrong under common random numbers.
    return RegretEligibility(
        True, "", best_ev=best.ev_bb, actual_ev=actual.ev_bb,
        estimated_regret=best.ev_bb - actual.ev_bb, uncertainty=None,
        uncertainty_source="unavailable", source=best.source)


__all__ = ["RegretEligibility", "regret_eligibility", "paired_difference",
           "INELIGIBLE_REASONS"]
