"""Strict EV-regret semantics for session review.

A difference between two EV numbers is only evidence about a decision when
both numbers were produced by the same methodology AND that methodology is
what actually produced the recommendation. Solver frequencies are NOT EVs:
when ``DecisionReport.method`` is a solver lookup, any ``ev_bb`` on the
candidates is a separately computed rollout *estimate* ("used for EVs only"
in the source cascade) and must never be read as what the MCCFR strategy
optimised. This module therefore exposes one gatekeeper,
:func:`regret_eligibility`, and the rest of the analysis suite is required
to go through it.

The product phrase for an eligible difference is **ESTIMATED DECISION
REGRET** (in big blinds) — an estimate under the rollout model, not money
lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..decision.report import DecisionReport

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
    estimated_regret: Optional[float] = None   # best_ev - actual_ev (bb); may be ~0 or
    uncertainty: Optional[float] = None    # 1 SE of the difference, when both SEs exist
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
                       actual_label: Optional[str]) -> RegretEligibility:
    """Whether ``best EV - actual EV`` is a valid estimate for this decision.

    Valid only when (1) the EVs are the actual decision basis — the report's
    method is the Monte Carlo rollout; (2) the hero's real action maps onto
    an analysed candidate; (3) both candidates carry an EV; and (4) both EVs
    come from the same methodology (identical ``CandidateAction.source``).
    """
    if report.recommended is None or not report.candidates:
        return RegretEligibility(False, "NO_RECOMMENDATION")
    code = _ev_is_decision_basis(report.method)
    if code is not None:
        return RegretEligibility(False, code)
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
    unc = None
    if best.ev_se_bb is not None and actual.ev_se_bb is not None:
        unc = math.sqrt(best.ev_se_bb ** 2 + actual.ev_se_bb ** 2)
    return RegretEligibility(
        True, "", best_ev=best.ev_bb, actual_ev=actual.ev_bb,
        estimated_regret=best.ev_bb - actual.ev_bb, uncertainty=unc,
        source=best.source)


__all__ = ["RegretEligibility", "regret_eligibility", "INELIGIBLE_REASONS"]
