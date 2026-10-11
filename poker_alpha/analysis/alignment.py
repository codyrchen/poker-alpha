"""Strategy alignment: how a real action sits inside a solver mix.

A low-frequency choice is NOT automatically wrong — a mixed strategy plays
its 5% actions 5% of the time. This module classifies the hero's action by
its frequency in the trained strategy and keeps the language neutral.

Thresholds (documented, used verbatim in tests):

    COMMON       frequency >= 0.25      "Common in solver mix"
    MIXED        0.05 <= f < 0.25       "Reasonable but infrequent"
    RARE         0 < f < 0.05           "Rare in solver mix"
    NOT_IN_MIX   f == 0 or no mapping   "Not represented / translated"
    UNCERTAIN    gate LOW CONFIDENCE or REJECTED   "Solver uncertain"
    NOT_SOLVER   method is not a solver lookup (no alignment defined)

``surprise_bits`` is ``-log2(frequency)`` — *policy surprise*, an
information measure of how unusual the choice is under the mix. It is not a
regret and must never be presented as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..decision.report import DecisionReport

COMMON_AT = 0.25
RARE_BELOW = 0.05

CATEGORY_TEXT = {
    "COMMON": "Common in solver mix",
    "MIXED": "Reasonable but infrequent",
    "RARE": "Rare in solver mix",
    "NOT_IN_MIX": "Not represented / translated",
    "UNCERTAIN": "Solver uncertain",
    "NOT_SOLVER": "No solver mix for this decision",
}


@dataclass(frozen=True)
class StrategyAlignment:
    category: str                         # key of CATEGORY_TEXT
    frequency: Optional[float] = None     # actual action's solver frequency
    highest_label: Optional[str] = None   # the mix's most frequent action
    highest_frequency: Optional[float] = None
    surprise_bits: Optional[float] = None  # -log2(frequency), policy surprise

    @property
    def text(self) -> str:
        return CATEGORY_TEXT[self.category]

    def display(self) -> str:
        if self.frequency is None:
            return self.text
        return f"{self.text} — {self.frequency:.0%}"


def solver_state(report: DecisionReport) -> str:
    """Gate state of the trained strategy for this report:
    ACCEPTED / LOW CONFIDENCE / REJECTED / OFF (mirrors the Play Mode
    semantics; a fallback's confidence is never a solver state)."""
    info = report.details.get("solver", {}) if report.details else {}
    conf = info.get("confidence", "not configured")
    if info.get("used"):
        return "LOW CONFIDENCE" if conf == "SOLVER_LOW_CONFIDENCE" else "ACCEPTED"
    return "OFF" if conf == "not configured" else "REJECTED"


def strategy_alignment(report: DecisionReport,
                       actual_label: Optional[str]) -> StrategyAlignment:
    """Classify the hero's real action against the report's solver mix."""
    m = report.method.lower()
    if not ("solver" in m or "abstraction" in m):
        return StrategyAlignment("NOT_SOLVER")
    mix = dict(report.recommended_mix or {})
    hi_label, hi_f = None, None
    if mix:
        hi_label = max(mix, key=mix.get)
        hi_f = mix[hi_label]
    if solver_state(report) in ("LOW CONFIDENCE", "REJECTED"):
        f = mix.get(actual_label) if actual_label else None
        return StrategyAlignment("UNCERTAIN", frequency=f,
                                 highest_label=hi_label, highest_frequency=hi_f,
                                 surprise_bits=-math.log2(f) if f else None)
    f = mix.get(actual_label, 0.0) if actual_label else 0.0
    if f <= 0.0:
        return StrategyAlignment("NOT_IN_MIX", frequency=0.0,
                                 highest_label=hi_label, highest_frequency=hi_f)
    cat = "COMMON" if f >= COMMON_AT else ("MIXED" if f >= RARE_BELOW else "RARE")
    return StrategyAlignment(cat, frequency=f, highest_label=hi_label,
                             highest_frequency=hi_f, surprise_bits=-math.log2(f))


__all__ = ["StrategyAlignment", "strategy_alignment", "solver_state",
           "CATEGORY_TEXT", "COMMON_AT", "RARE_BELOW"]
