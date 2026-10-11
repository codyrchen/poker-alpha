"""Mapping a real action onto a report's candidates — with honesty about
how good the mapping is.

Nearest-size mapping is fine for strategy/abstraction comparison, but it is
NOT enough to treat a candidate's EV as the EV of the action the user
actually took: a 60%-pot bet is not a 75%-pot bet. Every mapping therefore
carries a quality:

    EXACT       the candidate IS the action taken (same kind, same size
                within tolerance) — EV comparisons are legitimate
    TRANSLATED  nearest aggressive size; usable for strategy-alignment
                display (clearly labeled), NEVER for EV regret
    UNMAPPED    nothing comparable was analysed

Session review avoids TRANSLATED where it can: for off-menu bet/raise
sizes it re-analyses with the hero's exact size injected as an extra
rollout candidate (``ActionAbstraction.extra_targets``), which shares the
same common-random-number samples as the rest of the menu — making the
paired comparison legitimate instead of approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..decision.report import DecisionReport

EXACT = "EXACT"
TRANSLATED = "TRANSLATED"
UNMAPPED = "UNMAPPED"

#: |amount_to - actual| at or below this counts as the same size (chips).
SIZE_TOLERANCE = 0.01

#: Label used when session review injects the hero's real size into the
#: evaluated menu.
ACTUAL_SIZE_LABEL = "actual_size"

AGGRESSIVE = ("bet", "raise", "all_in")


@dataclass(frozen=True)
class ActionMapping:
    label: Optional[str]          # candidate label, or None when UNMAPPED
    quality: str                  # EXACT / TRANSLATED / UNMAPPED
    distance: Optional[float] = None   # |candidate size - actual size|, chips
    actual_amount: Optional[float] = None
    candidate_amount: Optional[float] = None

    @property
    def usable_for_regret(self) -> bool:
        return self.quality == EXACT and self.label is not None

    def note(self) -> str:
        if self.quality == TRANSLATED:
            return (f"translated: actual {self.actual_amount:g} → nearest "
                    f"evaluated {self.candidate_amount:g} "
                    f"(Δ{self.distance:g})")
        return self.quality.lower()


def map_action(report: DecisionReport, kind: str,
               amount: Optional[float],
               tolerance: float = SIZE_TOLERANCE) -> ActionMapping:
    """Map the hero's real action onto the report's candidates."""
    by_label = {c.label: c for c in report.candidates}
    if kind in ("fold", "check", "call"):
        if kind in by_label:
            return ActionMapping(kind, EXACT)
        # a call that puts the hero all-in is labeled "call" with kind
        # all_in by the menu; accept kind-level matches for passive actions
        for c in report.candidates:
            if c.kind == kind:
                return ActionMapping(c.label, EXACT)
        return ActionMapping(None, UNMAPPED)

    aggressive = [c for c in report.candidates if c.kind in AGGRESSIVE]
    if not aggressive:
        return ActionMapping(None, UNMAPPED, actual_amount=amount)

    if kind == "all_in":
        for c in report.candidates:
            if c.kind == "all_in":
                return ActionMapping(c.label, EXACT)
        # fall through: treat as a sized raise to `amount`

    if amount is None:
        return ActionMapping(None, UNMAPPED)
    best = min(aggressive, key=lambda c: abs(c.amount_to - amount))
    dist = abs(best.amount_to - amount)
    quality = EXACT if dist <= tolerance else TRANSLATED
    return ActionMapping(best.label, quality, distance=dist,
                         actual_amount=amount,
                         candidate_amount=best.amount_to)


__all__ = ["ActionMapping", "map_action", "EXACT", "TRANSLATED", "UNMAPPED",
           "SIZE_TOLERANCE", "ACTUAL_SIZE_LABEL", "AGGRESSIVE"]
