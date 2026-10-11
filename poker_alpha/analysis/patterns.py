"""Pattern detection over review records — tendencies, not verdicts.

Every detector requires a defensible minimum sample and uses graded
language tied to the count:

    occurrences >= MIN_PATTERN (3)   ->  "Potential pattern"
    occurrences >= MIN_REPEATED (5)  ->  "Repeated deviation"

Detectors only look where the evidence is meaningful: frequency-based
tendencies use decisions where the solver was ACCEPTED (the mix is
trusted); off-tree and rejected-region detectors count the gate's own
signals. Nothing here claims EV loss.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .review import DecisionRecord

MIN_PATTERN = 3
MIN_REPEATED = 5

AGGRESSIVE = ("bet", "raise", "all_in")
PASSIVE = ("check", "call", "fold")


@dataclass(frozen=True)
class Pattern:
    kind: str            # machine code, e.g. "rare_fold"
    label: str           # graded: Potential pattern / Repeated deviation
    description: str
    count: int
    hands: Tuple[int, ...]            # hand indices, for drill-down
    breakdown: Dict[str, int]         # e.g. by street or position


def _grade(count: int) -> str:
    return "Repeated deviation" if count >= MIN_REPEATED else "Potential pattern"


def _pattern(kind: str, desc: str, recs: Sequence[DecisionRecord],
             by) -> Pattern:
    return Pattern(kind=kind, label=_grade(len(recs)), description=desc,
                   count=len(recs),
                   hands=tuple(sorted({r.hand_index for r in recs})),
                   breakdown=dict(Counter(by(r) or "—" for r in recs)))


def detect_patterns(records: Sequence[DecisionRecord]) -> List[Pattern]:
    out: List[Pattern] = []

    # 1. rare choices inside TRUSTED solver mixes, split by action family
    trusted = [r for r in records if r.solver_state == "ACCEPTED"
               and r.alignment.category in ("RARE", "NOT_IN_MIX")]
    fams = {"rare_fold": [r for r in trusted if r.actual_kind == "fold"],
            "rare_call": [r for r in trusted if r.actual_kind in ("call", "check")],
            "rare_aggressive": [r for r in trusted
                                if r.actual_kind in AGGRESSIVE]}
    descs = {"rare_fold": "folds that are rare or absent in the trained mix",
             "rare_call": "passive actions rare or absent in the trained mix",
             "rare_aggressive": "aggressive actions rare or absent in the "
                                "trained mix"}
    for kind, recs in fams.items():
        if len(recs) >= MIN_PATTERN:
            out.append(_pattern(kind, descs[kind], recs, lambda r: r.street))

    # 2. direction of deviation vs the mix's highest-frequency action
    passive_dev = [r for r in trusted
                   if r.actual_kind in PASSIVE
                   and r.alignment.highest_label
                   and any(a in r.alignment.highest_label for a in AGGRESSIVE)]
    aggro_dev = [r for r in trusted
                 if r.actual_kind in AGGRESSIVE
                 and r.alignment.highest_label in ("check", "call", "fold")]
    if len(passive_dev) >= MIN_PATTERN:
        out.append(_pattern("passive_deviation",
                            "passive choices where the mix leans aggressive",
                            passive_dev, lambda r: r.street))
    if len(aggro_dev) >= MIN_PATTERN:
        out.append(_pattern("aggressive_deviation",
                            "aggressive choices where the mix leans passive",
                            aggro_dev, lambda r: r.street))

    # 3. position- and street-concentrated review flags
    flagged = [r for r in records if r.category in
               ("HIGH-PRIORITY EV REVIEW", "STRATEGY REVIEW")]
    by_pos: Dict[str, List[DecisionRecord]] = defaultdict(list)
    for r in flagged:
        by_pos[r.position or "—"].append(r)
    for pos, recs in sorted(by_pos.items()):
        if len(recs) >= MIN_PATTERN:
            out.append(_pattern(
                f"position_{pos.lower()}",
                f"review-flagged decisions concentrated at {pos}",
                recs, lambda r: r.street))
    pre = [r for r in flagged if r.street == "Preflop"]
    post = [r for r in flagged if r.street != "Preflop"]
    for name, recs in (("preflop_flags", pre), ("postflop_flags", post)):
        if len(recs) >= MIN_PATTERN:
            out.append(_pattern(
                name, f"review-flagged decisions are "
                      f"{'preflop' if recs is pre else 'postflop'}-heavy",
                recs, lambda r: r.position))

    # 4. repeated off-tree sizing (the gate translated the observed sizes)
    off = [r for r in records if r.off_tree]
    if len(off) >= MIN_PATTERN:
        out.append(_pattern(
            "off_tree_sizing",
            "bet sizes repeatedly off the trained tree (translated lookups)",
            off, lambda r: r.street))

    # 5. repeated solver-rejected regions
    rej = [r for r in records if r.solver_state == "REJECTED"]
    if len(rej) >= MIN_PATTERN:
        out.append(_pattern(
            "solver_rejected_regions",
            "spots repeatedly landing where the gate rejects the strategy",
            rej, lambda r: r.street))

    return out


__all__ = ["Pattern", "detect_patterns", "MIN_PATTERN", "MIN_REPEATED"]
