"""Session summary and confidence report over review records.

Aggregates NEVER mix methodologies: estimated regret is averaged only over
decisions where :mod:`.regret` found a valid same-method comparison, and
every aggregate carries its sample count.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from .review import DecisionRecord

LOW_SAMPLE_BELOW = 20   # decisions; below this the summary carries a warning


@dataclass(frozen=True)
class GroupStats:
    n: int = 0
    eligible_regret_n: int = 0
    total_regret_bb: float = 0.0

    @property
    def avg_regret_bb(self) -> Optional[float]:
        if not self.eligible_regret_n:
            return None
        return self.total_regret_bb / self.eligible_regret_n


@dataclass(frozen=True)
class SessionSummary:
    hands: int
    decisions: int
    solver_accepted: int
    solver_low_confidence: int
    solver_rejected: int
    solver_off: int
    rollout_decisions: int
    heuristic_decisions: int
    refused_decisions: int
    regret_eligible: int
    total_estimated_regret_bb: Optional[float]
    avg_estimated_regret_bb: Optional[float]
    categories: Dict[str, int] = field(default_factory=dict)
    by_street: Dict[str, GroupStats] = field(default_factory=dict)
    by_position: Dict[str, GroupStats] = field(default_factory=dict)
    by_source: Dict[str, GroupStats] = field(default_factory=dict)
    by_solver_state: Dict[str, GroupStats] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


def _group(records: Sequence[DecisionRecord], key) -> Dict[str, GroupStats]:
    buckets: Dict[str, list] = defaultdict(list)
    for r in records:
        buckets[key(r) or "—"].append(r)
    out: Dict[str, GroupStats] = {}
    for k, rs in sorted(buckets.items()):
        elig = [r for r in rs if r.regret.eligible
                and r.regret.estimated_regret is not None]
        out[k] = GroupStats(n=len(rs), eligible_regret_n=len(elig),
                            total_regret_bb=sum(r.regret.estimated_regret
                                                for r in elig))
    return out


def summarize(records: Sequence[DecisionRecord], hands: int,
              extra_warnings: Sequence[str] = ()) -> SessionSummary:
    states = Counter(r.solver_state for r in records)
    sources = Counter(r.source_kind for r in records)
    cats = Counter(r.category for r in records)
    elig = [r for r in records if r.regret.eligible
            and r.regret.estimated_regret is not None]
    total = sum(r.regret.estimated_regret for r in elig) if elig else None
    warnings = list(extra_warnings)
    if len(records) < LOW_SAMPLE_BELOW:
        warnings.append(
            f"low sample: only {len(records)} hero decisions — every "
            f"aggregate here is anecdote, not evidence")
    return SessionSummary(
        hands=hands, decisions=len(records),
        solver_accepted=states.get("ACCEPTED", 0),
        solver_low_confidence=states.get("LOW CONFIDENCE", 0),
        solver_rejected=states.get("REJECTED", 0),
        solver_off=states.get("OFF", 0),
        rollout_decisions=sources.get("rollout", 0),
        heuristic_decisions=sources.get("heuristic", 0),
        refused_decisions=sources.get("none", 0),
        regret_eligible=len(elig),
        total_estimated_regret_bb=total,
        avg_estimated_regret_bb=(total / len(elig)) if elig else None,
        categories=dict(cats),
        by_street=_group(records, lambda r: r.street),
        by_position=_group(records, lambda r: r.position),
        by_source=_group(records, lambda r: r.source_kind),
        by_solver_state=_group(records, lambda r: r.solver_state),
        warnings=tuple(warnings))


def confidence_report(records: Sequence[DecisionRecord],
                      recognition_confidences: Sequence[float] = ()) -> Dict:
    """Session confidence facts. Observer recognition reliability and solver
    trust are reported as SEPARATE sections and never combined into one
    number."""
    n = len(records)
    states = Counter(r.solver_state for r in records)
    sources = Counter(r.source_kind for r in records)
    confs = Counter(r.confidence for r in records)
    out = {
        "decisions": n,
        "solver_gate": {k: states.get(k, 0) for k in
                        ("ACCEPTED", "LOW CONFIDENCE", "REJECTED", "OFF")},
        "decision_source": {k: sources.get(k, 0) for k in
                            ("solver", "rollout", "heuristic", "none")},
        "decision_confidence": dict(confs),
        "warnings": [],
    }
    if recognition_confidences:
        rc = list(recognition_confidences)
        out["recognition"] = {
            "frames": len(rc),
            "min": min(rc), "mean": sum(rc) / len(rc),
            "note": ("observer recognition reliability; independent of and "
                     "never combined with solver trust"),
        }
    if n < LOW_SAMPLE_BELOW:
        out["warnings"].append(f"low sample size ({n} decisions)")
    return out


__all__ = ["GroupStats", "SessionSummary", "summarize", "confidence_report",
           "LOW_SAMPLE_BELOW"]
