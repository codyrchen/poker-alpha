"""Decision stability: sweep one variable and find where the decision
changes.

Built on the what-if machinery (same single-variable discipline, same
validation); every point is a real pipeline output. The result is a
description of boundaries — where the highest-frequency action changes,
where the solver gate degrades (ACCEPTED → LOW CONFIDENCE → REJECTED),
and where the decision method changes — presented as *decision
stability*, not causal inference.

Example reading of a result:

    "call stays the top action from 1.65 to 4.5 bb; at 6.0 the lookup is
    low confidence; at 9.0 it is rejected and PokerAlpha falls back to
    the rollout."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..decision import DecisionConfig
from .whatif import Variant, WhatIfError, analyze_variant, apply_whatif


@dataclass(frozen=True)
class SweepPoint:
    value: str
    top_action: Optional[str]
    solver_state: str
    method_kind: str                 # solver / rollout / heuristic
    distribution: Dict[str, float]


@dataclass(frozen=True)
class Boundary:
    kind: str                        # "top_action" | "solver_state" | "method"
    at_value: str                    # first value where the change holds
    before: str
    after: str

    def text(self) -> str:
        what = {"top_action": "highest-frequency action",
                "solver_state": "solver gate",
                "method": "decision method"}[self.kind]
        return f"{what} changes {self.before} → {self.after} at {self.at_value}"


@dataclass(frozen=True)
class StabilityResult:
    variable: str
    points: Tuple[SweepPoint, ...]
    boundaries: Tuple[Boundary, ...]
    rejected: Dict[str, str]
    summary: str


def _kind(method: str) -> str:
    m = method.lower()
    if "solver" in m or "abstraction" in m:
        return "solver"
    if "rollout" in m:
        return "rollout"
    return "heuristic"


def _point(v: Variant) -> SweepPoint:
    dist = {l: (f or 0.0) for l, f in v.top_actions}
    top = max(dist, key=dist.get) if dist else None
    return SweepPoint(v.value, top, v.solver_state, _kind(v.method), dist)


def sweep(base: dict, variable: str, values: Sequence,
          config: Optional[DecisionConfig] = None,
          solver=None) -> StabilityResult:
    """Analyse the base state at every value; report the boundaries."""
    points: List[SweepPoint] = []
    rejected: Dict[str, str] = {}
    for v in values:
        try:
            st = apply_whatif(base, variable, v)
        except WhatIfError as exc:
            rejected[str(v)] = str(exc)
            continue
        points.append(_point(analyze_variant(st, variable, v, config=config,
                                             solver=solver)))
    boundaries: List[Boundary] = []
    for prev, cur in zip(points, points[1:]):
        if cur.top_action != prev.top_action:
            boundaries.append(Boundary("top_action", cur.value,
                                       str(prev.top_action),
                                       str(cur.top_action)))
        if cur.solver_state != prev.solver_state:
            boundaries.append(Boundary("solver_state", cur.value,
                                       prev.solver_state, cur.solver_state))
        if cur.method_kind != prev.method_kind:
            boundaries.append(Boundary("method", cur.value,
                                       prev.method_kind, cur.method_kind))
    if not points:
        summary = "no valid points in the sweep"
    elif not boundaries:
        p = points[0]
        summary = (f"stable: {p.top_action} stays the highest-frequency "
                   f"action across {variable} ∈ "
                   f"[{points[0].value}, {points[-1].value}] "
                   f"(gate {p.solver_state} throughout)")
    else:
        summary = "; ".join(b.text() for b in boundaries)
    return StabilityResult(variable, tuple(points), tuple(boundaries),
                           rejected, summary)


__all__ = ["SweepPoint", "Boundary", "StabilityResult", "sweep"]
