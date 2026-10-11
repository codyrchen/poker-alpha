"""Decision Surface: where does the recommended strategy change?

A 2D grid over two state parameters (e.g. opponent bet size × effective
stack). EVERY cell is a real engine analysis of a fully re-validated
state — there is no interpolation, no smoothing, and no invented cells:
a parameter combination that does not describe a possible poker state is
stored as invalid with its reason.

Built on the what-if machinery, so each cell differs from the base state
in exactly the two swept variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..decision import DecisionConfig
from .stability import _kind  # solver / rollout / heuristic
from .whatif import WhatIfError, analyze_variant, apply_whatif


@dataclass(frozen=True)
class SurfaceCell:
    x: float
    y: float
    valid: bool
    reason: str = ""                      # when invalid
    top_action: Optional[str] = None
    mix: Dict[str, float] = None          # label -> probability (top 5)
    solver_state: str = ""
    method_kind: str = ""
    confidence: str = ""
    warnings: int = 0


def generate_surface(base: dict, x_var: str, x_values: Sequence[float],
                     y_var: str, y_values: Sequence[float],
                     config: Optional[DecisionConfig] = None,
                     solver=None,
                     progress=None) -> List[List[SurfaceCell]]:
    """Grid of real analyses: rows indexed by y, columns by x."""
    if x_var == y_var:
        raise ValueError("x and y must sweep different variables")
    rows: List[List[SurfaceCell]] = []
    for y in y_values:
        row: List[SurfaceCell] = []
        for x in x_values:
            try:
                st = apply_whatif(base, y_var, y)
                st = apply_whatif(st, x_var, x)
            except WhatIfError as exc:
                row.append(SurfaceCell(float(x), float(y), False,
                                       reason=str(exc)))
                continue
            v = analyze_variant(st, f"{x_var}x{y_var}", f"{x},{y}",
                                config=config, solver=solver)
            dist = {l: round(f or 0.0, 4) for l, f in v.top_actions[:5]}
            top = max(dist, key=dist.get) if dist else None
            row.append(SurfaceCell(
                float(x), float(y), True, top_action=top, mix=dist,
                solver_state=v.solver_state, method_kind=_kind(v.method),
                confidence=v.confidence, warnings=len(v.warnings)))
            if progress:
                progress(x, y)
        rows.append(row)
    return rows


def surface_to_dict(cells: List[List[SurfaceCell]], base: dict,
                    x_var: str, y_var: str) -> dict:
    return {
        "x_var": x_var, "y_var": y_var,
        "x_values": [c.x for c in cells[0]] if cells else [],
        "y_values": [row[0].y for row in cells],
        "base_state": base,
        "cells": [[{
            "valid": c.valid, "reason": c.reason, "top": c.top_action,
            "mix": c.mix, "gate": c.solver_state, "method": c.method_kind,
            "confidence": c.confidence, "warnings": c.warnings}
            for c in row] for row in cells],
        "semantics": ("every cell is a real engine analysis of a "
                      "re-validated state; no interpolation"),
    }


__all__ = ["SurfaceCell", "generate_surface", "surface_to_dict"]
