"""Decision-path view: how a lookup travelled through the abstraction.

Explains WHY the gate trusts or rejects a solver lookup, using only
information the engine actually recorded — nothing is reconstructed or
guessed. For non-solver decisions the path shows the cascade that was
taken instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..decision.report import DecisionReport
from ..decision.solver_gate import REASONS
from .alignment import solver_state


@dataclass(frozen=True)
class PathRow:
    label: str
    value: str
    note: str = ""


def decision_path(report: DecisionReport, state=None) -> List[PathRow]:
    rows: List[PathRow] = []
    if state is not None:
        rows.append(PathRow("Street / position",
                            f"{state.street.name.title()}"
                            + (f" · {state.hero_position}"
                               if state.hero_position else "")))
        hist = ", ".join(f"s{a.seat} {a.kind}"
                         + (f" {a.amount:g}" if a.amount else "")
                         for a in state.action_history) or "no actions yet"
        rows.append(PathRow("Action history", hist))
    info = (report.details or {}).get("solver", {})
    st = solver_state(report)
    key = (report.details or {}).get("infoset")
    if key:
        rows.append(PathRow("Abstract infoset key", str(key),
                            "the imperfect-recall encoder's bucket for this "
                            "state; strategically different hands can share it"))
        visits = (report.details or {}).get("visits")
        if visits is not None:
            rows.append(PathRow("Training visits", f"{visits:,.0f}"))
        menu = [(c.label, c.amount_to) for c in report.candidates]
        rows.append(PathRow("Abstract action menu",
                            " · ".join(f"{l} (to {t:g})" if t else l
                                       for l, t in menu)))
    rows.append(PathRow("Gate verdict", st))
    for code in info.get("reasons", ()):
        rows.append(PathRow(f"· {code}", REASONS.get(code, code)))
    signals = {k: v for k, v in info.items()
               if isinstance(v, (int, float)) and k != "used"}
    if signals:
        rows.append(PathRow("Gate signals",
                            ", ".join(f"{k}={v:g}" for k, v in signals.items())))
    if "OFF_TREE_TRANSLATION" in info.get("reasons", ()):
        rows.append(PathRow(
            "Size translation",
            "observed bet sizes were translated to the nearest sizes of the "
            "trained tree (log-space nearest)",
            "the mix shown answers the TRANSLATED spot, not the literal one"))
    if not key:
        cascade = (report.details or {}).get("source_cascade", [])
        for step in cascade:
            rows.append(PathRow(step.get("source", "?"),
                                step.get("status", ""),
                                step.get("reason", "")))
    return rows


__all__ = ["PathRow", "decision_path"]
