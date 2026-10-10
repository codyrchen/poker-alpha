"""Pure presentation helpers shared by the UI (no Streamlit import)."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..decision.report import DecisionReport
from ..holdem.observed import ObservedTableState
from ..poker.cards import card_str


def _cards(cs) -> str:
    return " ".join(card_str(c) for c in cs) if cs else "-"


def _bb(obs: ObservedTableState, x: Optional[float]) -> str:
    return "?" if x is None else f"{x / obs.big_blind:.1f} BB"


def state_rows(obs: ObservedTableState) -> List[Dict[str, str]]:
    """Key facts about the hero's spot, in display order."""
    eff = obs.effective_stack
    return [
        {"field": "Hero", "value": _cards(obs.hero_cards)},
        {"field": "Board", "value": _cards(obs.board)},
        {"field": "Street", "value": obs.street.name.title()},
        {"field": "Position", "value": obs.hero_position or "?"},
        {"field": "Hero stack", "value": _bb(obs, obs.hero.stack)},
        {"field": "Effective stack", "value": _bb(obs, eff)},
        {"field": "Pot", "value": _bb(obs, obs.pot)},
        {"field": "Amount to call", "value": _bb(obs, obs.amount_to_call)},
        {"field": "SPR", "value": "?" if obs.spr is None else f"{obs.spr:.2f}"},
    ]


def seat_rows(obs: ObservedTableState) -> List[Dict[str, object]]:
    pos = obs.positions()
    out = []
    for s in obs.seats:
        if not s.occupied:
            continue
        status = "folded" if s.folded else ("all-in" if s.all_in else "in hand")
        out.append({"seat": s.seat, "position": pos.get(s.seat, ""),
                    "name": s.name, "stack (BB)": None if s.stack is None
                    else round(s.stack / obs.big_blind, 2),
                    "bet (BB)": round(s.current_bet / obs.big_blind, 2),
                    "status": ("HERO " if s.seat == obs.hero_seat else "") + status})
    return out


def candidate_rows(report: DecisionReport) -> List[Dict[str, object]]:
    rows = []
    for c in report.candidates:
        rows.append({
            "action": c.label,
            "frequency": None if c.probability is None else round(100 * c.probability, 1),
            "EV (BB)": None if c.ev_bb is None else round(c.ev_bb, 2),
            "± SE (BB)": None if c.ev_se_bb is None else round(c.ev_se_bb, 2),
            "samples": c.samples,
            "source": c.source,
            "note": c.note,
        })
    return rows


def range_rows(report: DecisionReport) -> List[Dict[str, object]]:
    return [{"seat": r.seat, "position": r.position, "line": r.line,
             "model": r.model, "live combos": r.live_combos,
             "effective combos": round(r.effective_combos),
             "entropy (bits)": round(r.entropy_bits, 2),
             "top classes": ", ".join(f"{c} {p:.0%}" for c, p in r.top_classes[:6])}
            for r in report.opponent_ranges]


def headline(report: DecisionReport) -> str:
    if report.recommended is None:
        return "No recommendation (see warnings)"
    return (f"Suggested: {report.recommended}  ·  method: {report.method}  ·  "
            f"confidence: {report.confidence}")


def solver_status(report: DecisionReport) -> str:
    """One line on whether the solver strategy was used and why (not)."""
    info = report.details.get("solver") or {}
    if not info or info.get("confidence") == "not configured":
        return "Solver strategy: not loaded"
    reasons = ", ".join(info.get("reasons") or [])
    if info.get("used"):
        return f"Solver strategy used · gate {info.get('confidence')}" + (f" ({reasons})" if reasons else "")
    fallback = report.method
    return f"Solver strategy not used: {reasons or 'unknown reason'} · fallback: {fallback}"


def source_rows(report: DecisionReport) -> List[Dict[str, object]]:
    rows = []
    for c in report.details.get("source_cascade") or []:
        rows.append({"source": c.get("source"), "status": c.get("status"),
                     "reasons": ", ".join(c.get("reasons") or ([c["code"]] if c.get("code") else [])),
                     "detail": c.get("reason", "")})
    return rows


def solver_signal_rows(report: DecisionReport) -> List[Dict[str, object]]:
    info = report.details.get("solver") or {}
    return [{"signal": k, "value": info.get(k)} for k in
            ("confidence", "visits", "seed_disagreement", "movement", "collision") if k in info]


def uncertainty_rows(report: DecisionReport) -> List[Dict[str, object]]:
    return [{"source of uncertainty": k, "detail": str(v)} for k, v in report.uncertainty.items()]
