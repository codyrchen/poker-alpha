"""JSON (de)serialization of reports and hand summaries for storage."""

from __future__ import annotations

from typing import Any, Dict

from ..decision.report import DecisionReport
from ..opponent.statistics import HandSummary


def report_to_dict(r: DecisionReport) -> Dict[str, Any]:
    return {
        "state_summary": r.state_summary, "hero_equity": r.hero_equity,
        "hero_equity_se": r.hero_equity_se, "pot_bb": r.pot_bb,
        "to_call_bb": r.to_call_bb, "pot_odds": r.pot_odds, "spr": r.spr,
        "effective_stack_bb": r.effective_stack_bb,
        "opponent_ranges": [{
            "seat": o.seat, "position": o.position, "line": o.line,
            "live_combos": o.live_combos, "entropy_bits": o.entropy_bits,
            "effective_combos": o.effective_combos,
            "top_classes": [list(x) for x in o.top_classes], "model": o.model}
            for o in r.opponent_ranges],
        "candidates": [{
            "label": c.label, "kind": c.kind, "amount_to": c.amount_to,
            "added": c.added, "probability": c.probability, "ev_bb": c.ev_bb,
            "ev_se_bb": c.ev_se_bb, "samples": c.samples, "source": c.source,
            "note": c.note} for c in r.candidates],
        "recommended": r.recommended, "recommended_mix": r.recommended_mix,
        "mix_meaning": r.mix_meaning, "method": r.method,
        "confidence": r.confidence, "warnings": list(r.warnings),
    }


def summary_to_dict(s: HandSummary) -> Dict[str, Any]:
    return {
        "hand_index": s.hand_index,
        "players": {str(k): v for k, v in s.players.items()},
        "positions": {str(k): v for k, v in s.positions.items()},
        "actions": [{"street": a.street, "seat": a.seat, "kind": a.kind,
                     "amount": getattr(a, "amount", 0.0)} for a in s.actions],
        "big_blind_seat": s.big_blind_seat, "big_blind": s.big_blind,
        "showdown_seats": list(s.showdown_seats),
        "won": {str(k): v for k, v in s.won.items()},
    }


class _Act:
    __slots__ = ("street", "seat", "kind", "amount")

    def __init__(self, street, seat, kind, amount):
        self.street, self.seat, self.kind, self.amount = street, seat, kind, amount


def summary_from_dict(d: Dict[str, Any]) -> HandSummary:
    return HandSummary(
        hand_index=int(d["hand_index"]),
        players={int(k): v for k, v in d["players"].items()},
        positions={int(k): v for k, v in d["positions"].items()},
        actions=[_Act(a["street"], a["seat"], a["kind"], a["amount"])
                 for a in d["actions"]],
        big_blind_seat=d["big_blind_seat"], big_blind=d["big_blind"],
        showdown_seats=tuple(d["showdown_seats"]),
        won={int(k): v for k, v in d["won"].items()})
