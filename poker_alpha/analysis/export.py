"""Local exports of a session review: JSON / CSV / Markdown.

Everything stays on disk, chosen by the user — no upload anywhere.
Exports preserve provenance (method, source, solver gate state and reason
codes), regret eligibility with its reason, and confidence; they never
collapse those into one score.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Dict, Optional, Sequence

from .patterns import Pattern
from .regret import INELIGIBLE_REASONS
from .review import DecisionRecord, SessionReview
from .summary import SessionSummary

EXPORT_FORMAT = "pokeralpha.session_review/v1"

CSV_COLUMNS = [
    "hand", "decision", "street", "position", "hero_cards", "board",
    "pot_bb", "to_call_bb", "effective_stack_bb",
    "actual", "actual_amount", "actual_label", "pokeralpha",
    "actual_frequency", "method", "source", "solver_state",
    "decision_confidence", "regret_eligible", "regret_reason",
    "estimated_regret_bb", "regret_uncertainty_bb", "regret_source",
    "alignment", "policy_surprise_bits", "category", "off_tree",
    "mapping_quality", "mapping_distance",
]


def record_to_dict(r: DecisionRecord) -> Dict:
    return {
        "hand": r.hand_index, "decision": r.decision_index,
        "street": r.street, "position": r.position,
        "hero_cards": r.hero_cards, "board": r.board,
        "pot_bb": r.pot_bb, "to_call_bb": r.to_call_bb,
        "effective_stack_bb": r.effective_stack_bb,
        "actual": r.actual_kind, "actual_amount": r.actual_amount,
        "actual_label": r.actual_label, "pokeralpha": r.recommended,
        "actual_frequency": r.actual_frequency,
        "method": r.method, "source": r.source_kind,
        "solver_state": r.solver_state,
        "decision_confidence": r.confidence,
        "gate_reasons": list(r.gate_reasons),
        "warnings": list(r.warnings),
        "regret": {
            "eligible": r.regret.eligible,
            "reason": r.regret.reason,
            "reason_text": INELIGIBLE_REASONS.get(r.regret.reason, ""),
            "best_ev_bb": r.regret.best_ev,
            "actual_ev_bb": r.regret.actual_ev,
            "estimated_regret_bb": r.regret.estimated_regret,
            "uncertainty_bb": r.regret.uncertainty,
            "uncertainty_source": r.regret.uncertainty_source,
            "source": r.regret.source,
        },
        "action_mapping": {
            "quality": r.mapping_quality,
            "distance": r.mapping_distance,
        },
        "alignment": {
            "category": r.alignment.category,
            "text": r.alignment.text,
            "frequency": r.alignment.frequency,
            "highest_label": r.alignment.highest_label,
            "highest_frequency": r.alignment.highest_frequency,
            "policy_surprise_bits": r.alignment.surprise_bits,
        },
        "category": r.category, "off_tree": r.off_tree,
    }


def review_to_json(review: SessionReview, summary: SessionSummary,
                   patterns: Sequence[Pattern] = ()) -> str:
    def gs(d):
        return {k: {"n": v.n, "regret_eligible": v.eligible_regret_n,
                    "total_estimated_regret_bb": v.total_regret_bb,
                    "avg_estimated_regret_bb": v.avg_regret_bb}
                for k, v in d.items()}

    payload = {
        "format": EXPORT_FORMAT,
        "semantics": {
            "estimated_regret": ("ESTIMATED DECISION REGRET under the rollout "
                                 "model; only present where the EVs were the "
                                 "actual decision basis, same-method, and the "
                                 "exact action taken was itself evaluated; "
                                 "uncertainty is the paired-difference SE "
                                 "(common random numbers), never an "
                                 "independent-SE combination"),
            "alignment": "solver-mix frequency of the action actually taken",
        },
        "summary": {
            "hands": summary.hands, "decisions": summary.decisions,
            "solver_accepted": summary.solver_accepted,
            "solver_low_confidence": summary.solver_low_confidence,
            "solver_rejected": summary.solver_rejected,
            "solver_off": summary.solver_off,
            "rollout_decisions": summary.rollout_decisions,
            "heuristic_decisions": summary.heuristic_decisions,
            "refused_decisions": summary.refused_decisions,
            "regret_eligible": summary.regret_eligible,
            "total_estimated_regret_bb": summary.total_estimated_regret_bb,
            "avg_estimated_regret_bb": summary.avg_estimated_regret_bb,
            "categories": summary.categories,
            "by_street": gs(summary.by_street),
            "by_position": gs(summary.by_position),
            "by_source": gs(summary.by_source),
            "by_solver_state": gs(summary.by_solver_state),
            "warnings": list(summary.warnings),
        },
        "patterns": [{
            "kind": p.kind, "label": p.label, "description": p.description,
            "count": p.count, "hands": list(p.hands),
            "breakdown": p.breakdown} for p in patterns],
        "decisions": [record_to_dict(r) for r in review.records],
        "review_warnings": list(review.warnings),
    }
    return json.dumps(payload, indent=1)


def review_to_csv(review: SessionReview) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for r in review.records:
        w.writerow({
            "hand": r.hand_index, "decision": r.decision_index,
            "street": r.street, "position": r.position or "",
            "hero_cards": r.hero_cards, "board": r.board,
            "pot_bb": r.pot_bb, "to_call_bb": r.to_call_bb,
            "effective_stack_bb": r.effective_stack_bb,
            "actual": r.actual_kind, "actual_amount": r.actual_amount,
            "actual_label": r.actual_label or "",
            "pokeralpha": r.recommended or "",
            "actual_frequency": r.actual_frequency,
            "method": r.method, "source": r.source_kind,
            "solver_state": r.solver_state,
            "decision_confidence": r.confidence,
            "regret_eligible": r.regret.eligible,
            "regret_reason": r.regret.reason,
            "estimated_regret_bb": r.regret.estimated_regret,
            "regret_uncertainty_bb": r.regret.uncertainty,
            "regret_source": r.regret.source,
            "alignment": r.alignment.category,
            "policy_surprise_bits": r.alignment.surprise_bits,
            "category": r.category, "off_tree": r.off_tree,
            "mapping_quality": r.mapping_quality,
            "mapping_distance": r.mapping_distance,
        })
    return buf.getvalue()


def _fmt(x: Optional[float], spec: str = ".2f") -> str:
    return "—" if x is None else format(x, spec)


def review_to_markdown(review: SessionReview, summary: SessionSummary,
                       patterns: Sequence[Pattern] = ()) -> str:
    s = summary
    lines = [
        "# PokerAlpha session review", "",
        f"{s.hands} hands · {s.decisions} hero decisions", "",
        "## Summary", "",
        "| metric | value |", "|---|---|",
        f"| solver accepted | {s.solver_accepted} |",
        f"| solver low confidence | {s.solver_low_confidence} |",
        f"| solver rejected | {s.solver_rejected} |",
        f"| rollout decisions | {s.rollout_decisions} |",
        f"| heuristic decisions | {s.heuristic_decisions} |",
        f"| refused | {s.refused_decisions} |",
        f"| decisions with valid EV regret | {s.regret_eligible} |",
        f"| total ESTIMATED decision regret | "
        f"{_fmt(s.total_estimated_regret_bb)} BB "
        f"(n={s.regret_eligible}) |",
        f"| average ESTIMATED decision regret | "
        f"{_fmt(s.avg_estimated_regret_bb)} BB |",
        "",
        "Estimated regret is a same-method rollout estimate, not money lost;",
        "decisions without a valid comparison are never averaged in.",
        "",
    ]
    for w in s.warnings:
        lines.append(f"> ⚠ {w}")
    if patterns:
        lines += ["", "## Patterns", ""]
        for p in patterns:
            lines.append(f"- **{p.label}** ({p.count}× in hands "
                         f"{', '.join(map(str, p.hands[:8]))}"
                         f"{'…' if len(p.hands) > 8 else ''}): {p.description}")
    lines += ["", "## Decisions", "",
              "| hand | street | pos | actual | PokerAlpha | freq | solver | "
              "est. regret | category |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in review.records:
        freq = "—" if r.actual_frequency is None else f"{r.actual_frequency:.0%}"
        reg = (_fmt(r.regret.estimated_regret) + " BB"
               if r.regret.eligible else "unavailable")
        lines.append(
            f"| {r.hand_index} | {r.street} | {r.position or '—'} | "
            f"{r.actual_label or r.actual_kind} | {r.recommended or '—'} | "
            f"{freq} | {r.solver_state} | {reg} | {r.category} |")
    return "\n".join(lines) + "\n"


__all__ = ["EXPORT_FORMAT", "CSV_COLUMNS", "record_to_dict",
           "review_to_json", "review_to_csv", "review_to_markdown"]
