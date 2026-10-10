"""Decision report data model.

Every number carries its provenance (``source``) and, where it is an
estimate, its uncertainty. Sources:

* ``"solver"`` — frequencies from a trained heads-up abstract strategy at an
  exactly matching abstract information set.
* ``"interpolated abstraction"`` — solver frequencies after translating
  off-tree bet sizes onto the abstraction.
* ``"Monte Carlo rollout"`` — range-based simulation of opponent responses
  (approximate; never labelled GTO).
* ``"heuristic fallback"`` — closed-form pot-odds/equity reasoning only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SOURCES = ("solver", "interpolated abstraction", "Monte Carlo rollout",
           "heuristic fallback")


@dataclass(frozen=True)
class CandidateAction:
    label: str                    # abstract label, e.g. "bet_75", "call"
    kind: str                     # fold/check/call/bet/raise/all_in
    amount_to: float              # street commitment after acting (chips)
    added: float                  # chips put in by this action
    probability: Optional[float]  # recommended frequency (see report.mix_meaning)
    ev_bb: Optional[float]        # EV in big blinds, relative to folding now
    ev_se_bb: Optional[float]     # standard error of ev_bb
    samples: int = 0
    source: str = "heuristic fallback"
    note: str = ""


@dataclass(frozen=True)
class RangeSummary:
    seat: int
    position: str
    line: str
    live_combos: int
    entropy_bits: float
    effective_combos: float
    top_classes: Tuple[Tuple[str, float], ...]
    model: str = ""


@dataclass(frozen=True)
class DecisionReport:
    state_summary: str
    hero_equity: Optional[float]
    hero_equity_se: Optional[float]
    pot_bb: float
    to_call_bb: float
    pot_odds: Optional[float]
    spr: Optional[float]
    effective_stack_bb: Optional[float]
    opponent_ranges: Tuple[RangeSummary, ...]
    candidates: Tuple[CandidateAction, ...]
    recommended: Optional[str]
    recommended_mix: Dict[str, float]
    mix_meaning: str              # how to read the probabilities
    method: str                   # dominant source
    confidence: str               # low / medium / high
    warnings: Tuple[str, ...] = ()
    details: Dict[str, object] = field(default_factory=dict)
    # Separate uncertainty sources: observation, range_estimation, sampling,
    # abstraction, response_model (see recommend._uncertainty).
    uncertainty: Dict[str, object] = field(default_factory=dict)

    def candidate(self, label: str) -> CandidateAction:
        for c in self.candidates:
            if c.label == label:
                return c
        raise KeyError(label)

    def format(self) -> str:
        lines = [self.state_summary]
        if self.hero_equity is not None:
            eq = f"Equity: {self.hero_equity:.1%}"
            if self.hero_equity_se:
                eq += f" (±{self.hero_equity_se:.1%})"
            lines.append(eq)
        if self.pot_odds is not None:
            lines.append(f"Pot odds: {self.pot_odds:.1%}")
        if self.spr is not None:
            lines.append(f"SPR: {self.spr:.2f}")
        for r in self.opponent_ranges:
            top = ", ".join(f"{c} {p:.0%}" for c, p in r.top_classes[:5])
            lines.append(f"  seat {r.seat} {r.position} [{r.line}] "
                         f"~{r.effective_combos:.0f} eff. combos: {top}")
        lines.append("")
        lines.append(f"{'Action':<14}{'Freq':>8}{'EV (BB)':>12}{'±SE':>8}  source")
        for c in self.candidates:
            freq = "" if c.probability is None else f"{c.probability:.0%}"
            ev = "n/a" if c.ev_bb is None else f"{c.ev_bb:+.2f}"
            se = "" if c.ev_se_bb is None else f"{c.ev_se_bb:.2f}"
            lines.append(f"{c.label:<14}{freq:>8}{ev:>12}{se:>8}  {c.source}")
        lines.append("")
        lines.append(f"Recommendation: {self.recommended} "
                     f"(method: {self.method}; confidence: {self.confidence})")
        lines.append(f"Frequencies mean: {self.mix_meaning}")
        cascade = self.details.get("source_cascade") or []
        if cascade:
            lines.append("Decision sources (priority solver -> rollout -> heuristic):")
            for c in cascade:
                extra = f" [{c['code']}]" if c.get("code") else ""
                why = f": {c['reason']}" if c.get("reason") else ""
                lines.append(f"  {c['source']}: {c['status']}{extra}{why}")
        if self.uncertainty:
            lines.append("Uncertainty by source:")
            for k, v in self.uncertainty.items():
                lines.append(f"  {k}: {v}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)
