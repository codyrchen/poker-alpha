"""Trainer / quiz mode: show a state, take an answer, reveal PokerAlpha.

Spots come from a deterministic library executed through the REAL pipeline
(release strategy + gate + rollouts) — labels in the library describe what
the engine actually does there and are verified by tests, never assumed.

Scoring language follows the research semantics:

* accepted solver mixes grade by the chosen action's frequency —
  "Common in solver mix — 64%", "Mixed option — 18%",
  "Rare in solver mix — 2%" — never binary right/wrong;
* rollout decisions with a valid same-method comparison show
  ESTIMATED DECISION REGRET;
* uncertain solver states (low confidence / rejected) are review-only:
  "PokerAlpha uncertain — review only", with no penalty.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..decision import DecisionConfig, DecisionReport
from ..pipeline import analyze, observe_manual
from .alignment import solver_state, strategy_alignment
from .regret import regret_eligibility

# ---------------------------------------------------------------------------
# deterministic spot library (states, not canned answers — the engine runs)
# ---------------------------------------------------------------------------

_PRE = [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
        {"street": 0, "seat": 1, "kind": "call", "amount": 1.5}]
_TO_TURN = _PRE + [{"street": 1, "seat": 1, "kind": "check"},
                   {"street": 1, "seat": 0, "kind": "check"}]
_TO_RIVER = _TO_TURN + [{"street": 2, "seat": 1, "kind": "check"},
                        {"street": 2, "seat": 0, "kind": "check"}]


def _hu(hero_cards, board, pot, actions, committed, stacks, bets=(0.0, 0.0)):
    return {"num_seats": 2, "hero_seat": 0, "dealer": 0,
            "small_blind": 0.5, "big_blind": 1.0,
            "hero_cards": hero_cards, "board": board, "pot": pot, "actor": 0,
            "seats": [{"stack": stacks[0], "committed": committed[0],
                       "bet": bets[0]},
                      {"stack": stacks[1], "committed": committed[1],
                       "bet": bets[1]}],
            "actions": actions}


@dataclass(frozen=True)
class TrainerSpot:
    name: str
    prompt: str                   # what to think about, no spoilers
    state: dict
    covers: Tuple[str, ...]       # streets / engine states this spot exercises


SPOT_LIBRARY: Tuple[TrainerSpot, ...] = (
    TrainerSpot(
        "River value spot, checked to you",
        "Top pair on a dry runout; villain has checked every street.",
        _hu("Ah Kd", "Qs 7d 2c Th 3h", 5.0,
            _TO_RIVER + [{"street": 3, "seat": 1, "kind": "check"}],
            (2.5, 2.5), (97.5, 97.5)),
        ("river", "solver_accepted")),
    TrainerSpot(
        "River, facing a half-pot bet",
        "Middling made hand; the size is between the trained tree's sizes.",
        _hu("9h 9c", "Qs 7d 2c Th 3h", 7.5,
            _TO_RIVER + [{"street": 3, "seat": 1, "kind": "bet",
                          "amount": 2.5}],
            (2.5, 5.0), (97.5, 95.0), bets=(0.0, 2.5)),
        ("river", "solver_low_confidence", "off_tree")),
    TrainerSpot(
        "Preflop open decision",
        "First action on the button, 100bb deep.",
        _hu("Ad Qs", "", 1.5, [], (0.5, 1.0), (99.5, 99.0), bets=(0.5, 1.0)),
        ("preflop", "gate_rejected", "rollout_fallback")),
    TrainerSpot(
        "Preflop, facing a 3-bet",
        "You opened, the big blind re-raised to 9bb.",
        _hu("Ts Td", "", 11.5,
            [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
             {"street": 0, "seat": 1, "kind": "raise", "amount": 9.0}],
            (2.5, 9.0), (97.5, 91.0), bets=(2.5, 9.0)),
        ("preflop", "rollout_fallback")),
    TrainerSpot(
        "Flop c-bet decision",
        "You raised preflop and were called; the flop checks to you.",
        _hu("Ah Kd", "Qs 7d 2c", 5.0,
            _PRE + [{"street": 1, "seat": 1, "kind": "check"}],
            (2.5, 2.5), (97.5, 97.5)),
        ("flop", "solver_low_confidence")),
    TrainerSpot(
        "Turn after flop goes check/check",
        "Second barrel opportunity on a board that improved you.",
        _hu("Ah Kd", "Qs 7d 2c Kh", 5.0,
            _TO_TURN + [{"street": 2, "seat": 1, "kind": "check"}],
            (2.5, 2.5), (97.5, 97.5)),
        ("turn", "solver_low_confidence")),
)


# ---------------------------------------------------------------------------
# answering and feedback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainerFeedback:
    chosen: str
    grade: str          # alignment category, REGRET, or UNCERTAIN
    headline: str       # the one line shown first
    detail: str         # neutral explanation
    solver_state: str
    method: str
    frequency: Optional[float] = None
    estimated_regret_bb: Optional[float] = None


def run_spot(spot: TrainerSpot, config: Optional[DecisionConfig] = None,
             solver=None) -> DecisionReport:
    return analyze(observe_manual(spot.state), config=config, solver=solver)


def answer_options(report: DecisionReport) -> List[str]:
    return [c.label for c in report.candidates]


def evaluate_answer(report: DecisionReport, chosen: str) -> TrainerFeedback:
    st = solver_state(report)
    method = report.method
    m = method.lower()
    solver_answer = "solver" in m or "abstraction" in m
    if solver_answer and st == "LOW CONFIDENCE":
        align = strategy_alignment(report, chosen)
        f = align.frequency
        extra = ""
        if f is not None:
            extra = f" (your action: {f:.0%} of the downgraded mix)"
        return TrainerFeedback(
            chosen=chosen, grade="UNCERTAIN",
            headline="PokerAlpha uncertain — review only",
            detail=("The confidence gate does not fully trust the trained "
                    "strategy here, so your choice is not graded." + extra),
            solver_state=st, method=method, frequency=f)
    if solver_answer and st == "ACCEPTED":
        align = strategy_alignment(report, chosen)
        f = align.frequency or 0.0
        pct = "<1%" if 0 < f < 0.005 else f"{f:.0%}"
        if align.category == "COMMON":
            head = f"Common in solver mix — {pct}"
        elif align.category == "MIXED":
            head = f"Mixed option — {pct}"
        elif align.category == "RARE":
            head = f"Rare in solver mix — {pct}"
        else:
            head = "Not in the trained mix"
        hi = align.highest_label
        detail = ("A mixed strategy plays several actions on purpose; "
                  "frequency is how often this one is chosen.")
        if hi and hi != chosen:
            detail += (f" The mix's most frequent action is {hi} "
                       f"({align.highest_frequency:.0%}).")
        return TrainerFeedback(chosen=chosen, grade=align.category,
                               headline=head, detail=detail,
                               solver_state=st, method=method, frequency=f)
    # rollout / heuristic path (incl. fallback after a gate rejection)
    reg = regret_eligibility(report, chosen)
    if reg.eligible:
        r = reg.estimated_regret
        head = ("Matches the rollout's best action" if abs(r) < 1e-9 else
                f"Estimated decision regret {r:+.2f} BB"
                + (f" ±{reg.uncertainty:.2f}" if reg.uncertainty else ""))
        detail = ("An estimate under the rollout model — the same "
                  "methodology that produced the recommendation; "
                  "not money lost.")
        if st == "REJECTED":
            detail += (" The trained strategy was withheld here, so this "
                       "grades you against the rollout fallback only.")
        return TrainerFeedback(
            chosen=chosen, grade="REGRET", headline=head,
            detail=detail, solver_state=st, method=method,
            estimated_regret_bb=r)
    return TrainerFeedback(
        chosen=chosen, grade="UNSCORED",
        headline="Not scored",
        detail=f"EV regret unavailable — {reg.reason_text}.",
        solver_state=st, method=method)


# ---------------------------------------------------------------------------
# local session stats (no login, no database, no cloud)
# ---------------------------------------------------------------------------

@dataclass
class TrainerStats:
    attempted: int = 0
    by_grade: Counter = field(default_factory=Counter)
    by_street: Counter = field(default_factory=Counter)
    regret_n: int = 0
    regret_total_bb: float = 0.0
    uncertain: int = 0

    def record(self, spot: TrainerSpot, fb: TrainerFeedback) -> None:
        self.attempted += 1
        self.by_grade[fb.grade] += 1
        street = next((c for c in spot.covers
                       if c in ("preflop", "flop", "turn", "river")), "?")
        self.by_street[street] += 1
        if fb.grade == "UNCERTAIN":
            self.uncertain += 1
        if fb.estimated_regret_bb is not None:
            self.regret_n += 1
            self.regret_total_bb += fb.estimated_regret_bb

    @property
    def avg_regret_bb(self) -> Optional[float]:
        return self.regret_total_bb / self.regret_n if self.regret_n else None

    def summary(self) -> Dict:
        return {"spots_attempted": self.attempted,
                "alignment": dict(self.by_grade),
                "by_street": dict(self.by_street),
                "eligible_regret_n": self.regret_n,
                "avg_estimated_regret_bb": self.avg_regret_bb,
                "uncertain_or_unscored": self.uncertain
                + self.by_grade.get("UNSCORED", 0)}


__all__ = ["TrainerSpot", "SPOT_LIBRARY", "TrainerFeedback", "TrainerStats",
           "run_spot", "answer_options", "evaluate_answer"]
