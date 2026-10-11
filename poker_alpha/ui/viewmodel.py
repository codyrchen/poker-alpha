"""View models for the Play Mode UI (pure Python, no Streamlit import).

Transforms the existing :class:`~poker_alpha.decision.report.DecisionReport`
and :class:`~poker_alpha.holdem.observed.ObservedTableState` into small,
typed presentation models. Rendering (Streamlit today, possibly a web or
overlay frontend later) consumes these; no solver logic lives here and no
solver output is recomputed or invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..decision.report import CandidateAction, DecisionReport
from ..holdem.observed import ObservedTableState
from ..poker.cards import card_str

SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RED_SUITS = ("h", "d")

#: Human wording for the machine-readable gate / cascade reason codes
#: (poker_alpha.decision.solver_gate.REASONS). Only translates codes that
#: actually exist; unknown codes fall through verbatim.
REASON_TEXT = {
    "LOW_VISIT_COUNT": "this situation was rarely reached in training",
    "HIGH_SEED_DISAGREEMENT": "independently trained runs disagree here",
    "UNSTABLE_ACROSS_CHECKPOINTS": "the policy was still moving between checkpoints",
    "HIGH_COLLISION_DISPERSION": "the abstraction merges different hands here",
    "CONFIG_MISMATCH": "the strategy file does not match this configuration",
    "INCOMPATIBLE_CHECKPOINT": "the strategy file could not be used",
    "UNSEEN_STATE": "this situation never came up in training",
    "OUTSIDE_ABSTRACTION": "this spot cannot be mapped into the trained game",
    "KNOWN_PATHOLOGICAL_BUCKET": "this situation is flagged by the strategy audit",
    "NO_STABILITY_DATA": "no stability data exists for this situation",
    "OFF_TREE_TRANSLATION": "the real bet sizes were translated onto the trained tree",
    "ILLEGAL_SIZE_MASS": "much of the trained strategy used sizes that are not legal here",
    "STREET_ABSTRACTION_ERROR": "abstraction error is measured to be material on this street",
    "SOLVER_ERROR": "the strategy lookup failed",
    # refusal codes (recommend_action declines to fabricate a decision)
    "HERO_CARDS_UNKNOWN": "hero cards are not known",
    "NOT_HERO_TURN": "it is not the hero's turn",
    "HERO_STACK_UNKNOWN": "the hero stack could not be read",
    "HERO_FOLDED": "the hero has folded",
    "NO_OPPONENTS": "no live opponents remain",
    "HERO_ALL_IN": "the hero is already all-in",
}


def reason_text(code: str) -> str:
    return REASON_TEXT.get(code, code.replace("_", " ").lower())


# ---------------------------------------------------------------------------
# cards / hand state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardVM:
    rank: str      # "A", "T", "7"
    suit: str      # "s" | "h" | "d" | "c"

    @property
    def symbol(self) -> str:
        return SUIT_SYMBOLS[self.suit]

    @property
    def red(self) -> bool:
        return self.suit in RED_SUITS

    @property
    def text(self) -> str:
        return f"{self.rank}{self.symbol}"


def card_vm(code: int) -> CardVM:
    s = card_str(code)
    return CardVM(s[0], s[1])


@dataclass(frozen=True)
class HandVM:
    """Compact 'current hand' summary for the top of Play Mode."""

    hero: Tuple[CardVM, ...]
    board: Tuple[CardVM, ...]
    street: str
    position: Optional[str]
    pot_bb: Optional[float]
    to_call_bb: Optional[float]
    effective_bb: Optional[float]
    spr: Optional[float]

    @property
    def facing(self) -> str:
        if self.to_call_bb is None:
            return "?"
        return "unopened" if self.to_call_bb <= 0 else f"{self.to_call_bb:g} BB to call"


def hand_vm(obs: ObservedTableState) -> HandVM:
    def bb(x):
        return None if x is None else x / obs.big_blind

    return HandVM(
        hero=tuple(card_vm(c) for c in (obs.hero_cards or ())),
        board=tuple(card_vm(c) for c in (obs.board or ())),
        street=obs.street.name.title(),
        position=obs.hero_position,
        pot_bb=bb(obs.pot),
        to_call_bb=bb(obs.amount_to_call),
        effective_bb=bb(obs.effective_stack),
        spr=obs.spr,
    )


# ---------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionVM:
    label: str                      # raw candidate label ("bet_75")
    display: str                    # "Bet 6.2 BB" / "Check" / "Fold"
    frequency: Optional[float]      # 0..1
    ev_bb: Optional[float]
    ev_se_bb: Optional[float]
    source: str
    recommended: bool
    note: str = ""

    @property
    def frequency_pct(self) -> Optional[int]:
        return None if self.frequency is None else round(100 * self.frequency)

    @property
    def bar_width_pct(self) -> float:
        """Exact proportional bar width; 0 whenever the displayed number is
        0%, so the bar and the number can never contradict each other."""
        if self.frequency is None or self.frequency_pct == 0:
            return 0.0
        return max(0.0, 100.0 * self.frequency)


def _display_action(c: CandidateAction, big_blind_pot: Optional[float]) -> str:
    kind = c.kind.replace("_", "-")
    if kind in ("fold", "check", "call", "all-in"):
        name = {"fold": "Fold", "check": "Check", "call": "Call",
                "all-in": "All-in"}[kind]
        if kind == "call" and c.added > 0:
            return f"Call {c.added:g} BB"
        if kind == "all-in" and c.amount_to > 0:
            return f"All-in {c.amount_to:g} BB"
        return name
    verb = "Raise to" if kind == "raise" else "Bet"
    size = c.amount_to if kind == "raise" else c.added
    return f"{verb} {size:g} BB"


CONFIDENCE_LABELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


@dataclass(frozen=True)
class AbstentionVM:
    """The deliberate 'recommendation withheld / downgraded' presentation."""

    withheld: bool                 # True = no solver recommendation at all
    title: str
    reasons: Tuple[str, ...]       # human-readable
    reason_codes: Tuple[str, ...]  # raw codes for the technical expander
    fallback: Optional[str]        # 'Monte Carlo rollout' / 'heuristic' / None


@dataclass(frozen=True)
class RefusalVM:
    """No recommendation at all: the state itself cannot be trusted."""

    codes: Tuple[str, ...]
    reasons: Tuple[str, ...]

    @property
    def message(self) -> str:
        return self.reasons[0].capitalize() if self.reasons else \
            "The observed state is incomplete"


def refusal_vm(report: DecisionReport) -> Optional[RefusalVM]:
    codes = tuple(report.details.get("refusal_codes") or ())
    if report.method != "none" and not codes:
        return None
    if report.method != "none":
        return None
    return RefusalVM(codes, tuple(reason_text(c) for c in codes))


@dataclass(frozen=True)
class RecommendationVM:
    actions: Tuple[ActionVM, ...]          # frequency-descending
    recommended: Optional[ActionVM]
    method: str                             # dominant source (report.method)
    source_kind: str                        # 'solver' | 'rollout' | 'heuristic'
    confidence: str                         # LOW / MEDIUM / HIGH
    mix_meaning: str
    ev_edge_bb: Optional[float]             # recommended EV − best alternative EV
    abstention: Optional[AbstentionVM]
    warnings: Tuple[str, ...]
    equity: Optional[float]
    equity_se: Optional[float]
    pot_odds: Optional[float]
    refusal: Optional["RefusalVM"] = None
    #: True only when the candidate EVs are the actual decision basis
    #: (rollout/heuristic selection). Solver frequencies come from training;
    #: any attached EVs there are separate rollout ESTIMATES and must not be
    #: presented as the reason for the solver's mix.
    ev_is_decision_basis: bool = False
    #: Gate state of the trained strategy: ACCEPTED / LOW CONFIDENCE /
    #: REJECTED / OFF (not configured).
    solver_state: str = "OFF"
    #: Tag for the emphasized action: solver output is a distribution, so
    #: its top action is "HIGHEST FREQUENCY", not "RECOMMENDED".
    recommended_tag: str = "RECOMMENDED"


def _source_kind(method: str) -> str:
    m = (method or "").lower()
    if "solver" in m or "abstraction" in m:
        return "solver"
    if "rollout" in m:
        return "rollout"
    return "heuristic"


def _ev_edge(actions: List[ActionVM]) -> Optional[float]:
    rec = [a for a in actions if a.recommended and a.ev_bb is not None]
    rest = [a.ev_bb for a in actions if not a.recommended and a.ev_bb is not None]
    if not rec or not rest:
        return None
    return rec[0].ev_bb - max(rest)


def _solver_info(report: DecisionReport) -> Dict[str, object]:
    return dict(report.details.get("solver") or {})


def _solver_state(info: Dict[str, object]) -> str:
    conf = str(info.get("confidence", ""))
    if info.get("used"):
        if conf == "SOLVER_ACCEPT":
            return "ACCEPTED"
        if conf == "SOLVER_LOW_CONFIDENCE":
            return "LOW CONFIDENCE"
        return "ACCEPTED"
    if conf == "not configured" or not info:
        return "OFF"
    return "REJECTED"


def abstention_vm(report: DecisionReport) -> Optional[AbstentionVM]:
    """The abstention/downgrade state, from the report's actual fields.

    * Solver rejected (gate or lookup) -> withheld solver advice, fallback
      shown.
    * Solver used at low confidence  -> downgraded (not withheld).
    * No solver configured/loaded is reported as unavailable, not as a gate
      decision.
    """
    info = _solver_info(report)
    codes = tuple(info.get("reasons") or ())
    kind = _source_kind(report.method)
    if info and not info.get("used") and info.get("confidence") != "not configured":
        fallback = {"rollout": "Monte Carlo rollout",
                    "heuristic": "pot-odds heuristic"}.get(kind)
        if report.recommended is None:
            fallback = None
        return AbstentionVM(
            withheld=True,
            title="Solver recommendation withheld",
            reasons=tuple(reason_text(c) for c in codes) or
                    ("the confidence gate rejected this lookup",),
            reason_codes=codes,
            fallback=fallback)
    if info.get("used") and str(info.get("confidence", "")).upper().endswith("LOW_CONFIDENCE"):
        return AbstentionVM(
            withheld=False,
            title="Low-confidence solver advice",
            reasons=tuple(reason_text(c) for c in codes) or
                    ("the confidence gate downgraded this lookup",),
            reason_codes=codes,
            fallback=None)
    return None


def recommendation_vm(report: DecisionReport) -> RecommendationVM:
    actions = []
    for c in report.candidates:
        actions.append(ActionVM(
            label=c.label,
            display=_display_action(c, report.pot_bb),
            frequency=c.probability,
            ev_bb=c.ev_bb,
            ev_se_bb=c.ev_se_bb,
            source=c.source,
            recommended=(c.label == report.recommended),
            note=c.note))
    # Stable sort: frequency descending, ties keep the pipeline's own
    # deterministic candidate order (fold/check/call/bets).
    actions.sort(key=lambda a: -(a.frequency if a.frequency is not None else -1.0))
    rec = next((a for a in actions if a.recommended), None)
    kind = _source_kind(report.method)
    info = _solver_info(report)
    return RecommendationVM(
        actions=tuple(actions),
        recommended=rec,
        method=report.method,
        source_kind=kind,
        confidence=CONFIDENCE_LABELS.get(report.confidence, report.confidence.upper()),
        mix_meaning=report.mix_meaning,
        ev_edge_bb=_ev_edge(actions),
        abstention=abstention_vm(report),
        warnings=tuple(report.warnings),
        ev_is_decision_basis=(kind == "rollout"),
        solver_state=_solver_state(info),
        recommended_tag=("HIGHEST FREQUENCY" if kind == "solver"
                         else "RECOMMENDED"),
        equity=report.hero_equity,
        equity_se=report.hero_equity_se,
        pot_odds=report.pot_odds,
        refusal=refusal_vm(report),
    )


# ---------------------------------------------------------------------------
# compact summary (future desktop overlay)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactSummary:
    """The minimal recommendation view a small overlay could render."""

    lines: Tuple[Tuple[str, str], ...]   # (action text, "64%")
    confidence: str                      # "HIGH CONFIDENCE" etc.
    withheld: bool

    def text(self) -> str:
        body = "\n".join(f"{a:<16}{f:>5}" for a, f in self.lines)
        return f"{body}\n\n{self.confidence}"


def compact_summary(report: DecisionReport, max_lines: int = 3) -> CompactSummary:
    vm = recommendation_vm(report)
    if vm.refusal is not None:
        return CompactSummary((), "WAITING FOR A RELIABLE STATE", True)
    if vm.abstention is not None and vm.abstention.withheld and vm.recommended is None:
        return CompactSummary((), "RECOMMENDATION WITHHELD", True)
    lines = tuple(
        (a.display, "" if a.frequency_pct is None else f"{a.frequency_pct}%")
        for a in vm.actions[:max_lines])
    return CompactSummary(lines, f"{vm.confidence} CONFIDENCE", False)


# ---------------------------------------------------------------------------
# live-state / error states
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveStatusVM:
    recognition: Optional[float]        # critical-state confidence 0..1
    ok: bool
    waiting_for: Optional[str]          # one human line when not ok
    paused: bool = False


#: live_check problem substrings -> the smallest useful user instruction.
_WAIT_RULES: Tuple[Tuple[str, str], ...] = (
    ("hero", "Waiting for a reliable hero-card read"),
    ("pot", "Waiting for a reliable pot read"),
    ("board", "Waiting for a reliable board read"),
    ("stack", "Waiting for reliable stack reads"),
    ("confidence", "Waiting for recognition confidence to recover"),
)


def waiting_message(problems: List[str]) -> str:
    joined = " ".join(problems).lower()
    for needle, message in _WAIT_RULES:
        if needle in joined:
            return message
    return "Waiting for a reliable table read"


def live_status_vm(ok: bool, critical_confidence: Optional[float],
                   problems: Optional[List[str]] = None,
                   paused: bool = False) -> LiveStatusVM:
    return LiveStatusVM(
        recognition=critical_confidence,
        ok=ok and not paused,
        waiting_for=None if (ok and not paused) else
            ("Observer paused" if paused else waiting_message(problems or [])),
        paused=paused)


# ---------------------------------------------------------------------------
# "why" panel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhyVM:
    source_line: str
    gate_line: Optional[str]
    gate_reasons: Tuple[str, ...]
    ev_line: Optional[str]              # only when EVs are the decision basis
    ev_estimates_note: Optional[str]    # provenance-labeled estimates otherwise
    uncertainty: Tuple[Tuple[str, str], ...]
    warnings: Tuple[str, ...]


def why_vm(report: DecisionReport) -> WhyVM:
    vm = recommendation_vm(report)
    info = _solver_info(report)
    source_line = {
        "solver": "Frequencies come from the trained heads-up strategy "
                  "(2M-iteration MCCFR, experimental).",
        "rollout": "Values come from Monte Carlo rollouts against modelled "
                   "opponent ranges — an estimate, not an equilibrium.",
        "heuristic": "Only closed-form pot-odds / equity reasoning was "
                     "available for this spot.",
    }[vm.source_kind]
    gate_line = None
    if info and info.get("confidence") not in (None, "not configured"):
        status = str(info.get("confidence"))
        gate_line = ("Confidence gate: " + status.replace("SOLVER_", "")
                     .replace("_", " ").lower())
    ev_line = None
    ev_note = None
    if vm.ev_is_decision_basis:
        if vm.recommended is not None and vm.recommended.ev_bb is not None:
            ev_line = f"{vm.recommended.display}: {vm.recommended.ev_bb:+.2f} BB"
            if vm.ev_edge_bb is not None:
                ev_line += f" ({vm.ev_edge_bb:+.2f} BB vs the next-best action)"
    else:
        # EVs attached to a solver mix are separate rollout ESTIMATES — they
        # did not produce the frequencies and are labeled as such.
        with_ev = [a for a in vm.actions if a.ev_bb is not None]
        if with_ev:
            parts = ", ".join(f"{a.display} {a.ev_bb:+.2f} BB" for a in with_ev[:4])
            ev_note = ("Rollout EV estimates (computed separately — they did "
                       "NOT produce the solver's frequencies): " + parts)
    return WhyVM(
        source_line=source_line,
        gate_line=gate_line,
        gate_reasons=tuple(reason_text(c) for c in (info.get("reasons") or ())),
        ev_line=ev_line,
        ev_estimates_note=ev_note,
        uncertainty=tuple((k, str(v)) for k, v in report.uncertainty.items()),
        warnings=tuple(report.warnings),
    )
