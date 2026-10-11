"""Offline session review: run the real engine over every hero decision.

Input is the canonical hand-history representation (``pokeralpha.hand/v1``
event lists, see :mod:`poker_alpha.history`). Each hero decision point is
replayed into an :class:`ObservedTableState` and analysed by the actual
decision engine (solver -> rollout -> heuristic, same code path as Play
Mode) — nothing is scored from heuristics invented here.

Review categories are transparent and threshold-documented, not a single
magical score:

    HIGH-PRIORITY EV REVIEW
        valid same-method regret (see :mod:`.regret`) with
        estimated_regret >= HIGH_REGRET_BB (0.5 bb) and, when an
        uncertainty is measured, regret >= HIGH_REGRET_SIGMA (2) times it.
    STRATEGY REVIEW
        solver ACCEPTED and the action taken is rare (< 5%) or absent in
        the trained mix.
    UNCERTAIN
        solver LOW CONFIDENCE or REJECTED — the engine itself does not
        trust its answer enough to grade yours.
    UNSCORED
        the engine refused the state, or the action could not be mapped.
    OK
        everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..decision import DecisionConfig, DecisionReport
from ..history import replay_hand
from ..pipeline import Observation, analyze
from ..poker.cards import card_str
from .alignment import StrategyAlignment, solver_state, strategy_alignment
from .mapping import ACTUAL_SIZE_LABEL, ActionMapping, map_action
from .regret import RegretEligibility, regret_eligibility

HIGH_REGRET_BB = 0.5      # bb of eligible regret before HIGH-PRIORITY
HIGH_REGRET_SIGMA = 2.0   # and at least this many SEs above zero

CATEGORIES = ("HIGH-PRIORITY EV REVIEW", "STRATEGY REVIEW", "UNCERTAIN",
              "UNSCORED", "OK")


@dataclass(frozen=True)
class DecisionRecord:
    """Everything the review table / summary needs for one hero decision."""

    hand_index: int
    decision_index: int
    street: str
    position: Optional[str]
    hero_cards: str
    board: str
    pot_bb: Optional[float]
    to_call_bb: Optional[float]
    effective_stack_bb: Optional[float]
    actual_kind: str
    actual_amount: Optional[float]
    actual_label: Optional[str]           # mapped onto the report's candidates
    recommended: Optional[str]            # engine's primary/highest-frequency
    actual_frequency: Optional[float]     # solver-mix frequency, when meaningful
    method: str
    source_kind: str                      # solver / rollout / heuristic / none
    solver_state: str                     # ACCEPTED / LOW CONFIDENCE / REJECTED / OFF
    confidence: str                       # decision confidence (low/medium/high)
    gate_reasons: Tuple[str, ...]
    warnings: Tuple[str, ...]
    regret: RegretEligibility
    alignment: StrategyAlignment
    category: str                         # one of CATEGORIES
    off_tree: bool                        # bet sizes were translated onto the tree
    mapping_quality: str = "EXACT"        # EXACT / TRANSLATED / UNMAPPED
    mapping_distance: Optional[float] = None  # |evaluated - actual| size, chips
    report: Optional[DecisionReport] = field(default=None, compare=False,
                                             repr=False)
    state: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SessionReview:
    records: Tuple[DecisionRecord, ...]
    hands: int
    warnings: Tuple[str, ...] = ()


def _source_kind(method: str) -> str:
    m = method.lower()
    if "solver" in m or "abstraction" in m:
        return "solver"
    if "rollout" in m:
        return "rollout"
    if m in ("", "none"):
        return "none"
    return "heuristic"


def categorize(regret: RegretEligibility, align: StrategyAlignment,
               state: str, method: str) -> str:
    """Apply the documented thresholds (module docstring)."""
    if _source_kind(method) == "none":
        return "UNSCORED"
    if regret.eligible and regret.estimated_regret is not None:
        r = regret.estimated_regret
        strong = r >= HIGH_REGRET_BB and (
            regret.uncertainty is None or r >= HIGH_REGRET_SIGMA * regret.uncertainty)
        if strong:
            return "HIGH-PRIORITY EV REVIEW"
    if state == "ACCEPTED" and align.category in ("RARE", "NOT_IN_MIX"):
        return "STRATEGY REVIEW"
    if state in ("LOW CONFIDENCE", "REJECTED"):
        return "UNCERTAIN"
    if regret.reason == "ACTION_NOT_COMPARABLE" and align.category in (
            "NOT_SOLVER", "NOT_IN_MIX"):
        return "UNSCORED"
    return "OK"


def review_decision(report: DecisionReport, state, hand_index: int,
                    decision_index: int, actual_kind: str,
                    actual_amount: Optional[float]) -> DecisionRecord:
    """Build one record from a report the engine already produced.

    The real action is mapped with explicit quality (EXACT / TRANSLATED /
    UNMAPPED); EV regret only ever uses an EXACT mapping — a nearest-size
    translation is shown for strategy alignment but never scored as if the
    candidate EV were the EV of the action actually taken.
    """
    mapping = map_action(report, actual_kind, actual_amount)
    label = mapping.label
    reg = regret_eligibility(report, label, mapping)
    align = strategy_alignment(report, label)
    st = solver_state(report)
    info = report.details.get("solver", {}) if report.details else {}
    reasons = tuple(info.get("reasons", ()))
    cards = " ".join(card_str(c) for c in state.hero_cards) \
        if state.hero_cards else ""
    return DecisionRecord(
        hand_index=hand_index, decision_index=decision_index,
        street=state.street.name.title(), position=state.hero_position,
        hero_cards=cards,
        board=" ".join(card_str(c) for c in state.board),
        pot_bb=state.to_bb(state.pot), to_call_bb=state.to_bb(state.amount_to_call),
        effective_stack_bb=state.to_bb(state.effective_stack),
        actual_kind=actual_kind, actual_amount=actual_amount,
        actual_label=label, recommended=report.recommended,
        actual_frequency=align.frequency if align.category != "NOT_SOLVER" else None,
        method=report.method, source_kind=_source_kind(report.method),
        solver_state=st, confidence=report.confidence,
        gate_reasons=reasons, warnings=tuple(report.warnings),
        regret=reg, alignment=align,
        category=categorize(reg, align, st, report.method),
        off_tree="OFF_TREE_TRANSLATION" in reasons,
        mapping_quality=mapping.quality, mapping_distance=mapping.distance,
        report=report, state=state)


def review_hands(hands: Sequence[Sequence], config: Optional[DecisionConfig] = None,
                 solver=None, progress=None) -> SessionReview:
    """Run the engine over every hero decision of every hand.

    ``hands`` is a sequence of event lists (``poker_alpha.history.load_hands``).
    ``solver`` overrides ``config.solver`` exactly as in ``pipeline.analyze``.
    """
    records: List[DecisionRecord] = []
    warnings: List[str] = []
    for hi, events in enumerate(hands):
        try:
            res = replay_hand(events)
        except Exception as exc:          # a broken hand must not kill the review
            warnings.append(f"hand {hi}: replay failed: {exc}")
            continue
        warnings.extend(f"hand {hi}: {w}" for w in res.warnings)
        for di, d in enumerate(res.decisions):
            obs = Observation(replace(d.state, source="hand_history"),
                              "hand_history")
            cfg = config or DecisionConfig()
            if d.action in ("bet", "raise") and d.amount is not None:
                # Evaluate the hero's EXACT size alongside the menu, on the
                # same common-random-number samples, so regret for it is a
                # legitimate paired comparison instead of a nearest-size
                # approximation (mapping.py).
                cfg = replace(cfg, abstraction=replace(
                    cfg.abstraction,
                    extra_targets=((ACTUAL_SIZE_LABEL, float(d.amount)),)))
            report = analyze(obs, config=cfg, solver=solver)
            records.append(review_decision(report, d.state, hi, di,
                                           d.action, d.amount))
            if progress:
                progress(hi, di)
    return SessionReview(tuple(records), hands=len(hands),
                         warnings=tuple(warnings))


__all__ = ["DecisionRecord", "SessionReview", "CATEGORIES", "HIGH_REGRET_BB",
           "HIGH_REGRET_SIGMA", "review_hands", "review_decision",
           "categorize"]
