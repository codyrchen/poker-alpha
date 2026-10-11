"""What-if analysis: controlled, single-variable counterfactuals.

One variable changes at a time, the resulting state is fully re-validated,
and every variant is re-analysed by the real engine. Variants are compared
by action distribution / solver gate state / method / confidence — NEVER by
EV across different decision methods (a rollout EV and a solver mix are not
the same kind of number).

Supported variables (``WHATIF_VARIABLES``):

    villain_bet      the size the villain is currently betting (bb)
    pot              the pot before hero acts (bb)
    effective_stack  both starting stacks scaled to a new effective depth
    hero_cards       the hero's hand
    board_card       append the next (turn/river) card

Invalid results — duplicate cards, negative amounts, impossible stacks,
illegal street transitions — raise :class:`WhatIfError` instead of being
analysed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..decision import DecisionConfig, DecisionReport
from ..holdem import ManualStateAdapter, validate
from ..pipeline import observe_manual, analyze
from .alignment import solver_state

WHATIF_VARIABLES = ("villain_bet", "pot", "effective_stack", "hero_cards",
                    "board_card")

_CARDS = {r + s for r in "AKQJT98765432" for s in "shdc"}


class WhatIfError(ValueError):
    """The requested variant does not describe a possible poker state."""


def _cards_of(text: str) -> List[str]:
    return [c for c in str(text).replace(",", " ").split() if c]


def _check_cards(cards: Sequence[str]) -> None:
    for c in cards:
        if c not in _CARDS:
            raise WhatIfError(f"{c!r} is not a card")
    if len(set(cards)) != len(cards):
        raise WhatIfError("duplicate cards")


def _villain_seat(base: dict) -> int:
    hero = base.get("hero_seat", 0)
    if base.get("num_seats") != 2:
        raise WhatIfError("villain_bet what-if supports heads-up states only")
    return 1 - hero


def apply_whatif(base: dict, variable: str, value) -> dict:
    """A new manual-state dict with exactly one variable changed."""
    if variable not in WHATIF_VARIABLES:
        raise WhatIfError(f"unknown what-if variable {variable!r}; "
                          f"supported: {WHATIF_VARIABLES}")
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    out["seats"] = [dict(s) for s in base["seats"]]

    if variable == "hero_cards":
        cards = _cards_of(value)
        if len(cards) != 2:
            raise WhatIfError("hero_cards needs exactly two cards")
        _check_cards(cards + _cards_of(out.get("board", "")))
        out["hero_cards"] = " ".join(cards)

    elif variable == "pot":
        pot = float(value)
        bets = sum(float(s.get("bet", 0.0)) for s in out["seats"])
        if pot <= 0:
            raise WhatIfError("pot must be positive")
        if pot < bets:
            raise WhatIfError(f"pot {pot:g} is below the chips currently bet "
                              f"({bets:g})")
        out["pot"] = pot

    elif variable == "effective_stack":
        eff = float(value)
        if eff <= 0:
            raise WhatIfError("effective stack must be positive")
        for s in out["seats"]:
            if s.get("stack") is None:
                raise WhatIfError("stacks unknown; cannot rescale")
            behind = eff - float(s.get("bet", 0.0))
            if behind < 0:
                raise WhatIfError("effective stack below current bets")
            s["stack"] = behind

    elif variable == "villain_bet":
        new = float(value)
        if new < 0:
            raise WhatIfError("bet cannot be negative")
        v = _villain_seat(base)
        seat = out["seats"][v]
        old = float(seat.get("bet", 0.0))
        stack = seat.get("stack")
        if stack is None:
            raise WhatIfError("villain stack unknown")
        if new - old > float(stack):
            raise WhatIfError(f"villain cannot bet {new:g}: only "
                              f"{float(stack) + old:g} available")
        seat["bet"] = new
        seat["stack"] = float(stack) + old - new
        if seat.get("committed") is not None:
            seat["committed"] = float(seat["committed"]) + new - old
        out["pot"] = float(out.get("pot", 0.0)) + new - old
        if out["pot"] <= 0:
            raise WhatIfError("resulting pot is not positive")
        # keep the recorded action history consistent with the new size
        acts = [dict(a) for a in out.get("actions", [])]
        for a in reversed(acts):
            if a.get("seat") == v and a.get("kind") in ("bet", "raise"):
                a["amount"] = new
                break
        out["actions"] = acts

    elif variable == "board_card":
        card = str(value).strip()
        board = _cards_of(out.get("board", ""))
        if len(board) not in (3, 4):
            raise WhatIfError("board_card needs an existing flop or turn "
                              f"(3 or 4 cards), found {len(board)}")
        hero = _cards_of(out.get("hero_cards", ""))
        _check_cards(board + hero + [card])
        out["board"] = " ".join(board + [card])
        # a new card means a new street: current bets return to zero and the
        # recorded history must already close the previous street
        for s in out["seats"]:
            if float(s.get("bet", 0.0)) != 0.0:
                raise WhatIfError("cannot deal the next card while bets are "
                                  "outstanding on the current street")

    st = ManualStateAdapter.from_dict(out)
    errors = [i for i in validate(st) if i.severity == "error"]
    if errors:
        raise WhatIfError("; ".join(i.message for i in errors))
    return out


@dataclass(frozen=True)
class Variant:
    variable: str
    value: str
    state: dict
    report: DecisionReport
    solver_state: str
    method: str
    confidence: str
    top_actions: Tuple[Tuple[str, Optional[float]], ...]   # (label, freq)
    warnings: Tuple[str, ...]

    def strategy_text(self) -> str:
        parts = []
        for label, f in self.top_actions:
            parts.append(label if f is None else f"{label} {f:.0%}")
        return " / ".join(parts) if parts else "—"


def analyze_variant(state_dict: dict, variable: str, value,
                    config: Optional[DecisionConfig] = None,
                    solver=None) -> Variant:
    report = analyze(observe_manual(state_dict), config=config, solver=solver)
    tops = tuple(sorted(((c.label, c.probability) for c in report.candidates),
                        key=lambda x: -(x[1] or 0.0))[:4])
    return Variant(variable=variable, value=str(value), state=state_dict,
                   report=report, solver_state=solver_state(report),
                   method=report.method, confidence=report.confidence,
                   top_actions=tops, warnings=tuple(report.warnings))


def compare_whatif(base: dict, variable: str, values: Sequence,
                   config: Optional[DecisionConfig] = None,
                   solver=None) -> Tuple[List[Variant], Dict[str, str]]:
    """Analyse the base plus one variant per value. Returns the variants and
    a map of rejected values -> reason for the ones that were invalid."""
    variants = [analyze_variant(base, variable, "base", config=config,
                                solver=solver)]
    rejected: Dict[str, str] = {}
    for v in values:
        try:
            st = apply_whatif(base, variable, v)
        except WhatIfError as exc:
            rejected[str(v)] = str(exc)
            continue
        variants.append(analyze_variant(st, variable, v, config=config,
                                        solver=solver))
    return variants, rejected


__all__ = ["WHATIF_VARIABLES", "WhatIfError", "apply_whatif",
           "analyze_variant", "compare_whatif", "Variant"]
