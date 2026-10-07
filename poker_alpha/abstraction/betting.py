"""Solver-facing betting (action) abstraction for no-limit Hold'em.

No-limit betting has a continuum of legal wager sizes; solvers need a small
discrete menu. This module owns the two directions of that translation and is
deliberately independent of any rules engine — it only needs the numeric
:class:`BettingContext` of the decision:

* **abstract → concrete**: :meth:`ActionAbstraction.to_concrete` turns e.g.
  ``bet_75`` into a legal wager respecting pot geometry, the amount to call,
  the minimum legal raise and the all-in ceiling.
* **concrete → abstract**: :meth:`ActionAbstraction.translate` maps an
  *observed* wager (an opponent's real bet size) onto the menu, using the
  pseudo-harmonic mapping of Ganzfried & Sandholm (2013), which is
  less exploitable than nearest-size rounding. :meth:`nearest` gives the
  deterministic most-likely label.

Sizing conventions
------------------
All amounts are in chips (typically big blinds).

* A **bet** of fraction ``f`` (no outstanding wager) commits ``f × pot``.
* A **raise** of fraction ``f`` is a pot-relative raise: hero first calls, then
  adds ``f × (pot + to_call)``. ``raise_100`` is the classic "pot-sized raise".
* Sizes below the legal minimum are lifted to the minimum; sizes at or above
  hero's stack (or above what any opponent can call) collapse into ``all_in``.
  Menu entries that resolve to the same concrete amount are de-duplicated,
  keeping the first (smallest-label) one, so the menu never offers two names
  for one wager.

Assumptions (documented, not hidden)
------------------------------------
* The minimum raise increment is the larger of the big blind and the last
  full bet/raise increment on the street (standard NLHE). A short all-in
  below a full raise is always legal.
* A wager larger than every opponent's remaining stack is pointless (the
  excess is returned); the menu caps at the largest amount someone can call.

PERFECT-RECALL NOTE
-------------------
Do **not** key information sets by "which menu was legal" — two different
histories can expose the same menu, and merging them destroys perfect recall.
Callers must keep the (abstracted) action *sequence* in the information-set
key; :meth:`translate` exists so an observed concrete action can be appended
to that sequence as an abstract token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import format_number

_EPS = 1e-9


@dataclass(frozen=True)
class BettingContext:
    """Numeric facts about a betting decision.

    ``pot`` includes every chip committed so far on all streets (including
    the current street's outstanding bets). ``hero_street`` is hero's
    commitment on this street, ``to_call`` the extra hero needs to match the
    largest commitment. ``max_opponent_stack`` is the largest remaining
    stack (behind, after their current commitment) among opponents still
    able to act or call.
    """

    pot: float
    to_call: float
    hero_stack: float
    max_opponent_stack: float
    hero_street: float = 0.0
    min_raise_increment: float = 1.0
    big_blind: float = 1.0

    def __post_init__(self) -> None:
        for name in ("pot", "to_call", "hero_stack", "max_opponent_stack",
                     "hero_street", "min_raise_increment", "big_blind"):
            if getattr(self, name) < -_EPS:
                raise ValueError(f"{name} must be non-negative")

    @property
    def current_bet(self) -> float:
        """Largest street commitment hero faces (raise-to reference)."""
        return self.hero_street + self.to_call

    @property
    def facing_bet(self) -> bool:
        return self.to_call > _EPS

    @property
    def max_raise_to(self) -> float:
        """Largest meaningful street commitment (all-in or opponent cap)."""
        hero_max = self.hero_street + self.hero_stack
        callable_max = self.current_bet + self.max_opponent_stack
        return min(hero_max, callable_max)

    @property
    def min_raise_to(self) -> float:
        """Smallest legal full bet/raise, as a street commitment."""
        inc = max(self.min_raise_increment, self.big_blind)
        return self.current_bet + inc


@dataclass(frozen=True)
class ConcreteAction:
    """A legal wager: ``kind`` in fold/check/call/bet/raise/all_in.

    ``raise_to`` is hero's total street commitment after acting; ``add`` the
    chips moved from stack to pot by this action.
    """

    kind: str
    raise_to: float
    add: float


@dataclass(frozen=True)
class ActionAbstraction:
    """A configurable discrete action menu.

    ``bet_fractions`` / ``raise_fractions`` are pot fractions (see module
    docstring for the conventions). Labels are ``bet_<pct>`` / ``raise_<pct>``
    with ``pct = round(100 × fraction)``.
    """

    bet_fractions: Tuple[float, ...] = (0.33, 0.75, 1.0)
    raise_fractions: Tuple[float, ...] = (0.75, 1.0)
    include_all_in: bool = True
    version: int = 1

    def signature(self) -> str:
        bets = ",".join(format_number(f) for f in self.bet_fractions)
        raises = ",".join(format_number(f) for f in self.raise_fractions)
        return (f"ActionAbstraction:v{self.version}:bets={bets}:"
                f"raises={raises}:allin={int(self.include_all_in)}")

    # -- menu ----------------------------------------------------------------

    @staticmethod
    def _label(prefix: str, frac: float) -> str:
        return f"{prefix}_{int(round(frac * 100))}"

    def _sized(self, ctx: BettingContext) -> List[Tuple[str, float]]:
        """Candidate ``(label, raise_to)`` sizes before legality filtering."""
        if ctx.facing_bet:
            base = ctx.pot + ctx.to_call
            return [(self._label("raise", f), ctx.current_bet + f * base)
                    for f in self.raise_fractions]
        return [(self._label("bet", f), ctx.hero_street + f * ctx.pot)
                for f in self.bet_fractions]

    def menu(self, ctx: BettingContext) -> List[Tuple[str, ConcreteAction]]:
        """Legal abstract actions with their concrete wagers, in menu order."""
        out: List[Tuple[str, ConcreteAction]] = []
        if ctx.facing_bet:
            out.append(("fold", ConcreteAction("fold", ctx.hero_street, 0.0)))
            call_add = min(ctx.to_call, ctx.hero_stack)
            kind = "all_in" if call_add >= ctx.hero_stack - _EPS else "call"
            out.append(("call", ConcreteAction(kind, ctx.hero_street + call_add,
                                               call_add)))
        else:
            out.append(("check", ConcreteAction("check", ctx.hero_street, 0.0)))

        cap = ctx.max_raise_to
        can_raise = (ctx.hero_stack > ctx.to_call + _EPS
                     and ctx.max_opponent_stack > _EPS
                     and cap > ctx.current_bet + _EPS)
        if not can_raise:
            return out
        verb = "raise" if ctx.facing_bet else "bet"
        seen = set()
        all_in_to = ctx.hero_street + ctx.hero_stack
        for label, target in self._sized(ctx):
            target = max(target, ctx.min_raise_to)
            if target >= cap - _EPS:
                continue  # covered by all_in (or the opponent cap)
            key = round(target, 9)
            if key in seen:
                continue
            seen.add(key)
            out.append((label, ConcreteAction(verb, target,
                                              target - ctx.hero_street)))
        if self.include_all_in:
            kind = "all_in" if cap >= all_in_to - _EPS else verb
            out.append(("all_in", ConcreteAction(kind, cap, cap - ctx.hero_street)))
        return out

    def legal_actions(self, ctx: BettingContext) -> List[str]:
        return [label for label, _ in self.menu(ctx)]

    def to_concrete(self, label: str, ctx: BettingContext) -> ConcreteAction:
        for name, action in self.menu(ctx):
            if name == label:
                return action
        raise ValueError(f"abstract action {label!r} not legal here; menu is "
                         f"{self.legal_actions(ctx)}")

    # -- observed concrete -> abstract ----------------------------------------

    def translate(self, kind: str, raise_to: float,
                  ctx: BettingContext) -> Dict[str, float]:
        """Map an observed action onto the menu as label probabilities.

        ``kind`` is fold/check/call/bet/raise/all_in; ``raise_to`` the
        observed total street commitment (ignored for fold/check/call).
        Wagers are compared as pot fractions with the pseudo-harmonic
        mapping between the two neighbouring menu sizes.
        """
        menu = self.menu(ctx)
        labels = [m[0] for m in menu]
        if kind == "fold":
            return {"fold": 1.0}
        if kind in ("check", "call"):
            return {"call" if ctx.facing_bet else "check": 1.0}
        if raise_to >= ctx.max_raise_to - _EPS and "all_in" in labels:
            return {"all_in": 1.0}
        sized = [(lab, act.raise_to) for lab, act in menu
                 if act.kind in ("bet", "raise", "all_in")
                 and lab not in ("call",)]
        if not sized:
            return {labels[-1]: 1.0}

        def frac(to: float) -> float:
            base = ctx.pot + ctx.to_call
            return (to - ctx.current_bet) / base if base > 0 else 0.0

        x = frac(raise_to)
        points = sorted((frac(to), lab) for lab, to in sized)
        if x <= points[0][0]:
            return {points[0][1]: 1.0}
        if x >= points[-1][0]:
            return {points[-1][1]: 1.0}
        for (a, la), (b, lb) in zip(points, points[1:]):
            if a <= x <= b:
                if b - a < _EPS:
                    return {la: 1.0}
                p_a = (b - x) * (1 + a) / ((b - a) * (1 + x))
                p_a = min(max(p_a, 0.0), 1.0)
                out = {la: p_a, lb: 1.0 - p_a}
                return {k: v for k, v in out.items() if v > 0.0}
        return {points[-1][1]: 1.0}  # unreachable; defensive

    def nearest(self, kind: str, raise_to: float, ctx: BettingContext) -> str:
        """Most likely label under :meth:`translate` (ties → smaller size)."""
        probs = self.translate(kind, raise_to, ctx)
        order = {label: i for i, label in enumerate(self.legal_actions(ctx))}
        return max(probs.items(), key=lambda kv: (kv[1], -order.get(kv[0], 0)))[0]
