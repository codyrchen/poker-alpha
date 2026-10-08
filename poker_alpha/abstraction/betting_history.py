"""Betting-history abstraction for the heads-up solver game (Phase 26D).

The exact action history is the second source of key-space explosion
(Phase 25: 2,158 distinct histories in a 2,000-hand corpus). This module
summarizes it into strategically meaningful context:

* ``street`` and the acting ``position`` (0 = button/SB, 1 = BB);
* ``initiative`` — who made the last bet/raise on an *earlier* street,
  relative to the actor: ``own`` / ``opp`` / ``none``;
* ``raises`` — bets/raises made so far on this street (0..cap);
* ``facing`` — bet-size class the actor faces: ``none`` or ``small`` (<= 0.5
  pot), ``medium`` (<= 0.9), ``large`` (<= 1.25), ``over`` (> 1.25),
  ``allin``; the fraction is chips added over the call / pot after the call;
* ``own_prior`` — actor's last action this street: ``none`` / ``check`` /
  ``call`` / ``aggr``;
* ``spr`` — effective stack / pot at the start of the street, bucketed
  (edges 1, 3, 8): pot geometry;
* ``legal`` — the exact legal-action tokens. Including them makes every
  abstract key map to ONE legal-action set (required for regret tables to be
  well defined). It is one field among many, never the only one: two states
  are never merged *because* their menus match.

This summary forgets earlier streets' exact sequences (keeping initiative),
so any encoder using it is IMPERFECT RECALL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..games.holdem import BIG_BLIND, SMALL_BLIND, _tokens

BETTING_HISTORY_VERSION = 1
SIZE_EDGES = ((0.5, "small"), (0.9, "medium"), (1.25, "large"))


def size_class(frac: float, all_in: bool) -> str:
    if all_in:
        return "allin"
    for edge, name in SIZE_EDGES:
        if frac <= edge + 1e-9:
            return name
    return "over"


@dataclass(frozen=True)
class BettingContext:
    street: int
    position: int
    initiative: str
    raises: int
    facing: str
    own_prior: str
    spr: int
    legal: str

    def key(self) -> str:
        return (f"i{self.initiative[0]}r{self.raises}f{self.facing}"
                f"a{self.own_prior[0]}s{self.spr}|{self.legal}")


def betting_context(game: Any, state: Any, spr_edges=(1.0, 3.0, 8.0)) -> BettingContext:
    from .cards import bucketize

    stack = game.starting_stack
    total = [SMALL_BLIND, BIG_BLIND]
    actor = game.current_player(state)
    last_aggr_prev = None          # last aggressor on earlier streets
    street_last_aggr = None
    street_spr = None
    facing = "none"
    raises = 0
    own_prior = "none"
    n_streets = len(state.streets)
    for si, street_actions in enumerate(state.streets):
        if si > 0 and street_last_aggr is not None:
            last_aggr_prev = street_last_aggr
        street_last_aggr = None
        paid = [SMALL_BLIND, BIG_BLIND] if si == 0 else [0.0, 0.0]
        to_act = 0 if si == 0 else 1
        if si == n_streets - 1:
            pot = total[0] + total[1]
            eff = stack - max(total)
            street_spr = bucketize(eff / pot if pot > 0 else 99.0, spr_edges)
        raises = 0
        facing = "none"
        own_prior = "none"
        for tok in _tokens(street_actions):
            me, opp = to_act, 1 - to_act
            owe = paid[opp] - paid[me]
            mystack = stack - total[me]
            if tok == "f":
                add = 0.0
            elif tok == "c":
                add = min(owe, mystack)
            elif tok == "a":
                add = mystack
            else:
                pot_now = total[0] + total[1]
                add = owe + game.bet_fractions[tok] * (pot_now + owe)
            if tok in ("a",) or tok.startswith("b"):
                pot_after_call = total[0] + total[1] + owe
                frac = (add - owe) / pot_after_call if pot_after_call > 0 else 0.0
                cls = size_class(frac, tok == "a")
                raises += 1
                street_last_aggr = me
                if me != actor:
                    facing = cls
                kind = "aggr"
            else:
                kind = "call" if (tok == "c" and owe > 1e-9) else ("check" if tok == "c" else "fold")
                if me != actor:
                    facing = "none"
            if me == actor:
                own_prior = kind
                facing = "none"
            paid[me] += add
            total[me] += add
            to_act = opp
    if last_aggr_prev is None:
        initiative = "none"
    else:
        initiative = "own" if last_aggr_prev == actor else "opp"
    legal = ".".join(game.legal_actions(state))
    return BettingContext(n_streets - 1, actor, initiative, raises, facing,
                          own_prior, street_spr, legal)
