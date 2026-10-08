"""Method A: look up a trained heads-up abstract strategy for an observed spot.

Translates an :class:`ObservedTableState` into the abstract heads-up game
(:class:`~poker_alpha.games.holdem.HoldemGame`) and reads the average
strategy at the hero's information set. The lookup *refuses* (returns
``None`` with a reason) whenever the observed spot is outside what the
strategy was trained on: not heads-up, different stack depth or blind
structure, unknown history, or a too-rarely visited information set.

Off-tree bet sizes are mapped to the nearest abstract size (in pot-relative
terms); such lookups are marked ``exact=False`` and reported with source
``"interpolated abstraction"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Tuple

from ..games.holdem import HoldemGame, HoldemState
from ..holdem.observed import ObservedTableState
from ..holdem.state import Street

Strategy = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class SolverLookup:
    infoset_key: str
    visits: float
    exact: bool
    actions: Tuple[Tuple[str, str, str, float, float], ...]
    # (token, label, kind, raise_to_bb, probability)


# Why a solver strategy was not used (``LookupMiss.code``).
REJECTION_CODES = {
    "config_mismatch": "strategy was trained under a different solver config / encoder / game",
    "incompatible_checkpoint": "strategy file unreadable or of an unsupported format",
    "out_of_abstraction": "observed spot cannot be mapped into the abstract game",
    "unvisited": "abstract information set never visited in training",
    "insufficient_visits": "abstract information set visited too rarely to trust",
}


@dataclass(frozen=True)
class LookupMiss:
    reason: str
    code: str = "out_of_abstraction"


class SolverStrategyProvider:
    def __init__(self, game: HoldemGame, strategy: Strategy,
                 visits: Optional[Mapping[str, float]] = None,
                 min_visits: float = 20.0,
                 stack_tolerance: float = 0.05,
                 description: str = "abstract heads-up strategy") -> None:
        self.game = game
        self.strategy = strategy
        self.visits = visits or {}
        self.min_visits = min_visits
        self.stack_tolerance = stack_tolerance
        self.description = description

    # -- loading ---------------------------------------------------------------

    @classmethod
    def from_artifact(cls, path, config=None, min_visits: float = 20.0):
        """Provider from a strategy artifact (``solvers.strategy_artifact``)
        trained under ``config`` (default: the locked ``PRIMARY_CONFIG``).

        Returns a :class:`LookupMiss` (``config_mismatch`` or
        ``incompatible_checkpoint``) instead of raising when it cannot be used.
        """
        from ..solver_config import PRIMARY_CONFIG
        from ..solvers.strategy_artifact import StrategyArtifactError, load_artifact

        config = config or PRIMARY_CONFIG
        game = config.build_game()
        try:
            art = load_artifact(path, game)
        except StrategyArtifactError as exc:
            code = "config_mismatch" if "signature mismatch" in str(exc) else "incompatible_checkpoint"
            return LookupMiss(f"strategy artifact rejected: {exc}", code)
        recall = "perfect recall" if config.perfect_recall else \
            "IMPERFECT-RECALL abstraction, no equilibrium guarantee"
        desc = (f"MCCFR average strategy, {art.meta.get('iterations', '?')} iterations, "
                f"seed {art.meta.get('seed', '?')}, {recall}")
        return cls(game, art.strategy, art.visits, min_visits=min_visits, description=desc)

    @classmethod
    def from_checkpoint(cls, path, config=None, min_visits: float = 20.0):
        """Provider from a full training checkpoint; same rejection rules."""
        from ..solver_config import PRIMARY_CONFIG
        from ..solvers.serialize import CheckpointError, load_checkpoint

        config = config or PRIMARY_CONFIG
        game = config.build_game()
        try:
            solver = load_checkpoint(path, game)
        except CheckpointError as exc:
            code = "config_mismatch" if "mismatch" in str(exc) else "incompatible_checkpoint"
            return LookupMiss(f"checkpoint rejected: {exc}", code)
        strategy, visits = {}, {}
        for k, n in solver.infosets.items():
            v = float(n.strategy_sum.sum())
            if v > 0:
                strategy[k] = dict(zip(n.actions, (float(x) for x in n.average_strategy())))
                visits[k] = v
        recall = "perfect recall" if config.perfect_recall else \
            "IMPERFECT-RECALL abstraction, no equilibrium guarantee"
        return cls(game, strategy, visits, min_visits=min_visits,
                   description=f"MCCFR checkpoint, {solver.iterations} iterations, {recall}")

    # -- translation -----------------------------------------------------------

    def _to_state(self, obs: ObservedTableState):
        game = self.game
        occupied = [s for s in obs.seats if s.occupied and not s.sitting_out]
        if len(occupied) != 2:
            return LookupMiss("strategy is heads-up only")
        if abs(obs.small_blind / obs.big_blind - 0.5) > 1e-9 or obs.ante:
            return LookupMiss("blind structure differs from the abstraction")
        if obs.hero_cards is None:
            return LookupMiss("hero cards unknown")
        bb = obs.big_blind
        starts = []
        for s in occupied:
            if s.stack is None or s.committed_total is None:
                return LookupMiss("stacks/contributions unknown")
            starts.append((s.stack + s.committed_total) / bb)
        eff = min(starts)
        if abs(eff - game.starting_stack) > self.stack_tolerance * game.starting_stack:
            return LookupMiss(f"effective stack {eff:.1f}BB outside trained "
                              f"{game.starting_stack:g}BB")
        btn = obs.dealer
        other = next(s.seat for s in occupied if s.seat != btn)
        player_of = {btn: 0, other: 1}
        hero_p = player_of[obs.hero_seat]
        dead = set(obs.hero_cards) | set(obs.board)
        filler = tuple(c for c in range(52) if c not in dead)[:2]
        holes = (obs.hero_cards, filler) if hero_p == 0 else (filler, obs.hero_cards)
        board_by_street = {0: 0, 1: 3, 2: 4, 3: 5}
        state = HoldemState(holes=holes, board=(), streets=("",),
                            contrib=(0.5, 1.0))
        exact = True
        for a in obs.action_history:
            if a.kind == "post":
                continue
            street = int(a.street)
            while state.street < street:
                n = board_by_street[state.street + 1]
                if len(obs.board) < n:
                    return LookupMiss("board missing for observed street")
                state = replace(state, board=tuple(obs.board[:n]),
                                streets=state.streets + ("",))
            if game.is_terminal(state) or game.is_chance(state):
                return LookupMiss("history leaves the abstract tree")
            if game.current_player(state) != player_of.get(a.seat, -1):
                return LookupMiss("observed action order differs")
            token, was_exact = self._match(state, a.kind, a.amount / bb)
            if token is None:
                return LookupMiss(f"cannot map observed {a.kind}")
            exact = exact and was_exact
            state = game.next_state(state, token)
        current = int(min(obs.street, Street.RIVER))
        while state.street < current:
            n = board_by_street[state.street + 1]
            if len(obs.board) < n or not game.is_chance(state):
                return LookupMiss("history incomplete for the current street")
            state = replace(state, board=tuple(obs.board[:n]),
                            streets=state.streets + ("",))
        if game.is_terminal(state) or game.is_chance(state):
            return LookupMiss("no decision in the abstract tree")
        if game.current_player(state) != hero_p:
            return LookupMiss("hero is not to act in the abstract tree")
        return state, exact

    def _match(self, state: HoldemState, kind: str, raise_to_bb: float):
        game = self.game
        legal = game.legal_actions(state)
        if kind == "fold":
            return ("f", True) if "f" in legal else (None, False)
        if kind in ("check", "call"):
            return "c", True
        street_paid, total, me, _ = game._replay(state)
        sized = []
        for tok in legal:
            if tok in ("f", "c"):
                continue
            nxt = game.next_state(state, tok)
            to = street_paid[me] + (nxt.contrib[me] - state.contrib[me])
            sized.append((tok, to))
        if not sized:
            return None, False
        if kind == "all_in":
            return ("a", True) if "a" in legal else (sized[-1][0], False)
        for tok, to in sized:
            if abs(to - raise_to_bb) < 1e-6:
                return tok, True
        # Nearest size in log space (pot-geometry-insensitive, scale-free).
        tok = min(sized, key=lambda t: abs(math.log(max(t[1], 1e-9))
                                           - math.log(max(raise_to_bb, 1e-9))))[0]
        return tok, False

    # -- lookup ------------------------------------------------------------------

    def lookup(self, obs: ObservedTableState):
        res = self._to_state(obs)
        if isinstance(res, LookupMiss):
            return res
        state, exact = res
        key = self.game.infoset_key(state)
        probs = self.strategy.get(key)
        if probs is None:
            return LookupMiss("information set never visited in training", "unvisited")
        visits = float(self.visits.get(key, 0.0))
        if self.visits and visits < self.min_visits:
            return LookupMiss(f"information set visited only {visits:g} times "
                              f"(< {self.min_visits:g})", "insufficient_visits")
        street_paid, total, me, _ = self.game._replay(state)
        owe = street_paid[1 - me] - street_paid[me]
        out = []
        for tok in self.game.legal_actions(state):
            p = float(probs.get(tok, 0.0))
            if tok == "f":
                out.append((tok, "fold", "fold", street_paid[me], p))
                continue
            nxt = self.game.next_state(state, tok)
            to = street_paid[me] + (nxt.contrib[me] - state.contrib[me])
            if tok == "c":
                label = "call" if owe > 1e-9 else "check"
                out.append((tok, label, label, to, p))
            elif tok == "a":
                out.append((tok, "all_in", "all_in", to, p))
            else:
                verb = "raise" if owe > 1e-9 or state.street == 0 else "bet"
                pct = int(round(self.game.bet_fractions[tok] * 100))
                out.append((tok, f"{verb}_{pct}", verb, to, p))
        return SolverLookup(key, visits, exact, tuple(out))
