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
from .solver_gate import (ILLEGAL_MASS_LOW, REASONS, REJECT, GateThresholds, KeyStats, downgrade,
                          gate)

Strategy = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class SolverLookup:
    infoset_key: str
    visits: float
    exact: bool
    actions: Tuple[Tuple[str, str, str, float, float], ...]
    # (token, label, kind, raise_to_bb, probability)
    gate: Optional[dict] = None       # solver_gate.GateDecision.to_dict()


# Why a solver strategy was not used (``LookupMiss.code``): the reason codes
# of the solver-use gate (decision/solver_gate.py).
REJECTION_CODES = REASONS


@dataclass(frozen=True)
class LookupMiss:
    reason: str
    code: str = "OUTSIDE_ABSTRACTION"
    reasons: Tuple[str, ...] = ()
    gate: Optional[dict] = None


class SolverStrategyProvider:
    def __init__(self, game: HoldemGame, strategy: Strategy,
                 visits: Optional[Mapping[str, float]] = None,
                 min_visits: float = 20.0,
                 stack_tolerance: float = 0.05,
                 description: str = "abstract heads-up strategy",
                 confidence=None, thresholds=None) -> None:
        self.game = game
        self.strategy = strategy
        self.visits = visits or {}
        self.min_visits = min_visits
        self.stack_tolerance = stack_tolerance
        self.description = description
        # Solver-use gate: a ConfidenceTable (per-key visits / movement /
        # seed disagreement / collision) and thresholds. Without a table the
        # gate still applies the visit threshold and reports NO_STABILITY_DATA.
        self.confidence = confidence
        self.thresholds = thresholds or GateThresholds(
            reject_visits_below=min_visits, low_visits_below=min_visits)

    # -- loading ---------------------------------------------------------------

    @classmethod
    def from_artifact(cls, path, config=None, min_visits: Optional[float] = None,
                      confidence_path=None, use_gate: bool = True):
        """Provider from a strategy artifact (``solvers.strategy_artifact``)
        trained under ``config`` (default: inferred from the artifact's signature).

        Returns a :class:`LookupMiss` (``CONFIG_MISMATCH`` or
        ``INCOMPATIBLE_CHECKPOINT``) instead of raising when it cannot be used.
        """
        from ..solvers.strategy_artifact import (StrategyArtifactError, load_artifact,
                                                 read_config_signature)

        if config is None:
            # Infer the config from the artifact's own signature (known
            # configs only); unknown signatures are rejected as mismatches.
            from ..solver_config import config_for_signature
            try:
                sig = read_config_signature(path)
            except StrategyArtifactError as exc:
                return LookupMiss(f"strategy artifact rejected: {exc}", "INCOMPATIBLE_CHECKPOINT")
            config = config_for_signature(sig)
            if config is None:
                return LookupMiss(f"strategy artifact trained under unknown config "
                                  f"{sig!r}", "CONFIG_MISMATCH")
        game = config.build_game()
        try:
            art = load_artifact(path, game)
        except StrategyArtifactError as exc:
            code = "CONFIG_MISMATCH" if "signature mismatch" in str(exc) else "INCOMPATIBLE_CHECKPOINT"
            return LookupMiss(f"strategy artifact rejected: {exc}", code)
        recall = "perfect recall" if config.perfect_recall else \
            "IMPERFECT-RECALL abstraction, no equilibrium guarantee"
        desc = (f"MCCFR average strategy, {art.meta.get('iterations', '?')} iterations, "
                f"seed {art.meta.get('seed', '?')}, {recall}")
        table, thresholds = None, None
        if use_gate:
            from pathlib import Path

            from .solver_gate import ConfidenceTable, GateThresholds
            cpath = Path(confidence_path) if confidence_path else \
                Path(path).with_name(Path(path).stem + "_confidence.npz")
            if cpath.exists():
                try:
                    table = ConfidenceTable.load(cpath, config.signature())
                except ValueError as exc:
                    return LookupMiss(f"confidence table rejected: {exc}", "CONFIG_MISMATCH")
            thresholds = GateThresholds.calibrated()
            if min_visits is not None:
                from dataclasses import replace as _r
                thresholds = _r(thresholds, reject_visits_below=min_visits,
                                low_visits_below=max(min_visits, thresholds.low_visits_below))
        mv = 20.0 if min_visits is None else min_visits
        return cls(game, art.strategy, art.visits, min_visits=mv, description=desc,
                   confidence=table, thresholds=thresholds)

    @classmethod
    def from_checkpoint(cls, path, config=None, min_visits: float = 20.0):
        """Provider from a full training checkpoint; same rejection rules."""
        from ..solver_config import PRIMARY_CONFIG
        from ..solvers.serialize import CheckpointError, load_checkpoint

        if config is None:
            import numpy as np

            from ..solver_config import config_for_signature
            try:
                with np.load(path, allow_pickle=False) as z:
                    sig = str(z["solver_config"][()]) if "solver_config" in z.files else ""
            except (OSError, ValueError) as exc:
                return LookupMiss(f"checkpoint rejected: {exc}", "INCOMPATIBLE_CHECKPOINT")
            config = config_for_signature(sig) or PRIMARY_CONFIG
        game = config.build_game()
        try:
            solver = load_checkpoint(path, game)
        except CheckpointError as exc:
            code = "CONFIG_MISMATCH" if "mismatch" in str(exc) else "INCOMPATIBLE_CHECKPOINT"
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
            return LookupMiss("information set never visited in training", "UNSEEN_STATE",
                              ("UNSEEN_STATE",))
        visits = float(self.visits.get(key, 0.0)) if self.visits else float("inf")
        stats = self.confidence.get(key) if self.confidence is not None else None
        if stats is None:
            stats = KeyStats(visits)
        decision = gate(stats, self.thresholds,
                        pathological=bool(self.confidence and key in self.confidence.pathological))
        if decision.status == REJECT:
            first = decision.reasons[0]
            detail = ", ".join(f"{k}={v}" for k, v in decision.signals.items() if v is not None)
            return LookupMiss(f"solver gate rejected {first} ({detail})", first, decision.reasons,
                              decision.to_dict())
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
                if tok[0] == "x":
                    m = self.game.preflop_raise_multiples[tok]
                    out.append((tok, f"raise_to_{m:g}x", "raise", to, p))
                else:
                    pct = int(round(self.game.bet_fractions[tok] * 100))
                    out.append((tok, f"{verb}_{pct}", verb, to, p))
        # Never recommend an abstract size that is illegal in real NLHE (the
        # v1 abstraction offers bets/raises below the minimum): drop them,
        # renormalize, and report the removed mass.
        level = street_paid[1 - me]
        min_inc = self.game._min_increment(state.streets)
        legal_out, removed = [], 0.0
        for row in out:
            tok, to, p = row[0], row[3], row[4]
            if tok not in ("f", "c", "a") and to - level < min_inc - 1e-9:
                removed += p
                continue
            legal_out.append(row)
        gate_d = decision.to_dict()
        if removed > 0 or len(legal_out) < len(out):
            rest = sum(r[4] for r in legal_out)
            if rest <= 0:
                return LookupMiss("all solver mass is on sizes below the NLHE minimum",
                                  "OUTSIDE_ABSTRACTION", ("OUTSIDE_ABSTRACTION",), gate_d)
            legal_out = [r[:4] + (r[4] / rest,) for r in legal_out]
            gate_d["signals"]["illegal_size_mass_removed"] = round(removed, 4)
            if removed >= ILLEGAL_MASS_LOW:
                gate_d = downgrade(gate_d, "ILLEGAL_SIZE_MASS")
        if not exact:
            gate_d = downgrade(gate_d, "OFF_TREE_TRANSLATION")
        return SolverLookup(key, visits, exact, tuple(legal_out), gate_d)
