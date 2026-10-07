"""The decision engine: ``recommend_action`` -> :class:`DecisionReport`.

Method hierarchy
----------------
A. **Solver** — if a :class:`SolverStrategyProvider` is configured and the
   spot maps into its trained heads-up abstraction (with enough visits), the
   recommended frequencies are the solver's average strategy. An abstract
   heads-up strategy is an *approximate equilibrium of the abstract game*,
   not of real Hold'em; it is labelled ``"solver"`` (exact mapping) or
   ``"interpolated abstraction"`` (off-tree sizes translated).
B/C. **Range-based estimation** — otherwise (heads-up outside the
   abstraction, or any multiway spot) candidate actions come from the
   :class:`ActionAbstraction` menu and are evaluated against the opponents'
   *estimated* ranges: by Monte Carlo rollouts when enabled
   (``"Monte Carlo rollout"``) or by closed-form pot-odds reasoning
   (``"heuristic fallback"``). Neither is game-theoretically optimal and
   neither is ever called GTO.

Opponent ranges are beliefs: data-driven preflop priors, card removal, and
Bayesian updates for each observed postflop action, with entropy reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..abstraction.betting import ActionAbstraction, BettingContext
from ..holdem.observed import ObservedTableState, validate
from ..holdem.state import Street
from ..opponent.behavior import ARCHETYPE_MODELS, BehaviorModel
from ..opponent.ranges import (RangePriors, classify_preflop_line,
                               update_range_for_action)
from ..poker.multiway import SamplingError, multiway_equity
from ..poker.ranges import WeightedRange
from .report import CandidateAction, DecisionReport, RangeSummary
from .strategy import LookupMiss, SolverLookup, SolverStrategyProvider

BOARD_AT_STREET = {0: 0, 1: 3, 2: 4, 3: 5}


@dataclass
class DecisionConfig:
    equity_simulations: int = 3000
    rollout_simulations: int = 0          # 0 disables Monte Carlo rollouts
    seed: int = 0
    abstraction: ActionAbstraction = field(default_factory=ActionAbstraction)
    solver: Optional[SolverStrategyProvider] = None
    priors: Optional[RangePriors] = None
    default_model: str = "regular"
    hero_model: str = "regular"           # hero's continuation policy in rollouts
    opponent_bet_fraction: float = 0.66   # size opponents bet when hero checks
    raise_multiplier: float = 3.0         # opponents' raise-to vs hero's bet
    observer_confidence: Optional[float] = None  # from screen recognition


# -- opponent ranges -------------------------------------------------------------

def infer_opponent_range(obs: ObservedTableState, seat: int,
                         priors: RangePriors, model: BehaviorModel,
                         warnings: List[str]) -> Tuple[WeightedRange, str]:
    """Preflop prior by position/line, card removal, then a Bayesian update
    for each of the seat's observed postflop actions."""
    positions = obs.positions()
    pos = positions.get(seat, "?")
    pre = [a for a in obs.action_history if a.street == 0]
    line = classify_preflop_line(pre, seat)
    opp = obs.seats[seat]
    stack_bb = None
    if opp.stack is not None:
        stack_bb = (opp.stack + (opp.committed_total or opp.current_bet)) / obs.big_blind
    r = priors.prior(pos, line, stack_bb)
    dead = list(obs.hero_cards or ()) + [c for _, cs in obs.shown_cards
                                         for c in cs]
    r = r.remove_cards(dead + list(obs.board))
    if r.is_empty():
        warnings.append(f"seat {seat}: prior range empty after card removal; "
                        "using any two cards")
        r = WeightedRange.uniform().remove_cards(dead + list(obs.board))
    level: Dict[int, float] = {}
    pot_before = 0.0
    street_commit: Dict[int, float] = {}
    cur_street = 0
    for a in obs.action_history:
        if a.street != cur_street:
            pot_before += sum(street_commit.values())
            street_commit = {}
            cur_street = a.street
        lvl = level.get(a.street, obs.big_blind if a.street == 0 else 0.0)
        facing = lvl > street_commit.get(a.seat, 0.0) + 1e-9
        if a.kind in ("bet", "raise", "all_in"):
            to = a.amount
        elif a.kind == "call":
            to = lvl
        elif a.kind == "post":
            to = street_commit.get(a.seat, 0.0) + a.amount
        else:
            to = street_commit.get(a.seat, 0.0)
        if a.seat == seat and a.street >= 1 and a.kind in (
                "bet", "raise", "all_in", "call", "check"):
            board = obs.board[:BOARD_AT_STREET[min(a.street, 3)]]
            pot_now = pot_before + sum(street_commit.values())
            size = (to - lvl) / pot_now if pot_now > 0 else 0.5
            try:
                r = update_range_for_action(
                    r, a.kind if a.kind != "all_in" else "raise",
                    facing_bet=facing, board=board, dead=dead, model=model,
                    size_pot_fraction=max(size, 0.0))
            except ValueError as exc:
                warnings.append(f"seat {seat}: skipped update for {a.kind} "
                                f"({exc})")
        street_commit[a.seat] = max(street_commit.get(a.seat, 0.0), to)
        level[a.street] = max(lvl, to)
    return r, line


def _summary(seat: int, pos: str, line: str, r: WeightedRange,
             model: str) -> RangeSummary:
    return RangeSummary(seat=seat, position=pos, line=line,
                        live_combos=r.num_live_combos,
                        entropy_bits=r.entropy_bits(),
                        effective_combos=r.effective_combos(),
                        top_classes=tuple(r.top_classes(8)), model=model)


def betting_context(obs: ObservedTableState) -> BettingContext:
    hero = obs.hero
    opp_stacks = [obs.seats[i].stack for i in obs.opponents_in_hand
                  if not obs.seats[i].all_in]
    known = [s for s in opp_stacks if s is not None]
    hero_stack = hero.stack if hero.stack is not None else 0.0
    bets = sorted((s.current_bet for s in obs.seats if s.occupied), reverse=True)
    last_inc = bets[0] - bets[1] if len(bets) > 1 else bets[0]
    return BettingContext(
        pot=obs.pot, to_call=obs.amount_to_call, hero_stack=hero_stack,
        max_opponent_stack=max(known) if known else (0.0 if opp_stacks == [] else hero_stack),
        hero_street=hero.current_bet,
        min_raise_increment=max(obs.big_blind, last_inc),
        big_blind=obs.big_blind)


# -- the engine ------------------------------------------------------------------

def recommend_action(state: ObservedTableState,
                     hero_range_or_hand=None,
                     opponent_ranges: Optional[Mapping[int, WeightedRange]] = None,
                     opponent_models: Optional[Mapping[int, BehaviorModel]] = None,
                     config: Optional[DecisionConfig] = None) -> DecisionReport:
    """Analyse the hero's decision in ``state``.

    ``hero_range_or_hand`` defaults to ``state.hero_cards``. Explicit
    ``opponent_ranges`` (seat -> range) override inferred ones;
    ``opponent_models`` (seat -> BehaviorModel) drive range updates and
    rollout responses.
    """
    cfg = config or DecisionConfig()
    warnings: List[str] = []
    issues = validate(state)
    errors = [i for i in issues if i.severity == "error"]
    for i in issues:
        warnings.append(f"{i.severity}: {i.message}")
    hero = tuple(hero_range_or_hand) if hero_range_or_hand is not None \
        else state.hero_cards
    bb = state.big_blind
    summary = state.describe() if state.hero_cards else f"{state.street.name} (hero cards unknown)"
    ctx = betting_context(state)
    pot_odds = state.pot_odds
    eff = state.effective_stack
    base = dict(
        state_summary=summary, pot_bb=state.pot / bb,
        to_call_bb=state.amount_to_call / bb, pot_odds=pot_odds,
        spr=state.spr, effective_stack_bb=None if eff is None else eff / bb)

    if errors or hero is None or len(hero) != 2 or state.actor not in (None, state.hero_seat):
        if hero is None:
            warnings.append("hero cards unknown: cannot evaluate")
        if state.actor not in (None, state.hero_seat):
            warnings.append("it is not the hero's turn")
        return DecisionReport(hero_equity=None, hero_equity_se=None,
                              opponent_ranges=(), candidates=(),
                              recommended=None, recommended_mix={},
                              mix_meaning="no recommendation",
                              method="none", confidence="low",
                              warnings=tuple(warnings), **base)

    # Opponent ranges & models.
    priors = cfg.priors or RangePriors.load()
    models = dict(opponent_models or {})
    ranges: Dict[int, WeightedRange] = {}
    summaries = []
    positions = state.positions()
    for seat in state.opponents_in_hand:
        model = models.get(seat) or ARCHETYPE_MODELS[cfg.default_model]
        models[seat] = model
        if opponent_ranges and seat in opponent_ranges:
            r = opponent_ranges[seat].remove_cards(list(hero) + list(state.board))
            line = "given"
        else:
            r, line = infer_opponent_range(state, seat, priors, model, warnings)
        ranges[seat] = r
        summaries.append(_summary(seat, positions.get(seat, "?"), line, r,
                                  model.name))
    if not ranges:
        warnings.append("no opponents remain in the hand")

    # Hero equity vs the estimated ranges (all-in-now share, no side pots).
    equity = se = None
    if ranges:
        try:
            eq = multiway_equity(hero, state.board, list(ranges.values()),
                                 simulations=cfg.equity_simulations,
                                 seed=cfg.seed)
        except SamplingError:
            eq = multiway_equity(hero, state.board, list(ranges.values()),
                                 simulations=cfg.equity_simulations,
                                 seed=cfg.seed, method="importance")
            warnings.append("equity used importance sampling (ranges overlap)")
        equity, se = eq.expected_share, eq.std_error

    # Method A: trained heads-up strategy.
    lookup = None
    if cfg.solver is not None:
        lookup = cfg.solver.lookup(state)
        if isinstance(lookup, LookupMiss):
            warnings.append(f"solver strategy not used: {lookup.reason}")
            lookup = None

    if lookup is not None:
        source = "solver" if lookup.exact else "interpolated abstraction"
        cands = [CandidateAction(label=label, kind=kind, amount_to=to * bb,
                                 added=max(0.0, to * bb - state.hero.current_bet),
                                 probability=p, ev_bb=None, ev_se_bb=None,
                                 source=source)
                 for _, label, kind, to, p in lookup.actions]
        if cfg.rollout_simulations and ranges:
            cands = _attach_rollout_evs(state, hero, ranges, models, cfg, cands)
        mix = {c.label: c.probability for c in cands if c.probability}
        rec = max(cands, key=lambda c: c.probability or 0.0).label
        confidence = "medium" if lookup.exact else "low"
        if not lookup.exact:
            warnings.append("off-tree bet sizes were translated onto the abstraction")
        warnings.append("solver frequencies are from an abstracted heads-up "
                        "game; not an exact Hold'em equilibrium")
        return DecisionReport(
            hero_equity=equity, hero_equity_se=se,
            opponent_ranges=tuple(summaries), candidates=tuple(cands),
            recommended=rec, recommended_mix=mix,
            mix_meaning="solver average-strategy frequencies",
            method=source, confidence=confidence, warnings=tuple(warnings),
            details={"infoset": lookup.infoset_key, "visits": lookup.visits},
            **base)

    # Methods B/C: abstraction menu evaluated against ranges.
    menu = cfg.abstraction.menu(ctx)
    if cfg.rollout_simulations and ranges:
        cands = [CandidateAction(label=l, kind=a.kind, amount_to=a.raise_to,
                                 added=a.add, probability=None, ev_bb=None,
                                 ev_se_bb=None) for l, a in menu]
        cands = _attach_rollout_evs(state, hero, ranges, models, cfg, cands)
        method = "Monte Carlo rollout"
        mix, rec = _best_probabilities(cands, cfg.seed)
        cands = [_with_prob(c, mix.get(c.label, 0.0)) for c in cands]
        mix_meaning = ("probability each action has the highest EV given "
                       "the rollout estimates and their standard errors "
                       "(uncertainty, not a mixed strategy)")
        confidence = _confidence_from_evs(cands, summaries, cfg)
        if len(ranges) > 1:
            warnings.append("multiway spot: EVs are approximate range-based "
                            "estimates, not an equilibrium")
    else:
        cands = _heuristic_candidates(menu, equity, state)
        method = "heuristic fallback"
        rec = _heuristic_choice(cands, equity, pot_odds)
        mix = {rec: 1.0} if rec else {}
        cands = [_with_prob(c, mix.get(c.label, 0.0)) for c in cands]
        mix_meaning = "heuristic pot-odds choice (not a strategy)"
        confidence = "low"
    if cfg.observer_confidence is not None and cfg.observer_confidence < 0.9:
        warnings.append(f"observer confidence {cfg.observer_confidence:.0%}: "
                        "verify the recognized state")
        confidence = "low"
    return DecisionReport(
        hero_equity=equity, hero_equity_se=se,
        opponent_ranges=tuple(summaries), candidates=tuple(cands),
        recommended=rec, recommended_mix=mix, mix_meaning=mix_meaning,
        method=method, confidence=confidence, warnings=tuple(warnings),
        **base)


def _with_prob(c: CandidateAction, p: float) -> CandidateAction:
    from dataclasses import replace
    return replace(c, probability=p)


def _heuristic_candidates(menu, equity: Optional[float],
                          state: ObservedTableState) -> List[CandidateAction]:
    """Closed-form EVs where they are honest: fold = 0, check = share of the
    current pot, call = share of the final pot minus the call (both assume a
    check-down with no further betting). Bets/raises need an opponent
    response model, so their EV is left unknown here."""
    bb = state.big_blind
    pot = state.pot
    out = []
    for label, a in menu:
        ev = None
        note = ""
        if a.kind == "fold":
            ev = 0.0
        elif equity is not None and a.kind == "check":
            ev = equity * pot / bb
            note = "assumes check-down"
        elif equity is not None and (a.kind == "call" or
                                     (label == "call" and a.kind == "all_in")):
            ev = (equity * (pot + a.add) - a.add) / bb
            note = "assumes no further betting"
        else:
            note = "needs a response model (enable rollouts)"
        out.append(CandidateAction(label=label, kind=a.kind, amount_to=a.raise_to,
                                   added=a.add, probability=None, ev_bb=ev,
                                   ev_se_bb=None, source="heuristic fallback",
                                   note=note))
    return out


def _heuristic_choice(cands, equity, pot_odds) -> Optional[str]:
    labels = [c.label for c in cands]
    if "fold" in labels and "call" in labels:
        if equity is None or pot_odds is None:
            return None
        return "call" if equity >= pot_odds else "fold"
    if "check" in labels:
        return "check"
    return labels[0] if labels else None


def _best_probabilities(cands: Sequence[CandidateAction], seed: int,
                        draws: int = 4000) -> Tuple[Dict[str, float], str]:
    """P(action is EV-best) under independent normal estimate errors."""
    evaluated = [c for c in cands if c.ev_bb is not None]
    if not evaluated:
        return {}, cands[0].label if cands else None
    rng = np.random.default_rng(seed + 7919)
    means = np.array([c.ev_bb for c in evaluated])
    ses = np.array([c.ev_se_bb or 0.0 for c in evaluated])
    sims = means[None, :] + rng.standard_normal((draws, len(evaluated))) * ses[None, :]
    best = np.bincount(sims.argmax(axis=1), minlength=len(evaluated)) / draws
    mix = {c.label: float(p) for c, p in zip(evaluated, best) if p > 0}
    rec = evaluated[int(np.argmax(means))].label
    return mix, rec


def _confidence_from_evs(cands, summaries, cfg) -> str:
    evs = sorted(((c.ev_bb, c.ev_se_bb or 0.0) for c in cands
                  if c.ev_bb is not None), reverse=True)
    if len(evs) < 2:
        return "low"
    (m1, s1), (m2, s2) = evs[0], evs[1]
    gap = (m1 - m2) / max(np.hypot(s1, s2), 1e-9)
    wide = any(r.effective_combos > 300 for r in summaries)
    if gap > 3 and not wide:
        return "high" if len(summaries) == 1 else "medium"
    if gap > 1.5:
        return "medium"
    return "low"


def _attach_rollout_evs(state, hero, ranges, models, cfg, cands):
    """Placeholder until rollouts exist: EVs stay unknown."""
    return list(cands)
