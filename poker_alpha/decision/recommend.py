"""The decision engine: ``recommend_action`` -> :class:`DecisionReport`.

Method hierarchy
----------------
A. **Solver** — if a :class:`SolverStrategyProvider` is configured and the
   spot maps into its trained heads-up abstraction (with enough visits), the
   recommended frequencies are the solver's average strategy. The locked
   solver config uses an IMPERFECT-RECALL abstraction, so this is an
   abstract MCCFR strategy with no equilibrium guarantee — of the abstract
   game or of real Hold'em; it is labelled ``"solver"`` (exact mapping) or
   ``"interpolated abstraction"`` (off-tree sizes translated). When it is
   not used, the reason is recorded with a code from
   :data:`~.strategy.REJECTION_CODES` (config mismatch, incompatible
   checkpoint, out of abstraction, unvisited, insufficient visits).
B/C. **Range-based estimation** — otherwise (heads-up outside the
   abstraction, or any multiway spot) candidate actions come from the
   :class:`ActionAbstraction` menu and are evaluated against the opponents'
   *estimated* ranges: by Monte Carlo rollouts when enabled
   (``"Monte Carlo rollout"``) or by closed-form pot-odds reasoning
   (``"heuristic fallback"``). Neither is game-theoretically optimal and
   neither is ever called GTO.

Priority: solver -> Monte Carlo rollout -> heuristic. Every report carries
``details["source_cascade"]`` (what was tried, used or rejected, and why)
and ``uncertainty`` (observation, range estimation, sampling, abstraction
and response-model uncertainty, kept separate).

Opponent ranges are beliefs: data-driven preflop priors, card removal, and
Bayesian updates for each observed postflop action, with entropy reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


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
    # Set when a configured strategy could not be loaded (see
    # SolverStrategyProvider.from_artifact); reported in the source cascade.
    solver_unavailable: Optional[LookupMiss] = None
    priors: Optional[RangePriors] = None
    default_model: str = "regular"
    hero_model: str = "regular"           # hero's continuation policy in rollouts
    opponent_bet_fraction: float = 0.66   # size opponents bet when hero checks
    raise_multiplier: float = 3.0         # opponents' raise-to vs hero's bet
    rollout_response: str = "behavior"    # opponents' answer to a bet (rollout.RESPONSE_MODELS)
    rollout_response_params: Optional[Tuple[float, float]] = None
    rollout_depth: str = "street"         # "showdown": play later streets too (HU, slower)
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
    if r.is_empty():
        return RangeSummary(seat=seat, position=pos, line=line + " (EMPTY)",
                            live_combos=0, entropy_bits=0.0, effective_combos=0.0,
                            top_classes=(), model=model)
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

    # Refusal reasons (coded). Anything here means no recommendation at all:
    # guessing would fabricate a decision from a state we cannot trust.
    refuse: List[str] = sorted({i.code for i in errors})
    if hero is None or len(hero) != 2:
        refuse.append("HERO_CARDS_UNKNOWN")
        warnings.append("hero cards unknown: cannot evaluate")
    if state.actor not in (None, state.hero_seat):
        refuse.append("NOT_HERO_TURN")
        warnings.append("it is not the hero's turn")
    if not errors and state.hero.stack is None:
        refuse.append("HERO_STACK_UNKNOWN")
        warnings.append("hero stack unknown: bet sizes, all-in and SPR cannot be "
                        "evaluated — read or enter the hero's stack")
    if not errors and state.hero.folded:
        refuse.append("HERO_FOLDED")
    if not errors and not state.hero.folded and not state.opponents_in_hand:
        refuse.append("NO_OPPONENTS")
    if not errors and state.hero.all_in:
        refuse.append("HERO_ALL_IN")
        warnings.append("hero is all-in: no decision to make")
    if refuse:
        return DecisionReport(hero_equity=None, hero_equity_se=None,
                              opponent_ranges=(), candidates=(),
                              recommended=None, recommended_mix={},
                              mix_meaning="no recommendation",
                              method="none", confidence="low",
                              warnings=tuple(warnings),
                              details={"refusal_codes": refuse, "source_cascade": [
                                  {"source": "none", "status": "no decision",
                                   "codes": refuse,
                                   "reason": "invalid or incomplete state: " + ", ".join(refuse)}]},
                              uncertainty=_uncertainty(cfg, state, [], None, None, None, None),
                              **base)

    # Documented assumptions (recommendation still made, confidence capped).
    assumptions: List[str] = []
    if state.actor is None:
        assumptions.append("ACTOR_UNKNOWN")
        warnings.append("actor unknown: assuming it is the hero's turn")
    unknown_opp = [s for s in state.opponents_in_hand if state.seats[s].stack is None]
    if unknown_opp:
        assumptions.append("OPPONENT_STACK_UNKNOWN")
        warnings.append("opponent stack unknown (seat " + ", ".join(map(str, unknown_opp))
                        + "): assumed to cover the hero; all-in and SPR are approximate")

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
    cascade: List[Dict[str, object]] = []
    equity = se = None
    empty = [s for s, r in ranges.items() if r.is_empty()]
    if empty:
        warnings.append("opponent range empty after card removal (seat "
                        + ", ".join(map(str, empty)) + "): equity and rollouts unavailable")
        cascade.append({"source": "equity", "status": "failed", "code": "RANGE_EMPTY",
                        "seats": empty})
    elif ranges:
        try:
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
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            warnings.append(f"equity estimate failed ({type(exc).__name__}: {exc})")
            cascade.append({"source": "equity", "status": "failed", "code": "EQUITY_ERROR",
                            "reason": f"{type(exc).__name__}: {exc}"})
    usable_ranges = bool(ranges) and not empty

    # Method A: trained heads-up strategy.
    lookup = None
    solver_info: Dict[str, object] = {"used": False, "confidence": "not configured", "reasons": []}
    if cfg.solver is not None:
        try:
            lookup = cfg.solver.lookup(state)
        except Exception as exc:  # noqa: BLE001 - a broken solver must not crash analysis
            lookup = LookupMiss(f"solver lookup failed ({type(exc).__name__}: {exc})",
                                code="SOLVER_ERROR")
        if isinstance(lookup, LookupMiss):
            warnings.append(f"solver strategy not used: {lookup.reason}")
            cascade.append({"source": "solver", "status": "rejected",
                            "code": lookup.code, "reason": lookup.reason,
                            "reasons": list(lookup.reasons or (lookup.code,)),
                            "gate": lookup.gate})
            solver_info = {"used": False, "confidence": "rejected",
                           "reasons": list(lookup.reasons or (lookup.code,)),
                           **((lookup.gate or {}).get("signals") or {})}
            lookup = None
    elif cfg.solver_unavailable is not None:
        miss = cfg.solver_unavailable
        warnings.append(f"solver strategy not used: {miss.reason}")
        cascade.append({"source": "solver", "status": "rejected", "code": miss.code,
                        "reason": miss.reason, "reasons": [miss.code]})
        solver_info = {"used": False, "confidence": "rejected", "reasons": [miss.code]}
    else:
        cascade.append({"source": "solver", "status": "not configured"})

    if lookup is not None:
        source = "solver" if lookup.exact else "interpolated abstraction"
        cands = [CandidateAction(label=label, kind=kind, amount_to=to * bb,
                                 added=max(0.0, to * bb - state.hero.current_bet),
                                 probability=p, ev_bb=None, ev_se_bb=None,
                                 source=source)
                 for _, label, kind, to, p in lookup.actions]
        gate_d = lookup.gate or {"status": "SOLVER_ACCEPT", "reasons": [], "signals": {}}
        cascade.append({"source": source, "status": "used",
                        "infoset": lookup.infoset_key, "visits": lookup.visits,
                        "strategy": cfg.solver.description, "gate": gate_d})
        solver_info = {"used": True, "confidence": gate_d["status"], "reasons": gate_d["reasons"],
                       **gate_d.get("signals", {})}
        rres = None
        if cfg.rollout_simulations and usable_ranges:
            try:
                cands, rres = _attach_rollout_evs(state, hero, ranges, models, cfg,
                                                  cands, source=source)
                cascade.append({"source": "Monte Carlo rollout", "status": "used for EVs only"})
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"rollout EVs unavailable ({type(exc).__name__}: {exc})")
                cascade.append({"source": "Monte Carlo rollout", "status": "failed",
                                "code": "ROLLOUT_ERROR",
                                "reason": f"{type(exc).__name__}: {exc}"})
        else:
            cascade.append({"source": "Monte Carlo rollout", "status": "not run"})
        cascade.append({"source": "heuristic fallback", "status": "not needed"})
        mix = {c.label: c.probability for c in cands if c.probability}
        rec = max(cands, key=lambda c: c.probability or 0.0).label
        confidence = "medium" if lookup.exact else "low"
        if gate_d["status"] != "SOLVER_ACCEPT":
            confidence = "low"
            warnings.append("solver confidence low: " + ", ".join(gate_d["reasons"]))
        if not lookup.exact:
            warnings.append("off-tree bet sizes were translated onto the abstraction")
        if rres is not None:
            best = max((c for c in cands if c.ev_bb is not None), key=lambda c: c.ev_bb,
                       default=None)
            if best is not None and best.label != rec:
                try:
                    mean, se_d = rres.paired_difference(best.label, rec)
                except KeyError:
                    mean, se_d = 0.0, 0.0
                if mean > 2 * max(se_d, 1e-9):
                    warnings.append(
                        f"solver and rollouts disagree: solver prefers {rec}, rollout EV "
                        f"favours {best.label} by {mean:.2f} BB (±{se_d:.2f}); both are "
                        "approximations — treat this spot as uncertain")
                    confidence = "low"
                    cascade.append({"source": "consistency check", "status": "conflict",
                                    "reason": f"rollout best {best.label}, solver {rec}"})
        warnings.append("solver frequencies are from an abstracted heads-up "
                        "game (imperfect-recall abstraction): not a Hold'em "
                        "equilibrium and not proven optimal")
        if assumptions:
            confidence = "low"
        if cfg.observer_confidence is not None and cfg.observer_confidence < 0.9:
            warnings.append(f"observer confidence {cfg.observer_confidence:.0%}: "
                            "verify the recognized state")
            confidence = "low"
        return DecisionReport(
            hero_equity=equity, hero_equity_se=se,
            opponent_ranges=tuple(summaries), candidates=tuple(cands),
            recommended=rec, recommended_mix=mix,
            mix_meaning=f"solver average-strategy frequencies ({cfg.solver.description})",
            method=source, confidence=confidence, warnings=tuple(warnings),
            details={"infoset": lookup.infoset_key, "visits": lookup.visits,
                     "source_cascade": cascade, "solver": solver_info,
                     "assumptions": assumptions},
            uncertainty=_uncertainty(cfg, state, summaries, se, rres, lookup, source),
            **base)

    # Methods B/C: abstraction menu evaluated against ranges.
    menu = cfg.abstraction.menu(ctx)
    res = rollout_failure = None
    if cfg.rollout_simulations and usable_ranges:
        cands = [CandidateAction(label=l, kind=a.kind, amount_to=a.raise_to,
                                 added=a.add, probability=None, ev_bb=None,
                                 ev_se_bb=None) for l, a in menu]
        try:
            cands, res = _attach_rollout_evs(state, hero, ranges, models, cfg, cands)
        except Exception as exc:  # noqa: BLE001 - fall back to the heuristic, reported
            rollout_failure = f"{type(exc).__name__}: {exc}"
            warnings.append(f"rollout failed ({rollout_failure}): heuristic fallback used")
    if res is not None:
        method = "Monte Carlo rollout"
        cascade.append({"source": method, "status": "used",
                        "simulations": cfg.rollout_simulations})
        cascade.append({"source": "heuristic fallback", "status": "not needed"})
        mix = dict(res.best_probability)
        rec = max(cands, key=lambda c: c.ev_bb).label
        cands = [_with_prob(c, mix.get(c.label, 0.0)) for c in cands]
        mix_meaning = ("paired-bootstrap probability that each action has "
                       "the highest rollout EV (estimation uncertainty, not "
                       "a mixed strategy)")
        confidence = _confidence_from_evs(cands, summaries, res)
        if len(ranges) > 1:
            warnings.append("multiway spot: EVs are approximate range-based "
                            "estimates, not an equilibrium")
        if rec in ("all_in",) and (state.spr or 0) > 3:
            warnings.append("all-in at high SPR: rollouts check down after one "
                            "response, so later-street value of smaller bets "
                            "is not credited; treat with caution")
    else:
        if rollout_failure is not None:
            cascade.append({"source": "Monte Carlo rollout", "status": "failed",
                            "code": "ROLLOUT_ERROR", "reason": rollout_failure})
        else:
            cascade.append({"source": "Monte Carlo rollout", "status": "not run",
                            "reason": "rollouts disabled (rollout_simulations=0)"
                            if not cfg.rollout_simulations else "no usable opponent ranges"})
        cands = _heuristic_candidates(menu, equity, state)
        method = "heuristic fallback"
        cascade.append({"source": method, "status": "used"})
        rec = _heuristic_choice(cands, equity, pot_odds)
        if rec is None:
            method = "none"
            warnings.append("no recommendation: facing a bet with no equity estimate")
            cascade[-1] = {"source": "heuristic fallback", "status": "no decision",
                           "code": "NO_EQUITY"}
        elif equity is None:
            warnings.append(f"heuristic chose {rec} without an equity estimate "
                            "(checking is never worse than folding)")
        mix = {rec: 1.0} if rec else {}
        cands = [_with_prob(c, mix.get(c.label, 0.0)) for c in cands]
        mix_meaning = "heuristic pot-odds choice (not a strategy)"
        confidence = "low"
    if assumptions:
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
        details={"source_cascade": cascade, "solver": solver_info,
                 "assumptions": assumptions},
        uncertainty=_uncertainty(cfg, state, summaries, se, res, None, method),
        **base)


def _uncertainty(cfg, state, summaries, equity_se, rollout_res, lookup, method):
    """Separate the sources of uncertainty instead of one blended number."""
    u: Dict[str, object] = {}
    if cfg.observer_confidence is None:
        u["observation"] = f"state given as input (source: {state.source or 'unknown'}); not recognized from pixels"
    else:
        u["observation"] = (f"screen recognition, critical-field confidence "
                            f"{cfg.observer_confidence:.2f}; real PokerNow accuracy not validated")
    if summaries:
        u["range_estimation"] = "; ".join(
            f"seat {r.seat} {r.position} [{r.line}]: ~{r.effective_combos:.0f} effective combos, "
            f"entropy {r.entropy_bits:.1f} bits (belief, model {r.model})" for r in summaries)
    else:
        u["range_estimation"] = "no opponent ranges"
    samp = []
    if equity_se is not None:
        samp.append(f"equity standard error {equity_se:.3f}")
    if rollout_res is not None:
        ses = [e.se_bb for e in rollout_res.evs]
        if ses:
            samp.append(f"rollout EV standard error up to {max(ses):.2f} BB "
                        f"({rollout_res.simulations} simulations)")
    u["sampling"] = "; ".join(samp) if samp else "none"
    if lookup is not None:
        g = lookup.gate or {}
        sig = g.get("signals", {})
        extra = "; ".join(f"{k} {v}" for k, v in sig.items() if v is not None and k != "visits")
        u["abstraction"] = (f"solver infoset {lookup.infoset_key!r} with {lookup.visits:.0f} visits; "
                            f"{'exact' if lookup.exact else 'off-tree sizes translated'}; "
                            f"gate {g.get('status', 'n/a')}"
                            + (f" ({', '.join(g.get('reasons', []))})" if g.get("reasons") else "")
                            + (f"; {extra}" if extra else "")
                            + f"; {cfg.solver.description}")
    else:
        u["abstraction"] = "solver not used; candidate sizes from the betting abstraction menu"
    if method == "Monte Carlo rollout" or (lookup is not None and rollout_res is not None):
        u["response_model"] = (f"opponent responses from behaviour model(s), hero continuation "
                               f"'{cfg.hero_model}'; one response, then check-down")
    elif method == "heuristic fallback":
        u["response_model"] = "no response model: check/call EVs assume no further betting"
    elif lookup is not None:
        u["response_model"] = "none (solver frequencies; no EVs computed)"
    else:
        u["response_model"] = "n/a"
    return u


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


def _confidence_from_evs(cands, summaries, res) -> str:
    """High only when the best action beats the runner-up by > 3 paired SEs
    heads-up against a reasonably narrow range."""
    ranked = sorted((c for c in cands if c.ev_bb is not None),
                    key=lambda c: c.ev_bb, reverse=True)
    if len(ranked) < 2:
        return "low"
    mean, se = res.paired_difference(ranked[0].label, ranked[1].label)
    gap = mean / max(se, 1e-9)
    wide = any(r.effective_combos > 300 for r in summaries)
    if gap > 3 and not wide:
        return "high" if len(summaries) == 1 else "medium"
    if gap > 1.5:
        return "medium"
    return "low"


def _attach_rollout_evs(state, hero, ranges, models, cfg, cands,
                        source: Optional[str] = None):
    """Fill EV/SE/sample counts from common-random-number rollouts.

    Returns ``(candidates, best_probability)``."""
    from dataclasses import replace

    from .rollout import RolloutCandidate, rollout_action_evs

    rc = [RolloutCandidate(c.label, c.kind if c.label != "call" else "call",
                           c.amount_to) for c in cands]
    res = rollout_action_evs(
        state, hero, ranges, models, rc,
        simulations=cfg.rollout_simulations, seed=cfg.seed,
        hero_model=ARCHETYPE_MODELS[cfg.hero_model],
        opponent_bet_fraction=cfg.opponent_bet_fraction,
        raise_multiplier=cfg.raise_multiplier,
        response=cfg.rollout_response, response_params=cfg.rollout_response_params,
        depth=cfg.rollout_depth if len(ranges) == 1 else "street")
    out = []
    for c in cands:
        e = res.ev(c.label)
        note = c.note
        if source is not None:
            note = (note + "; " if note else "") + "EV from Monte Carlo rollout"
        if cfg.rollout_depth != "street" and len(ranges) != 1:
            note = (note + "; " if note else "") + "multiway: one-street rollout depth used"
        out.append(replace(c, ev_bb=e.ev_bb, ev_se_bb=e.se_bb,
                           samples=e.samples,
                           source=source or "Monte Carlo rollout", note=note))
    return out, res
