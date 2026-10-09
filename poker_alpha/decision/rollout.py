"""Local Monte Carlo rollouts comparing the hero's candidate actions.

For each sample we draw, once and shared by **every** candidate action
(common random numbers):

* opponents' hands jointly from their ranges (exact rejection sampling, no
  card collisions),
* the remaining board runout,
* uniform numbers driving each opponent's (and the hero's) responses.

Each candidate is then played out against that same sample:

1. hero takes the action (fold / check / call / bet X / raise X / all-in);
2. opponents respond with their :class:`BehaviorModel` given their hand's
   *current-street strength*, position order and bet size — fold, call or
   (at most one) raise to ``raise_multiplier ×`` the bet; after a check,
   opponents behind the hero may bet ``opponent_bet_fraction`` of the pot;
3. if raised (or bet into after checking) the hero continues per
   ``hero_model`` — a simple strength-threshold policy, not a solver;
4. the hand is then **checked down** to showdown, pots split with side-pot
   rules.

The EV is the hero's expected stack change versus folding now, in big
blinds, with its standard error and sample count. Because all actions share
samples, EV *differences* are far less noisy than the individual SEs
suggest; :attr:`RolloutResult.best_probability` is a paired bootstrap of
"which action has the highest mean".

Modelling limits (deliberate, documented): one response round, at most one
raise, no further betting on later streets (check-down), pre-existing pot
treated as one layer all remaining players are eligible for. These make the
numbers *approximate local EVs*, useful for comparing candidates, not
equilibrium values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..holdem.observed import ObservedTableState
from ..holdem.pots import Pot, award_pots, build_pots
from ..holdem.positions import clockwise_from
from ..holdem.state import Street
from ..opponent.behavior import ARCHETYPE_MODELS, BehaviorModel
from ..opponent.ranges import strength_vector
from ..poker.cards import codes
from ..poker.evaluator import evaluate_best_codes
from ..poker.ranges import (COMBO_INDEX, COMBOS, NUM_COMBOS, WeightedRange,
                            combo_cdf, draw_combo)


class RolloutError(RuntimeError):
    pass


@dataclass(frozen=True)
class RolloutCandidate:
    label: str
    kind: str          # fold/check/call/bet/raise/all_in
    raise_to: float    # hero street commitment after the action (chips)


@dataclass(frozen=True)
class ActionEV:
    label: str
    ev_bb: float
    se_bb: float
    samples: int


@dataclass(frozen=True)
class RolloutResult:
    evs: Tuple[ActionEV, ...]
    best_probability: Dict[str, float]   # paired bootstrap P(highest mean)
    simulations: int
    acceptance_rate: float
    differences: Dict[Tuple[str, str], Tuple[float, float]]

    def ev(self, label: str) -> ActionEV:
        for e in self.evs:
            if e.label == label:
                return e
        raise KeyError(label)

    def paired_difference(self, a: str, b: str) -> Tuple[float, float]:
        """Mean and SE of EV(a) - EV(b) using the shared samples."""
        return self.differences[(a, b)]


def _action_order(obs: ObservedTableState) -> List[int]:
    """Seats in this street's acting order."""
    occupied = [s.seat for s in obs.seats if s.occupied and not s.sitting_out]
    n = obs.num_seats
    if obs.street == Street.PREFLOP:
        if len(occupied) == 2:
            start = obs.dealer
        else:
            after = [s for s in clockwise_from((obs.dealer + 1) % n, n) if s in occupied]
            start = after[2 % len(after)]  # seat after the big blind
    else:
        start = (obs.dealer + 1) % n
    return [s for s in clockwise_from(start, n) if s in occupied]


def rollout_action_evs(obs: ObservedTableState, hero: Sequence,
                       ranges: Mapping[int, WeightedRange],
                       models: Mapping[int, BehaviorModel],
                       candidates: Sequence[RolloutCandidate],
                       simulations: int = 2000, seed: int = 0,
                       hero_model: Optional[BehaviorModel] = None,
                       opponent_bet_fraction: float = 0.66,
                       raise_multiplier: float = 3.0,
                       bootstrap: int = 1000,
                       max_attempts_factor: int = 200,
                       response: str = "behavior",
                       response_params: Optional[Sequence[float]] = None,
                       depth: str = "street") -> RolloutResult:
    """``response`` selects how opponents answer a bet (see :data:`RESPONSE_MODELS`):
    "behavior" (default: BehaviorModel on absolute hand strength), "mdf_range"
    (defend the top 1/(1+s) of their own range, no raises) or "mdf_calibrated"
    (defend share a + b/(1+s) with ``response_params`` = (a, b)).

    ``depth`` (see :data:`DEPTHS`): "street" (default, fast) resolves the
    current street and checks the hand down; "showdown" (heads-up only,
    ~10-50x slower) also plays one bet / call-or-fold round on every later
    street with the behaviour models on that street's hand strength (no
    raises on later streets)."""
    if response not in RESPONSE_MODELS:
        raise ValueError(f"response must be one of {RESPONSE_MODELS}")
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    fold_share = _fold_share(response, response_params)
    rng = np.random.default_rng(seed)
    hero_c = codes(hero)
    board = list(obs.board)
    seats = [s for s in _action_order(obs) if s in ranges]
    if not seats:
        raise RolloutError("no opponents to roll out against")
    hero_model = hero_model or ARCHETYPE_MODELS["regular"]
    bb = obs.big_blind
    pot0 = obs.pot
    hero_s = obs.hero
    hero_stack = hero_s.stack if hero_s.stack is not None else 0.0
    h0 = hero_s.current_bet
    level0 = obs.max_bet
    order = _action_order(obs)
    hero_pos = order.index(obs.hero_seat) if obs.hero_seat in order else 0
    behind = {s for s in order[hero_pos + 1:]}
    opp_stack = {}
    opp_bet = {}
    for s in seats:
        st = obs.seats[s]
        opp_stack[s] = st.stack if st.stack is not None else hero_stack
        opp_bet[s] = st.current_bet
    dead = set(hero_c) | set(board)
    probs = []
    for s in seats:
        w = ranges[s].weights.copy()
        for c in dead:
            w[(COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)] = 0.0
        if w.sum() <= 0:
            raise RolloutError(f"seat {s} range is empty")
        probs.append(combo_cdf(w / w.sum()))
    sv = strength_vector(board) if board else strength_vector(())
    hero_strength = float(sv[COMBO_INDEX[tuple(sorted(hero_c))]])
    rel_tab = {}
    if fold_share is not None:
        for s, cdf in zip(seats, probs):
            w = np.diff(np.concatenate([[0.0], cdf]))
            rel_tab[s] = _range_percentile(np.nan_to_num(sv, nan=0.0), w)
    need = 5 - len(board)
    deep_on = depth == "showdown" and len(seats) == 1 and need > 0
    if depth == "showdown" and len(seats) != 1:
        raise RolloutError("depth='showdown' supports heads-up spots only")
    hero_i = COMBO_INDEX[tuple(sorted(hero_c))]
    hero_last = obs.dealer == obs.hero_seat        # HU: the button acts last postflop
    future = [n for n in (3, 4, 5) if n > len(board)]
    sv_cache: Dict[tuple, np.ndarray] = {}

    def street_strength(b: tuple) -> np.ndarray:
        if b not in sv_cache:
            sv_cache[b] = strength_vector(list(b))
        return sv_cache[b]

    n_opp = len(seats)
    labels = [c.label for c in candidates]
    results = np.zeros((simulations, len(candidates)))
    attempts = 0
    k = 0
    max_attempts = simulations * max_attempts_factor
    while k < simulations:
        attempts += 1
        if attempts > max_attempts:
            raise RolloutError("could not sample collision-free opponent hands")
        idx = [draw_combo(rng, c) for c in probs]
        cards = [int(c) for i in idx for c in COMBOS[i]]
        if len(set(cards)) != len(cards):
            continue
        used = dead | set(cards)
        live = [c for c in range(52) if c not in used]
        run = [live[j] for j in rng.choice(len(live), size=need, replace=False)] \
            if need else []
        full = board + run
        hv = evaluate_best_codes(hero_c + full)
        ov = {s: evaluate_best_codes([int(COMBOS[i, 0]), int(COMBOS[i, 1])] + full)
              for s, i in zip(seats, idx)}
        strength = {s: float(sv[i]) for s, i in zip(seats, idx)}
        rel = {s: float(rel_tab[s][i]) for s, i in zip(seats, idx)} if rel_tab else None
        u_resp = {s: float(x) for s, x in zip(seats, rng.random(n_opp))}
        u_bet = {s: float(x) for s, x in zip(seats, rng.random(n_opp))}
        u_hero = float(rng.random())
        deep = None
        if deep_on:
            boards = [tuple(board + run[:n - len(board)]) for n in future]
            deep = _deep_continuation(boards, street_strength, hero_i, idx[0], seats[0],
                                      rng.random((len(boards), 4)), hero_last, models,
                                      hero_model, pot0, hero_stack, opp_stack,
                                      opponent_bet_fraction)
        for j, cand in enumerate(candidates):
            results[k, j] = _play(cand, hv, ov, strength, u_resp, u_bet, u_hero,
                                  seats, behind, models, hero_model,
                                  hero_strength, pot0, h0, level0, hero_stack,
                                  opp_stack, opp_bet, opponent_bet_fraction,
                                  raise_multiplier, rel, fold_share, deep) / bb
        k += 1

    means = results.mean(axis=0)
    ses = results.std(axis=0, ddof=1) / math.sqrt(simulations) \
        if simulations > 1 else np.full(len(candidates), np.nan)
    evs = tuple(ActionEV(l, float(m), float(s), simulations)
                for l, m, s in zip(labels, means, ses))
    boot_rng = np.random.default_rng(seed + 104729)
    wins = np.zeros(len(candidates))
    for _ in range(bootstrap):
        sel = boot_rng.integers(0, simulations, size=simulations)
        wins[int(np.argmax(results[sel].mean(axis=0)))] += 1
    best = {l: float(w / bootstrap) for l, w in zip(labels, wins) if w > 0}
    diffs = {}
    for a in range(len(labels)):
        for b in range(len(labels)):
            if a != b:
                d = results[:, a] - results[:, b]
                diffs[(labels[a], labels[b])] = (
                    float(d.mean()), float(d.std(ddof=1) / math.sqrt(simulations)))
    return RolloutResult(evs, best, simulations, simulations / attempts, diffs)


RESPONSE_MODELS = ("behavior", "mdf_range", "mdf_calibrated")
DEPTHS = ("street", "showdown")


def _deep_continuation(boards, street_strength, hero_i, opp_i, s, u, hero_last, models,
                       hero_model, pot0, hero_stack, opp_stack, bet_frac):
    """Later streets for depth="showdown" (heads-up): on each board one
    player may bet ``bet_frac`` x pot (capped by the stacks); the other calls
    or folds (a model "raise" counts as a call). Mutates ``add`` / ``in_hand``."""
    def run(add, in_hand):
        for k, b in enumerate(boards):
            if not (in_hand["hero"] and in_hand[s]):
                return
            rem_h, rem_o = hero_stack - add["hero"], opp_stack[s] - add[s]
            if min(rem_h, rem_o) <= 1e-9:
                return                                    # all-in: run it out
            sv = street_strength(b)
            st = {"hero": float(sv[hero_i]), s: float(sv[opp_i])}
            mdl = {"hero": hero_model, s: models[s]}
            first, second = (s, "hero") if hero_last else ("hero", s)
            bet = min(bet_frac * (pot0 + sum(add.values())), rem_h, rem_o)
            for bettor, caller, ub, uc in ((first, second, u[k, 0], u[k, 1]),
                                           (second, first, u[k, 2], u[k, 3])):
                if _respond(mdl[bettor], st[bettor], False, bet_frac, ub) != "bet":
                    continue
                if _respond(mdl[caller], st[caller], True, bet_frac, uc) == "fold":
                    in_hand[caller] = False               # uncalled bet never leaves
                    return
                add[bettor] += bet
                add[caller] += bet
                break
    return run


def _fold_share(response: str, params):
    """Share of its own range a responder folds to a bet of ``size`` x pot."""
    if response == "behavior":
        return None
    if response == "mdf_range":
        return lambda size: size / (1.0 + size)
    if params is None or len(params) != 2:
        raise ValueError("mdf_calibrated needs response_params = (a, b)")
    a, b = float(params[0]), float(params[1])
    return lambda size: 1.0 - min(1.0, max(0.0, a + b / (1.0 + size)))


def _range_percentile(strength: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Percentile of every combo's strength within the weighted range."""
    order = np.argsort(strength, kind="stable")
    cw = np.cumsum(weights[order])
    tot = cw[-1] if cw[-1] > 0 else 1.0
    out = np.empty(len(strength))
    out[order] = (cw - weights[order] / 2.0) / tot
    return out


def _respond(model: BehaviorModel, s: float, facing: bool, size: float,
             u: float) -> str:
    p = model.probabilities(np.array([s]), facing, size)
    if not facing:
        return "bet" if u < float(p["bet"][0]) else "check"
    pr, pf = float(p["raise"][0]), float(p["fold"][0])
    if u < pr:
        return "raise"
    if u < pr + pf:
        return "fold"
    return "call"


def _settle(hv, ov, in_hand: Dict[str, bool], add: Dict[str, float],
            street_bet: Dict[str, float], pot0: float) -> float:
    """Hero's winnings (chips).

    Each player's *street total* (bet already on the table + chips added now)
    is layered into main/side pots with the engine's rules; money from
    earlier streets (``pot0`` minus current street bets) is one layer every
    remaining player is eligible for.
    """
    players = ["hero"] + [s for s in ov]
    live = [p for p in players if in_hand[p]]
    total_add = sum(add.values())
    if live == ["hero"]:
        return pot0 + total_add
    if "hero" not in live:
        return 0.0
    # Integer units for the shared pot code: 1/100 chip, times 2520 = lcm(1..9)
    # so ties among any number of winners split exactly (no odd-chip bias).
    scale = 100 * 2520
    contrib = [int(round((street_bet[p] + add[p]) * scale)) for p in players]
    folded = [not in_hand[p] for p in players]
    dead = pot0 - sum(street_bet.values())
    live_idx = tuple(i for i, p in enumerate(players) if in_hand[p])
    pots = [Pot(int(round(dead * scale)), live_idx)] if dead > 1e-9 else []
    if any(contrib):
        pots += build_pots(contrib, folded)
    values = [hv] + [ov[s] for s in ov]
    won = award_pots(pots, lambda i: values[i], list(range(len(players))))
    return won.get(0, 0) / scale


def _play(cand, hv, ov, strength, u_resp, u_bet, u_hero, seats, behind,
          models, hero_model, hero_strength, pot0, h0, level0, hero_stack,
          opp_stack, opp_bet, bet_frac, raise_mult, rel=None, fold_share=None,
          deep=None) -> float:
    if cand.kind == "fold":
        return 0.0

    def opp(s, facing, size, u):
        if facing and fold_share is not None:
            return "fold" if rel[s] < fold_share(size) else "call"
        return _respond(models[s], strength[s], facing, size, u)
    in_hand = {"hero": True, **{s: True for s in seats}}
    add = {"hero": 0.0, **{s: 0.0 for s in seats}}
    to_call0 = max(0.0, level0 - h0)

    def hero_put(to: float) -> None:
        add["hero"] = min(max(0.0, to - h0), hero_stack)

    def opp_put(s, to: float) -> None:
        add[s] = min(max(0.0, to - opp_bet[s]), opp_stack[s])

    can_act = {s: opp_stack[s] > 1e-9 for s in seats}
    street_bet = {"hero": h0, **{s: opp_bet[s] for s in seats}}

    if cand.kind == "check":
        bettor = None
        for s in seats:
            if s in behind and can_act[s] and bettor is None:
                if opp(s, False, bet_frac, u_bet[s]) == "bet":
                    bettor = s
        if bettor is not None:
            size = bet_frac * pot0
            opp_put(bettor, size)
            for s in seats:
                if s == bettor or not can_act[s]:
                    continue
                r = opp(s, True, bet_frac, u_resp[s])
                if r == "fold":
                    in_hand[s] = False
                else:
                    opp_put(s, size)
            r = _respond(hero_model, hero_strength, True, bet_frac, u_hero)
            if r == "fold":
                return 0.0
            hero_put(size)
        if deep is not None:
            deep(add, in_hand)
        return _settle(hv, ov, in_hand, add, street_bet, pot0) - add["hero"]

    if cand.kind in ("call",) or (cand.kind == "all_in" and cand.raise_to <= level0 + 1e-9):
        hero_put(min(level0, h0 + hero_stack))
        size = to_call0 / max(pot0, 1e-9)
        for s in seats:
            if opp_bet[s] + 1e-9 >= level0 or not can_act[s]:
                add[s] = 0.0
                continue
            r = opp(s, True, size, u_resp[s])
            if r == "fold":
                in_hand[s] = False
            else:
                opp_put(s, level0)
        if deep is not None:
            deep(add, in_hand)
        return _settle(hv, ov, in_hand, add, street_bet, pot0) - add["hero"]

    # bet / raise / all-in above the current level
    target = min(cand.raise_to, h0 + hero_stack)
    hero_put(target)
    size = (target - level0) / max(pot0 + to_call0, 1e-9)
    level = target
    raised_by = None
    called_at: Dict[int, float] = {}
    for s in seats:
        if not can_act[s]:
            continue
        r = opp(s, True, size, u_resp[s])
        if r == "fold":
            in_hand[s] = False
        elif r == "raise" and raised_by is None and \
                opp_stack[s] + opp_bet[s] > level + 1e-9:
            new = min(raise_mult * target, opp_stack[s] + opp_bet[s])
            opp_put(s, new)
            level = opp_bet[s] + add[s]
            raised_by = s
        else:
            opp_put(s, level)
            called_at[s] = opp_bet[s] + add[s]
    if raised_by is not None:
        # Earlier callers face the raise once more (same random draw).
        rsize = (level - target) / max(pot0 + sum(add.values()), 1e-9)
        for s, at in called_at.items():
            if at + 1e-9 < level and can_act[s] and opp_stack[s] > add[s] + 1e-9:
                r = opp(s, True, rsize, u_resp[s])
                if r == "fold":
                    in_hand[s] = False
                else:
                    opp_put(s, level)
        r = _respond(hero_model, hero_strength, True, rsize, u_hero)
        if r == "fold" and add["hero"] < hero_stack - 1e-9:
            in_hand["hero"] = False
            return -add["hero"]
        hero_put(level)
    if deep is not None:
        deep(add, in_hand)
    return _settle(hv, ov, in_hand, add, street_bet, pot0) - add["hero"]
