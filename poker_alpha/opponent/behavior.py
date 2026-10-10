"""Configurable behavioural action models: ``P(action | hand, situation)``.

These supply the likelihood term of the Bayesian range update when no solved
strategy covers a spot. A model maps each combo's *strength* (a number in
[0, 1], e.g. :func:`~poker_alpha.poker.ranges.preflop_strength` or
:func:`~poker_alpha.poker.ranges.strength_percentiles`) to action
probabilities with smooth logistic thresholds:

* not facing a bet: ``P(bet) = bluff + (1 - bluff) * σ((s - t_bet) / temp)``;
* facing a bet: ``P(raise)`` analogous with ``t_raise``; ``P(fold) =
  (1 - P(raise)) * σ((t_fold - s) / temp)``; ``P(call)`` takes the rest.

Bet size enters through *pot odds*, which keeps the model bounded however
large the wager: facing a bet of ``f`` × pot the caller needs equity
``r(f) = f / (1 + 2f)`` (0.25 for half pot, -> 0.5 for huge overbets), and
the fold threshold moves by ``pot_odds_sensitivity × (r(f) − 0.25)``, the
raise (or, facing an all-in, call-off) threshold by
``raise_sensitivity × (r(f) − 0.25)``, and raise-bluffs shrink to zero as
``r`` approaches 0.5 — so nobody "calls off" a 100 BB shove with the top
half of hands, but premium hands still continue. When
betting, a bounded ``size_sensitivity × clip(f − 0.5, −0.5, 1.5)`` shift
makes bigger bets come from a stronger range. Both are modelling assumptions
exposed as parameters.

Archetype parameters are illustrative defaults, not fitted to data. They are
data, so users can supply their own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

import numpy as np

AGGRESSIVE = ("bet", "raise", "all_in")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class BehaviorModel:
    name: str = "regular"
    t_bet: float = 0.65
    t_raise: float = 0.85
    t_fold: float = 0.45
    bluff_bet: float = 0.10
    bluff_raise: float = 0.04
    temperature: float = 0.06
    size_sensitivity: float = 0.15
    pot_odds_sensitivity: float = 2.0
    raise_sensitivity: float = 0.4

    def probabilities(self, strength: np.ndarray, facing_bet: bool,
                      size_pot_fraction: float = 0.5) -> Dict[str, np.ndarray]:
        """Per-combo action probabilities. NaN strengths (blocked combos)
        produce NaN rows, which callers treat as impossible combos."""
        s = np.asarray(strength, dtype=np.float64)
        f = max(float(size_pot_fraction), 0.0)
        temp = self.temperature
        if not facing_bet:
            shift = self.size_sensitivity * min(max(f - 0.5, -0.5), 1.5)
            bet = self.bluff_bet + (1 - self.bluff_bet) * _sigmoid(
                (s - (self.t_bet + shift)) / temp)
            return {"bet": bet, "check": 1.0 - bet}
        d = f / (1.0 + 2.0 * f) - 0.25          # required equity - 0.25
        t_fold = self.t_fold + self.pot_odds_sensitivity * d
        t_raise = min(self.t_raise + self.raise_sensitivity * d, 0.995)
        bluff = self.bluff_raise * max(0.0, 1.0 - 4.0 * d)
        raise_ = bluff + (1 - bluff) * _sigmoid((s - t_raise) / temp)
        fold = (1.0 - raise_) * _sigmoid((t_fold - s) / temp)
        return {"raise": raise_, "fold": fold, "call": 1.0 - raise_ - fold}

    def likelihood(self, action: str, strength: np.ndarray, facing_bet: bool,
                   size_pot_fraction: float = 0.5) -> np.ndarray:
        """``P(action | combo)`` vector with blocked (NaN) combos at 0."""
        probs = self.probabilities(strength, facing_bet, size_pot_fraction)
        key = action
        if action in AGGRESSIVE:
            key = "raise" if facing_bet else "bet"
        elif action == "check" and facing_bet:
            raise ValueError("cannot check facing a bet")
        elif action == "call" and not facing_bet:
            key = "check"
        if key not in probs:
            raise ValueError(f"action {action!r} impossible here")
        return np.nan_to_num(probs[key], nan=0.0)


ARCHETYPE_MODELS: Dict[str, BehaviorModel] = {
    "regular": BehaviorModel(),
    "nit": BehaviorModel("nit", t_bet=0.75, t_raise=0.92, t_fold=0.6,
                         bluff_bet=0.04, bluff_raise=0.01),
    "calling_station": BehaviorModel("calling_station", t_bet=0.8,
                                     t_raise=0.95, t_fold=0.15,
                                     bluff_bet=0.03, bluff_raise=0.01),
    "maniac": BehaviorModel("maniac", t_bet=0.4, t_raise=0.6, t_fold=0.3,
                            bluff_bet=0.35, bluff_raise=0.2,
                            temperature=0.1),
}


class StrategyLikelihood:
    """Likelihood from an explicit per-combo policy, e.g. a solver lookup.

    ``policy(combo_index) -> {action: probability}``; combos it cannot
    answer for return ``None`` and fall back to ``fallback`` (if given).
    """

    def __init__(self, policy: Callable[[int], Optional[Dict[str, float]]],
                 fallback: Optional[BehaviorModel] = None) -> None:
        self.policy = policy
        self.fallback = fallback

    def likelihood(self, action: str, strength: np.ndarray, facing_bet: bool,
                   size_pot_fraction: float = 0.5) -> np.ndarray:
        fb = None
        if self.fallback is not None:
            fb = self.fallback.likelihood(action, strength, facing_bet,
                                          size_pot_fraction)
        out = np.zeros(len(strength))
        for i in range(len(strength)):
            probs = self.policy(i)
            if probs is None:
                out[i] = 0.0 if fb is None else fb[i]
            else:
                out[i] = probs.get(action, 0.0)
        return out
