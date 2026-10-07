"""Configurable behavioural action models: ``P(action | hand, situation)``.

These supply the likelihood term of the Bayesian range update when no solved
strategy covers a spot. A model maps each combo's *strength* (a number in
[0, 1], e.g. :func:`~poker_alpha.poker.ranges.preflop_strength` or
:func:`~poker_alpha.poker.ranges.strength_percentiles`) to action
probabilities with smooth logistic thresholds:

* not facing a bet: ``P(bet) = bluff + (1 - bluff) * σ((s - t_bet) / temp)``;
* facing a bet: ``P(raise)`` analogous with ``t_raise``; ``P(fold) =
  (1 - P(raise)) * σ((t_fold - s) / temp)``; ``P(call)`` takes the rest.

Bigger bets shift thresholds up by ``size_sensitivity × (size − 0.5 pot)``
(stronger hands bet bigger) — a modelling assumption, exposed as a parameter.

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

    def probabilities(self, strength: np.ndarray, facing_bet: bool,
                      size_pot_fraction: float = 0.5) -> Dict[str, np.ndarray]:
        """Per-combo action probabilities. NaN strengths (blocked combos)
        produce NaN rows, which callers treat as impossible combos."""
        s = np.asarray(strength, dtype=np.float64)
        shift = self.size_sensitivity * (size_pot_fraction - 0.5)
        temp = self.temperature
        if not facing_bet:
            bet = self.bluff_bet + (1 - self.bluff_bet) * _sigmoid(
                (s - (self.t_bet + shift)) / temp)
            return {"bet": bet, "check": 1.0 - bet}
        raise_ = self.bluff_raise + (1 - self.bluff_raise) * _sigmoid(
            (s - self.t_raise) / temp)
        fold = (1.0 - raise_) * _sigmoid(((self.t_fold + shift) - s) / temp)
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
