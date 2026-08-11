"""Synthetic opponent archetypes: configurable deviations from equilibrium.

An archetype is built by *tilting* a base (equilibrium) strategy with
multiplicative biases on recognizable action classes, then renormalizing at
every information set:

* ``fold_multiplier``  — weight on fold actions when facing a bet,
* ``raise_multiplier`` — weight on bet/raise actions,
* ``call_multiplier``  — weight on check/call actions,
* ``bluff_multiplier`` — *additional* multiplier on bet/raise weight when the
  hand is weak (so bluffing can be tuned independently of value aggression).

Multipliers of 1.0 leave the equilibrium untouched; 0 removes the action class
entirely (mass renormalizes to the rest). Because the tilt operates on a real
equilibrium strategy, archetypes stay structurally sensible — a Nit still bets
the nuts — while deviating in the measured, parameterized way the adaptive
agent is supposed to detect and exploit.

Weak-hand detection is game-specific and supplied as a callable so archetypes
work on any game; a Leduc implementation is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

Strategy = Dict[str, Dict[str, float]]

# Action classification for games with an unambiguous fold/call/raise
# alphabet (Leduc: 'f'/'c'/'r'; abstracted Hold'em: 'f'/'c'/'b*'/'a').
# Kuhn is excluded: its 'p'/'b' tokens change meaning with context (a pass is
# a fold only when facing a bet), so a static classification would mislabel.
_FOLDISH = {"f"}
_RAISEISH = {"r", "b50", "b100", "b200", "a"}
_CALLISH = {"c"}


@dataclass(frozen=True)
class OpponentConfig:
    """Behavioral tilt applied to an equilibrium strategy."""

    name: str = "custom"
    fold_multiplier: float = 1.0
    raise_multiplier: float = 1.0
    call_multiplier: float = 1.0
    bluff_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for field in ("fold_multiplier", "raise_multiplier",
                      "call_multiplier", "bluff_multiplier"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")


# The six standard archetypes from the project brief.
ARCHETYPES: Dict[str, OpponentConfig] = {
    "balanced": OpponentConfig(name="balanced"),
    "nit": OpponentConfig(name="nit", fold_multiplier=1.8,
                          raise_multiplier=0.5, bluff_multiplier=0.3),
    "calling_station": OpponentConfig(name="calling_station",
                                      fold_multiplier=0.25,
                                      call_multiplier=1.6,
                                      raise_multiplier=0.7),
    "maniac": OpponentConfig(name="maniac", raise_multiplier=2.5,
                             bluff_multiplier=2.5, fold_multiplier=0.5),
    "bluff_heavy": OpponentConfig(name="bluff_heavy", bluff_multiplier=3.0),
    "passive": OpponentConfig(name="passive", raise_multiplier=0.3,
                              call_multiplier=1.5),
}


def leduc_is_weak(infoset_key: str) -> bool:
    """Weak-hand heuristic for Leduc infoset keys (``rank|public|history``).

    Weak = holding the jack unpaired, or the queen unpaired after the flop.
    """
    rank_s, public, _ = infoset_key.split("|")
    rank = int(rank_s)
    if public != "?" and rank == int(public):
        return False  # paired: never weak
    if rank == 0:
        return True
    return rank == 1 and public != "?"


def tilt_strategy(
    base: Strategy,
    config: OpponentConfig,
    is_weak: Optional[Callable[[str], bool]] = None,
) -> Strategy:
    """Return ``base`` tilted by ``config``, renormalized per infoset.

    ``is_weak`` classifies infoset keys as weak-hand spots for the bluff
    multiplier; if omitted, the bluff multiplier is not applied.
    """
    tilted: Strategy = {}
    for key, probs in base.items():
        weak = bool(is_weak(key)) if is_weak is not None else False
        weights: Dict[str, float] = {}
        for action, p in probs.items():
            if action in _RAISEISH:
                w = config.raise_multiplier
                if weak:
                    w *= config.bluff_multiplier
            elif action in _FOLDISH:
                w = config.fold_multiplier
            elif action in _CALLISH:
                w = config.call_multiplier
            else:
                w = 1.0
            weights[action] = p * w
        total = sum(weights.values())
        if total <= 0.0:
            # Everything zeroed out: fall back to the base distribution.
            tilted[key] = dict(probs)
        else:
            tilted[key] = {a: w / total for a, w in weights.items()}
    return tilted
