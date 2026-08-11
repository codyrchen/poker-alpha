"""Simulated Leduc matches between two strategies, with hero-visible logging.

The simulator is omniscient (it deals the cards), but the records it hands to
the learning agent contain only what that agent could legitimately see: its
own card, the public card, the opponent's public actions, and the opponent's
private card *only when a showdown reveals it*. This firewall is what makes
the downstream Bayesian updates honest.

Seats alternate every hand (the button passes back and forth, as in real
heads-up play), so positional asymmetry cancels out of long-run win rates.
Utilities are reported from the hero's perspective in chips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..games.leduc import LeducPoker, LeducState, rank
from .beliefs import HandObservation, ObservedDecision

Strategy = Dict[str, Dict[str, float]]
# A strategy provider lets the hero adapt between hands: it is called with the
# 0-based hand index before each hand and returns the strategy to play.
StrategyProvider = Callable[[int], Strategy]


@dataclass(frozen=True)
class HandRecord:
    """One simulated hand: hero-visible observation plus outcome."""

    hand_index: int
    hero_seat: int
    hero_utility: float  # chips, hero perspective
    observation: HandObservation
    went_to_showdown: bool


def _static(strategy: Strategy) -> StrategyProvider:
    return lambda _hand: strategy


def _sample_action(probs: Dict[str, float], legal: Sequence[str],
                   rng: np.random.Generator) -> str:
    weights = np.array([max(probs.get(a, 0.0), 0.0) for a in legal])
    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(legal))
        total = float(len(legal))
    return legal[int(rng.choice(len(legal), p=weights / total))]


def play_hand(
    game: LeducPoker,
    hero_seat: int,
    hero_strategy: Strategy,
    opp_strategy: Strategy,
    rng: np.random.Generator,
    hand_index: int = 0,
) -> HandRecord:
    """Play one Leduc hand; return the hero-visible record."""
    # Sample the deal.
    outcomes = game.chance_outcomes(game.root())
    idx = int(rng.choice(len(outcomes), p=[p for p, _ in outcomes]))
    state: LeducState = outcomes[idx][1]

    opp_seat = 1 - hero_seat
    opp_decisions: List[ObservedDecision] = []

    while not game.is_terminal(state):
        if game.is_chance(state):
            outs = game.chance_outcomes(state)
            j = int(rng.choice(len(outs), p=[p for p, _ in outs]))
            state = outs[j][1]
            continue
        player = game.current_player(state)
        key = game.infoset_key(state)
        legal = game.legal_actions(state)
        strat = hero_strategy if player == hero_seat else opp_strategy
        probs = strat.get(key)
        if probs is None:
            probs = {a: 1.0 / len(legal) for a in legal}
        action = _sample_action(probs, legal, rng)
        if player == opp_seat:
            public = "?" if state.public is None else str(state.public // 2)
            opp_decisions.append((public, state.history, action))
        state = game.next_state(state, action)

    u0 = game.utility(state)
    hero_utility = u0 if hero_seat == 0 else -u0
    # A fold is always the last action of a terminal Leduc history.
    showdown = not state.history.endswith("f")
    observation = HandObservation(
        hero_rank=rank(state.cards[hero_seat]),
        public_rank=None if state.public is None else state.public // 2,
        opp_decisions=tuple(opp_decisions),
        revealed_opp_rank=rank(state.cards[opp_seat]) if showdown else None,
    )
    return HandRecord(
        hand_index=hand_index,
        hero_seat=hero_seat,
        hero_utility=hero_utility,
        observation=observation,
        went_to_showdown=showdown,
    )


def simulate_match(
    game: LeducPoker,
    hero: "Strategy | StrategyProvider",
    opponent: "Strategy | StrategyProvider",
    hands: int,
    rng: np.random.Generator,
    on_hand: Optional[Callable[[HandRecord], None]] = None,
) -> List[HandRecord]:
    """Simulate ``hands`` hands, alternating seats; return all records.

    ``hero``/``opponent`` may be fixed strategies or per-hand providers (the
    hook the adaptive agent uses to update between hands). ``on_hand`` is
    called after each hand with its record — e.g. to feed a belief update.
    """
    hero_provider = hero if callable(hero) else _static(hero)
    opp_provider = opponent if callable(opponent) else _static(opponent)
    records: List[HandRecord] = []
    for i in range(hands):
        record = play_hand(
            game,
            hero_seat=i % 2,
            hero_strategy=hero_provider(i),
            opp_strategy=opp_provider(i),
            rng=rng,
            hand_index=i,
        )
        records.append(record)
        if on_hand is not None:
            on_hand(record)
    return records


def match_summary(records: Sequence[HandRecord]) -> Dict[str, float]:
    """Aggregate a match into headline numbers (chips; bb assumed = 1 ante)."""
    utils = np.array([r.hero_utility for r in records], dtype=float)
    n = len(utils)
    mean = float(utils.mean()) if n else 0.0
    std = float(utils.std(ddof=1)) if n > 1 else 0.0
    return {
        "hands": float(n),
        "total_chips": float(utils.sum()),
        "mean_chips_per_hand": mean,
        "chips_per_100": 100.0 * mean,
        "std_per_hand": std,
        "std_error_per_100": 100.0 * std / np.sqrt(n) if n > 1 else 0.0,
        "showdown_rate": float(np.mean([r.went_to_showdown for r in records]))
        if n else 0.0,
    }
