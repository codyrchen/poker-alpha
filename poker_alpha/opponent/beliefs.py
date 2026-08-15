"""Bayesian identification of opponent type from observed public actions.

The agent never sees the opponent's private card (except at a showdown), so a
hand's evidence is the opponent's *public* action sequence. For a candidate
strategy ``sigma`` the likelihood of one hand's observations marginalizes over
the private cards the opponent could hold:

    P(actions | sigma) = sum_r  w(r) * prod_t sigma(a_t | I_t(r))

where ``w(r)`` is the card-removal-adjusted prior over the opponent's hidden
rank and ``I_t(r)`` the information set the opponent would occupy at decision
``t`` if they held rank ``r``. When the hand reaches showdown the revealed
rank replaces the marginalization for that hand.

:class:`ArchetypeBelief` maintains a discrete posterior over named candidate
strategies via accumulated log-likelihoods — exactly Bayes' rule, numerically
stable in log space. This is the "which opponent am I facing?" inference of
the identification experiment, and its posterior mixture is what the adaptive
agent best-responds to.

The likelihood bookkeeping here is Leduc-specific (rank arithmetic and infoset
key layout); the posterior machinery is game-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Strategy = Dict[str, Dict[str, float]]

# One opponent decision as seen publicly: the infoset key the opponent would
# hold *given a hypothetical private rank r* is f"{r}|{public}|{history}".
# We store the public parts and the action taken.
ObservedDecision = Tuple[str, str, str]  # (public_rank_or_"?", history, action)


@dataclass(frozen=True)
class HandObservation:
    """Hero-visible record of one hand, for belief updates."""

    hero_rank: int
    public_rank: Optional[int]  # None if the hand ended preflop
    opp_decisions: Sequence[ObservedDecision]
    revealed_opp_rank: Optional[int] = None  # set when showdown reached


def _rank_weights(hero_rank: int, public_rank: Optional[int]) -> Dict[int, float]:
    """Prior over the opponent's hidden rank, adjusted for card removal.

    Leduc has two cards of each of three ranks. The hero's card removes one of
    its rank; the public card (once dealt) removes one of its rank.
    """
    counts = {0: 2, 1: 2, 2: 2}
    counts[hero_rank] -= 1
    if public_rank is not None:
        counts[public_rank] -= 1
    total = sum(counts.values())
    return {r: c / total for r, c in counts.items() if c > 0}


def leduc_hand_likelihood(strategy: Strategy, obs: HandObservation) -> float:
    """P(opponent's observed actions | they play ``strategy``) for one hand."""
    if obs.revealed_opp_rank is not None:
        ranks = {obs.revealed_opp_rank: 1.0}
    else:
        ranks = _rank_weights(obs.hero_rank, obs.public_rank)

    total = 0.0
    for rank, weight in ranks.items():
        prob = weight
        for public, history, action in obs.opp_decisions:
            key = f"{rank}|{public}|{history}"
            probs = strategy.get(key)
            if probs is None:
                # Unvisited infoset in the candidate: uniform default, matching
                # the evaluator's convention.
                prob *= 1.0 / 3.0
            else:
                prob *= probs.get(action, 0.0)
        total += prob
    return total


@dataclass
class ArchetypeBelief:
    """Discrete Bayesian posterior over named candidate opponent strategies.

    ``decay`` is a forgetting factor applied to the accumulated log-likelihood
    before folding in each new hand's evidence. At the default ``decay=1.0``
    nothing is discounted: this is the *stationary* baseline used throughout
    the project so far, and it is exact Bayesian updating -- every hand ever
    seen counts equally forever, which is correct only if the opponent's type
    never changes.

    ``decay<1.0`` gives a *recency-aware* posterior: expanding the recursion
    ``LL_t = decay * LL_{t-1} + ll_t`` shows hand ``j``, observed ``t-j`` hands
    ago, contributes with weight ``decay**(t-j)`` -- old evidence is
    exponentially down-weighted, with an effective memory horizon of roughly
    ``1 / (1 - decay)`` hands. This trades identification precision against a
    stationary opponent (more noise in the posterior at any given hand count)
    for the ability to notice and recover from a regime change (see
    ``experiments/regime_change.py`` for the measured tradeoff, including its
    cost: a smaller decay makes the posterior more prone to spurious
    "false-switch" flips from ordinary sampling noise even when the opponent
    never actually changes).

    A rolling evidence window (keep the raw last ``W`` hands, recompute the
    likelihood from scratch) was the other natural design; exponential
    forgetting was chosen because it drops in as an O(1) per-hand update to
    the existing scalar log-likelihood accumulator, with no history buffer.
    """

    candidates: Dict[str, Strategy]
    decay: float = 1.0
    log_likelihood: Dict[str, float] = field(default_factory=dict)
    hands_observed: int = 0
    _floor: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        for name in self.candidates:
            self.log_likelihood.setdefault(name, 0.0)

    def update(self, obs: HandObservation) -> None:
        """Discount past evidence by ``decay``, then fold in this hand's."""
        for name, strategy in self.candidates.items():
            like = leduc_hand_likelihood(strategy, obs)
            self.log_likelihood[name] = (
                self.decay * self.log_likelihood[name]
                + float(np.log(max(like, self._floor))))
        self.hands_observed += 1

    def posterior(self) -> Dict[str, float]:
        """Normalized posterior over candidates (uniform prior)."""
        names = list(self.candidates)
        logs = np.array([self.log_likelihood[n] for n in names])
        logs -= logs.max()  # stabilize
        probs = np.exp(logs)
        probs /= probs.sum()
        return {n: float(p) for n, p in zip(names, probs)}

    def map_estimate(self) -> str:
        post = self.posterior()
        return max(post, key=post.get)

    def entropy(self) -> float:
        """Posterior entropy in nats; 0 = certain, log(K) = uniform."""
        p = np.array(list(self.posterior().values()))
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def confidence(self) -> float:
        """1 − normalized entropy: 0 when uniform, → 1 when certain."""
        k = len(self.candidates)
        if k <= 1:
            return 1.0
        return 1.0 - self.entropy() / float(np.log(k))

    def mixture_strategy(self) -> Strategy:
        """Posterior-weighted mixture of candidate strategies.

        This is the Bayes-optimal single-strategy summary of "what my opponent
        plays": the estimate the exploitative agent best-responds to.
        """
        post = self.posterior()
        keys = set()
        for strategy in self.candidates.values():
            keys.update(strategy)
        mixture: Strategy = {}
        for key in keys:
            acc: Dict[str, float] = {}
            for name, strategy in self.candidates.items():
                probs = strategy.get(key)
                if probs is None:
                    continue
                for action, p in probs.items():
                    acc[action] = acc.get(action, 0.0) + post[name] * p
            total = sum(acc.values())
            if total > 0:
                mixture[key] = {a: v / total for a, v in acc.items()}
        return mixture
