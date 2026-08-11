"""Bayesian opponent modeling with Beta-Bernoulli posteriors.

Each observable behavioral tendency — fold-vs-bet frequency, raise frequency,
bluff-at-showdown frequency — is modeled as a Bernoulli parameter with a Beta
prior:

    p ~ Beta(alpha, beta)

After observing ``s`` occurrences in ``n`` opportunities the posterior is
``Beta(alpha + s, beta + n - s)``: conjugacy makes the update exact and O(1).

The model's job is not just a point estimate but *calibrated uncertainty*:
3 folds out of 4 and 300 out of 400 share a posterior mean near 0.75, but
their credible intervals differ enormously — and the adaptive agent keys its
exploitation off that difference, not the mean alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from scipy import stats as sps


@dataclass
class BetaPosterior:
    """Posterior over one Bernoulli behavioral parameter."""

    alpha: float = 1.0  # uniform Beta(1, 1) prior by default
    beta: float = 1.0

    def update(self, occurred: bool) -> None:
        if occurred:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def update_counts(self, successes: int, failures: int) -> None:
        if successes < 0 or failures < 0:
            raise ValueError("counts must be non-negative")
        self.alpha += successes
        self.beta += failures

    @property
    def observations(self) -> float:
        return self.alpha + self.beta - 2.0  # net of the uniform prior

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Central credible interval at the given level."""
        lo = (1.0 - level) / 2.0
        dist = sps.beta(self.alpha, self.beta)
        return float(dist.ppf(lo)), float(dist.ppf(1.0 - lo))


# The behavioral dimensions the model tracks. Each maps an observation
# opportunity to a yes/no outcome, so all are Beta-Bernoulli.
TENDENCIES = ("fold_vs_bet", "raise_freq", "call_vs_bet", "bluff_at_showdown")


@dataclass
class OpponentModel:
    """Posteriors over an opponent's behavioral tendencies."""

    posteriors: Dict[str, BetaPosterior] = field(
        default_factory=lambda: {t: BetaPosterior() for t in TENDENCIES})

    def observe(self, tendency: str, occurred: bool) -> None:
        self.posteriors[tendency].update(occurred)

    def posterior_mean(self, tendency: str) -> float:
        return self.posteriors[tendency].mean()

    def posterior_variance(self, tendency: str) -> float:
        return self.posteriors[tendency].variance()

    def confidence_interval(self, tendency: str,
                            level: float = 0.95) -> Tuple[float, float]:
        return self.posteriors[tendency].confidence_interval(level)

    def observations(self, tendency: str) -> float:
        return self.posteriors[tendency].observations
