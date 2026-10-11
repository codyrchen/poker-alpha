"""Adaptive agents for the generic agent framework (Phase II, Program 15).

:class:`BayesArchetypeAgent` keeps a posterior over a finite set of candidate
opponent strategies (behavior strategies over the opponent's information-set
keys) and plays an exact best response to the posterior.

Information used: its own information-set keys during play and, after each
hand, the PUBLIC action sequence plus its own payoff (``observe_episode``).
It never sees the opponent's private cards. The likelihood of a hand under
candidate ``k`` is

    P(opponent's observed actions | k) = sum_c P(c | own private info) *
                                          prod_t pi_k(I_t(c))[a_t]

marginalizing the opponent's hidden private state ``c`` given what the
agent itself held. ``private_states(own_key)`` and ``opp_key(c, history)``
come from a small game adapter (Kuhn and Leduc provided).

Best response to a posterior over *types* is computed exactly by converting
the type mixture to its realization-equivalent behavior strategy (Kuhn's
theorem): ``pi(I)[a] = sum_k w_k rho_k(I) pi_k(I)[a] / sum_k w_k rho_k(I)``,
where ``rho_k(I)`` is type k's own reach of ``I``.

Recency: with ``decay < 1`` the log-likelihoods are discounted every hand
(``L_k <- decay * L_k + log p_k(hand)``) so the posterior can track an
opponent who changes. ``decay = 1`` is the stationary Bayesian.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..games.base import Game
from ..solvers.evaluation import best_response

Strategy = Mapping[str, Mapping[str, float]]


# --------------------------------------------------------------------------
# game adapters: hidden private states and key construction
# --------------------------------------------------------------------------

class KuhnAdapter:
    """Kuhn keys are ``<card><history>`` (history over 'p'/'b')."""

    n_cards = 3

    def own_private(self, key: str) -> int:
        return int(key[0])

    def opponent_private_dist(self, own: int) -> Dict[int, float]:
        others = [c for c in range(self.n_cards) if c != own]
        return {c: 1.0 / len(others) for c in others}

    def history_string(self, public: Sequence[Tuple[int, str]]) -> List[Tuple[int, str, str]]:
        """[(seat, history_before_action, action)]"""
        out, h = [], ""
        for seat, a in public:
            out.append((seat, h, a))
            h += a
        return out

    def opp_key(self, c: int, history_before: str) -> str:
        return f"{c}{history_before}"


class BayesArchetypeAgent:
    """Posterior over candidate opponent strategies + exact Bayes best response."""

    def __init__(self, game: Game, candidates: Dict[str, Strategy], adapter=None, decay: float = 1.0,
                 refresh: int = 20, prior: Optional[Dict[str, float]] = None, name: str = "bayes",
                 fallback: Optional[Strategy] = None) -> None:
        self.game = game
        self.candidates = dict(candidates)
        self.names = list(self.candidates)
        self.adapter = adapter or KuhnAdapter()
        self.decay = float(decay)
        self.refresh = int(refresh)
        self.prior = prior or {n: 1.0 / len(self.names) for n in self.names}
        self.name = name
        self.fallback = fallback
        self.reset()

    # -- protocol -----------------------------------------------------------
    def reset(self) -> None:
        self.loglik = {n: math.log(self.prior[n]) for n in self.names}
        self.hands = 0
        self._own_private: Optional[int] = None
        self._policy: Dict[int, Dict[str, str]] = {}
        self._dirty = True
        self.posterior_trace: List[Dict[str, float]] = []

    def posterior(self) -> Dict[str, float]:
        m = max(self.loglik.values())
        w = {n: math.exp(v - m) for n, v in self.loglik.items()}
        z = sum(w.values())
        return {n: v / z for n, v in w.items()}

    def choose_action(self, observation: str, legal_actions: List[str], rng: random.Random) -> str:
        self._own_private = self.adapter.own_private(observation)
        seat = self._seat_of(observation)
        if self._dirty or seat not in self._policy:
            self._recompute()
        a = self._policy[seat].get(observation)
        return a if a in legal_actions else legal_actions[0]

    def observe_episode(self, seat: int, public: Tuple[Tuple[int, str], ...], payoff: float) -> None:
        own = self._own_private
        if own is None:
            return
        dist = self.adapter.opponent_private_dist(own)
        steps = [(s, h, a) for s, h, a in self.adapter.history_string(public) if s != seat]
        for n in self.names:
            strat = self.candidates[n]
            lik = 0.0
            for c, pc in dist.items():
                p = pc
                for _, h, a in steps:
                    probs = strat.get(self.adapter.opp_key(c, h))
                    p *= (probs.get(a, 0.0) if probs else 0.5)
                lik += p
            self.loglik[n] = self.decay * self.loglik[n] + math.log(max(lik, 1e-12))
        self.hands += 1
        self._own_private = None
        if self.hands % self.refresh == 0:
            self._dirty = True
            self.posterior_trace.append(self.posterior())

    def metadata(self) -> Dict[str, str]:
        return {"algorithm": f"Bayesian posterior over {len(self.names)} candidate strategies + exact Bayes best "
                             f"response (decay {self.decay:g}, refresh every {self.refresh} hands)",
                "information": "own infoset keys + public actions and own payoff after each hand",
                "adaptive": "yes", "source": "exact best response on the mixture's realization-equivalent strategy",
                "limitations": "opponent assumed to be one of the candidates (misspecification not modelled)"}

    # -- internals ------------------------------------------------------------
    def _seat_of(self, key: str) -> int:
        # Kuhn: the acting seat is determined by history length parity
        return (len(key) - 1) % 2

    def mixture_strategy(self) -> Dict[str, Dict[str, float]]:
        """Realization-equivalent behavior strategy of the posterior mixture."""
        w = self.posterior()
        reach = {n: self._own_reach(self.candidates[n]) for n in self.names}
        keys = set().union(*[set(self.candidates[n]) for n in self.names])
        out = {}
        for k in keys:
            num: Dict[str, float] = {}
            den = 0.0
            for n in self.names:
                probs = self.candidates[n].get(k)
                if not probs:
                    continue
                r = w[n] * reach[n].get(k, 0.0)
                den += r
                for a, p in probs.items():
                    num[a] = num.get(a, 0.0) + r * p
            if den > 0:
                out[k] = {a: v / den for a, v in num.items()}
            else:
                probs = next((self.candidates[n][k] for n in self.names if k in self.candidates[n]), None)
                out[k] = dict(probs)
        return out

    def _own_reach(self, strat: Strategy) -> Dict[str, float]:
        """Each infoset's reach under the strategy's own action probabilities
        (product over that player's earlier actions; chance excluded)."""
        g = self.game
        reach: Dict[str, float] = {}

        def walk(s, r):
            if g.is_terminal(s):
                return
            if g.is_chance(s):
                for _, c in g.chance_outcomes(s):
                    walk(c, r)
                return
            k = g.infoset_key(s)
            p = g.current_player(s)
            reach[k] = r[p]
            probs = strat.get(k)
            for a in g.legal_actions(s):
                nr = list(r)
                nr[p] *= (probs.get(a, 0.0) if probs else 1.0 / len(g.legal_actions(s)))
                walk(g.next_state(s, a), nr)
        walk(g.root(), [1.0, 1.0])
        return reach

    def _recompute(self) -> None:
        mix = self.mixture_strategy()
        self._policy = {}
        for seat in (0, 1):
            _, pol = best_response(self.game, mix, seat)
            self._policy[seat] = pol
        self._dirty = False


class SwitchingAgent:
    """Non-stationary opponent: plays ``first`` for ``switch_at`` hands of a
    match, then ``second`` (counts hands through observe_episode)."""

    def __init__(self, first, second, switch_at: int, name: str = "switching") -> None:
        self.first, self.second, self.switch_at, self.name = first, second, switch_at, name
        self.reset()

    def reset(self) -> None:
        self.hands = 0
        for a in (self.first, self.second):
            a.reset()

    def choose_action(self, observation, legal_actions, rng):
        ag = self.first if self.hands < self.switch_at else self.second
        return ag.choose_action(observation, legal_actions, rng)

    def observe_episode(self, seat, public, payoff):
        self.hands += 1

    def metadata(self):
        return {"algorithm": f"switches strategy after {self.switch_at} hands", "information": "own infoset key",
                "adaptive": "no (scheduled change)", "source": "composition", "limitations": "test opponent"}
