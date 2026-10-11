"""Zero-sum Bayesian signaling game: a non-poker hidden-information game
with ONE-SIDED private information and a message channel.

Structure (different from Kuhn / Leduc / the auction duel):

* chance draws the sender's private type ``t`` from a prior;
* the sender (player 0) observes ``t`` and sends a message ``m``;
  sending ``m`` costs ``cost[t][m]`` (paid to the receiver; zero-sum);
* the receiver (player 1) observes ONLY the message and picks an action
  ``a``; the receiver earns ``payoff[t][a]`` from the sender.

Player 0's utility is ``-(payoff[t][a] + cost[t][m])``.

Theory used as a test oracle: with zero message costs (cheap talk) in a
zero-sum game, communication cannot help either side. The sender can always
babble (send the same mixture regardless of type), and the receiver can
always ignore the message. Hence the game value to the receiver equals
``max_a sum_t prior[t] * payoff[t][a]``, the no-information value. With type-dependent
costs, informative (separating) messages can arise and the value changes.

Information-set keys: sender ``S<t>``, receiver ``R<m>`` (globally unique
across players).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .base import Game


@dataclass(frozen=True)
class SignalState:
    t: Optional[int]
    m: Optional[int]
    a: Optional[int]


class SignalingGame(Game):
    def __init__(self, prior: Sequence[float] = (0.5, 0.5),
                 payoff: Sequence[Sequence[float]] = ((1.0, -1.0), (-1.0, 1.0)),
                 cost: Optional[Sequence[Sequence[float]]] = None, n_messages: int = 2) -> None:
        self.prior = tuple(float(p) for p in prior)
        if abs(sum(self.prior) - 1.0) > 1e-9 or any(p < 0 for p in self.prior):
            raise ValueError("prior must be a probability vector")
        self.n_types = len(self.prior)
        self.payoff = tuple(tuple(float(x) for x in row) for row in payoff)
        if len(self.payoff) != self.n_types:
            raise ValueError("payoff needs one row per type")
        self.n_actions = len(self.payoff[0])
        self.n_messages = int(n_messages)
        if cost is None:
            cost = [[0.0] * self.n_messages for _ in range(self.n_types)]
        self.cost = tuple(tuple(float(x) for x in row) for row in cost)
        if len(self.cost) != self.n_types or any(len(r) != self.n_messages for r in self.cost):
            raise ValueError("cost must be n_types x n_messages")

    def signature(self) -> str:
        return (f"SignalingGame:v1:prior={self.prior}:payoff={self.payoff}:cost={self.cost}"
                f":messages={self.n_messages}")

    def root(self):
        return SignalState(None, None, None)

    def is_chance(self, s):
        return s.t is None

    def chance_outcomes(self, s) -> List[Tuple[float, SignalState]]:
        return [(p, SignalState(t, None, None)) for t, p in enumerate(self.prior) if p > 0]

    def is_terminal(self, s):
        return s.a is not None

    def utility(self, s):
        return -(self.payoff[s.t][s.a] + self.cost[s.t][s.m])

    def current_player(self, s):
        return 0 if s.m is None else 1

    def infoset_key(self, s):
        return f"S{s.t}" if s.m is None else f"R{s.m}"

    def legal_actions(self, s):
        if s.m is None:
            return [f"m{i}" for i in range(self.n_messages)]
        return [f"a{i}" for i in range(self.n_actions)]

    def next_state(self, s, action):
        if s.m is None:
            return SignalState(s.t, int(action[1:]), None)
        return SignalState(s.t, s.m, int(action[1:]))

    def no_information_value_receiver(self) -> float:
        """Receiver's value if messages are ignored (cheap-talk game value)."""
        return max(sum(p * self.payoff[t][a] for t, p in enumerate(self.prior)) for a in range(self.n_actions))
