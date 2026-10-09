"""Scripted synthetic hand timelines for observer state-machine tests.

Pot convention of the generic layout: the displayed pot INCLUDES the bets in
front of the players. ``expected`` events come from the script's poker
semantics (not from the tracker's own diff logic)."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from poker_alpha.observer.synthetic import SyntheticSeat, SyntheticTable


@dataclass
class Hand:
    stacks: List[float]
    dealer: int
    hero: Tuple[str, str]
    sb: float = 0.5
    bb: float = 1.0
    bets: List[float] = field(default_factory=list)
    in_hand: List[bool] = field(default_factory=list)
    all_in: List[bool] = field(default_factory=list)
    board: Tuple[str, ...] = ()
    pot: float = 0.0
    states: List[SyntheticTable] = field(default_factory=list)
    expected: List[Tuple[str, Optional[int]]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.stacks)
        self.bets = self.bets or [0.0] * n
        self.in_hand = self.in_hand or [True] * n
        self.all_in = self.all_in or [False] * n

    def snap(self, actor=None):
        seats = [SyntheticSeat(f"p{i}", 0.0 if self.all_in[i] else round(self.stacks[i], 2),
                               bet=round(self.bets[i], 2), in_hand=self.in_hand[i],
                               all_in=self.all_in[i]) for i in range(len(self.stacks))]
        self.states.append(SyntheticTable(seats=seats, dealer=self.dealer, hero_cards=self.hero,
                                          board=self.board, pot=round(self.pot, 2), actor=actor))

    def _put(self, s, to):
        delta = to - self.bets[s]
        self.stacks[s] -= delta
        self.bets[s] = to
        self.pot += delta

    # -- actions ------------------------------------------------------------------
    def post(self):
        n = len(self.stacks)
        sb = self.dealer if n == 2 else (self.dealer + 1) % n
        bb = (sb + 1) % n
        self._put(sb, self.sb)
        self._put(bb, self.bb)
        self.snap()
        return self

    def bet(self, s, to):
        self._put(s, to)
        self.expected.append(("bet", s))
        self.snap()
        return self

    def call(self, s):
        return self.bet(s, max(self.bets))

    def check(self, s):
        self.snap(actor=s)          # nothing visible changes but the actor
        return self

    def fold(self, s, hero_seat=0):
        self.in_hand[s] = False
        # The hero's own cards stay on screen after a fold (the generic renderer
        # keeps them; PokerNow dims them), so a hero fold is not observable from
        # the cards: KNOWN LIMITATION, no event expected.
        if s != hero_seat:
            self.expected.append(("fold", s))
        self.snap()
        return self

    def allin(self, s):
        to = self.bets[s] + self.stacks[s]
        self._put(s, to)
        self.all_in[s] = True
        self.expected.append(("bet", s))
        self.snap()
        return self

    def deal(self, *cards):
        self.bets = [0.0] * len(self.stacks)      # collected (already in the pot)
        self.board = self.board + tuple(cards)
        self.expected.append(("board", None))
        self.snap()
        return self

    def award(self, *winners):
        share = self.pot / len(winners)
        for w in winners:
            self.stacks[w] += share
            self.all_in[w] = False
            self.expected.append(("stack_increase", w))
        for i in range(len(self.stacks)):
            if self.all_in[i] and self.stacks[i] < 1e-9:
                self.all_in[i] = False             # busted all-in shows a 0 stack
        self.pot = 0.0
        self.bets = [0.0] * len(self.stacks)
        self.snap()
        return self

    def award_split(self, amounts):
        """Side pots: amounts[seat] won."""
        for w, a in amounts.items():
            self.stacks[w] += a
            self.all_in[w] = False
            self.expected.append(("stack_increase", w))
        self.pot = 0.0
        self.bets = [0.0] * len(self.stacks)
        self.snap()
        return self


def next_hand(prev: Hand, hero, dealer=None) -> Hand:
    n = len(prev.stacks)
    h = Hand(list(prev.stacks), (prev.dealer + 1) % n if dealer is None else dealer, hero,
             prev.sb, prev.bb)
    h.expected.append(("new_hand", None))
    return h


def frames(states, repeat=3):
    out = []
    for st in states:
        out += [st] * repeat
    return out
