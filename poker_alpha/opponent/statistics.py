"""Hold'em player statistics with posterior uncertainty and recency weighting.

Every statistic is a Bernoulli rate observed through *opportunities* (a
spot where the action was possible) and *occurrences*. Following the rest of
this package, each rate is a Beta posterior — so "75% after 4 hands" and
"75% after 400 hands" report the same mean but very different credible
intervals — and old evidence can be down-weighted geometrically: an
observation made ``k`` hands ago counts ``decay**k`` (``decay=1`` keeps
everything; effective memory ≈ ``1 / (1 - decay)`` hands), exactly the
forgetting scheme of :class:`~poker_alpha.opponent.beliefs.ArchetypeBelief`.

Statistics (opportunity → occurrence):

==================  =======================================================
vpip                every dealt hand except a BB walk → voluntarily put chips in preflop
pfr                 every dealt hand except a BB walk → raised preflop
three_bet           facing exactly one preflop raise → re-raised
fold_to_three_bet   opened, then faced a 3-bet → folded
four_bet            facing a 3-bet → re-raised
limp                acting first-in in an unopened pot (not a blind) → called
cold_call           facing one raise with no chips voluntarily in yet → called
flop_cbet           preflop aggressor, first to bet on the flop → bet
fold_to_flop_cbet   facing the preflop aggressor's flop c-bet → folded
turn_barrel         c-bet the flop, unbet turn → bet
river_barrel        barrelled the turn, unbet river → bet
check_raise         checked, then faced a bet on the same postflop street → raised
fold_to_raise       bet/raised postflop, then faced a raise → folded
aggression          any postflop bet/raise/call/fold → was a bet or raise (AFq)
wtsd                saw the flop → reached showdown
wsd                 reached showdown → won chips there (W$SD)
==================  =======================================================

Scopes: ``"all"``, ``"pos:<POSITION>"`` and, for postflop stats,
``"street:<flop|turn|river>"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scipy import stats as sps

STATS = ("vpip", "pfr", "three_bet", "fold_to_three_bet", "four_bet", "limp",
         "cold_call", "flop_cbet", "fold_to_flop_cbet", "turn_barrel",
         "river_barrel", "check_raise", "fold_to_raise", "aggression", "wtsd",
         "wsd")
STREET_NAMES = {1: "flop", 2: "turn", 3: "river"}


@dataclass
class DecayedBeta:
    """Beta posterior over a rate with geometric forgetting of evidence."""

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    successes: float = 0.0       # decayed
    failures: float = 0.0        # decayed
    raw_successes: int = 0       # undecayed, for display
    raw_opportunities: int = 0
    last_hand: Optional[int] = None

    def _decay_to(self, hand: int, decay: float) -> None:
        if self.last_hand is not None and decay < 1.0 and hand > self.last_hand:
            f = decay ** (hand - self.last_hand)
            self.successes *= f
            self.failures *= f
        self.last_hand = hand if self.last_hand is None else max(self.last_hand, hand)

    def observe(self, occurred: bool, hand: int, decay: float = 1.0) -> None:
        self._decay_to(hand, decay)
        if occurred:
            self.successes += 1.0
            self.raw_successes += 1
        else:
            self.failures += 1.0
        self.raw_opportunities += 1

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.successes

    @property
    def beta(self) -> float:
        return self.prior_beta + self.failures

    @property
    def effective_samples(self) -> float:
        return self.successes + self.failures

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def interval(self, level: float = 0.95) -> Tuple[float, float]:
        lo = (1.0 - level) / 2.0
        d = sps.beta(self.alpha, self.beta)
        return float(d.ppf(lo)), float(d.ppf(1.0 - lo))

    def raw_rate(self) -> Optional[float]:
        if not self.raw_opportunities:
            return None
        return self.raw_successes / self.raw_opportunities


@dataclass(frozen=True)
class StatEstimate:
    stat: str
    scope: str
    raw_successes: int
    raw_opportunities: int
    effective_samples: float
    mean: float                  # posterior mean (shrunk toward the prior)
    low: float                   # credible interval
    high: float
    level: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def format(self) -> str:
        if self.raw_opportunities == 0:
            return f"{self.stat}: no data"
        raw = self.raw_successes / self.raw_opportunities
        return (f"{self.stat}: {raw:.0%} ({self.raw_successes}/"
                f"{self.raw_opportunities}; posterior {self.mean:.0%}, "
                f"{self.level:.0%} CI {self.low:.0%}-{self.high:.0%})")


@dataclass(frozen=True)
class HandSummary:
    """What statistics need from one completed hand.

    ``actions`` are ordered records with ``street`` (0-3), ``seat``,
    ``kind`` (post/fold/check/call/bet/raise/all_in) and ``amount``
    (street raise-to for bet/raise/all_in). ``showdown_seats`` reached
    showdown; ``won`` maps seat → chips won at showdown.
    """

    hand_index: int
    players: Mapping[int, str]          # seat -> player id
    positions: Mapping[int, str]        # seat -> position name
    actions: Sequence
    big_blind_seat: Optional[int] = None
    big_blind: float = 0.0
    showdown_seats: Tuple[int, ...] = ()
    won: Mapping[int, float] = field(default_factory=dict)


def _normalized_actions(actions: Sequence,
                        big_blind: float = 0.0) -> List[Tuple[int, int, str]]:
    """Resolve ``all_in`` into call/bet/raise using street bet levels.

    Preflop, any voluntary bet is a raise (the blinds are the opening bet).
    The preflop level starts at ``big_blind`` (or the largest ``post``) so an
    all-in for no more than the current bet is recognised as a call.
    """
    out: List[Tuple[int, int, str]] = []
    level: Dict[int, float] = {0: float(big_blind)}
    for a in actions:
        street, kind = int(a.street), a.kind
        cur = level.get(street, 0.0)
        amount = float(getattr(a, "amount", 0.0) or 0.0)
        if kind == "post":
            level[street] = max(cur, amount)
            continue
        if kind == "all_in":
            if amount > cur + 1e-9:
                kind = "bet" if cur <= 1e-9 else "raise"
            else:
                kind = "call"
        if kind in ("bet", "raise"):
            level[street] = max(cur, amount)
            kind = "raise" if (street == 0 or cur > 1e-9) else "bet"
        out.append((street, int(a.seat), kind))
    return out


def opportunities(hand: HandSummary) -> List[Tuple[int, str, str, bool]]:
    """``(seat, stat, scope_street_or_"", occurred)`` for every opportunity."""
    acts = _normalized_actions(hand.actions, hand.big_blind)
    out: List[Tuple[int, str, str, bool]] = []
    seats = list(hand.players)

    # -- preflop ---------------------------------------------------------
    pre = [(s, k) for st, s, k in acts if st == 0]
    voluntary = {s: False for s in seats}
    raised = {s: False for s in seats}
    raises = 0
    opener = None
    pfa = None
    for seat, kind in pre:
        if raises == 0 and kind in ("call", "raise") and seat != hand.big_blind_seat:
            out.append((seat, "limp", "", kind == "call"))
        if raises == 1 and not voluntary[seat] and kind in ("call", "raise", "fold"):
            out.append((seat, "cold_call", "", kind == "call"))
        if raises == 1 and seat != opener:
            out.append((seat, "three_bet", "", kind == "raise"))
        if raises == 2 and kind in ("call", "raise", "fold"):
            out.append((seat, "four_bet", "", kind == "raise"))
            if seat == opener:
                out.append((seat, "fold_to_three_bet", "", kind == "fold"))
        if kind in ("call", "raise"):
            voluntary[seat] = True
        if kind == "raise":
            raised[seat] = True
            raises += 1
            pfa = seat
            if raises == 1:
                opener = seat
    walk = all(k == "fold" for _, k in pre) and len(pre) == len(seats) - 1
    for seat in seats:
        if walk and seat == hand.big_blind_seat:
            continue
        out.append((seat, "vpip", "", voluntary[seat]))
        out.append((seat, "pfr", "", raised[seat]))

    # -- postflop --------------------------------------------------------
    saw_flop = {s for st, s, _ in acts if st >= 1}
    folded_pre = {s for s, k in pre if k == "fold"}
    if any(st >= 1 for st, _, _ in acts) or hand.showdown_seats:
        saw_flop |= {s for s in seats if s not in folded_pre}
    cbet_by: Dict[int, int] = {}       # street -> seat that bet as continuation
    for street in (1, 2, 3):
        sname = STREET_NAMES[street]
        st_acts = [(s, k) for st, s, k in acts if st == street]
        bets = 0
        checked = set()
        aggressed = set()          # seats that bet/raised this street
        last_aggressor = None
        first_bettor = None
        for seat, kind in st_acts:
            if kind in ("bet", "raise", "call", "fold"):
                out.append((seat, "aggression", sname, kind in ("bet", "raise")))
            if bets == 0:
                # continuation-bet style opportunities
                if street == 1 and seat == pfa:
                    out.append((seat, "flop_cbet", sname, kind == "bet"))
                if street == 2 and cbet_by.get(1) == seat:
                    out.append((seat, "turn_barrel", sname, kind == "bet"))
                if street == 3 and cbet_by.get(2) == seat:
                    out.append((seat, "river_barrel", sname, kind == "bet"))
            else:
                if seat in checked:
                    out.append((seat, "check_raise", sname, kind == "raise"))
                    checked.discard(seat)
                if seat in aggressed and seat != last_aggressor:
                    out.append((seat, "fold_to_raise", sname, kind == "fold"))
                if (street == 1 and first_bettor is not None
                        and first_bettor == pfa and bets == 1
                        and seat != pfa and kind in ("call", "raise", "fold")):
                    out.append((seat, "fold_to_flop_cbet", sname, kind == "fold"))
            if kind == "check":
                checked.add(seat)
            if kind in ("bet", "raise"):
                bets += 1
                if bets == 1:
                    first_bettor = seat
                    if (street == 1 and seat == pfa) or \
                            (street > 1 and cbet_by.get(street - 1) == seat):
                        cbet_by[street] = seat
                aggressed.add(seat)
                last_aggressor = seat
    for seat in saw_flop:
        out.append((seat, "wtsd", "", seat in hand.showdown_seats))
    for seat in hand.showdown_seats:
        out.append((seat, "wsd", "", hand.won.get(seat, 0.0) > 0))
    return out


class PlayerStatistics:
    """Accumulates :data:`STATS` per player and scope."""

    def __init__(self, decay: float = 1.0, prior_alpha: float = 1.0,
                 prior_beta: float = 1.0) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.prior = (prior_alpha, prior_beta)
        self.counters: Dict[Tuple[str, str, str], DecayedBeta] = {}
        self.hands: Dict[str, int] = {}

    def _counter(self, player: str, stat: str, scope: str) -> DecayedBeta:
        key = (player, stat, scope)
        c = self.counters.get(key)
        if c is None:
            c = DecayedBeta(*self.prior)
            self.counters[key] = c
        return c

    def add_hand(self, hand: HandSummary) -> None:
        for seat, player in hand.players.items():
            self.hands[player] = self.hands.get(player, 0) + 1
        for seat, stat, street, occurred in opportunities(hand):
            player = hand.players[seat]
            scopes = ["all"]
            pos = hand.positions.get(seat)
            if pos:
                scopes.append(f"pos:{pos}")
            if street:
                scopes.append(f"street:{street}")
            for scope in scopes:
                self._counter(player, stat, scope).observe(
                    occurred, hand.hand_index, self.decay)

    def add_hands(self, hands: Iterable[HandSummary]) -> None:
        for h in hands:
            self.add_hand(h)

    def estimate(self, player: str, stat: str, scope: str = "all",
                 level: float = 0.95) -> StatEstimate:
        if stat not in STATS:
            raise ValueError(f"unknown stat {stat!r}")
        c = self.counters.get((player, stat, scope)) or DecayedBeta(*self.prior)
        lo, hi = c.interval(level)
        return StatEstimate(stat, scope, c.raw_successes, c.raw_opportunities,
                            c.effective_samples, c.mean(), lo, hi, level)

    def summary(self, player: str, scope: str = "all") -> Dict[str, StatEstimate]:
        return {s: self.estimate(player, s, scope) for s in STATS}

    def players(self) -> List[str]:
        return sorted(self.hands)
