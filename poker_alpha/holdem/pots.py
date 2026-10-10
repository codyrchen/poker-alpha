"""Side-pot construction and pot awarding (pure functions, integer chips).

Pots are built by layering contribution levels: for each distinct
contribution level among *live* (non-folded) players, everyone's chips up to
that level form a pot whose eligible winners are the live players who
contributed at least that much. Folded ("dead") chips fill the layers they
reach but their owners are never eligible.

Chips contributed above every other player's level form a layer only the
contributor is eligible for — an uncalled bet, which is simply returned by
being "won" back.

All amounts are integers so chip conservation is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: Tuple[int, ...]   # seat indices, ascending


def build_pots(contributions: Sequence[int], folded: Sequence[bool]) -> List[Pot]:
    """Main pot first, then side pots in increasing contribution level.

    ``contributions[i]`` is seat ``i``'s total chips committed this hand
    (blinds and antes included). Adjacent layers with identical eligibility
    are merged, so the result is the conventional main + side pots.
    """
    if len(contributions) != len(folded):
        raise ValueError("contributions and folded must align")
    if any(int(c) != c or c < 0 for c in contributions):
        raise ValueError("contributions must be non-negative integers")
    live_levels = sorted({c for c, f in zip(contributions, folded)
                          if not f and c > 0})
    pots: List[Pot] = []
    prev = 0
    for level in live_levels:
        amount = sum(max(0, min(c, level) - prev) for c in contributions)
        eligible = tuple(i for i, (c, f) in enumerate(zip(contributions, folded))
                         if not f and c >= level)
        if amount > 0:
            if pots and pots[-1].eligible == eligible:
                pots[-1] = Pot(pots[-1].amount + amount, eligible)
            else:
                pots.append(Pot(amount, eligible))
        prev = level
    # Dead chips above every live level (a folded player out-committed all
    # live players — only possible with forced bets) go to the top pot.
    leftover = sum(max(0, c - prev) for c in contributions)
    if leftover:
        if not pots:
            raise ValueError("chips in the pot but no live player")
        pots[-1] = Pot(pots[-1].amount + leftover, pots[-1].eligible)
    return pots


def award_pots(pots: Sequence[Pot],
               strength: Callable[[int], tuple],
               seat_order: Sequence[int]) -> Dict[int, int]:
    """Split each pot among its best eligible hands.

    ``strength(seat)`` returns a comparable hand value (higher wins).
    ``seat_order`` lists seats in odd-chip priority order (conventionally
    clockwise starting left of the button); indivisible chips go one at a
    time to the winners earliest in that order.
    """
    rank = {s: i for i, s in enumerate(seat_order)}
    won: Dict[int, int] = {}
    for pot in pots:
        if len(pot.eligible) == 1:
            winners = [pot.eligible[0]]
        else:
            values = {s: strength(s) for s in pot.eligible}
            best = max(values.values())
            winners = sorted((s for s, v in values.items() if v == best),
                             key=lambda s: rank[s])
        share, odd = divmod(pot.amount, len(winners))
        for i, s in enumerate(winners):
            won[s] = won.get(s, 0) + share + (1 if i < odd else 0)
    return won
