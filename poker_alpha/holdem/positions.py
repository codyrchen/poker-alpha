"""Table positions for 2-9 handed Hold'em."""

from __future__ import annotations

from typing import Dict, List, Tuple

# Names for the seats between the big blind and the button, by count.
_MIDDLE = {
    0: [],
    1: ["UTG"],
    2: ["UTG", "CO"],
    3: ["UTG", "HJ", "CO"],
    4: ["UTG", "LJ", "HJ", "CO"],
    5: ["UTG", "UTG+1", "LJ", "HJ", "CO"],
    6: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"],
}


def blind_seats(num_seats: int, dealer: int) -> Tuple[int, int]:
    """``(small blind seat, big blind seat)``. Heads-up the button is the SB."""
    if not 2 <= num_seats <= 9:
        raise ValueError("2-9 seats supported")
    if num_seats == 2:
        return dealer, (dealer + 1) % 2
    return (dealer + 1) % num_seats, (dealer + 2) % num_seats


def position_names(num_seats: int, dealer: int) -> Dict[int, str]:
    """Seat -> position name (BTN, SB, BB, UTG, ..., CO)."""
    sb, bb = blind_seats(num_seats, dealer)
    if num_seats == 2:
        return {dealer: "BTN", bb: "BB"}
    names = {dealer: "BTN", sb: "SB", bb: "BB"}
    middle = _MIDDLE[num_seats - 3]
    for i, name in enumerate(middle):
        names[(bb + 1 + i) % num_seats] = name
    return names


def clockwise_from(start: int, num_seats: int) -> List[int]:
    """Seats in clockwise order beginning at ``start``."""
    return [(start + i) % num_seats for i in range(num_seats)]
