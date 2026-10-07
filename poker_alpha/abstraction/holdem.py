"""Hold'em information-state encoders for the heads-up solver game.

* :class:`RawHoldemEncoder` — lossless; byte-for-byte the historical
  :class:`~poker_alpha.games.holdem.HoldemGame` key
  ``"{player}|{sorted hole codes}|{board codes}|{street histories}"``.
* :class:`ToyHoldemEncoder` — an intentionally coarse, *demonstration-only*
  encoder proving the abstraction plumbing works. See its warning.

Real abstractions (169 preflop classes, postflop feature buckets) live in
:mod:`poker_alpha.abstraction.cards` and build on this seam.
"""

from __future__ import annotations

from typing import Any

from ..poker.evaluator import evaluate_best_codes


def _acting_player_and_hole(game: Any, state: Any):
    player = game.current_player(state)
    return player, state.holes[player]


class RawHoldemEncoder:
    """Lossless encoder reproducing the original Hold'em infoset key."""

    VERSION = 1

    def encode(self, game: Any, state: Any) -> str:
        player, hole = _acting_player_and_hole(game, state)
        hole_s = ",".join(str(c) for c in sorted(hole))
        board = ",".join(str(c) for c in state.board)
        history = "/".join(state.streets)
        return f"{player}|{hole_s}|{board}|{history}"

    def signature(self) -> str:
        return f"RawHoldemEncoder:v{self.VERSION}"


class ToyHoldemEncoder:
    """Deliberately lossy encoder for plumbing tests ONLY.

    Key: ``player | card bucket | street | full action history``, where the
    card bucket is

    * preflop: ``P`` (pair) / ``S`` (suited) / ``O`` (offsuit) plus whether the
      high card is broadway (T+),
    * postflop: the made-hand category of the best five cards (0..8).

    The exact board is dropped and the hand is reduced to a handful of
    buckets, so thousands of raw states share one key.

    PERFECT-RECALL WARNING: the postflop bucket does not include the preflop
    bucket the player held, so a player "forgets" private information they
    had earlier in the hand. This is an imperfect-recall abstraction; any
    strategy trained on it is NOT an equilibrium of Hold'em or of a
    perfect-recall abstraction of it and must never be presented as one.
    """

    VERSION = 1

    def encode(self, game: Any, state: Any) -> str:
        player, hole = _acting_player_and_hole(game, state)
        if not state.board:
            r1, r2 = hole[0] % 13, hole[1] % 13
            if r1 == r2:
                kind = "P"
            elif hole[0] // 13 == hole[1] // 13:
                kind = "S"
            else:
                kind = "O"
            bucket = kind + ("B" if max(r1, r2) >= 8 else "x")
        else:
            bucket = str(evaluate_best_codes(list(hole) + list(state.board))[0])
        history = "/".join(state.streets)
        return f"{player}|{bucket}|{len(state.board)}|{history}"

    def signature(self) -> str:
        return f"ToyHoldemEncoder:v{self.VERSION}"
