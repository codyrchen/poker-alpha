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


class HoldemBucketEncoder:
    """First defensible heads-up Hold'em card abstraction.

    Key layout::

        {player}|{preflop class}[/{flop bucket}[/{turn bucket}[/{river bucket}]]]|{exact action history}

    * Preflop the hand is its canonical 169-class label — lossless up to suit
      isomorphism.
    * Each postflop street adds ``e{equity bucket}{texture code}s{SPR bucket}``
      where equity is the deterministic, suit-invariant equity vs a random
      hand (:func:`~poker_alpha.abstraction.cards.hand_equity`), texture is
      :func:`~poker_alpha.abstraction.cards.texture_code` and SPR is the
      effective stack / pot at the start of that street.
    * The exact action history is preserved, and the buckets of *every*
      earlier street are kept, so the abstraction has perfect recall of the
      player's own abstract signals and all actions. (It is still an
      abstraction: different hands in one bucket are forced to play alike.)

    Equity bucketing is uniform in ``[0, 1]``; this is a simple, inspectable
    choice, not an optimal one (no potential-aware or distribution-aware
    clustering yet).
    """

    VERSION = 1

    def __init__(self, equity_buckets: int = 10,
                 spr_edges=(1.0, 3.0, 8.0),
                 texture_version: int = 1,
                 equity_samples: int = 200) -> None:
        if equity_buckets < 1:
            raise ValueError("equity_buckets must be >= 1")
        if list(spr_edges) != sorted(spr_edges):
            raise ValueError("spr_edges must be increasing")
        self.equity_buckets = int(equity_buckets)
        self.spr_edges = tuple(float(x) for x in spr_edges)
        self.texture_version = int(texture_version)
        self.equity_samples = int(equity_samples)

    def signature(self) -> str:
        from .base import format_number

        spr = ",".join(format_number(x) for x in self.spr_edges)
        return (f"HoldemBucketEncoder:v{self.VERSION}"
                f":equity={self.equity_buckets}"
                f":samples={self.equity_samples}"
                f":spr={spr}"
                f":texture=v{self.texture_version}"
                f":history=exact")

    def street_bucket(self, hole, board, spr: float) -> str:
        from .cards import bucketize, board_texture, hand_equity, texture_code

        eq = hand_equity(hole, board, self.equity_samples)
        e = min(int(eq * self.equity_buckets), self.equity_buckets - 1)
        tex = texture_code(board_texture(board), self.texture_version)
        return f"e{e}{tex}s{bucketize(spr, self.spr_edges)}"

    def encode(self, game: Any, state: Any) -> str:
        from .cards import preflop_class

        player, hole = _acting_player_and_hole(game, state)
        parts = [preflop_class(hole)]
        board_sizes = (3, 4, 5)
        for street in range(1, len(state.streets)):
            n = board_sizes[street - 1]
            if len(state.board) < n:
                break
            parts.append(self.street_bucket(hole, state.board[:n],
                                            _street_start_spr(game, state, street)))
        history = "/".join(state.streets)
        return f"{player}|{'/'.join(parts)}|{history}"


def _street_start_spr(game: Any, state: Any, street: int) -> float:
    """Effective stack / pot at the start of ``street`` (heads-up game)."""
    from dataclasses import replace

    prefix = replace(state, streets=state.streets[:street] + ("",))
    _, total, _, _ = game._replay(prefix)
    pot = total[0] + total[1]
    eff = game.starting_stack - max(total)
    return eff / pot if pot > 0 else float("inf")
