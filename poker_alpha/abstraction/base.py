"""Information-state encoder seam.

A solver identifies information sets by string keys. *What goes into that key*
is an abstraction decision, separate from the true game state: the raw key
keeps every card and action, a bucketed key merges strategically "similar"
states so the solver can generalize across them.

:class:`InformationStateEncoder` is the seam. A game that accepts an encoder
calls ``encoder.encode(game, state)`` from its ``infoset_key`` and reports
``encoder.signature()`` from ``encoder_signature`` so solver checkpoints are
bound to the abstraction that produced them.

Signatures
----------
Signatures are deterministic, explicit, versioned and configuration-aware —
``Name:vN[:param=value...]`` built from explicit formatting, never ``repr``.
Two encoders whose keys could differ for any state must have different
signatures.

PERFECT-RECALL WARNING
----------------------
CFR's convergence guarantees assume *perfect recall*: a player never forgets
information they once had, including their own past actions and the private
information they held when taking them. An encoder that merges states with
different relevant histories (e.g. forgets which preflop hand class led to the
current flop bucket) produces an *imperfect-recall* abstraction. CFR can still
be run on it and often works well in practice, but the result carries no
equilibrium guarantee in the original game. Lossy encoders in this package
state explicitly whether they preserve perfect recall.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InformationStateEncoder(Protocol):
    """Maps a game state to the acting player's information-set key."""

    def encode(self, game: Any, state: Any) -> str:
        """Key for the player to act at ``state``."""
        ...

    def signature(self) -> str:
        """Deterministic, versioned, configuration-aware identifier."""
        ...


def format_number(x: float) -> str:
    """Deterministic number formatting for signatures (no ``repr``)."""
    x = float(x)
    if x == 0.0:
        x = 0.0
    return format(x, ".12g")
