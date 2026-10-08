"""Phase 26 candidate encoders for the heads-up solver game.

Two families, each available with an exact or an abstracted betting history:

``TransitionHoldemEncoder`` (26B) — aims for perfect recall.
    ``player | preflop class / flop state / turn transition / river transition | history``

    The flop state is ``strength draw nut`` from :mod:`.features`; later
    streets record how the hand *changed* (improved / same / weakened, draw
    gained / completed / missed / kept, became / lost the nuts, major texture
    shift) instead of a fresh absolute bucket, so the remembered card history
    multiplies by a small transition alphabet rather than by full buckets.
    With ``history="exact"`` every earlier abstract observation and every own
    action stays recoverable from the key (perfect recall by construction,
    audited empirically). With ``history="abstract"`` it is imperfect recall.

``CompactHoldemEncoder`` (26C) — IMPERFECT RECALL, NO STANDARD CFR
EQUILIBRIUM GUARANTEE.
    ``street | position | strength draw nut [blocker on river] | texture | betting context``

    Forgets the preflop class and every earlier street's card state; keeps
    only the current strategic state. With ``history="abstract"`` (default)
    the betting part is :class:`~.betting_history.BettingContext`; with
    ``history="exact"`` it is the exact action string.

Both are deterministic, use only cached exact features (no Monte Carlo at
lookup time) and have explicit versioned signatures.
"""

from __future__ import annotations

from typing import Any

from .betting_history import BETTING_HISTORY_VERSION, betting_context
from .features import CARD_FEATURES_VERSION, TRANSITION_VERSION, card_features, transition_label

_BOARD_AT = (0, 3, 4, 5)


def _history(game: Any, state: Any, mode: str) -> str:
    if mode == "exact":
        return "/".join(state.streets)
    return betting_context(game, state).key()


class TransitionHoldemEncoder:
    """Transition-aware card history (see module docstring)."""

    VERSION = 1

    def __init__(self, history: str = "exact") -> None:
        if history not in ("exact", "abstract"):
            raise ValueError("history must be 'exact' or 'abstract'")
        self.history = history

    @property
    def perfect_recall_by_design(self) -> bool:
        return self.history == "exact"

    def signature(self) -> str:
        return (f"TransitionHoldemEncoder:v{self.VERSION}:cards=v{CARD_FEATURES_VERSION}"
                f":transitions=v{TRANSITION_VERSION}:history={self.history}"
                + (f":betting=v{BETTING_HISTORY_VERSION}" if self.history == "abstract" else ""))

    def card_part(self, hole, board) -> str:
        pre = card_features(hole).made
        parts = [pre]
        if len(board) >= 3:
            f = card_features(hole, board[:3])
            parts.append(f"{f.strength}{f.draw}{f.nut}")
            prev = f
            for n in (4, 5):
                if len(board) < n:
                    break
                cur = card_features(hole, board[:n])
                parts.append(transition_label(prev, cur))
                prev = cur
        return "/".join(parts)

    def encode(self, game: Any, state: Any) -> str:
        p = game.current_player(state)
        return f"{p}|{self.card_part(state.holes[p], state.board)}|{_history(game, state, self.history)}"


class CompactHoldemEncoder:
    """Current-state encoder. IMPERFECT RECALL — NO STANDARD CFR EQUILIBRIUM
    GUARANTEE (see module docstring)."""

    VERSION = 1
    perfect_recall_by_design = False

    def __init__(self, history: str = "abstract", texture: bool = True,
                 river_blockers: bool = True) -> None:
        if history not in ("exact", "abstract"):
            raise ValueError("history must be 'exact' or 'abstract'")
        self.history = history
        self.texture = texture
        self.river_blockers = river_blockers

    def signature(self) -> str:
        return (f"CompactHoldemEncoder:v{self.VERSION}:cards=v{CARD_FEATURES_VERSION}"
                f":history={self.history}:texture={int(self.texture)}"
                f":river_blockers={int(self.river_blockers)}"
                + (f":betting=v{BETTING_HISTORY_VERSION}" if self.history == "abstract" else "")
                + ":recall=imperfect")

    def card_part(self, hole, board) -> str:
        f = card_features(hole, board)
        if not board:
            return f.made
        s = f"{f.strength}{f.draw}{f.nut}"
        if self.river_blockers and len(board) == 5:
            s += f"b{f.blocker}"
        if self.texture:
            s += f"|{f.texture}"
        return s

    def encode(self, game: Any, state: Any) -> str:
        p = game.current_player(state)
        street = len(state.streets) - 1
        return (f"{street}|{p}|{self.card_part(state.holes[p], state.board)}|"
                f"{_history(game, state, self.history)}")
