"""Recognition accuracy against ground truth.

Measures card accuracy, numeric-field accuracy and whole-state accuracy of a
:class:`PokerNowStyleAdapter` on labelled frames. Numbers produced on the
synthetic renderer describe the synthetic renderer only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .pokernow import FrameObservation, PokerNowStyleAdapter
from .synthetic import SyntheticTable


@dataclass(frozen=True)
class AccuracyReport:
    frames: int
    card_accuracy: float
    numeric_accuracy: float
    discrete_accuracy: float      # dealer / actor / in-hand / occupancy
    state_accuracy: float         # every field in the frame correct
    mean_conf_correct: float
    mean_conf_wrong: float
    errors: Tuple[str, ...]


def frame_errors(obs: FrameObservation, truth: SyntheticTable, hero_seat: int):
    """``(field, kind, correct, confidence)`` for each checked field."""
    out = []

    def add(name, kind, ok):
        out.append((name, kind, bool(ok), obs.confidence(name)))

    for i in range(2):
        want = truth.hero_cards[i] if i < len(truth.hero_cards) else None
        add(f"hero_card_{i}", "card", obs.value(f"hero_card_{i}") == want)
    for i in range(5):
        want = truth.board[i] if i < len(truth.board) else None
        add(f"board_{i}", "card", obs.value(f"board_{i}") == want)
    add("pot", "numeric", obs.value("pot") == (truth.pot if truth.pot > 0 else 0.0))
    for s, seat in enumerate(truth.seats):
        occ = seat.stack is not None
        add(f"seat{s}.occupied", "discrete", obs.value(f"seat{s}.occupied") == occ)
        if not occ:
            continue
        want_stack = 0.0 if seat.all_in else seat.stack
        add(f"seat{s}.stack", "numeric", obs.value(f"seat{s}.stack") == want_stack)
        add(f"seat{s}.all_in", "discrete", obs.value(f"seat{s}.all_in") == seat.all_in)
        add(f"seat{s}.bet", "numeric", obs.value(f"seat{s}.bet") == seat.bet)
        in_hand = seat.in_hand if s != hero_seat else bool(truth.hero_cards)
        add(f"seat{s}.in_hand", "discrete", obs.value(f"seat{s}.in_hand") == in_hand)
    add("dealer", "discrete", obs.value("dealer") == truth.dealer)
    add("actor", "discrete", obs.value("actor") == truth.actor)
    return out


def evaluate(adapter: PokerNowStyleAdapter,
             frames: Iterable[Tuple[object, SyntheticTable]]) -> AccuracyReport:
    rows: List[tuple] = []
    states_ok = n = 0
    errors: List[str] = []
    for image, truth in frames:
        obs = adapter.read_frame(image)
        res = frame_errors(obs, truth, adapter.cal.hero_seat)
        rows.extend(res)
        n += 1
        bad = [r for r in res if not r[2]]
        states_ok += not bad
        errors.extend(f"frame {n - 1}: {r[0]}={obs.value(r[0])!r}" for r in bad[:3])

    def acc(kind):
        sel = [r for r in rows if r[1] == kind]
        return sum(r[2] for r in sel) / len(sel) if sel else float("nan")

    good = [r[3] for r in rows if r[2]]
    wrong = [r[3] for r in rows if not r[2]]
    return AccuracyReport(
        frames=n, card_accuracy=acc("card"), numeric_accuracy=acc("numeric"),
        discrete_accuracy=acc("discrete"), state_accuracy=states_ok / max(n, 1),
        mean_conf_correct=sum(good) / len(good) if good else float("nan"),
        mean_conf_wrong=sum(wrong) / len(wrong) if wrong else float("nan"),
        errors=tuple(errors[:20]))
