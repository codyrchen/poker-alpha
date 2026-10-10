"""Phase 18: observation fusion rules, manual correction, state output."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from poker_alpha.decision import DecisionConfig, recommend_action  # noqa: E402
from poker_alpha.holdem import Street, validate  # noqa: E402
from poker_alpha.observer.fusion import StateTracker  # noqa: E402
from poker_alpha.observer.pokernow import (FieldReading,  # noqa: E402
                                           FrameObservation,
                                           PokerNowStyleAdapter,
                                           default_layout)
from poker_alpha.observer.synthetic import (SyntheticSeat,  # noqa: E402
                                            SyntheticTable, render_table)

pytestmark = pytest.mark.vision
CAL = default_layout(3, 0)


def table(**kw):
    seats = kw.pop("seats", None) or [SyntheticSeat("hero", 98.0),
                                      SyntheticSeat("a", 99.5),
                                      SyntheticSeat("b", 99.0, bet=0)]
    base = dict(seats=seats, dealer=0, hero_cards=("As", "Ks"), board=(),
                pot=3.0, actor=0)
    base.update(kw)
    return SyntheticTable(**base)


def frame(t: SyntheticTable, conf=0.8, ts=0.0, overrides=None) -> FrameObservation:
    """What a perfect-but-not-certain recognizer would report for ``t``."""
    f = {}
    put = lambda k, v, c=conf: f.__setitem__(k, FieldReading(v, c, k, ts))  # noqa: E731
    put("pot", t.pot)
    for i in range(5):
        put(f"board_{i}", t.board[i] if i < len(t.board) else None)
    for i in range(2):
        put(f"hero_card_{i}", t.hero_cards[i] if t.hero_cards else None)
    for s, seat in enumerate(t.seats):
        occ = seat.stack is not None
        put(f"seat{s}.occupied", occ)
        put(f"seat{s}.stack", (0.0 if seat.all_in else seat.stack) if occ else None)
        put(f"seat{s}.all_in", seat.all_in)
        put(f"seat{s}.bet", seat.bet)
        put(f"seat{s}.in_hand", seat.in_hand if s else bool(t.hero_cards))
    put("dealer", t.dealer)
    put("actor", t.actor)
    for k, v in (overrides or {}).items():
        put(k, *v) if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], float) \
            else put(k, v)
    return FrameObservation(ts, (0, 0, 1, 1), f)


def feed(tr, t, n=3, **kw):
    for i in range(n):
        tr.update(frame(t, ts=float(i), **kw))


def test_cards_need_agreement_and_board_cannot_change():
    tr = StateTracker(CAL, 0.5, 1.0)
    tr.update(frame(table()))
    assert tr.snapshot().hero_cards == (None, None)       # 1 frame: not yet
    feed(tr, table(), n=2)
    assert tr.snapshot().hero_cards == ("As", "Ks")
    flop = table(board=("Qs", "Js", "4h"), pot=6.0)
    feed(tr, flop, n=3)
    assert tr.snapshot().board == ("Qs", "Js", "4h")
    # One frame misreads the first board card: rejected and flagged.
    tr.update(frame(flop, overrides={"board_0": ("Qh", 0.95)}))
    assert tr.snapshot().board == ("Qs", "Js", "4h")
    assert any("board_0" in w for w in tr.tracked().warnings)


def test_board_fills_in_order():
    tr = StateTracker(CAL, 0.5, 1.0)
    weird = table(board=())
    feed(tr, weird, n=3, overrides={"board_3": "9c"})
    assert tr.snapshot().board == ()
    assert any("before earlier board" in w for w in tr.tracked().warnings)


def _bet(pot, bet):
    return table(pot=pot, seats=[SyntheticSeat("hero", 98.0),
                                 SyntheticSeat("a", 99.5 - bet, bet=0.5 + bet),
                                 SyntheticSeat("b", 99.0, bet=1.0)])


def test_pot_needs_confirmation_or_high_confidence():
    tr = StateTracker(CAL, 0.5, 1.0)
    feed(tr, table(pot=3.0))
    tr.update(frame(_bet(33.0, 30.0), conf=0.6))       # a bet: the pot may move
    assert tr.snapshot().pot == 3.0
    tr.update(frame(_bet(33.0, 30.0), conf=0.6))
    assert tr.snapshot().pot == 33.0
    tr.update(frame(_bet(40.0, 37.0), conf=0.95))
    assert tr.snapshot().pot == 40.0


def test_pot_change_without_betting_is_held_and_absurd_jumps_rejected():
    tr = StateTracker(CAL, 0.5, 1.0)
    feed(tr, table(pot=3.0))
    feed(tr, table(pot=8.0), n=3)                       # nobody bet: misread?
    assert tr.snapshot().pot == 3.0
    assert any("held pot change" in w for w in tr.tracked().warnings)
    feed(tr, table(pot=8.0), n=1)                       # persistent: accepted, flagged
    assert tr.snapshot().pot == 8.0
    feed(tr, table(pot=800.0), n=8)                     # more than all chips in play
    assert tr.snapshot().pot == 8.0
    assert any("breaks chip conservation" in w for w in tr.tracked().warnings)


def test_stack_increase_without_award_is_held():
    tr = StateTracker(CAL, 0.5, 1.0)
    feed(tr, table())
    up = table(seats=[SyntheticSeat("hero", 98.0), SyntheticSeat("a", 150.0),
                      SyntheticSeat("b", 99.0)])
    feed(tr, up, n=2)
    assert tr.snapshot().stacks[1] == 99.5
    assert any("held stack increase" in w for w in tr.tracked().warnings)
    feed(tr, up, n=3)                       # persistent: accepted, flagged
    assert tr.snapshot().stacks[1] == 150.0
    assert any("unexplained" in w for w in tr.tracked().warnings)


def test_stack_increase_with_award_is_accepted():
    tr = StateTracker(CAL, 0.5, 1.0)
    feed(tr, table(pot=20.0))
    won = table(pot=0.0, seats=[SyntheticSeat("hero", 118.0), SyntheticSeat("a", 89.5),
                                SyntheticSeat("b", 89.0)])
    feed(tr, won, n=2)
    assert tr.snapshot().stacks[0] == 118.0


def test_dealer_moves_only_between_hands():
    tr = StateTracker(CAL, 0.5, 1.0)
    flop = table(board=("Qs", "Js", "4h"))
    feed(tr, flop)
    feed(tr, table(board=("Qs", "Js", "4h"), dealer=1))
    assert tr.snapshot().dealer == 0
    assert any("dealer move" in w for w in tr.tracked().warnings)
    nxt = table(board=(), dealer=1, hero_cards=("7d", "7c"))
    feed(tr, nxt, n=4)
    assert tr.snapshot().dealer == 1 and tr.hand_number == 1
    assert tr.snapshot().board == () and tr.snapshot().hero_cards == ("7d", "7c")


def test_pause_correct_resume():
    tr = StateTracker(CAL, 0.5, 1.0)
    feed(tr, table(pot=3.0))
    tr.pause()
    feed(tr, table(pot=99.0), n=5)
    assert tr.snapshot().pot == 3.0
    tr.correct("pot", 4.5)
    tr.resume()
    feed(tr, table(pot=99.0), n=5)
    assert tr.snapshot().pot == 4.5
    assert "pot" in tr.tracked().manual_fields
    tr.release("pot")
    feed(tr, table(pot=12.0), n=4)          # unpinned again: tracked (held, then accepted)
    assert tr.snapshot().pot == 12.0
    with pytest.raises(KeyError):
        tr.correct("nonsense", 1)


def test_events_and_observed_state_feed_the_decision_engine():
    tr = StateTracker(CAL, 0.5, 1.0)
    pre = table(pot=1.5, seats=[SyntheticSeat("hero", 100.0),
                                SyntheticSeat("a", 99.5, bet=0.5),
                                SyntheticSeat("b", 99.0, bet=1.0)])
    feed(tr, pre)
    raised = table(pot=4.5, actor=1, seats=[
        SyntheticSeat("hero", 97.0, bet=3.0), SyntheticSeat("a", 99.5, bet=0.5),
        SyntheticSeat("b", 99.0, bet=1.0)])
    feed(tr, raised)
    kinds = [(e.kind, e.seat) for e in tr.events]
    assert ("bet", 0) in kinds and ("stack_decrease", 0) in kinds
    assert tr.actions[-1].kind == "raise" and tr.actions[-1].amount == 3.0
    folded = table(pot=4.5, actor=2, seats=[
        SyntheticSeat("hero", 97.0, bet=3.0), SyntheticSeat("a", 99.5, bet=0.5,
                                                            in_hand=False),
        SyntheticSeat("b", 99.0, bet=1.0)])
    feed(tr, folded)
    assert ("fold", 1) in [(e.kind, e.seat) for e in tr.events]
    # BB raises: hero to act facing a raise.
    facing = table(pot=13.5, actor=0, seats=[
        SyntheticSeat("hero", 97.0, bet=3.0), SyntheticSeat("a", 99.5, bet=0.5,
                                                            in_hand=False),
        SyntheticSeat("b", 90.0, bet=10.0)])
    feed(tr, facing)
    obs = tr.to_observed_state()
    assert obs.street == Street.PREFLOP and obs.amount_to_call == 7.0
    assert not [i for i in validate(obs) if i.severity == "error"]
    assert 0 < tr.critical_confidence() <= 1
    rep = recommend_action(obs, config=DecisionConfig(
        equity_simulations=300, observer_confidence=tr.critical_confidence()))
    assert rep.recommended in ("call", "fold")
    assert any("observer confidence" in w for w in rep.warnings)


def test_rendered_frames_end_to_end():
    cal = default_layout(6, 0)
    ad = PokerNowStyleAdapter(cal)
    tr = StateTracker(cal, 0.5, 1.0)
    t = SyntheticTable(seats=[SyntheticSeat("hero", 96.5), SyntheticSeat("sb", 99.5, in_hand=False),
                              SyntheticSeat("bb", 90.0, bet=4.5), SyntheticSeat("u", 100.0, in_hand=False),
                              SyntheticSeat("h", 100.0, in_hand=False), SyntheticSeat("c", 100.0, in_hand=False)],
                       dealer=0, hero_cards=("As", "Ks"), board=("Qs", "Js", "4h"),
                       pot=13.5, actor=0)
    for i in range(4):
        tr.update(ad.read_frame(render_table(t, cal, noise=4, seed=i), timestamp=float(i)))
    obs = tr.to_observed_state()
    assert obs.hero_cards is not None and len(obs.board) == 3
    assert obs.pot == 13.5 and obs.amount_to_call == 4.5
