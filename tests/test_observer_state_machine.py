"""Phase 51: observer state machine on scripted synthetic hand timelines.

Each scripted hand is rendered state by state (3 frames per state unless a
test perturbs the cadence) and replayed through the PokerNow-style adapter
and the StateTracker. The tracker's events must match the script's actions
(bet / call -> "bet", fold, street dealt -> "board", pot won ->
"stack_increase", next hand -> "new_hand") and its confirmed state must end
equal to the true final state. Synthetic frames only.
"""

import random
from collections import Counter

import pytest

pytest.importorskip("PIL")

from observer_timelines import Hand, next_hand  # noqa: E402

from poker_alpha.observer.fusion import StateTracker  # noqa: E402
from poker_alpha.observer.pokernow import PokerNowStyleAdapter, default_layout  # noqa: E402
from poker_alpha.observer.synthetic import render_table  # noqa: E402

pytestmark = pytest.mark.vision
COUNTED = ("bet", "fold", "board", "stack_increase", "new_hand")
_CACHE = {}


def _obs(adapter, cal, state, t):
    key = (cal.num_seats, repr(state))
    if key not in _CACHE:
        _CACHE[key] = adapter.read_frame(render_table(state, cal))
    obs = _CACHE[key]
    return obs


def replay(hands, repeat=3, cadence=None, timestamps=None):
    """Feed every state of ``hands``; ``cadence(i)`` = frames for state i."""
    n = len(hands[0].stacks)
    cal = default_layout(n, 0)
    ad = PokerNowStyleAdapter(cal)
    tr = StateTracker(cal, hands[0].sb, hands[0].bb)
    got, expected = [], []
    t = 0.0
    for hi, h in enumerate(hands):
        expected += h.expected
        for si, st in enumerate(h.states):
            k = cadence(len(got), si) if cadence else repeat
            for _ in range(k):
                obs = _obs(ad, cal, st, t)
                obs.timestamp = timestamps(t) if timestamps else t
                for e in tr.update(obs):
                    if e.kind in COUNTED and not (si == 0 and e.kind == "bet"):
                        got.append((e.kind, e.seat if e.kind in ("bet", "fold",
                                                                  "stack_increase") else None))
                t += 1.0
    return tr, got, expected


def final_ok(tr, hand):
    last = hand.states[-1]
    snap = tr.snapshot()
    stacks = tuple(0.0 if s.all_in else s.stack for s in last.seats)
    assert snap.stacks == stacks, (snap.stacks, stacks)
    assert snap.pot == last.pot and snap.board == tuple(last.board)


def H(dealer=1, hero=("As", "Kd"), stacks=(100.0, 100.0)):
    return Hand(list(stacks), dealer, hero).post()


SCENARIOS = {
    "preflop fold": lambda: [H().fold(1).award(0)],
    "limp / check": lambda: [H().call(1).check(0).deal("Qs", "Jh", "4c").check(0).check(1)],
    "raise / fold": lambda: [H().bet(1, 3.0).fold(0).award(1)],
    "raise / call -> flop": lambda: [H().bet(1, 3.0).call(0).deal("Qs", "Jh", "4c")],
    "flop check / check -> turn": lambda: [
        H().call(1).check(0).deal("Qs", "Jh", "4c").check(0).check(1).deal("2d")],
    "flop bet / fold": lambda: [
        H().call(1).check(0).deal("Qs", "Jh", "4c").bet(0, 1.5).fold(1).award(0)],
    "flop bet / call -> turn -> river -> showdown": lambda: [
        H().bet(1, 3.0).call(0).deal("Qs", "Jh", "4c").bet(0, 4.0).call(1).deal("2d")
        .check(0).check(1).deal("9s").bet(0, 10.0).call(1).award(0)],
    "all-in and call, runout": lambda: [
        H().allin(1).call(0).deal("Qs", "Jh", "4c").deal("2d").deal("9s").award(1)],
    "split pot": lambda: [
        H().bet(1, 3.0).call(0).deal("Qs", "Jh", "4c").deal("2d").deal("9s").award(0, 1)],
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_heads_up_scripts(name):
    hands = SCENARIOS[name]()
    tr, got, expected = replay(hands)
    assert Counter(got) == Counter(expected), (got, expected)
    final_ok(tr, hands[-1])


def test_consecutive_hands_and_new_hand_events():
    h1 = H().bet(1, 3.0).fold(0).award(1)
    h2 = next_hand(h1, ("7c", "7d")).post().call(0).check(1).deal("Kc", "8h", "3s")
    h3 = next_hand(h2, ("2c", "9d")).post().fold(1).award(0)
    tr, got, expected = replay([h1, h2, h3])
    assert Counter(got) == Counter(expected), (got, expected)
    assert tr.hand_number == 2
    final_ok(tr, h3)


def test_multiway_fold_all_in_side_pot_and_award():
    h = Hand([100.0, 40.0, 100.0], 0, ("As", "Kd")).post()
    h.bet(0, 6.0).allin(1).fold(2).call(0)            # seat 1 all-in for 40
    h.deal("Qs", "Jh", "4c").deal("2d").deal("9s")
    h.award_split({0: h.pot})                          # hero wins the (single) pot
    tr, got, expected = replay([h])
    assert Counter(got) == Counter(expected), (got, expected)
    final_ok(tr, h)


def test_multiway_side_pot_two_winners():
    h = Hand([100.0, 20.0, 100.0], 0, ("As", "Kd")).post()
    h.allin(1).call(2).call(0)                         # 3-way for 20
    h.deal("Qs", "Jh", "4c").bet(2, 10.0).call(0)      # side pot between 0 and 2
    h.deal("2d").deal("9s")
    main = 60.0
    side = h.pot - main
    h.award_split({1: main, 2: side})
    tr, got, expected = replay([h])
    assert Counter(got) == Counter(expected), (got, expected)
    final_ok(tr, h)


def test_delayed_and_missing_frames():
    """States shown 3-5 frames with random extra frames; same events."""
    hands = SCENARIOS["flop bet / call -> turn -> river -> showdown"]()
    rnd = random.Random(0)
    tr, got, expected = replay(hands, cadence=lambda i, si: rnd.choice([3, 4, 5]))
    assert Counter(got) == Counter(expected)
    final_ok(tr, hands[-1])


def test_transient_one_frame_states_are_not_events():
    """A state visible for a single frame (an animation glitch) is ignored."""
    h = H().bet(1, 3.0)
    glitch = h.states[-1]
    h2 = Hand([100.0, 100.0], 1, ("As", "Kd")).post()
    states = [h2.states[0], glitch, h2.states[0]]
    cal = default_layout(2, 0)
    ad = PokerNowStyleAdapter(cal)
    tr = StateTracker(cal, 0.5, 1.0)
    events = []
    for st, k in zip(states, (4, 1, 4)):
        for _ in range(k):
            events += tr.update(_obs(ad, cal, st, 0.0))
    assert not [e for e in events if e.kind in COUNTED]


def test_duplicated_frames_do_not_duplicate_events():
    hands = SCENARIOS["raise / call -> flop"]()
    tr, got, expected = replay(hands, repeat=12)
    assert Counter(got) == Counter(expected)


def test_out_of_order_timestamps_do_not_matter():
    hands = SCENARIOS["raise / call -> flop"]()
    a = replay(hands)
    b = replay(hands, timestamps=lambda t: 1000.0 - t)       # time runs backwards
    assert a[1] == b[1] and a[0].snapshot().stacks == b[0].snapshot().stacks
