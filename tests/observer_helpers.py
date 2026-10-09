"""Shared helpers for observer session tests: synthetic tables and a fake screen.

Synthetic frames only — no real PokerNow data."""

from poker_alpha.observer.live import CaptureSettings, LiveObserverSession
from poker_alpha.observer.pokernow import default_layout
from poker_alpha.observer.session import TestSessionRecorder
from poker_alpha.observer.synthetic import SyntheticSeat, SyntheticTable, render_table

CAL = default_layout(2, 0)
MON = {"index": 1, "left": 0, "top": 0, "width": 1280, "height": 800}


def table(board=(), pot=1.5, dealer=1, bet1=0.0, hero=("As", "Kd"), stacks=(99.0, 98.0)):
    return SyntheticTable(seats=[SyntheticSeat("h", stacks[0]),
                                 SyntheticSeat("v", stacks[1], bet=bet1)],
                          dealer=dealer, hero_cards=hero, board=board, pot=pot, actor=0)


class Screen:
    """Plays a fixed list of frames; ``None`` = capture failure."""

    def __init__(self, frames):
        self.frames, self.i = list(frames), 0

    def capture(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        if f is None:
            raise PermissionError("screen recording denied")
        return f


def frames_of(*tables, repeat=3, cal=CAL):
    out = []
    for t in tables:
        out += [render_table(t, cal)] * repeat
    return out


def record(root, frames, policy=None, limits=None, cal=CAL, actions=None, blinds=(0.5, 1.0)):
    """Run a recorder over ``frames``; ``actions[i]`` = callable(session, recorder)
    executed before step i (e.g. reset, recalibrate)."""
    scr = Screen(frames)
    s = LiveObserverSession(source_factory=lambda m, r: scr)
    s.configure(CaptureSettings(MON, None), cal, *blinds)
    rec = TestSessionRecorder(root, policy, limits)
    rec.start(cal, MON, None)
    for i in range(len(frames)):
        if actions and i in actions:
            actions[i](s, rec)
        rec.on_step(s, s.step(now=float(i)))
    return rec, s
