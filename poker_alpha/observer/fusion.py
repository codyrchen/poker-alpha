"""Observation fusion: noisy frames -> a trustworthy tracked table state.

:class:`StateTracker` owns one :class:`~poker_alpha.observer.tracker.FieldTracker`
per field and adds poker-aware stability rules on top of per-field
smoothing. Impossible transitions are *rejected and reported*, never
silently accepted:

* hero cards must agree across ``card_confirm_frames`` frames;
* the pot updates only on high confidence or repeated agreement;
* a board card cannot disappear or change mid-hand, and cards appear in
  order (flop, turn, river);
* a stack may only grow when the pot is awarded (pot drops) or a new hand
  starts;
* the dealer button only moves between hands (a move is accepted as the
  start of a new hand only together with a cleared board).

The user stays in control: :meth:`pause` stops ingesting frames,
:meth:`correct` pins a field to a typed value (confidence 1, source
"manual"), :meth:`release` hands it back to recognition, :meth:`resume`
continues. :meth:`to_observed_state` produces the normalized
:class:`~poker_alpha.holdem.observed.ObservedTableState` plus per-field
confidences and warnings for the decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..holdem.observed import (ObservedAction, ObservedSeat,
                               ObservedTableState)
from ..holdem.state import Street
from ..poker.cards import card_code
from .calibration import TableCalibration
from .pokernow import FieldReading, FrameObservation
from .tracker import FieldTracker, TableEvent, TableSnapshot, infer_events


@dataclass
class TrackerConfig:
    confirm_frames: int = 2
    card_confirm_frames: int = 3
    high_confidence: float = 0.9
    min_confidence: float = 0.3
    stack_increase_confirm: int = 4
    # A confirmed hero card can only change within a hand after this many
    # agreeing frames (normally a new hand clears the cards first).
    hero_change_confirm: int = 6
    # A bet can only shrink right after a collection (pot rose / board grew,
    # within this many frames) or a new hand; otherwise after
    # ``stack_increase_confirm`` agreeing frames.
    collection_window: int = 3


@dataclass(frozen=True)
class TrackedTableState:
    snapshot: TableSnapshot
    field_confidence: Dict[str, float]
    manual_fields: Tuple[str, ...]
    hand_number: int
    paused: bool
    warnings: Tuple[str, ...]


class StateTracker:
    def __init__(self, calibration: TableCalibration, small_blind: float,
                 big_blind: float, config: Optional[TrackerConfig] = None) -> None:
        self.cal = calibration
        self.sb, self.bb = float(small_blind), float(big_blind)
        self.cfg = config or TrackerConfig()
        self.paused = False
        self.hand_number = 0
        self.events: List[TableEvent] = []
        self.flags: List[str] = []
        self.actions: List[ObservedAction] = []
        self.timestamp: Optional[float] = None
        self._prev: Optional[TableSnapshot] = None
        self._frame_no = 0
        self._last_collection = -10**9       # frame number of the last pot collection
        self.fields: Dict[str, FieldTracker] = {}
        c = self.cfg
        for name in ["pot", "dealer", "actor"]:
            self.fields[name] = FieldTracker(name, c.confirm_frames,
                                             c.high_confidence, c.min_confidence)
        for i in range(5):
            self.fields[f"board_{i}"] = FieldTracker(
                f"board_{i}", c.card_confirm_frames, c.high_confidence,
                c.min_confidence, always_confirm=True)
        for i in range(2):
            self.fields[f"hero_card_{i}"] = FieldTracker(
                f"hero_card_{i}", c.card_confirm_frames, c.high_confidence,
                c.min_confidence, always_confirm=True)
        for s in range(calibration.num_seats):
            for k in ("stack", "bet", "in_hand", "occupied", "all_in"):
                name = f"seat{s}.{k}"
                self.fields[name] = FieldTracker(name, c.confirm_frames,
                                                 c.high_confidence, c.min_confidence)

    # -- user control --------------------------------------------------------

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def correct(self, name: str, value) -> None:
        """Pin ``name`` to a user-supplied value until :meth:`release`."""
        if name not in self.fields:
            raise KeyError(name)
        self.fields[name].pin(value)
        self.flags.append(f"manual correction: {name} = {value!r}")

    def release(self, name: str) -> None:
        self.fields[name].unpin()

    # -- ingestion -----------------------------------------------------------

    def _stable(self, name: str, default=None):
        f = self.fields[name]
        return f.stable if f.has_value else default

    def _new_hand(self, reason: str) -> None:
        self.hand_number += 1
        self.actions.clear()
        for name, f in self.fields.items():
            if name.startswith(("board_", "hero_card_")) or name.endswith(".bet"):
                if not f.pinned:
                    keep = name.startswith("hero_card_")   # new cards already seen count
                    f.has_value, f.stable = False, None
                    if not keep:
                        f.candidate, f.candidate_count = None, 0
        self.events.append(TableEvent("new_hand", detail=reason))

    def update(self, frame: FrameObservation) -> List[TableEvent]:
        """Ingest one frame; return the events it produced."""
        if self.paused:
            return []
        self.timestamp = frame.timestamp
        before = len(self.events)
        fr = frame.fields
        self._frame_no += 1
        board_len_before = sum(1 for i in range(5) if self._stable(f"board_{i}") is not None)

        # New-hand detection first: a confirmed dealer move with an empty
        # board starts a hand; a move with cards still out is suspicious.
        d = fr.get("dealer")
        cur_dealer = self._stable("dealer")
        if d is not None and d.value is not None and cur_dealer is not None \
                and d.value != cur_dealer:
            board_empty = all(fr[f"board_{i}"].value is None for i in range(5))
            df = self.fields["dealer"]
            if df.update(d) and board_empty:
                self._new_hand("dealer moved")
            elif df.has_value and df.stable != cur_dealer and not board_empty:
                df.accept(cur_dealer, df.stable_confidence)
                self.flags.append("rejected dealer move while board cards are out")
        elif d is not None:
            self.fields["dealer"].update(d)

        # Board: no disappearing/changed cards, fill in order.
        for i in range(5):
            name = f"board_{i}"
            reading = fr.get(name)
            if reading is None:
                continue
            stable = self._stable(name)
            if stable is not None and reading.value != stable:
                # A cleared board is a new hand only if NEW hero cards are
                # visible: an unreadable frame (every card "absent") is not.
                if reading.value is None and all(fr[f"board_{j}"].value is None
                                                 for j in range(5)) \
                        and all(fr[f"hero_card_{j}"].value is not None
                                and fr[f"hero_card_{j}"].value != self._stable(f"hero_card_{j}")
                                for j in range(2)):
                    self._new_hand("board cleared and hero cards changed")
                    break
                if reading.confidence >= self.cfg.min_confidence:
                    self.flags.append(f"rejected {name} change {stable}->{reading.value}")
                continue
            if stable is None and reading.value is not None:
                earlier_missing = any(
                    self._stable(f"board_{j}") is None
                    and (f"board_{j}" not in fr or fr[f"board_{j}"].value is None)
                    for j in range(i))
                if i >= 3 and earlier_missing:
                    self.flags.append(f"rejected {name} before earlier board cards")
                    continue
                self.fields[name].update(reading)
        for i in range(2):
            name = f"hero_card_{i}"
            if name not in fr:
                continue
            reading, f = fr[name], self.fields[name]
            stable = self._stable(name)
            if stable is not None and reading.value != stable and not f.pinned:
                # Hero cards cannot change (or vanish) within a hand; a new
                # hand clears them first. Hold the change (occlusion, glare,
                # a misread suit) unless it persists for many frames.
                if reading.confidence < self.cfg.min_confidence:
                    f.rejected += 1
                    continue
                if reading.value == f.candidate:
                    f.candidate_count += 1
                else:
                    f.candidate, f.candidate_count = reading.value, 1
                if f.candidate_count >= self.cfg.hero_change_confirm:
                    self.flags.append(f"accepted {name} change {stable}->{reading.value} "
                                      f"after {f.candidate_count} frames without a new hand")
                    f.accept(reading.value, reading.confidence, frame.timestamp)
                elif f.candidate_count == 1:
                    self.flags.append(f"held {name} change {stable}->{reading.value}: "
                                      "no new hand seen")
                continue
            f.update(reading)

        # Pot (award detection uses the pot dropping).
        pot_before = self._stable("pot")
        if "pot" in fr:
            self.fields["pot"].update(fr["pot"])
        pot_dropped = (pot_before is not None and self._stable("pot") is not None
                       and self._stable("pot") < pot_before)
        pot_rose = (pot_before is not None and self._stable("pot") is not None
                    and self._stable("pot") > pot_before)
        board_len = sum(1 for i in range(5) if self._stable(f"board_{i}") is not None)
        established = self._prev is not None and self._prev.hero_cards[0] is not None
        if pot_rose or (established and board_len > board_len_before):
            self._last_collection = self._frame_no
        collected = self._frame_no - self._last_collection < self.cfg.collection_window

        for s in range(self.cal.num_seats):
            in_hand = bool(self._stable(f"seat{s}.in_hand", False))
            for k in ("occupied", "all_in", "in_hand"):
                name = f"seat{s}.{k}"
                if name not in fr:
                    continue
                if k == "occupied" and in_hand and fr[name].value is False \
                        and self._stable(name) is True:
                    self._hold(name, fr[name], frame.timestamp,
                               f"seat {s} vacating while in the hand")
                    continue
                self.fields[name].update(fr[name])
            name = f"seat{s}.bet"
            reading = fr.get(name)
            if reading is not None:
                f = self.fields[name]
                stable = self._stable(name)
                if stable is not None and reading.value is not None and not f.pinned \
                        and reading.value < stable - 1e-9 and not collected:
                    # Bets only shrink when collected into the pot (pot rises /
                    # board grows) or at a new hand; hold anything else.
                    if reading.confidence < self.cfg.min_confidence:
                        f.rejected += 1
                    else:
                        if reading.value == f.candidate:
                            f.candidate_count += 1
                        else:
                            f.candidate, f.candidate_count = reading.value, 1
                        if f.candidate_count >= self.cfg.stack_increase_confirm:
                            self.flags.append(f"accepted bet decrease on seat {s} "
                                              f"({stable} -> {reading.value}) without a "
                                              f"collection after {f.candidate_count} frames")
                            f.accept(reading.value, reading.confidence, frame.timestamp)
                        elif f.candidate_count == 1:
                            self.flags.append(f"held bet decrease on seat {s} "
                                              f"({stable} -> {reading.value}): no collection seen")
                else:
                    f.update(reading)
            name = f"seat{s}.stack"
            reading = fr.get(name)
            if reading is None:
                continue
            stable = self._stable(name)
            if stable is not None and reading.value is None and in_hand:
                # an occluded / unreadable stack of a player in the hand
                self._hold(name, reading, frame.timestamp,
                           f"seat {s} stack disappearing while in the hand")
                continue
            if stable is not None and reading.value is not None and \
                    reading.value > stable + 1e-9 and not pot_dropped:
                f = self.fields[name]
                if reading.value == f.candidate:
                    f.candidate_count += 1
                else:
                    f.candidate, f.candidate_count = reading.value, 1
                if f.candidate_count >= self.cfg.stack_increase_confirm and \
                        reading.confidence >= self.cfg.min_confidence:
                    f.accept(reading.value, reading.confidence, frame.timestamp)
                    self.flags.append(f"accepted unexplained stack increase on "
                                      f"seat {s} after {f.candidate_count} frames")
                elif f.candidate_count == 1:
                    self.flags.append(f"held stack increase on seat {s} "
                                      f"({stable} -> {reading.value}): no award seen")
                continue
            self.fields[name].update(reading)
        self.fields["actor"].update(fr.get("actor", FieldReading(None, 0.0, "actor")))

        snap = self.snapshot()
        # Only diff against an *established* baseline: the jump from "nothing
        # confirmed yet" to the first stable frame is not a sequence of actions.
        if self._prev is not None and self._prev.pot is not None \
                and self._prev.hero_cards[0] is not None:
            new = infer_events(self._prev, snap)
            for e in new:
                if e.kind == "new_hand":
                    continue  # handled above with stronger evidence
                self.events.append(e)
                self._record_action(e, self._prev)
        self._prev = snap
        return self.events[before:]

    def _hold(self, name: str, reading: FieldReading, timestamp, what: str) -> None:
        """Hold an implausible change; accept it only if it persists for
        ``stack_increase_confirm`` frames (flagged either way)."""
        f = self.fields[name]
        if f.pinned:
            return
        if reading.confidence < self.cfg.min_confidence:
            f.rejected += 1
            return
        if reading.value == f.candidate:
            f.candidate_count += 1
        else:
            f.candidate, f.candidate_count = reading.value, 1
        if f.candidate_count >= self.cfg.stack_increase_confirm:
            self.flags.append(f"accepted {what} after {f.candidate_count} frames")
            f.accept(reading.value, reading.confidence, timestamp)
        elif f.candidate_count == 1:
            self.flags.append(f"held {what}")

    def _record_action(self, e: TableEvent, prev: TableSnapshot) -> None:
        street = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(prev.board), 0)
        if e.kind == "fold":
            self.actions.append(ObservedAction(street, e.seat, "fold"))
        elif e.kind == "bet":
            facing = max(prev.bets) > 0
            kind = "raise" if (facing or street == 0) else "bet"
            self.actions.append(ObservedAction(street, e.seat, kind, e.amount))

    # -- outputs -------------------------------------------------------------

    def snapshot(self) -> TableSnapshot:
        n = self.cal.num_seats
        board = tuple(self._stable(f"board_{i}") for i in range(5))
        board = tuple(c for c in board if c is not None)
        return TableSnapshot(
            pot=self._stable("pot"),
            board=board,
            hero_cards=(self._stable("hero_card_0"), self._stable("hero_card_1")),
            dealer=self._stable("dealer"),
            actor=self._stable("actor"),
            stacks=tuple(self._stable(f"seat{s}.stack") for s in range(n)),
            bets=tuple(float(self._stable(f"seat{s}.bet", 0.0) or 0.0) for s in range(n)),
            in_hand=tuple(bool(self._stable(f"seat{s}.in_hand", False)) for s in range(n)),
            occupied=tuple(bool(self._stable(f"seat{s}.occupied", False)) for s in range(n)),
            all_in=tuple(bool(self._stable(f"seat{s}.all_in", False)) for s in range(n)),
            timestamp=self.timestamp)

    def tracked(self) -> TrackedTableState:
        conf = {k: (f.stable_confidence if f.has_value else 0.0)
                for k, f in self.fields.items()}
        manual = tuple(sorted(k for k, f in self.fields.items() if f.pinned))
        return TrackedTableState(self.snapshot(), conf, manual, self.hand_number,
                                 self.paused, tuple(self.flags[-20:]))

    def critical_confidence(self) -> float:
        """Weakest confidence among fields a decision depends on."""
        hero = self.cal.hero_seat
        names = ["hero_card_0", "hero_card_1", "pot", f"seat{hero}.stack"]
        snap = self.snapshot()
        names += [f"board_{i}" for i in range(len(snap.board))]
        names += [f"seat{s}.bet" for s in range(self.cal.num_seats)
                  if snap.occupied[s] and snap.in_hand[s]]
        vals = []
        for n in names:
            f = self.fields[n]
            vals.append(1.0 if f.pinned else (f.stable_confidence if f.has_value else 0.0))
        return min(vals) if vals else 0.0

    def _pot_total(self, snap) -> Optional[float]:
        """All chips in the middle. Clients whose pot display leaves out the
        bets still in front of the players (PokerNow) get those added."""
        if snap.pot is None or self.cal.pot_includes_bets:
            return snap.pot
        return snap.pot + sum(b for s, b in enumerate(snap.bets)
                              if snap.occupied[s] and b)

    def to_observed_state(self) -> ObservedTableState:
        snap = self.snapshot()
        seats = []
        for s in range(self.cal.num_seats):
            occupied = snap.occupied[s]
            seats.append(ObservedSeat(
                seat=s, occupied=occupied,
                stack=snap.stacks[s] if occupied else None,
                current_bet=snap.bets[s] if occupied else 0.0,
                folded=occupied and not snap.in_hand[s],
                all_in=snap.all_in[s]))
        hero = tuple(card_code(c) for c in snap.hero_cards if c is not None)
        street = {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN,
                  5: Street.RIVER}.get(len(snap.board), Street.PREFLOP)
        dealer = snap.dealer if snap.dealer is not None else self.cal.hero_seat
        # Bets visible on the table are facts even when the frames that showed
        # them being made were missed (e.g. a single screenshot). Add them as
        # actions in ascending size order (the true order is unknown); blinds
        # (preflop bets up to the big blind) are posts, not actions.
        actions = list(self.actions)
        cur = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(snap.board), 0)
        recorded = {a.seat for a in actions if a.street == cur}
        visible = sorted((snap.bets[s], s) for s in range(self.cal.num_seats)
                         if snap.occupied[s] and snap.bets[s] > 0
                         and s not in recorded and s != self.cal.hero_seat
                         and not (cur == 0 and snap.bets[s] <= self.bb))
        level = self.bb if cur == 0 else 0.0
        for amount, seat in visible:
            if amount > level + 1e-9:
                kind = "raise" if (cur == 0 or level > 0) else "bet"
                level = amount
            else:
                kind = "call"
            actions.append(ObservedAction(cur, seat, kind, amount))
        return ObservedTableState(
            num_seats=self.cal.num_seats, hero_seat=self.cal.hero_seat,
            dealer=dealer, seats=tuple(seats), street=street,
            small_blind=self.sb, big_blind=self.bb,
            hero_cards=hero if len(hero) == 2 else None,
            board=tuple(card_code(c) for c in snap.board),
            pot_total=self._pot_total(snap), actor=snap.actor,
            action_history=tuple(actions),
            hand_id=f"observed-{self.hand_number}",
            timestamp=snap.timestamp, source="screen observer")
