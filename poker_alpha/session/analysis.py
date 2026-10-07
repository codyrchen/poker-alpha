"""Post-session analysis over a stored session.

Works on stored hands only, so it is equally usable after live-observed play
(where permitted) or purely post-hoc from hand histories and screenshots —
the mode to use wherever real-time assistance is not allowed.

* largest EV deviations — decisions where the action taken had a lower
  estimated EV than the best candidate (estimates carry SE; small gaps are
  noise, so each row reports the SE of the actual action's estimate);
* uncertain decisions — low-confidence reports or no clear best action;
* range-estimation changes — how much each opponent's range belief
  narrowed (entropy drop) between the hero's decisions within a hand;
* player tendency summaries — statistics rebuilt from stored hands;
* hands worth reviewing — the union of the above plus large pots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..opponent.statistics import PlayerStatistics, StatEstimate
from .serialize import summary_from_dict
from .store import SessionStore

KEY_STATS = ("vpip", "pfr", "three_bet", "flop_cbet", "fold_to_flop_cbet",
             "aggression", "wtsd", "wsd")


@dataclass(frozen=True)
class Deviation:
    hand_key: str
    street: str
    actual: str
    recommended: Optional[str]
    ev_loss_bb: float
    actual_se_bb: Optional[float]
    confidence: Optional[str]


@dataclass(frozen=True)
class RangeChange:
    hand_key: str
    seat: int
    from_street: str
    to_street: str
    entropy_before: float
    entropy_after: float

    @property
    def narrowing_bits(self) -> float:
        return self.entropy_before - self.entropy_after


@dataclass(frozen=True)
class SessionAnalysis:
    session_id: int
    hands: int
    decisions: int
    hero_net_bb: float
    deviations: Tuple[Deviation, ...]
    uncertain: Tuple[Tuple[str, str, str], ...]   # (hand, street, reason)
    range_changes: Tuple[RangeChange, ...]
    tendencies: Dict[str, Dict[str, StatEstimate]]
    review: Tuple[Tuple[str, str], ...]           # (hand, reason)

    def format(self) -> str:
        lines = [f"Session {self.session_id}: {self.hands} hands, "
                 f"{self.decisions} hero decisions, hero net "
                 f"{self.hero_net_bb:+.1f} BB"]
        lines.append("\nLargest EV deviations (estimates; check the SE):")
        for d in self.deviations[:10]:
            se = "" if d.actual_se_bb is None else f" ±{d.actual_se_bb:.2f}"
            lines.append(f"  {d.hand_key} {d.street}: played {d.actual}, "
                         f"best est. {d.recommended}; EV gap {d.ev_loss_bb:.2f} BB{se}"
                         f" [{d.confidence}]")
        if not self.deviations:
            lines.append("  none")
        lines.append("\nUncertain decisions:")
        lines.extend(f"  {h} {s}: {r}" for h, s, r in self.uncertain[:10])
        lines.append("\nRange narrowing between hero decisions:")
        for c in sorted(self.range_changes, key=lambda c: -c.narrowing_bits)[:10]:
            lines.append(f"  {c.hand_key} seat {c.seat} {c.from_street}->{c.to_street}: "
                         f"{c.entropy_before:.1f} -> {c.entropy_after:.1f} bits")
        lines.append("\nPlayer tendencies (raw count; posterior 95% CI):")
        for player, stats in sorted(self.tendencies.items()):
            parts = [s.format() for s in stats.values() if s.raw_opportunities]
            lines.append(f"  {player}: " + ("; ".join(parts) or "no data"))
        lines.append("\nHands worth reviewing:")
        lines.extend(f"  {h}: {r}" for h, r in self.review)
        return "\n".join(lines)


def analyze_session(store: SessionStore, session_id: int,
                    deviation_threshold_bb: float = 0.25,
                    big_pot_bb: float = 20.0) -> SessionAnalysis:
    hands = store.hands(session_id)
    decisions = store.decisions(session_id)
    devs: List[Deviation] = []
    uncertain: List[Tuple[str, str, str]] = []
    changes: List[RangeChange] = []
    review: Dict[str, List[str]] = {}
    prev_by_hand: Dict[str, dict] = {}
    for d in decisions:
        rep = d["report"]
        if rep is None:
            continue
        if d["ev_loss"] is not None and d["ev_loss"] > deviation_threshold_bb:
            se = None
            for c in rep["candidates"]:
                if c["label"] == d["actual_label"]:
                    se = c["ev_se_bb"]
            devs.append(Deviation(d["hand_key"], d["street"], d["actual_label"],
                                  d["recommended"], d["ev_loss"], se,
                                  d["confidence"]))
            review.setdefault(d["hand_key"], []).append(
                f"EV gap {d['ev_loss']:.2f} BB on {d['street'].lower()}")
        mix = rep.get("recommended_mix") or {}
        top = max(mix.values()) if mix else 0.0
        if rep["confidence"] == "low":
            reason = "low-confidence report"
            if mix and top < 0.6:
                reason += f" (best action only {top:.0%} likely best)"
            uncertain.append((d["hand_key"], d["street"], reason))
        prev = prev_by_hand.get(d["hand_key"])
        if prev is not None:
            before = {o["seat"]: o for o in prev["report"]["opponent_ranges"]}
            for o in rep["opponent_ranges"]:
                if o["seat"] in before:
                    changes.append(RangeChange(
                        d["hand_key"], o["seat"], prev["street"], d["street"],
                        before[o["seat"]]["entropy_bits"], o["entropy_bits"]))
        prev_by_hand[d["hand_key"]] = d

    stats = PlayerStatistics()
    net_bb = 0.0
    for h in hands:
        stats.add_hand(summary_from_dict(h["summary"]))
        if h["hero_net"] is not None:
            net_bb += h["hero_net"] / h["big_blind"]
            if abs(h["hero_net"]) / h["big_blind"] >= big_pot_bb:
                review.setdefault(h["hand_key"], []).append(
                    f"big pot ({h['hero_net'] / h['big_blind']:+.1f} BB)")
    for h, s, r in uncertain:
        review.setdefault(h, []).append(f"uncertain {s.lower()} decision")
    tendencies = {p: {s: stats.estimate(p, s) for s in KEY_STATS}
                  for p in stats.players()}
    devs.sort(key=lambda d: -d.ev_loss_bb)
    return SessionAnalysis(
        session_id=session_id, hands=len(hands), decisions=len(decisions),
        hero_net_bb=net_bb, deviations=tuple(devs), uncertain=tuple(uncertain),
        range_changes=tuple(changes), tendencies=tendencies,
        review=tuple((h, "; ".join(rs)) for h, rs in review.items()))
