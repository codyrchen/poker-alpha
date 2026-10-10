"""Replay a PokerAlpha JSON hand history and analyse every hero decision.

Usage::

    python -m poker_alpha.replay hand.json [--hero SEAT] [--recommend]
                                           [--rollouts N] [--seed S]

Without ``--recommend`` it prints the reconstructed hero decision points;
with it, a :class:`~poker_alpha.decision.report.DecisionReport` per decision
and the actual action beside the recommendation (post-hand analysis).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .history import load_hands, replay_hand


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--hero", type=int, default=None)
    p.add_argument("--recommend", action="store_true")
    p.add_argument("--rollouts", type=int, default=0)
    p.add_argument("--equity-sims", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    hands = load_hands(args.path)
    for k, events in enumerate(hands):
        res = replay_hand(events, hero_seat=args.hero, hand_index=k)
        print(f"=== hand {res.hand_id or k} ({res.started.num_seats} seats) ===")
        for w in res.warnings:
            print(f"WARNING: {w}")
        if not res.decisions:
            print("(no hero decisions)")
        for d in res.decisions:
            amount = "" if d.amount is None else f" {d.amount:g}"
            print(f"\n[{d.state.describe()}]")
            print(f"hero action: {d.action}{amount}")
            if args.recommend:
                from .decision import DecisionConfig, recommend_action

                rep = recommend_action(d.state, config=DecisionConfig(
                    equity_simulations=args.equity_sims,
                    rollout_simulations=args.rollouts, seed=args.seed))
                print(rep.format())
                actual = _actual_label(d, rep)
                if actual and rep.recommended:
                    print(f"actual: {actual}  recommended: {rep.recommended}")
        nets = ", ".join(f"seat {s}: {v:+g}" for s, v in sorted(res.net.items()))
        print(f"\nresult: {nets}")
    return 0


def _actual_label(decision, report) -> Optional[str]:
    """Closest candidate label to what the hero actually did."""
    kind = decision.action
    if kind in ("fold", "check", "call"):
        return kind
    best, gap = None, float("inf")
    for c in report.candidates:
        if c.kind in ("bet", "raise", "all_in") and decision.amount is not None:
            g = abs(c.amount_to - decision.amount)
            if g < gap:
                best, gap = c.label, g
    if kind == "all_in":
        return "all_in"
    return best


if __name__ == "__main__":
    sys.exit(main())
