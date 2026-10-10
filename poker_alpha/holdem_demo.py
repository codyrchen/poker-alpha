"""Deterministic end-to-end Hold'em decision demo (no poker site needed).

    python -m poker_alpha.holdem_demo [--rollouts N] [--seed S]

1. builds a realistic 6-max hand with the rules engine,
2. deals the hero's cards (opponents' cards exist only inside the simulator),
3. projects the table onto the hero's view (ObservedTableState),
4. infers opponent ranges from position and actions,
5. applies blockers (hero cards and board),
6. computes hero equity against those ranges,
7. computes pot odds and SPR,
8. generates candidate actions from the betting abstraction,
9. evaluates them with common-random-number rollouts,
10. prints the resulting DecisionReport.

The recommendation is an approximate, range-based estimate — not GTO.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .decision import DecisionConfig, recommend_action
from .decision.recommend import betting_context, infer_opponent_range
from .holdem import (Action, SimulationStateAdapter, TableConfig, apply_action,
                     deal_board, position_names, start_hand, validate)
from .opponent import ARCHETYPE_MODELS, RangePriors
from .poker.cards import card_code, card_str
from .poker.multiway import multiway_equity
from .poker.ranges import WeightedRange

CHIP = 0.5  # engine chips -> big blinds (blinds 1/2 chips = 0.5/1 BB)


def build_hand():
    """6-max, 100 BB. Seats: 0 BTN, 1 SB, 2 BB, 3 UTG, 4 HJ, 5 CO (hero)."""
    C = card_code
    holes = [(C("9d"), C("8d")), (C("Kc"), C("4h")), (C("Ts"), C("9s")),
             (C("7c"), C("2d")), (C("Qd"), C("Qc")), (C("Ah"), C("Js"))]
    s = start_hand([200] * 6, dealer=0, config=TableConfig(1, 2),
                   hole_cards=holes, hand_id="demo-6max")
    script = [Action.fold(),            # UTG
              Action.raise_to(5),       # HJ opens 2.5 BB
              Action.call(),            # CO (hero) flats
              Action.fold(),            # BTN
              Action.fold(),            # SB
              Action.call()]            # BB defends
    for a in script:
        s = apply_action(s, a)
    s = deal_board(s, [C("Qh"), C("Jd"), C("5s")])
    for a in (Action.check(),           # BB
              Action.bet(10)):          # HJ c-bets 5 BB into 7.5 BB
        s = apply_action(s, a)
    return s


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", type=int, default=1500)
    p.add_argument("--equity-sims", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    state = build_hand()
    hero = state.actor
    obs = SimulationStateAdapter.observe(state, hero, chip_unit=CHIP)
    names = position_names(state.num_seats, state.dealer)
    print("=== 1-3. Table (hero's view; opponents' cards hidden) ===")
    print(obs.describe())
    for s in obs.seats:
        tag = " <- hero" if s.seat == hero else ""
        status = "folded" if s.folded else "in hand"
        print(f"  seat {s.seat} {names[s.seat]:<4} stack {s.stack:6.1f} BB  "
              f"bet {s.current_bet:4.1f}  {status}{tag}")
    issues = validate(obs)
    print(f"  validation: {'ok' if not issues else [i.message for i in issues]}")

    print("\n=== 4-5. Opponent ranges (beliefs) and blockers ===")
    priors = RangePriors.load()
    warnings: List[str] = []
    ranges = {}
    for seat in obs.opponents_in_hand:
        r, line = infer_opponent_range(obs, seat, priors,
                                       ARCHETYPE_MODELS["regular"], warnings)
        raw = priors.prior(names[seat], line)
        ranges[seat] = r
        top = ", ".join(f"{c} {m:.0%}" for c, m in r.top_classes(6))
        print(f"  seat {seat} {names[seat]} [{line}]: prior {raw.num_live_combos} "
              f"combos -> {r.num_live_combos} after blockers/updates "
              f"(~{r.effective_combos():.0f} effective); top: {top}")
    hero_cards = obs.hero_cards
    print(f"  hero {card_str(hero_cards[0])}{card_str(hero_cards[1])} blocks "
          f"{WeightedRange.uniform().num_live_combos - WeightedRange.uniform().remove_cards(hero_cards).num_live_combos} "
          f"of 1326 combos")

    print("\n=== 6-7. Equity, pot odds, SPR ===")
    eq = multiway_equity(hero_cards, obs.board, list(ranges.values()),
                         simulations=args.equity_sims, seed=args.seed)
    print(f"  hero equity vs ranges: {eq.expected_share:.1%} ± {eq.std_error:.1%} "
          f"({eq.simulations} sims)")
    print(f"  pot {obs.pot:.1f} BB, to call {obs.amount_to_call:.1f} BB, "
          f"pot odds {obs.pot_odds:.1%}, SPR {obs.spr:.2f}")

    print("\n=== 8. Candidate actions (betting abstraction) ===")
    cfg = DecisionConfig(equity_simulations=args.equity_sims,
                         rollout_simulations=args.rollouts, seed=args.seed)
    for label, a in cfg.abstraction.menu(betting_context(obs)):
        print(f"  {label:<10} {a.kind:<7} to {a.raise_to:6.2f} BB (adds {a.add:6.2f})")

    print("\n=== 9-10. Evaluation and DecisionReport ===")
    report = recommend_action(obs, config=cfg)
    print(report.format())
    print("\n(Analysis only: PokerAlpha never acts. Multiway EVs are approximate.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
