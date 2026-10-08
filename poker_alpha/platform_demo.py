"""End-to-end demo of the normalized analysis path (Phase 31).

    python -m poker_alpha.platform_demo [--strategy results/strategy/holdem_v1.npz]
                                        [--rollouts N] [--source all|manual|...]

Runs every input source through the same path

    source -> ObservedTableState -> validate -> recommend_action -> DecisionReport

and prints each report with its decision-source cascade and uncertainty
breakdown:

* manual       — heads-up 100 BB spot typed as a dict (solver-eligible);
* simulation   — the 6-max rules-engine hand of ``holdem_demo`` (multiway:
                 the HU solver must refuse it; rollouts or heuristic answer);
* hand_history — first hero decision of ``tests/fixtures/hands/sample.json``;
* screenshot   — the synthetic fixture image ``tests/fixtures/table.png``
                 (read-only recognition; not real PokerNow accuracy).

The solver strategy is an abstract heads-up MCCFR strategy on an
IMPERFECT-RECALL abstraction: no equilibrium guarantee, not GTO.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "results" / "strategy" / "holdem_v1_seed0.npz"


def manual_hu_spot():
    return {"num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
            "big_blind": 1.0, "hero_cards": "Ah Qd", "board": "", "actor": 0,
            "seats": [{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                      {"stack": 99.0, "bet": 1.0, "committed": 1.0}],
            "pot": 1.5, "actions": [], "hand_id": "demo-hu"}


def main(argv: Optional[List[str]] = None) -> int:
    from .decision import DecisionConfig
    from .pipeline import (analyze, load_solver, observe_hand_history, observe_manual,
                           observe_screenshot, observe_simulation, validation_issues)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    p.add_argument("--min-visits", type=float, default=20.0)
    p.add_argument("--rollouts", type=int, default=400)
    p.add_argument("--equity-sims", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", default="all",
                   choices=("all", "manual", "simulation", "hand_history", "screenshot"))
    args = p.parse_args(argv)

    solver = load_solver(args.strategy, min_visits=args.min_visits)
    if solver is None or hasattr(solver, "code"):
        print(f"solver strategy unavailable: {getattr(solver, 'reason', 'none given')}")
    else:
        print(f"solver strategy: {solver.description}")
    cfg = DecisionConfig(equity_simulations=args.equity_sims,
                         rollout_simulations=args.rollouts, seed=args.seed)

    sources = []
    if args.source in ("all", "manual"):
        sources.append(("manual", lambda: observe_manual(manual_hu_spot())))
    if args.source in ("all", "simulation"):
        def sim():
            from .holdem_demo import CHIP, build_hand
            st = build_hand()
            return observe_simulation(st, st.actor, chip_unit=CHIP)
        sources.append(("simulation", sim))
    if args.source in ("all", "hand_history"):
        def hh():
            from .history import load_hands
            return observe_hand_history(load_hands(ROOT / "tests" / "fixtures" / "hands"
                                                   / "sample.json")[0])
        sources.append(("hand_history", hh))
    if args.source in ("all", "screenshot"):
        sources.append(("screenshot",
                        lambda: observe_screenshot(ROOT / "tests" / "fixtures" / "table.png")))

    for name, make in sources:
        print(f"\n{'=' * 72}\nSOURCE: {name}\n{'=' * 72}")
        obs = make()
        for issue in validation_issues(obs):
            print(f"validation {issue}")
        for n in obs.notes:
            print(f"note: {n}")
        rep = analyze(obs, replace(cfg), solver=solver)
        print(rep.format())
    print("\nRead-only analysis. Not GTO, not a solved game, not a profitable bot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
