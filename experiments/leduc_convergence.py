"""Experiment: CFR / CFR+ / MCCFR convergence on Leduc Poker.

Leduc is ~75x larger than Kuhn by decision nodes (3,780 vs ~50), so this is
where solver scaling starts to matter: full-traversal iterations get expensive
and MCCFR's sampled iterations start paying for themselves per unit of
wall-clock time.

Usage
-----
    python experiments/leduc_convergence.py --iterations 1000 --seed 42

Outputs (under --outdir, default ./results):
    data/leduc_convergence.csv / _runtime.csv
    figures/leduc_convergence.png / _time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poker_alpha.games.leduc import LeducPoker
from poker_alpha.solvers import CFRPlusSolver, CFRSolver, MCCFRSolver, expected_value
from poker_alpha.utils.comparison import compare_solvers
from poker_alpha.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=1_000,
                   help="full-traversal iterations for CFR/CFR+")
    p.add_argument("--mccfr-iterations", type=int, default=200_000,
                   help="sampled iterations for MCCFR (cheaper each)")
    p.add_argument("--snapshots", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    game = LeducPoker()

    # Exact solvers on their budget.
    df = compare_solvers(
        game,
        {"CFR": CFRSolver(game), "CFR+": CFRPlusSolver(game)},
        iterations=args.iterations,
        snapshots=args.snapshots,
        outdir=args.outdir,
        stem="leduc_convergence_exact",
        title="CFR vs CFR+ on Leduc Poker",
    )
    # MCCFR on its own (much larger) iteration budget; separate stem so the
    # per-iteration axes stay comparable within each figure.
    compare_solvers(
        game,
        {"MCCFR": MCCFRSolver(game, seed=args.seed)},
        iterations=args.mccfr_iterations,
        snapshots=args.snapshots,
        outdir=args.outdir,
        stem="leduc_convergence_mccfr",
        title="External-sampling MCCFR on Leduc Poker",
    )

    best = df[df["solver"] == "CFR+"].iloc[-1]
    solver = CFRPlusSolver(game)  # small re-solve for the value readout
    solver.train(min(args.iterations, 300))
    value = expected_value(game, solver.average_strategy())
    print(f"\nLeduc game value (player 0), CFR+ estimate: {value:+.4f}")
    print(f"CFR+ final exploitability at {int(best['iteration'])} iterations: "
          f"{best['exploitability']:.3e}")


if __name__ == "__main__":
    main()
