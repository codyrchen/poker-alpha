"""Experiment: CFR vs CFR+ vs external-sampling MCCFR on Kuhn Poker.

Trains all three solvers for the same number of iterations, snapshotting
exploitability on a log-spaced grid. Training is timed with the evaluation
clock stopped, so runtime reflects solving only.

Two views are produced: exploitability vs iterations (the algorithmic view)
and exploitability vs wall-clock seconds (the practical view). MCCFR's
iterations are much cheaper, so the two rankings can differ — on a game as
small as Kuhn the full tree is ~50 nodes and sampling buys little; the Leduc
experiment shows how that changes with game size.

Usage
-----
    python experiments/cfr_comparison.py --iterations 100000 --seed 42

Outputs (under --outdir, default ./results):
    data/cfr_comparison.csv / _runtime.csv
    figures/cfr_comparison.png / _time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poker_alpha.games import KuhnPoker
from poker_alpha.solvers import CFRPlusSolver, CFRSolver, MCCFRSolver
from poker_alpha.utils.comparison import compare_solvers
from poker_alpha.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--snapshots", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    game = KuhnPoker()
    compare_solvers(
        game,
        {
            "CFR": CFRSolver(game),
            "CFR+": CFRPlusSolver(game),
            "MCCFR": MCCFRSolver(game, seed=args.seed),
        },
        iterations=args.iterations,
        snapshots=args.snapshots,
        outdir=args.outdir,
        stem="cfr_comparison",
        title="CFR vs CFR+ vs MCCFR on Kuhn Poker",
    )


if __name__ == "__main__":
    main()
