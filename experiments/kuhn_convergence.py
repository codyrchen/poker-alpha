"""Experiment: CFR convergence on Kuhn Poker.

Trains vanilla CFR on Kuhn Poker, tracing exploitability against the number of
iterations, and reports the learned average strategy and game value. Compares
the converged value against the known Nash value of -1/18.

Usage
-----
    python experiments/kuhn_convergence.py --iterations 100000 --seed 42

Outputs (under --outdir, default ./results):
    data/kuhn_convergence.csv     iteration, exploitability
    figures/kuhn_convergence.png  exploitability vs iterations (log-log)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from poker_alpha.games import KuhnPoker
from poker_alpha.solvers import CFRSolver, exploitability, expected_value
from poker_alpha.utils.plotting import save_convergence_plot
from poker_alpha.utils.seeds import set_seed

KUHN_NASH_VALUE = -1.0 / 18.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--snapshot-every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    game = KuhnPoker()
    solver = CFRSolver(game)

    history = solver.train(
        iterations=args.iterations,
        snapshot_every=args.snapshot_every,
        evaluator=lambda strat: {"exploitability": exploitability(game, strat)},
    )

    avg = solver.average_strategy()
    value = expected_value(game, avg)
    final_expl = exploitability(game, avg)

    # Save raw data.
    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    csv_path = data_dir / "kuhn_convergence.csv"
    df.to_csv(csv_path, index=False)

    # Save figure (log-log makes the O(1/sqrt(T))-style decay legible).
    fig_path = save_convergence_plot(
        series=[("CFR", df["iteration"], df["exploitability"])],
        xlabel="iterations",
        ylabel="exploitability (chips)",
        title="CFR convergence on Kuhn Poker",
        path=args.outdir / "figures" / "kuhn_convergence.png",
        logx=True,
        logy=True,
    )

    print("== Kuhn Poker: CFR convergence ==")
    print(f"iterations           : {args.iterations:,}")
    print(f"game value (player 0): {value:+.6f}  (known Nash = {KUHN_NASH_VALUE:+.6f})")
    print(f"final exploitability : {final_expl:.6e}")
    print("\naverage strategy  (p = pass/check/fold, b = bet/call):")
    for key in sorted(avg):
        probs = ", ".join(f"{a}={p:.3f}" for a, p in avg[key].items())
        print(f"  {key:>4} : {probs}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
