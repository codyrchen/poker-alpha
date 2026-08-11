"""Experiment: CFR vs CFR+ on Kuhn Poker.

Trains both solvers for the same number of iterations, tracing exploitability
at snapshots, and compares convergence and wall-clock runtime. (MCCFR joins
this comparison in a later phase.)

Usage
-----
    python experiments/cfr_comparison.py --iterations 100000 --seed 42

Outputs (under --outdir, default ./results):
    data/cfr_comparison.csv          solver, iteration, exploitability
    data/cfr_comparison_runtime.csv  solver, iterations, seconds, iters_per_sec
    figures/cfr_comparison.png       exploitability vs iterations (log-log)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from poker_alpha.games import KuhnPoker
from poker_alpha.solvers import CFRPlusSolver, CFRSolver, exploitability
from poker_alpha.utils.plotting import save_convergence_plot
from poker_alpha.utils.seeds import set_seed


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

    solvers = {
        "CFR": CFRSolver(game),
        "CFR+": CFRPlusSolver(game),
    }

    rows = []
    runtimes = []
    for name, solver in solvers.items():
        evaluator = lambda strat: {"exploitability": exploitability(game, strat)}
        t0 = time.perf_counter()
        history = solver.train(args.iterations,
                               snapshot_every=args.snapshot_every,
                               evaluator=evaluator)
        elapsed = time.perf_counter() - t0
        for record in history:
            rows.append({"solver": name, **record})
        runtimes.append({
            "solver": name,
            "iterations": args.iterations,
            "seconds": round(elapsed, 3),
            "iters_per_sec": round(args.iterations / elapsed, 1),
        })
        final = history[-1]["exploitability"] if history else float("nan")
        print(f"{name:>5}: {elapsed:6.2f}s "
              f"({args.iterations / elapsed:,.0f} it/s), "
              f"final exploitability {final:.3e}")

    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "cfr_comparison.csv", index=False)
    pd.DataFrame(runtimes).to_csv(data_dir / "cfr_comparison_runtime.csv",
                                  index=False)

    series = []
    for name in solvers:
        sub = df[df["solver"] == name]
        series.append((name, sub["iteration"].tolist(),
                       sub["exploitability"].tolist()))
    fig_path = save_convergence_plot(
        series=series,
        xlabel="iterations",
        ylabel="exploitability (chips)",
        title="CFR vs CFR+ convergence on Kuhn Poker",
        path=args.outdir / "figures" / "cfr_comparison.png",
        logx=True,
        logy=True,
    )
    print(f"wrote {data_dir / 'cfr_comparison.csv'}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
