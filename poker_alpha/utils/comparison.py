"""Shared harness for solver-comparison experiments.

Trains several solvers on the same game with log-spaced exploitability
snapshots, timing the training in chunks with the evaluation clock stopped so
runtime reflects solving only. Used by the Kuhn and Leduc comparison
experiments (and any later game).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..games.base import Game
from ..solvers import exploitability
from ..solvers.cfr import CFRSolver
from .plotting import save_convergence_plot


def snapshot_grid(total: int, points: int) -> List[int]:
    """Log-spaced iteration checkpoints from 1 to ``total``."""
    return np.unique(np.geomspace(1, total, points).astype(int)).tolist()


def compare_solvers(
    game: Game,
    solvers: Dict[str, CFRSolver],
    iterations: int,
    snapshots: int,
    outdir: Path,
    stem: str,
    title: str,
) -> pd.DataFrame:
    """Run the comparison; write CSVs and two figures; return the tidy frame.

    Outputs under ``outdir``:
      data/{stem}.csv              solver, iteration, seconds, exploitability
      data/{stem}_runtime.csv      per-solver totals
      figures/{stem}.png           exploitability vs iterations (log-log)
      figures/{stem}_time.png      exploitability vs seconds (log-log)
    """
    grid = snapshot_grid(iterations, snapshots)
    rows = []
    totals = []
    for name, solver in solvers.items():
        trained = 0
        solve_seconds = 0.0
        for target in grid:
            chunk = target - trained
            if chunk > 0:
                t0 = time.perf_counter()
                solver.train(chunk)
                solve_seconds += time.perf_counter() - t0
                trained = target
            expl = exploitability(game, solver.average_strategy())
            rows.append({"solver": name, "iteration": trained,
                         "seconds": round(solve_seconds, 4),
                         "exploitability": expl})
        totals.append({
            "solver": name,
            "iterations": trained,
            "seconds": round(solve_seconds, 3),
            "iters_per_sec": round(trained / solve_seconds, 1),
            "infosets": len(solver.infosets),
        })
        print(f"{name:>6}: {solve_seconds:8.2f}s "
              f"({trained / solve_seconds:>9,.1f} it/s), "
              f"final exploitability {rows[-1]['exploitability']:.3e}")

    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(data_dir / f"{stem}.csv", index=False)
    pd.DataFrame(totals).to_csv(data_dir / f"{stem}_runtime.csv", index=False)

    by_iter, by_time = [], []
    for name in solvers:
        sub = df[df["solver"] == name]
        by_iter.append((name, sub["iteration"].tolist(),
                        sub["exploitability"].tolist()))
        by_time.append((name, sub["seconds"].tolist(),
                        sub["exploitability"].tolist()))
    fig1 = save_convergence_plot(
        by_iter, "iterations", "exploitability (chips)",
        f"{title} (per iteration)",
        outdir / "figures" / f"{stem}.png", logx=True, logy=True)
    fig2 = save_convergence_plot(
        by_time, "training wall-clock seconds", "exploitability (chips)",
        f"{title} (per second)",
        outdir / "figures" / f"{stem}_time.png", logx=True, logy=True)
    print(f"wrote {data_dir / f'{stem}.csv'}")
    print(f"wrote {fig1}")
    print(f"wrote {fig2}")
    return df
