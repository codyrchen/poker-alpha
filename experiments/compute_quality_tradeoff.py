"""How much strategy quality does a fixed compute budget buy?

The quant-relevant framing of phase 17: a decision system rarely gets to run
to convergence. It gets a latency budget. This experiment measures what
PokerAlpha can actually deliver inside one.

Two workloads, both on Leduc where the quality metric is exact:

**A. Solver quality vs wall-clock budget.** CFR+ and external-sampling MCCFR
are each trained in small chunks until a wall-clock budget is exhausted, then
scored by *exploitability* (distance from Nash, in chips) with the training
clock stopped so evaluation never counts against the budget. This is the
imperfect-information-game analogue of "how good a decision can I make in
X ms" — not a production trading latency benchmark.

**B. Equity accuracy vs latency.** Monte Carlo equity error shrinks as
1/sqrt(n), so halving the error costs 4x the compute. That makes evaluator
throughput directly purchasable as accuracy: the phase-17 speedup means a
fixed latency budget now buys ~7x more simulations, i.e. ~sqrt(7) ~ 2.6x
tighter error bars. Measured against a high-precision reference.

Usage
-----
    python experiments/compute_quality_tradeoff.py --seed 42

Outputs (under --outdir, default ./results):
    data/compute_quality_tradeoff.csv        solver quality per budget
    data/compute_quality_equity.csv          equity error per latency
    figures/compute_quality_tradeoff.png
    figures/compute_quality_equity.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from poker_alpha.games.leduc import LeducPoker
from poker_alpha.poker.equity import estimate_equity
from poker_alpha.solvers import CFRPlusSolver, CFRSolver, MCCFRSolver, exploitability
from poker_alpha.solvers.evaluation import expected_value
from poker_alpha.utils.plotting import save_convergence_plot
from poker_alpha.utils.seeds import set_seed

# Wall-clock budgets in seconds. The low end is bounded by the cost of a
# single Leduc full-tree traversal (~40 ms for CFR+), so budgets below that
# simply buy one iteration; the grid starts where the curve is meaningful.
BUDGETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

EQUITY_SIMS = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000)


def train_within_budget(solver, game, budget: float, chunk: int) -> dict:
    """Train in chunks until ``budget`` seconds of *training* time is spent.

    The evaluation clock is stopped: only ``solver.train`` time counts, so a
    budget means "compute spent deciding", not "compute spent measuring".
    """
    spent = 0.0
    iterations = 0
    while spent < budget:
        t0 = time.perf_counter()
        solver.train(chunk)
        spent += time.perf_counter() - t0
        iterations += chunk
    strategy = solver.average_strategy()
    return {
        "elapsed_seconds": spent,
        "iterations": iterations,
        "exploitability": exploitability(game, strategy),
        "expected_value": expected_value(game, strategy),
        "infosets": len(solver.infosets),
    }


def run_solver_budgets(game, seed: int, budgets, chunks: dict) -> pd.DataFrame:
    rows: List[dict] = []
    for name, factory in (
        ("CFR+", lambda: CFRPlusSolver(game)),
        ("CFR", lambda: CFRSolver(game)),
        ("MCCFR", lambda: MCCFRSolver(game, seed=seed)),
    ):
        for budget in budgets:
            solver = factory()  # fresh solver per budget: no warm start
            res = train_within_budget(solver, game, budget, chunks[name])
            rows.append({"solver": name, "budget_seconds": budget, **res})
            print(f"  {name:>6} @ {budget:6.2f}s -> {res['iterations']:>8,} iters, "
                  f"exploitability {res['exploitability']:.4e}")
    return pd.DataFrame(rows)


def run_equity_budgets(seed: int, sims_grid, reference_sims: int,
                       repeats: int) -> pd.DataFrame:
    """Equity error vs latency for a fixed flop spot.

    A single Monte Carlo run's error is itself one random draw, so the error
    at a given budget is estimated as the RMS over ``repeats`` independent
    seeds. That is what should track the 1/sqrt(n) law; any single seed
    wanders around it.
    """
    hero, board = ["As", "Ks"], ["Qs", "7d", "2c"]
    print(f"  reference: {reference_sims:,} simulations...")
    ref = estimate_equity(hero, board=board, simulations=reference_sims,
                          seed=seed + 991)
    print(f"  reference equity = {ref.equity:.5f} "
          f"(std err {ref.std_error:.5f})")

    rows: List[dict] = []
    for sims in sims_grid:
        errors, times = [], []
        for k in range(repeats):
            t0 = time.perf_counter()
            r = estimate_equity(hero, board=board, simulations=sims,
                                seed=seed + 1000 * k)
            times.append(time.perf_counter() - t0)
            errors.append(r.equity - ref.equity)
        errs = np.asarray(errors)
        mean_elapsed = float(np.mean(times))
        rows.append({
            "simulations": sims,
            "repeats": repeats,
            "mean_elapsed_seconds": mean_elapsed,
            "mean_elapsed_ms": mean_elapsed * 1e3,
            "rms_error_vs_reference": float(np.sqrt(np.mean(errs ** 2))),
            "max_abs_error": float(np.max(np.abs(errs))),
            "theoretical_std_error": r.std_error,
            "reference_equity": ref.equity,
            "reference_simulations": reference_sims,
            "throughput_sims_per_sec": sims / mean_elapsed,
        })
        print(f"  {sims:>8,} sims -> {mean_elapsed * 1e3:8.1f} ms, "
              f"rms err {rows[-1]['rms_error_vs_reference']:.5f}, "
              f"theory {r.std_error:.5f}")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    p.add_argument("--reference-sims", type=int, default=1_000_000)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--equity-repeats", type=int, default=12,
                   help="independent seeds per simulation count (RMS error)")
    p.add_argument("--skip-equity", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    game = LeducPoker()
    data_dir = args.outdir / "data"
    fig_dir = args.outdir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -- A. solver quality vs budget ----------------------------------------
    print("solver quality vs wall-clock budget (Leduc):")
    # Chunk sizes are per-solver so every solver checks its budget at a
    # similar granularity despite ~200x different per-iteration costs.
    chunks = {"CFR+": 1, "CFR": 1, "MCCFR": 200}
    df = run_solver_budgets(game, args.seed, args.budgets, chunks)
    csv_path = data_dir / "compute_quality_tradeoff.csv"
    df.to_csv(csv_path, index=False)

    series = [(name,
               df[df.solver == name]["elapsed_seconds"].tolist(),
               df[df.solver == name]["exploitability"].tolist())
              for name in df["solver"].unique()]
    fig1 = save_convergence_plot(
        series, "training wall-clock budget (seconds)",
        "exploitability (chips)",
        "Strategy quality per unit compute (Leduc)",
        fig_dir / "compute_quality_tradeoff.png", logx=True, logy=True)
    print(f"\nwrote {csv_path}\nwrote {fig1}")

    if args.skip_equity:
        return

    # -- B. equity accuracy vs latency --------------------------------------
    print("\nequity accuracy vs latency (AsKs on Qs7d2c):")
    eq = run_equity_budgets(args.seed, EQUITY_SIMS, args.reference_sims,
                            args.equity_repeats)
    eq_csv = data_dir / "compute_quality_equity.csv"
    eq.to_csv(eq_csv, index=False)

    eq_series = [
        (f"measured RMS error ({args.equity_repeats} seeds)",
         eq["mean_elapsed_ms"].tolist(),
         eq["rms_error_vs_reference"].tolist()),
        ("theoretical std error (1/sqrt n)", eq["mean_elapsed_ms"].tolist(),
         eq["theoretical_std_error"].tolist()),
    ]
    fig2 = save_convergence_plot(
        eq_series, "latency budget (milliseconds)", "equity error",
        "Monte Carlo equity: accuracy bought per millisecond",
        fig_dir / "compute_quality_equity.png", logx=True, logy=True)
    print(f"\nwrote {eq_csv}\nwrote {fig2}")


if __name__ == "__main__":
    main()
