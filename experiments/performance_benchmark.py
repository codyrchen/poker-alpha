"""Reproducible performance benchmarks for PokerAlpha's quantitative workloads.

Measures throughput and latency for the subsystems that dominate this
project's compute: hand evaluation, Monte Carlo equity, CFR-family solving,
exact strategy evaluation (best response / exploitability), and Bayesian
opponent-belief updates.

Method
------
Each workload runs a short **warmup** (excluded from timing, so import-time
lazy work and CPU frequency ramp-up do not pollute the first sample), then
``--reps`` timed repetitions on ``time.perf_counter()``. We report mean,
median, std, min and max seconds per repetition plus a derived throughput
(problem_size / mean_seconds). Median and min are the robust readings on a
noisy laptop: min is the closest thing to "the machine's real speed" since
scheduler noise only ever *adds* time, while mean/std expose that noise.

Every workload is seeded, so re-running reproduces the same *work* (not the
same timings). ``--label`` tags the rows so a before/after comparison can
concatenate two runs of this same script.

Because timings are machine- and load-sensitive, the before/after runs must be
taken back-to-back on an otherwise-idle machine; workloads untouched by an
optimization then act as a built-in control (they should come out unchanged,
and any drift in them bounds how much of a "speedup" elsewhere is real).

Usage
-----
    python experiments/performance_benchmark.py --label baseline
    python experiments/performance_benchmark.py --label optimized
    python experiments/performance_benchmark.py --compare baseline optimized

Outputs (under --outdir, default ./results):
    data/performance_<label>.csv                    one row per workload
    data/performance_comparison.csv                 (--compare) before vs after
    figures/performance_speedup.png                 (--compare) throughput chart
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from poker_alpha.games.kuhn import KuhnPoker
from poker_alpha.games.leduc import LeducPoker
from poker_alpha.opponent import (
    ARCHETYPES,
    ArchetypeBelief,
    leduc_is_weak,
    simulate_match,
    tilt_strategy,
)
from poker_alpha.poker.cards import NUM_CARDS
from poker_alpha.poker.equity import estimate_equity
from poker_alpha.poker.evaluator import evaluate_best, evaluate_five
from poker_alpha.solvers import CFRPlusSolver, CFRSolver, MCCFRSolver, exploitability
from poker_alpha.solvers.evaluation import profile_value
from poker_alpha.utils.plotting import save_grouped_bar_chart


@dataclass
class BenchRow:
    workload: str
    label: str
    problem_size: int
    unit: str
    reps: int
    mean_seconds: float
    median_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    throughput: float  # units per second, from mean
    throughput_best: float  # units per second, from min (least-noise reading)
    seed: int


def time_workload(
    name: str,
    fn: Callable[[], None],
    problem_size: int,
    unit: str,
    label: str,
    seed: int,
    reps: int,
    warmup: int = 1,
) -> BenchRow:
    """Warm up, then time ``fn`` ``reps`` times; summarize."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)

    mean_s = statistics.fmean(samples)
    row = BenchRow(
        workload=name,
        label=label,
        problem_size=problem_size,
        unit=unit,
        reps=reps,
        mean_seconds=mean_s,
        median_seconds=statistics.median(samples),
        std_seconds=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min_seconds=min(samples),
        max_seconds=max(samples),
        throughput=problem_size / mean_s if mean_s > 0 else float("nan"),
        throughput_best=problem_size / min(samples) if min(samples) > 0 else float("nan"),
        seed=seed,
    )
    print(f"  {name:<28} {row.throughput:>12,.0f} {unit}/s  "
          f"(mean {mean_s * 1e3:8.2f} ms, median {row.median_seconds * 1e3:8.2f} ms, "
          f"sd {row.std_seconds * 1e3:6.2f} ms)")
    return row


def random_seven_card_hands(n: int, seed: int) -> List[List[int]]:
    """``n`` distinct-card 7-card hands as integer codes."""
    rng = np.random.default_rng(seed)
    return [[int(c) for c in rng.choice(NUM_CARDS, size=7, replace=False)]
            for _ in range(n)]


def random_five_card_hands(n: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    return [[int(c) for c in rng.choice(NUM_CARDS, size=5, replace=False)]
            for _ in range(n)]


def build_comparison(outdir: Path, before_label: str, after_label: str) -> None:
    """Join two labelled benchmark runs into a comparison table and figure."""
    data_dir = outdir / "data"
    before = pd.read_csv(data_dir / f"performance_{before_label}.csv")
    after = pd.read_csv(data_dir / f"performance_{after_label}.csv")

    merged = before.merge(after, on="workload", suffixes=("_before", "_after"))
    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "workload": r["workload"],
            "unit": r["unit_before"],
            "problem_size": r["problem_size_before"],
            "seed": r["seed_before"],
            "implementation_before": before_label,
            "implementation_after": after_label,
            "mean_seconds_before": r["mean_seconds_before"],
            "mean_seconds_after": r["mean_seconds_after"],
            "median_seconds_before": r["median_seconds_before"],
            "median_seconds_after": r["median_seconds_after"],
            "throughput_before": r["throughput_before"],
            "throughput_after": r["throughput_after"],
            "speedup": r["throughput_after"] / r["throughput_before"],
        })
    comp = pd.DataFrame(rows).sort_values("speedup", ascending=False)
    path = data_dir / "performance_comparison.csv"
    comp.to_csv(path, index=False)

    print(f"{'workload':<26}{'before':>14}{'after':>14}{'speedup':>10}")
    for _, r in comp.iterrows():
        print(f"{r['workload']:<26}{r['throughput_before']:>14,.0f}"
              f"{r['throughput_after']:>14,.0f}{r['speedup']:>9.2f}x")

    # Throughput is compared on a log axis: the workloads span 45 it/s to
    # ~10^6 evals/s, so a linear axis would render everything but hand
    # evaluation as a flat line.
    labels = comp["workload"].tolist()
    save_grouped_bar_chart(
        labels,
        {before_label: comp["throughput_before"].tolist(),
         after_label: comp["throughput_after"].tolist()},
        "throughput (units/second, log scale)",
        f"PokerAlpha throughput: {before_label} vs {after_label}",
        outdir / "figures" / "performance_speedup.png",
        log_y=True,
    )
    print(f"\nwrote {path}")
    print(f"wrote {outdir / 'figures' / 'performance_speedup.png'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", type=str, default="baseline",
                   help="tag for these rows (e.g. baseline / optimized)")
    p.add_argument("--compare", type=str, nargs=2, metavar=("BEFORE", "AFTER"),
                   default=None,
                   help="join two existing labelled runs into a comparison")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-hands", type=int, default=20_000)
    p.add_argument("--equity-sims", type=int, default=5_000)
    p.add_argument("--kuhn-iters", type=int, default=2_000)
    p.add_argument("--leduc-iters", type=int, default=20)
    p.add_argument("--mccfr-iters", type=int, default=5_000)
    p.add_argument("--belief-hands", type=int, default=2_000)
    p.add_argument("--eq-iterations", type=int, default=300,
                   help="CFR+ iterations for the Leduc reference equilibrium")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--skip", type=str, nargs="*", default=[],
                   help="workload name substrings to skip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare is not None:
        build_comparison(args.outdir, args.compare[0], args.compare[1])
        return
    rows: List[BenchRow] = []

    def keep(name: str) -> bool:
        return not any(s in name for s in args.skip)

    def bench(name, fn, size, unit, warmup=1):
        if keep(name):
            rows.append(time_workload(name, fn, size, unit, args.label,
                                      args.seed, args.reps, warmup))

    print(f"PokerAlpha performance benchmark (label={args.label!r}, "
          f"reps={args.reps}, seed={args.seed})\n")

    # -- 1. hand evaluation --------------------------------------------------
    print("hand evaluation:")
    hands7 = random_seven_card_hands(args.eval_hands, args.seed)
    hands5 = random_five_card_hands(args.eval_hands, args.seed)

    def eval7():
        for h in hands7:
            evaluate_best(h)

    def eval5():
        for h in hands5:
            evaluate_five(h)

    bench("hand_eval_7card", eval7, args.eval_hands, "evals")
    bench("hand_eval_5card", eval5, args.eval_hands, "evals")

    # -- 2. Monte Carlo equity ----------------------------------------------
    print("\nmonte carlo equity:")
    bench("equity_preflop_uniform",
          lambda: estimate_equity(["As", "Ah"], simulations=args.equity_sims,
                                  seed=args.seed),
          args.equity_sims, "sims")
    bench("equity_flop_uniform",
          lambda: estimate_equity(["As", "Ks"], board=["Qs", "7d", "2c"],
                                  simulations=args.equity_sims, seed=args.seed),
          args.equity_sims, "sims")
    weighted_range = [(["Ac", "Ad"], 1.0), (["Kc", "Kd"], 1.0),
                      (["Qc", "Qd"], 1.0), (["Jc", "Jd"], 0.5),
                      (["Ac", "Kc"], 0.75)]
    bench("equity_weighted_range",
          lambda: estimate_equity(["As", "Ks"], opponent_range=weighted_range,
                                  simulations=args.equity_sims, seed=args.seed),
          args.equity_sims, "sims")

    # -- 3. CFR family -------------------------------------------------------
    print("\nsolvers (iterations/second):")
    kuhn = KuhnPoker()
    leduc = LeducPoker()

    def cfr_kuhn():
        CFRSolver(kuhn).train(args.kuhn_iters)

    def cfr_plus_kuhn():
        CFRPlusSolver(kuhn).train(args.kuhn_iters)

    def mccfr_kuhn():
        MCCFRSolver(kuhn, seed=args.seed).train(args.mccfr_iters)

    def cfr_leduc():
        CFRSolver(leduc).train(args.leduc_iters)

    def cfr_plus_leduc():
        CFRPlusSolver(leduc).train(args.leduc_iters)

    def mccfr_leduc():
        MCCFRSolver(leduc, seed=args.seed).train(args.mccfr_iters)

    bench("cfr_kuhn", cfr_kuhn, args.kuhn_iters, "iters")
    bench("cfr_plus_kuhn", cfr_plus_kuhn, args.kuhn_iters, "iters")
    bench("mccfr_kuhn", mccfr_kuhn, args.mccfr_iters, "iters")
    bench("cfr_leduc", cfr_leduc, args.leduc_iters, "iters")
    bench("cfr_plus_leduc", cfr_plus_leduc, args.leduc_iters, "iters")
    bench("mccfr_leduc", mccfr_leduc, args.mccfr_iters, "iters")

    # -- 4. exact evaluation -------------------------------------------------
    print("\nexact strategy evaluation:")
    solver = CFRPlusSolver(leduc)
    solver.train(args.eq_iterations)
    equilibrium = solver.average_strategy()
    archetypes = {name: tilt_strategy(equilibrium, cfg, leduc_is_weak)
                  for name, cfg in ARCHETYPES.items()}

    bench("exploitability_leduc",
          lambda: exploitability(leduc, equilibrium), 1, "calls")
    bench("profile_value_leduc",
          lambda: profile_value(leduc, equilibrium, archetypes["maniac"]),
          1, "calls")

    # -- 5. opponent-belief updates -----------------------------------------
    print("\nopponent modeling:")
    rng = np.random.default_rng(args.seed)
    records = simulate_match(leduc, equilibrium, archetypes["maniac"],
                             hands=args.belief_hands, rng=rng)
    observations = [r.observation for r in records]

    def belief_updates():
        belief = ArchetypeBelief(candidates=archetypes)
        for obs in observations:
            belief.update(obs)

    bench("belief_update", belief_updates, len(observations), "updates")

    def match_sim():
        simulate_match(leduc, equilibrium, archetypes["maniac"],
                       hands=args.belief_hands,
                       rng=np.random.default_rng(args.seed))

    bench("leduc_match_simulation", match_sim, args.belief_hands, "hands")

    # -- write ---------------------------------------------------------------
    df = pd.DataFrame([asdict(r) for r in rows])
    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"performance_{args.label}.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
