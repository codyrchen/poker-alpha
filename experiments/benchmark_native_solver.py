"""Python-vs-native MCCFR training benchmark (Phases 35, 83, 94).

Measures wall-clock training throughput of both backends on the release
config, in milestone blocks so cold start vs warm training is visible, with
repeated trials (median reported). Also times checkpoint save/load and
strategy export at the final table size.

    python experiments/benchmark_native_solver.py \
        [--python-iterations 10000] [--native-iterations 100000] \
        [--trials 3] [--out results/benchmarks/native_mccfr_v1.json]

Python blocks are capped (the reference backend is ~2 orders of magnitude
slower); the speedup is computed on the overlapping warm block range.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from poker_alpha.solver_config import RELEASE_CONFIG  # noqa: E402


def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1 << 20) if sys.platform == "darwin" else ru / (1 << 10)


def blocks_for(total: int):
    schedule = [1_000, 4_000, 5_000, 10_000, 30_000, 50_000, 100_000, 100_000]
    out, remaining = [], total
    for size in schedule:
        take = min(size, remaining)
        if take <= 0:
            break
        out.append(take)
        remaining -= take
    if remaining > 0:
        out.append(remaining)
    return out


def run_backend(backend: str, iterations: int, seed: int) -> dict:
    if backend == "python":
        solver = RELEASE_CONFIG.build_solver(seed=seed)
        train = lambda n: [solver.iterate() for _ in range(n)]  # noqa: E731
        infosets = lambda: len(solver.infosets)  # noqa: E731
    else:
        from poker_alpha.native import NativeMCCFRSolver

        solver = NativeMCCFRSolver(RELEASE_CONFIG, seed=seed)
        train = solver.train
        infosets = lambda: solver.metrics()["infosets"]  # noqa: E731

    milestones = []
    done = 0
    for block in blocks_for(iterations):
        t0 = time.perf_counter()
        train(block)
        dt = time.perf_counter() - t0
        done += block
        milestones.append({
            "from": done - block, "to": done,
            "seconds": round(dt, 3),
            "it_per_s": round(block / dt, 2),
            "infosets": int(infosets()),
        })
    result = {"backend": backend, "seed": seed, "iterations": done,
              "milestones": milestones, "rss_mb": round(rss_mb(), 1)}

    # Checkpoint / export timing at final size (native only; Python side
    # already has committed numbers in results/validation).
    if backend == "native":
        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            p = solver.save_checkpoint(Path(td) / "bench.npz")
            result["checkpoint_save_s"] = round(time.perf_counter() - t0, 3)
            result["checkpoint_mb"] = round(p.stat().st_size / 1e6, 2)
            t0 = time.perf_counter()
            from poker_alpha.native import NativeMCCFRSolver as N

            N.load_checkpoint(p, RELEASE_CONFIG)
            result["checkpoint_load_s"] = round(time.perf_counter() - t0, 3)
            t0 = time.perf_counter()
            a = solver.export_strategy(Path(td) / "bench_strategy.npz")
            result["export_strategy_s"] = round(time.perf_counter() - t0, 3)
            result["artifact_mb"] = round(a.stat().st_size / 1e6, 2)
    return result


def run_trial_subprocess(backend: str, iterations: int, seed: int) -> dict:
    """Each trial in a fresh process: no cache carry-over between trials."""
    code = (
        "import json, sys; sys.path.insert(0, %r);"
        "from experiments.benchmark_native_solver import run_backend;"
        "print(json.dumps(run_backend(%r, %d, %d)))"
        % (str(ROOT), backend, iterations, seed))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def median_warm_rate(trial_results, lo=10_000):
    """Median it/s over blocks at/after `lo` iterations (warm training)."""
    rates = []
    for r in trial_results:
        warm = [m["it_per_s"] for m in r["milestones"] if m["from"] >= lo]
        if warm:
            rates.append(statistics.median(warm))
    return statistics.median(rates) if rates else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-iterations", type=int, default=10_000)
    ap.add_argument("--native-iterations", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-python", action="store_true",
                    help="reuse the committed Python baseline instead")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/benchmarks/native_mccfr_v1.json")
    args = ap.parse_args()

    from poker_alpha.native import native_available, native_version

    if not native_available():
        raise SystemExit("native backend not installed (pip install ./cpp)")

    report = {
        "format": "pokeralpha.native_benchmark/v1",
        "machine": {"platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": sys.version.split()[0],
                    "numpy": np.__version__},
        "config_signature": RELEASE_CONFIG.signature(),
        "native_version": native_version(),
        "trials": args.trials,
        "python": [], "native": [],
    }

    for trial in range(args.trials):
        print(f"[trial {trial}] native {args.native_iterations} iterations...",
              flush=True)
        report["native"].append(run_trial_subprocess(
            "native", args.native_iterations, args.seed + trial))
        if not args.skip_python:
            print(f"[trial {trial}] python {args.python_iterations} iterations...",
                  flush=True)
            report["python"].append(run_trial_subprocess(
                "python", args.python_iterations, args.seed + trial))

    native_warm = median_warm_rate(report["native"])
    report["native_warm_it_per_s_median"] = native_warm
    if report["python"]:
        python_warm = median_warm_rate(report["python"], lo=5_000)
        report["python_warm_it_per_s_median"] = python_warm
        if python_warm and native_warm:
            report["speedup_warm"] = round(native_warm / python_warm, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("python", "native")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
