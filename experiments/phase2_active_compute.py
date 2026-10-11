"""Phase II Program 2: active compute allocation under equal budgets.

Compares standard external-sampling MCCFR with importance-corrected
prioritized root-chance sampling (poker_alpha.research.active_compute) on
exactly evaluable games, at equal node-traversal budgets (primary) and equal
wall time (secondary, interpolated), over many seeds.

    python experiments/phase2_active_compute.py [--quick] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.games.kuhn import KuhnPoker  # noqa: E402
from poker_alpha.games.leduc import LeducPoker  # noqa: E402
from poker_alpha.games.reduced_holdem import ReducedPreflopGame  # noqa: E402
from poker_alpha.research.active_compute import CountingGame, PrioritizedMCCFR  # noqa: E402
from poker_alpha.solvers import MCCFRSolver  # noqa: E402
from poker_alpha.solvers.evaluation import exploitability  # noqa: E402

OUT = ROOT / "results" / "phase2" / "active_compute.json"
METHODS = ["mccfr", "prio:uniform", "prio:positive_regret", "prio:scarcity", "prio:entropy",
           "prio:random_fixed", "prio:oracle_br_gain"]


def make_game(name):
    if name == "leduc":
        return LeducPoker()
    if name == "kuhn":
        return KuhnPoker()
    if name == "reduced_preflop":
        return ReducedPreflopGame(ranks="AKQJ", suits="sh", stack=8.0, equity_samples=600, seed=0)
    raise ValueError(name)


def run_job(args):
    game_name, method, seed, budgets, eps, refresh = args
    inner = make_game(game_name)
    game = CountingGame(inner)
    if method == "mccfr":
        solver = MCCFRSolver(game, seed=seed)
    else:
        sig = method.split(":", 1)[1]
        solver = PrioritizedMCCFR(game, seed=seed, signal=sig, eps=eps,
                                  refresh=(refresh * 4 if sig == "oracle_br_gain" else refresh),
                                  exact_game=inner)
    curve = []
    train_seconds = 0.0
    for b in budgets:
        t0 = time.perf_counter()
        while game.nodes < b:
            solver.iterate()
        train_seconds += time.perf_counter() - t0
        e = exploitability(inner, solver.average_strategy())
        curve.append({"budget_nodes": int(b), "nodes": int(game.nodes), "iterations": int(solver.iterations),
                      "train_seconds": round(train_seconds, 4), "exploitability": float(e)})
    extra = {}
    if method != "mccfr":
        extra = {"max_importance_weight": float(solver.max_weight),
                 "outcome_visit_counts": solver.outcome_visits.tolist(),
                 "final_q": solver.q.tolist()}
    return {"game": game_name, "method": method, "seed": seed, "curve": curve, **extra}


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows, budgets):
    games = sorted({r["game"] for r in rows})
    out = {}
    for g in games:
        out[g] = {"equal_nodes": {}, "equal_time": {}, "paired_vs_mccfr": {}}
        by = {}
        for r in rows:
            if r["game"] == g:
                by.setdefault(r["method"], []).append(r)
        base = by.get("mccfr", [])
        for m, runs in by.items():
            per_b = {}
            for bi, b in enumerate(budgets):
                e = np.array([x["curve"][bi]["exploitability"] for x in runs])
                t = np.array([x["curve"][bi]["train_seconds"] for x in runs])
                per_b[str(int(b))] = {"mean": float(e.mean()), "ci95": bootstrap_ci(e), "std": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
                                      "train_seconds_mean": float(t.mean())}
            out[g]["equal_nodes"][m] = per_b
            # paired vs baseline at each budget (same seeds)
            if base and m != "mccfr":
                paired = {}
                for bi, b in enumerate(budgets):
                    bm = {x["seed"]: x["curve"][bi]["exploitability"] for x in base}
                    d = np.array([bm[x["seed"]] - x["curve"][bi]["exploitability"] for x in runs if x["seed"] in bm])
                    paired[str(int(b))] = {"mean_diff": float(d.mean()), "ci95": bootstrap_ci(d),
                                           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                                           "relative": float(d.mean() / np.mean(list(bm.values())))}
                out[g]["paired_vs_mccfr"][m] = paired
        # equal wall time: interpolate each run's log-exploitability on a common time grid
        all_t = [c["train_seconds"] for runs in by.values() for x in runs for c in x["curve"]]
        grid = np.geomspace(max(min(all_t), 1e-3), max(all_t), 8)
        for m, runs in by.items():
            vals = []
            for x in runs:
                ts = np.array([c["train_seconds"] for c in x["curve"]])
                es = np.log(np.array([c["exploitability"] for c in x["curve"]]))
                vals.append(np.interp(np.log(grid), np.log(ts), es, left=np.nan, right=np.nan))
            vals = np.exp(np.array(vals))
            out[g]["equal_time"][m] = {f"{t:.4g}": (None if np.isnan(vals[:, i]).any() else float(vals[:, i].mean()))
                                       for i, t in enumerate(grid)}
    return out


def classify(summary, budgets):
    verdicts = {}
    for g, s in summary.items():
        best = None
        for m, paired in s["paired_vs_mccfr"].items():
            if m in ("prio:uniform", "prio:random_fixed", "prio:oracle_br_gain"):
                continue
            # count budgets where CI excludes zero in favour of the method
            wins = sum(1 for b in paired.values() if b["ci95"][0] > 0)
            losses = sum(1 for b in paired.values() if b["ci95"][1] < 0)
            verdicts[f"{g}:{m}"] = ("POSITIVE" if wins >= len(budgets) // 2 and losses == 0 else
                                    "NEGATIVE" if losses >= len(budgets) // 2 and wins == 0 else
                                    "NULL" if wins == 0 and losses == 0 else "INCONCLUSIVE")
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--refresh", type=int, default=50)
    ap.add_argument("--games", type=str, default="leduc,reduced_preflop")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed-base", type=int, default=0, help="independent seeds for confirmation runs")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    a = ap.parse_args()
    budgets = {"leduc": [2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6],
               "kuhn": [1e3, 3e3, 1e4, 3e4, 1e5, 3e5],
               "reduced_preflop": [2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6]}
    if a.quick:
        a.seeds = 2
        budgets = {k: v[:3] for k, v in budgets.items()}
    games = a.games.split(",")
    jobs = []
    for g in games:
        for m in a.methods.split(","):
            if m == "prio:oracle_br_gain" and g != "leduc":
                continue
            for s in range(a.seed_base, a.seed_base + a.seeds):
                jobs.append((g, m, s, budgets[g], a.eps, a.refresh))
    t0 = time.time()
    if a.workers > 1:
        with Pool(a.workers) as pool:
            rows = pool.map(run_job, jobs, chunksize=1)
    else:
        rows = [run_job(j) for j in jobs]
    summary = {g: summarize([r for r in rows if r["game"] == g], budgets[g])[g] for g in games}
    verdict = classify(summary, budgets)
    doc = {"format": "pokeralpha.phase2.active_compute/v1",
           "question": "Does importance-corrected prioritized root-chance sampling converge faster than "
                       "uniform ES-MCCFR at equal node traversals / wall time?",
           "methods": a.methods.split(","), "seed_base": a.seed_base, "eps": a.eps, "refresh_iterations": a.refresh, "seeds": a.seeds,
           "budgets_nodes": {g: budgets[g] for g in games},
           "metric": "exact exploitability (chips) of the average strategy; evaluation clock excluded from budgets",
           "summary": summary, "classification": verdict, "runs": rows,
           "seconds": round(time.time() - t0, 1),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "machine": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
           "limitations": ["root-chance prioritization only (deal level); deeper chance / action nodes sampled as usual",
                           "signals recomputed every `refresh` iterations from solver tables; oracle uses exact BR",
                           "equal-time comparison is interpolated between recorded budget checkpoints",
                           "small exact games; Python solver timings are not representative of the native backend"]}
    out = Path(a.out) if a.out else (OUT if not a.quick else OUT.with_name("active_compute_quick.json"))
    out.write_text(json.dumps(doc, indent=1, default=float))
    for g in games:
        print(g)
        for m, per_b in summary[g]["equal_nodes"].items():
            print(f"  {m:22s}", " ".join(f"{v['mean']:.4f}" for v in per_b.values()))
    print(json.dumps(verdict, indent=1))
    print("wrote", out, doc["seconds"], "s")


if __name__ == "__main__":
    main()
