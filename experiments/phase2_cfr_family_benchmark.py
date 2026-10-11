"""Phase II Program 3: CFR-family benchmark under three normalizations.

Algorithms: CFR (the repository's per-history online vanilla CFR), CFR+,
Discounted CFR, Linear CFR, external-sampling MCCFR, outcome-sampling MCCFR.
Games: Kuhn, Leduc, reduced preflop Hold'em, a small exact river subgame and
the zero-sum auction duel. Each run records (iterations, node visits, train
seconds, exact exploitability) at geometric node-visit checkpoints; the
summary reports exploitability at equal iterations, equal node visits and
equal wall time (interpolated in log-log between checkpoints), with the
evaluation clock excluded. Stochastic methods use many seeds.

Also re-measures the Phase I "CFR beats CFR+ under tight wall-clock budgets"
crossover on Leduc from the recorded curves.

    python experiments/phase2_cfr_family_benchmark.py [--quick] [--seeds 10]
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
sys.path.insert(0, str(ROOT / "experiments"))

from phase33_reduced_games import SUBGAMES, percentile_range  # noqa: E402

from poker_alpha.games.auction import AuctionDuel  # noqa: E402
from poker_alpha.games.kuhn import KuhnPoker  # noqa: E402
from poker_alpha.games.leduc import LeducPoker  # noqa: E402
from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame  # noqa: E402
from poker_alpha.research.active_compute import CountingGame  # noqa: E402
from poker_alpha.research.cfr_variants import DiscountedCFRSolver, OutcomeSamplingMCCFR  # noqa: E402
from poker_alpha.solvers import CFRSolver, MCCFRSolver  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import exploitability  # noqa: E402

OUT = ROOT / "results" / "phase2" / "cfr_family_benchmark.json"
DETERMINISTIC = ["cfr", "cfr+", "dcfr", "lcfr"]
STOCHASTIC = ["es_mccfr", "os_mccfr"]
GAMES = ["kuhn", "leduc", "reduced_preflop", "river", "auction"]
# node-visit budgets per game (geometric checkpoints)
MAX_NODES = {"kuhn": 3e5, "leduc": 2e7, "reduced_preflop": 1e7, "river": 2e6, "auction": 2e6}


def make_game(name):
    if name == "kuhn":
        return KuhnPoker()
    if name == "leduc":
        return LeducPoker()
    if name == "reduced_preflop":
        return ReducedPreflopGame(ranks="AKQJ", suits="sh", stack=8.0, equity_samples=600, seed=0)
    if name == "river":
        n, board, b0, b1 = SUBGAMES[0]
        return RiverSubgame(board, percentile_range(board, b0, 6), percentile_range(board, b1, 6),
                            pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2, name=n)
    if name == "auction":
        return AuctionDuel(n_values=5, n_bids=5)
    raise ValueError(name)


def make_solver(name, game, seed):
    if name == "cfr":
        return CFRSolver(game)
    if name == "cfr+":
        return CFRPlusSolver(game)
    if name == "dcfr":
        return DiscountedCFRSolver.preset(game, "dcfr")
    if name == "lcfr":
        return DiscountedCFRSolver.preset(game, "lcfr")
    if name == "es_mccfr":
        return MCCFRSolver(game, seed=seed)
    if name == "os_mccfr":
        return OutcomeSamplingMCCFR(game, seed=seed)
    raise ValueError(name)


def table_bytes(solver):
    return int(sum(n.regret_sum.nbytes + n.strategy_sum.nbytes + 200 for n in solver.infosets.values()))


def run_job(args):
    game_name, algo, seed, max_nodes, ratio = args
    inner = make_game(game_name)
    game = CountingGame(inner)
    solver = make_solver(algo, game, seed)
    checkpoints = []
    b = 500.0
    while b < max_nodes:
        checkpoints.append(b)
        b *= ratio
    checkpoints.append(float(max_nodes))
    curve = []
    train_seconds = 0.0
    for b in checkpoints:
        t0 = time.perf_counter()
        while game.nodes < b:
            solver.iterate()
        train_seconds += time.perf_counter() - t0
        e = exploitability(inner, solver.average_strategy())
        curve.append({"nodes": int(game.nodes), "iterations": int(solver.iterations),
                      "train_seconds": round(train_seconds, 5), "exploitability": float(e)})
    return {"game": game_name, "algorithm": algo, "seed": seed, "curve": curve,
            "infosets": len(solver.infosets), "table_bytes": table_bytes(solver)}


def interp_log(xs, ys, grid):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return np.exp(np.interp(np.log(grid), np.log(xs), np.log(np.maximum(ys, 1e-300)), left=np.nan, right=np.nan))


def summarize(rows):
    out = {}
    for g in sorted({r["game"] for r in rows}):
        rs = [r for r in rows if r["game"] == g]
        by = {}
        for r in rs:
            by.setdefault(r["algorithm"], []).append(r)
        # grids span the range every run covers (min of the maxima, max of the minima)
        max_it = min(r["curve"][-1]["iterations"] for r in rs)
        min_it = max(r["curve"][0]["iterations"] for r in rs)
        max_n = min(r["curve"][-1]["nodes"] for r in rs)
        min_n = max(r["curve"][0]["nodes"] for r in rs)
        max_t = min(r["curve"][-1]["train_seconds"] for r in rs)
        min_t = max(r["curve"][0]["train_seconds"] for r in rs)
        it_grid = np.unique(np.geomspace(max(min_it, 1), max(max_it, min_it + 1), 10).round()).tolist()
        n_grid = np.geomspace(min_n, max(max_n, min_n * 1.01), 10).tolist()
        t_grid = np.geomspace(max(min_t, 1e-4), max(max_t, min_t * 1.01), 10).tolist()
        gs = {"equal_iterations": {}, "equal_nodes": {}, "equal_time": {}, "memory": {}, "iterations_grid": it_grid,
              "nodes_grid": n_grid, "time_grid": t_grid}
        for algo, runs in by.items():
            stats = {}
            for label, grid, xkey in (("equal_iterations", it_grid, "iterations"), ("equal_nodes", n_grid, "nodes"),
                                      ("equal_time", t_grid, "train_seconds")):
                vals = np.array([interp_log([c[xkey] for c in r["curve"]], [c["exploitability"] for c in r["curve"]], grid)
                                 for r in runs])
                means, stds = [], []
                for i in range(len(grid)):
                    col = vals[:, i]
                    if np.isnan(col).any():
                        means.append(None)
                        stds.append(None)
                    else:
                        means.append(float(col.mean()))
                        stds.append(float(col.std(ddof=1)) if len(col) > 1 else 0.0)
                gs[label][algo] = {"mean": means, "std": stds, "runs": len(runs)}
            gs["memory"][algo] = {"infosets": int(np.mean([r["infosets"] for r in runs])),
                                  "table_bytes": int(np.mean([r["table_bytes"] for r in runs])),
                                  "nodes_per_iteration": float(np.mean([r["curve"][-1]["nodes"] / r["curve"][-1]["iterations"] for r in runs]))}
        out[g] = gs
    return out


def crossover(rows, budgets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)):
    """Phase I claim re-measurement: Leduc exploitability of CFR / CFR+ / ES-MCCFR
    at wall-clock budgets (seed-mean for MCCFR), from the recorded curves."""
    rs = [r for r in rows if r["game"] == "leduc" and r["algorithm"] in ("cfr", "cfr+", "es_mccfr")]
    if not rs:
        return None
    table = {}
    for algo in ("cfr", "cfr+", "es_mccfr"):
        runs = [r for r in rs if r["algorithm"] == algo]
        vals = np.array([interp_log([c["train_seconds"] for c in r["curve"]], [c["exploitability"] for c in r["curve"]],
                                    list(budgets)) for r in runs])
        table[algo] = [None if np.isnan(vals[:, i]).any() else float(vals[:, i].mean()) for i in range(len(budgets))]
    best = []
    for i in range(len(budgets)):
        cands = {a: table[a][i] for a in table if table[a][i] is not None}
        best.append(min(cands, key=cands.get) if cands else None)
    first_plus = next((b for b, w in zip(budgets, best) if w == "cfr+"), None)
    return {"budgets_seconds": list(budgets), "exploitability": table, "best": best,
            "first_budget_where_cfr_plus_wins": first_plus,
            "phase1_claim": "CFR better below ~0.5 s, CFR+ from ~1 s (RESEARCH.md 3.3; 4-vCPU container)",
            "note": "interpolated from node-checkpoint curves; wall time of this machine, evaluation excluded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--games", type=str, default=",".join(GAMES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    games = a.games.split(",")
    max_nodes = dict(MAX_NODES)
    ratio = 1.6
    if a.quick:
        a.seeds = 2
        max_nodes = {k: v / 100 for k, v in max_nodes.items()}
        ratio = 3.0
    jobs = []
    for g in games:
        for algo in DETERMINISTIC:
            jobs.append((g, algo, 0, max_nodes[g], ratio))
        for algo in STOCHASTIC:
            for s in range(a.seeds):
                jobs.append((g, algo, s, max_nodes[g], ratio))
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)
    summary = summarize(rows)
    doc = {"format": "pokeralpha.phase2.cfr_family_benchmark/v1",
           "algorithms": {"cfr": "poker_alpha.solvers.cfr.CFRSolver (vanilla CFR; per-history online regret updates "
                                 "within an iteration, see docs/phase2/cfr_benchmark.md)",
                          "cfr+": "poker_alpha.solvers.cfr_plus.CFRPlusSolver (alternating, clipped, linear averaging)",
                          "dcfr": "DiscountedCFRSolver (alpha 1.5, beta 0, gamma 2; Brown & Sandholm 2019)",
                          "lcfr": "DiscountedCFRSolver (1, 1, 1): linear CFR",
                          "es_mccfr": "external-sampling MCCFR (poker_alpha.solvers.mccfr)",
                          "os_mccfr": "outcome-sampling MCCFR, epsilon 0.6 (poker_alpha.research.cfr_variants)"},
           "games": {g: make_game(g).signature() for g in games},
           "seeds": a.seeds, "metric": "exact exploitability (game chip units), evaluation clock excluded",
           "node_visits": "is_terminal calls counted by CountingGame (one per visited node, chance nodes included)",
           "summary": summary, "crossover_leduc": crossover(rows), "runs": rows,
           "seconds": round(time.time() - t0, 1),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "machine": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                       "note": "jobs ran in parallel across cores; wall times are per-process CPU-bound timings"}}
    out = Path(a.out) if a.out else (OUT if not a.quick else OUT.with_name("cfr_family_benchmark_quick.json"))
    out.write_text(json.dumps(doc, indent=1, default=float))
    for g in games:
        print(g, "equal nodes (last grid point):",
              {k: (None if v["mean"][-1] is None else round(v["mean"][-1], 5)) for k, v in summary[g]["equal_nodes"].items()})
    print("crossover:", json.dumps(doc["crossover_leduc"], indent=0)[:600] if doc["crossover_leduc"] else None)
    print("wrote", out, doc["seconds"], "s")


if __name__ == "__main__":
    main()
