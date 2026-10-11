"""Phase II Program 1: adaptive abstraction refinement with capacity-matched
controls on exact river subgames.

Domain: 52-card river subgames (fixed board, explicit ranges, pot 10 /
stack 20, bets 50% + 100% + all-in, raise cap 2 - the Phase 33 subgame
family), solved exactly by the vectorized range solver. The abstraction is a
partition of each player's range into strength-ordered cells; one cell per
hand is the raw game. Error = exact exploitability (chips) of the lifted
strategy in the raw game.

Arms per instance (board x range-sampling seed):
  raw               one cell per hand (solver-error floor)
  baseline          K equal-count cells per player
  adaptive:<signal> K cells, then `splits` greedy rounds of (solve, measure,
                    split the worst cell at its strength median)
  oracle            same loop, but each round tries every split and keeps the
                    best (upper bound for one-step greedy selection)
  uniform_matched   equal-count cells with exactly the per-player cell counts
                    the adaptive br_gain arm ended with (same infoset count)
  random            `splits` random (player, cell) splits, several seeds

Usage:
  python experiments/phase2_adaptive_abstraction.py                 # discovery
  python experiments/phase2_adaptive_abstraction.py --confirm       # independent seeds + boards
  python experiments/phase2_adaptive_abstraction.py --quick
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

from phase33_reduced_games import SUBGAMES  # noqa: E402

from poker_alpha.poker.cards import card_str, codes  # noqa: E402
from poker_alpha.poker.evaluator import evaluate_best_codes  # noqa: E402
from poker_alpha.research.refinement import SIGNALS, RiverRefinementProblem, refine  # noqa: E402
from poker_alpha.research.river_solver import RiverRangeSolver  # noqa: E402

OUT = ROOT / "results" / "phase2" / "adaptive_abstraction.json"
CONFIRM_BOARDS = [
    ("paired_low_ace_river", ["7h", "7d", "2s", "Jc", "Ad"], [(0.0, 0.25), (0.85, 1.0)], [(0.3, 0.9)]),
    ("broadway_dry", ["Ah", "Kd", "Qs", "Tc", "3h"], [(0.2, 0.6), (0.9, 1.0)], [(0.4, 1.0)]),
    ("two_tone_middling", ["Jd", "8d", "6c", "3s", "9h"], [(0.1, 0.5), (0.8, 1.0)], [(0.3, 0.85)]),
]


def sample_range(board, bands, n, rng):
    """``n`` combos drawn uniformly (without replacement) from the given
    river-strength percentile bands, as ((a, b), weight) pairs."""
    b = codes(board)
    combos = [(a, c) for a in range(52) for c in range(a + 1, 52) if a not in b and c not in b]
    vals = sorted(((evaluate_best_codes([a, c] + b), (a, c)) for a, c in combos))
    m = len(vals)
    band = [h for i, (_, h) in enumerate(vals) if any(lo <= i / m < hi for lo, hi in bands)]
    idx = rng.choice(len(band), size=min(n, len(band)), replace=False)
    return [(band[i], 1.0) for i in sorted(idx)]


def run_instance(args):
    (name, board, b0, b1, seed, n_hands, k0, splits, iters, random_seeds, oracle) = args
    rng = np.random.default_rng(seed)
    r0 = sample_range(board, b0, n_hands, rng)
    r1 = sample_range(board, b1, n_hands, rng)
    solver = RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    prob = RiverRefinementProblem(solver, iters)
    t0 = time.perf_counter()
    out = {"name": name, "board": board, "seed": seed, "n_hands": [len(r0), len(r1)],
           "range0": [card_str(a) + card_str(b) for (a, b), _ in r0],
           "range1": [card_str(a) + card_str(b) for (a, b), _ in r1], "arms": {}}

    def rec(sv, **kw):
        d = {"exploitability": sv.exploitability, "num_infosets": sv.num_infosets,
             "cells_per_player": sv.cells_per_player, "solve_seconds": sv.seconds,
             "ev_p0": sv.extra.get("ev_p0")}
        d.update(kw)
        return d

    raw = prob.solve([np.arange(solver.n[0]), np.arange(solver.n[1])])
    out["arms"]["raw"] = rec(raw)
    base_cells = prob.uniform([k0, k0])
    base = prob.solve(base_cells)
    out["arms"]["baseline"] = rec(base)
    for sig in SIGNALS:
        if sig == "random":
            continue
        r = refine(prob, base_cells, splits, sig)
        out["arms"][f"adaptive:{sig}"] = rec(r["final"], trace=r["trace"], wall_seconds=r["wall_seconds"],
                                             solve_seconds_total=r["solve_seconds"])
    if oracle:
        r = refine(prob, base_cells, splits, "br_gain", oracle=True)
        out["arms"]["oracle"] = rec(r["final"], trace=r["trace"], wall_seconds=r["wall_seconds"],
                                    solve_seconds_total=r["solve_seconds"])
    # capacity-matched uniform: same per-player cell counts as adaptive:br_gain
    kk = out["arms"]["adaptive:br_gain"]["cells_per_player"]
    um = prob.solve(prob.uniform(kk))
    out["arms"]["uniform_matched"] = rec(um)
    # symmetric uniform with the same total number of cells (if even split)
    tot = sum(kk)
    us = prob.solve(prob.uniform([tot // 2, tot - tot // 2]))
    out["arms"]["uniform_symmetric"] = rec(us)
    rand = []
    for rs in range(random_seeds):
        r = refine(prob, base_cells, splits, "random", seed=1000 * seed + rs)
        rand.append(rec(r["final"], trace=r["trace"]))
    out["arms"]["random"] = {"runs": rand,
                             "exploitability_mean": float(np.mean([x["exploitability"] for x in rand])),
                             "exploitability_min": float(np.min([x["exploitability"] for x in rand])),
                             "exploitability_max": float(np.max([x["exploitability"] for x in rand])),
                             "num_infosets_mean": float(np.mean([x["num_infosets"] for x in rand]))}
    out["seconds"] = time.perf_counter() - t0
    return out


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(instances, arms):
    base = np.array([i["arms"]["baseline"]["exploitability"] for i in instances])
    raw = np.array([i["arms"]["raw"]["exploitability"] for i in instances])
    headroom = base - raw
    summary = {"instances": len(instances), "baseline_exploitability_mean": float(base.mean()),
               "raw_exploitability_mean": float(raw.mean()), "arms": {}}

    def series(arm):
        if arm == "random":
            return np.array([i["arms"]["random"]["exploitability_mean"] for i in instances])
        return np.array([i["arms"][arm]["exploitability"] for i in instances])

    for arm in arms:
        if not all(arm in i["arms"] for i in instances):
            continue
        e = series(arm)
        if arm == "random":
            inf = np.array([i["arms"]["random"]["num_infosets_mean"] for i in instances])
        else:
            inf = np.array([i["arms"][arm]["num_infosets"] for i in instances])
        frac = (base - e) / np.where(headroom > 0, headroom, np.nan)
        summary["arms"][arm] = {
            "exploitability_mean": float(e.mean()), "exploitability_ci95": bootstrap_ci(e),
            "reduction_vs_baseline_mean": float((base - e).mean()),
            "fraction_of_headroom_recovered_mean": float(np.nanmean(frac)),
            "fraction_of_headroom_recovered_median": float(np.nanmedian(frac)),
            "num_infosets_mean": float(inf.mean()),
            "improvement_per_added_infoset": float(np.mean((base - e) / np.maximum(inf - np.array(
                [i["arms"]["baseline"]["num_infosets"] for i in instances]), 1)))}
    # paired comparisons of each adaptive arm vs capacity-matched uniform and vs random
    controls = {"vs_uniform_matched": series("uniform_matched"),
                "vs_uniform_symmetric": series("uniform_symmetric"),
                "vs_random_mean": series("random")}
    summary["paired"] = {}
    for arm in arms:
        if not arm.startswith("adaptive") and arm != "oracle":
            continue
        if not all(arm in i["arms"] for i in instances):
            continue
        e = series(arm)
        summary["paired"][arm] = {}
        for label, ctrl in controls.items():
            d = ctrl - e      # positive = adaptive better (lower exploitability)
            summary["paired"][arm][label] = {
                "mean_diff": float(d.mean()), "ci95": bootstrap_ci(d),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                "relative_mean": float(d.mean() / ctrl.mean())}
    return summary


def classify(summary):
    p = summary["paired"].get("adaptive:br_gain")
    if p is None:
        return "INCONCLUSIVE"
    lo, hi = p["vs_uniform_matched"]["ci95"]
    lo_s, _ = p["vs_uniform_symmetric"]["ci95"]
    lo_r, hi_r = p["vs_random_mean"]["ci95"]
    if lo > 0 and lo_s > 0 and lo_r > 0 and p["vs_uniform_matched"]["relative_mean"] > 0.05:
        return "POSITIVE"
    if hi < 0:
        return "NEGATIVE"
    if lo <= 0 <= hi and abs(p["vs_uniform_matched"]["relative_mean"]) < 0.05:
        return "NULL"
    return "INCONCLUSIVE"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--confirm", action="store_true", help="independent seeds and boards")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hands", type=int, default=48)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.hands, a.splits, a.iters, a.random_seeds = 1, 12, 2, 60, 2
    boards = list(SUBGAMES) + (CONFIRM_BOARDS if a.confirm else [])
    if a.quick:
        boards = boards[:2]
    seed_base = 5000 if a.confirm else 0
    jobs = [(name, board, b0, b1, seed_base + s, a.hands, a.k0, a.splits, a.iters, a.random_seeds,
             not a.no_oracle)
            for name, board, b0, b1 in boards for s in range(a.seeds)]
    t0 = time.time()
    if a.workers > 1 and len(jobs) > 1:
        with Pool(a.workers) as pool:
            instances = pool.map(run_instance, jobs)
    else:
        instances = [run_instance(j) for j in jobs]
    arms = ["raw", "baseline", "uniform_matched", "uniform_symmetric", "random"] + \
        [f"adaptive:{s}" for s in SIGNALS if s != "random"] + ([] if a.no_oracle else ["oracle"])
    summary = summarize(instances, arms)
    doc = {
        "format": "pokeralpha.phase2.adaptive_abstraction/v1",
        "mode": "confirmation" if a.confirm else "discovery",
        "hypothesis": ("Splitting the abstraction cell with the largest measured exact error "
                       "(best-response gain) lowers raw-game exploitability more than spending the "
                       "same number of cells uniformly or at random."),
        "domain": {"game": "52-card river subgame, pot 10, stack 20, bets 50%/100%/all-in, raise cap 2, "
                           "OOP acts first; ranges sampled from strength-percentile bands",
                   "abstraction": "per-player partition of the range into strength-ordered cells; "
                                  "infoset = (betting node, cell); perfect recall (single street)",
                   "error_metric": "exact exploitability of the lifted strategy in the raw game (chips)",
                   "solver": "vectorized CFR+ (parity-tested vs poker_alpha.solvers.cfr_plus)"},
        "config": {"seeds": a.seeds, "seed_base": seed_base, "hands_per_player": a.hands, "k0": a.k0,
                   "splits": a.splits, "cfr_plus_iterations": a.iters, "random_control_seeds": a.random_seeds,
                   "oracle": not a.no_oracle, "boards": [b[0] for b in boards]},
        "signals": {s: "see poker_alpha/research/refinement.py" for s in SIGNALS},
        "summary": summary,
        "classification": classify(summary),
        "instances": instances,
        "seconds": round(time.time() - t0, 1),
        "commit": git_sha(),
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpu_count": os.cpu_count()},
        "limitations": [
            "single-street exact domain: findings need not transfer to multi-street imperfect-recall abstractions",
            "cells are contiguous in strength order; splits are at the cell's strength median only",
            "greedy one-split-per-round refinement; no lookahead except the oracle arm",
            "exploitability in chips of a pot-10 subgame; not comparable with full-game bb/100 numbers",
        ],
    }
    out = Path(a.out) if a.out else (OUT if not (a.quick or a.confirm) else
                                     OUT.with_name("adaptive_abstraction_" + ("quick" if a.quick else "confirm") + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"classification": doc["classification"], "summary": {k: v for k, v in summary.items()
                                                                            if k != "arms"},
                      "arms": {k: round(v["exploitability_mean"], 4) for k, v in summary["arms"].items()}},
                     indent=1))
    print("wrote", out, "in", doc["seconds"], "s")


if __name__ == "__main__":
    main()
