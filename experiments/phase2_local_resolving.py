"""Phase II Program 6: anytime decision-time resolving on exact river subgames.

For each instance (board x range seed), a coarse blueprint (K strength cells
per player, 400 CFR+ iterations) is resolved at full hand resolution in the
subtrees below every first action (resolver = IP, setting "ip_depth1") or
below every IP response to an OOP check (resolver = OOP, "oop_depth2"),
with the time budget split equally across subtrees. Arms:

  blueprint        static precomputed policy
  unsafe@B         unsafe subgame solving, budget B ms
  safe@B           re-solve gadget (Burch et al. 2014), budget B ms
  full@B           whole-game raw CFR+ from scratch for B ms (reference: what
                   the same compute buys without a blueprint; feasible only
                   because this domain is tiny)
  equilibrium      raw CFR+ 2000 iterations (lower-bound reference)

Metrics (chips; both reported, never mixed):
  br_value         exact value of the opponent's best response against the
                   resolver's strategy (worst case; lower is better)
  vs_blueprint     resolver's exact EV against an opponent who plays the
                   blueprint (exploitation of the assumed opponent; higher is better)

    python experiments/phase2_local_resolving.py [--quick]
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

from phase2_adaptive_abstraction import CONFIRM_BOARDS, sample_range  # noqa: E402
from phase33_reduced_games import SUBGAMES  # noqa: E402

from poker_alpha.research.resolving import opponent_br_value, resolve_all  # noqa: E402
from poker_alpha.research.river_solver import RiverRangeSolver, equal_count_cells  # noqa: E402

OUT = ROOT / "results" / "phase2" / "local_resolving.json"
BUDGETS = [10, 50, 100, 500, 1000, 5000]


def run_instance(args):
    name, board, b0, b1, seed, n_hands, k, budgets = args
    rng = np.random.default_rng(seed)
    r0 = sample_range(board, b0, n_hands, rng)
    r1 = sample_range(board, b1, n_hands, rng)

    def mk():
        return RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    ev = mk()
    eq_s = mk()
    eq_s.train(2000)
    eq = eq_s.lifted(eq_s.average_strategy())
    bp = mk()
    bp.set_cells([equal_count_cells(bp, 0, k), equal_count_cells(bp, 1, k)])
    bp.train(400)
    blue = bp.lifted(bp.average_strategy())
    T = ev.tree
    settings = {"ip_depth1": (1, list(T.nodes[0].children)),
                "oop_depth2": (0, [c for c in T.nodes[T.nodes[0].children[0]].children
                                   if T.nodes[c].player == 0])}
    out = {"name": name, "seed": seed, "k": k, "n_hands": [len(r0), len(r1)], "settings": {}}

    def vs_blue(lifted, resolver):
        mixed = [None if x is None else x.copy() for x in blue]
        for i in T.decision_nodes:
            if T.nodes[i].player == resolver:
                mixed[i] = lifted[i]
        return float(ev.profile_values(mixed, resolver).sum())

    for sname, (resolver, roots) in settings.items():
        res = {"resolver": resolver, "roots": [list(T.nodes[r].history) for r in roots],
               "equilibrium": {"br_value": opponent_br_value(ev, eq, resolver), "vs_blueprint": vs_blue(eq, resolver)},
               "blueprint": {"br_value": opponent_br_value(ev, blue, resolver), "vs_blueprint": vs_blue(blue, resolver)}}
        for mode in ("unsafe", "safe"):
            for b in budgets:
                r = resolve_all(mk, blue, roots, mode, b)
                res[f"{mode}@{b}"] = {"br_value": opponent_br_value(ev, r["lifted"], resolver),
                                      "vs_blueprint": vs_blue(r["lifted"], resolver),
                                      "iterations": r["iterations"]}
        for b in budgets:
            f = mk()
            n = f.train_for_seconds(b / 1000.0)
            lf = f.lifted(f.average_strategy())
            res[f"full@{b}"] = {"br_value": opponent_br_value(ev, lf, resolver), "vs_blueprint": vs_blue(lf, resolver),
                                "iterations": [n]}
        out["settings"][sname] = res
    return out


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(instances, budgets):
    summ = {}
    for k in sorted({i["k"] for i in instances}):
        for sname in instances[0]["settings"]:
            rows = [i["settings"][sname] for i in instances if i["k"] == k]
            base = np.array([r["blueprint"]["br_value"] for r in rows])
            eqv = np.array([r["equilibrium"]["br_value"] for r in rows])
            gap = base - eqv
            arms = ["blueprint", "equilibrium"] + [f"{m}@{b}" for m in ("unsafe", "safe", "full") for b in budgets]
            s = {}
            for a in arms:
                br = np.array([r[a]["br_value"] for r in rows])
                vb = np.array([r[a]["vs_blueprint"] for r in rows])
                d = base - br
                s[a] = {"br_value_mean": float(br.mean()), "br_reduction_vs_blueprint_mean": float(d.mean()),
                        "br_reduction_ci95": bootstrap_ci(d), "fraction_of_gap_closed_mean": float(np.mean(d / np.maximum(gap, 1e-9))),
                        "worse_than_blueprint_count": int((d < -1e-9).sum()), "n": len(rows),
                        "vs_blueprint_mean": float(vb.mean())}
            summ[f"k{k}:{sname}"] = s
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hands", type=int, default=48)
    ap.add_argument("--ks", type=str, default="3,6")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    budgets = BUDGETS
    boards = list(SUBGAMES) + list(CONFIRM_BOARDS)
    if a.quick:
        budgets, boards, a.seeds, a.hands = [10, 100], boards[:2], 1, 16
    jobs = [(n, b, b0, b1, 7000 + s, a.hands, int(k), budgets) for n, b, b0, b1 in boards
            for s in range(a.seeds) for k in a.ks.split(",")]
    t0 = time.time()
    with Pool(a.workers) as pool:
        inst = pool.map(run_instance, jobs, chunksize=1)
    summ = summarize(inst, budgets)
    doc = {"format": "pokeralpha.phase2.local_resolving/v1",
           "question": "Does resolving a subtree at decision time (full hand resolution, time budget) improve on a "
                       "coarse precomputed blueprint, and is it safe?",
           "domain": "52-card river subgames, pot 10, stack 20, bets 50%/100%/all-in, cap 2; blueprint = K strength cells",
           "assumptions": ["unsafe: opponent range at the subtree root = blueprint reach (opponent assumed to follow the blueprint)",
                           "safe: re-solve gadget; opponent may terminate with blueprint counterfactual value per hand "
                           "(Burch, Johanson & Bowling 2014); guarantee is asymptotic in the re-solve, measured here at finite budgets",
                           "resolver's own range = blueprint reach; only the resolver's strategy changes"],
           "budgets_ms": budgets, "workers": a.workers,
           "timing_note": f"{a.workers} instances ran concurrently; budgets are wall-clock per process; iterations recorded",
           "summary": summ, "instances": inst, "seconds": round(time.time() - t0, 1),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "machine": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()}}
    out = OUT if not a.quick else OUT.with_name("local_resolving_quick.json")
    out.write_text(json.dumps(doc, indent=1, default=float))
    for key, s in summ.items():
        print(key)
        for arm, v in s.items():
            print(f"  {arm:14s} br {v['br_value_mean']:.4f}  red {v['br_reduction_vs_blueprint_mean']:+.4f} "
                  f"gap_closed {v['fraction_of_gap_closed_mean']:+.3f} worse {v['worse_than_blueprint_count']}/{v['n']} "
                  f"vs_bp {v['vs_blueprint_mean']:+.4f}")
    print("wrote", out, doc["seconds"], "s")


if __name__ == "__main__":
    main()
