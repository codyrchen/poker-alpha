"""Phase 31: final benchmark — offline training cost vs online decision latency.

Offline numbers are read from the Phase 28/29 result files (training is not
re-run). Online latency is measured here: for each input source, the time
to build the ObservedTableState and the time for ``analyze`` with each
decision method (solver lookup, heuristic, Monte Carlo rollouts), median and
p95 over repeated runs, plus strategy-artifact load time.

Usage::

    python experiments/final_benchmark.py --strategy results/strategy/holdem_v1_seed0.npz
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision import DecisionConfig  # noqa: E402
from poker_alpha.history import load_hands  # noqa: E402
from poker_alpha.pipeline import (analyze, load_solver,  # noqa: E402
                                  observe_hand_history, observe_manual,
                                  observe_screenshot, observe_simulation)
from poker_alpha.platform_demo import manual_hu_spot  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def timed(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, {"median_ms": 1e3 * statistics.median(ts), "p95_ms": 1e3 * pct(ts, 0.95),
                 "reps": reps}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", type=Path, required=True)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "validation" / "final_benchmark.json")
    a = p.parse_args()

    solver, load = timed(lambda: load_solver(a.strategy), 3)
    from poker_alpha.holdem_demo import CHIP, build_hand

    st = build_hand()
    hands = load_hands(ROOT / "tests" / "fixtures" / "hands" / "sample.json")
    sources = {
        "manual (HU 100BB)": lambda: observe_manual(manual_hu_spot()),
        "simulation (6-max)": lambda: observe_simulation(st, st.actor, chip_unit=CHIP),
        "hand_history": lambda: observe_hand_history(hands[0]),
        "screenshot (synthetic, 3 frames)": lambda: observe_screenshot(
            ROOT / "tests" / "fixtures" / "table.png"),
    }
    methods = {
        "solver if eligible, else heuristic (equity 1500 sims)": (DecisionConfig(equity_simulations=1500), solver),
        "heuristic (equity 1500 sims, no solver)": (DecisionConfig(equity_simulations=1500), None),
        "rollout 400 sims (no solver)": (DecisionConfig(equity_simulations=1500, rollout_simulations=400), None),
    }
    online = {}
    for sname, make in sources.items():
        obs, t_obs = timed(make, max(3, a.reps // 3) if "screenshot" in sname else a.reps)
        row = {"observe": t_obs, "analyze": {}}
        for mname, (cfg, prov) in methods.items():
            rep, t = timed(lambda: analyze(obs, replace(cfg), solver=prov),
                           max(3, a.reps // 3) if "rollout" in mname else a.reps)
            t["method_used"] = rep.method
            row["analyze"][mname] = t
        online[sname] = row
        print(sname, json.dumps(row), flush=True)

    def read(name):
        f = ROOT / "results" / "validation" / name
        return json.loads(f.read_text()) if f.exists() else None

    bench = read("backend_benchmark.json")
    train = read("holdem_training_v1_300k.json") or read("holdem_training_v1.json")
    offline = {}
    if bench:
        offline["mccfr_iterations_per_second_single_core"] = bench["optimized"]["iterations_per_second"]
    if train:
        offline["training_runs"] = {
            s: {"iterations": rows[-1]["iterations"],
                "wall_clock_seconds": rows[-1]["wall_clock_seconds"],
                "checkpoint_mb": rows[-1]["checkpoint_mb"],
                "infosets": rows[-1]["infosets"]}
            for s, rows in train["convergence_proxies"].items()}
        offline["artifact_bytes"] = train["artifact"]["bytes"]
    doc = {"format": "pokeralpha.final_benchmark/v1",
           "machine": "4-vCPU cloud container, CPython, single process per measurement",
           "offline_training": offline,
           "online": {"strategy_artifact_load": load, "sources": online},
           "note": "online latency excludes training; solver lookups are dictionary reads after "
                   "a one-off artifact load; equity and rollouts dominate online time"}
    a.out.write_text(json.dumps(doc, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
