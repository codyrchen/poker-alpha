"""HiddenInfoBench: a reproducible hidden-information benchmark suite (Phase II, Program 12).

    python -m poker_alpha.bench                 # full (~2-4 min)
    python -m poker_alpha.bench --quick         # smoke (~30 s)
    python -m poker_alpha.bench --tasks kuhn_cfr_plus,leduc_cfr_plus --out bench.json

Every task declares its game, algorithm, seeds, budget, metric and an
*expected invariant* (a pass/fail check with a stated bound). The output is
one machine-readable document, format ``pokeralpha.bench/v1``. There is NO
single aggregate score: metrics are reported separately because they are
not commensurable (exploitability in chips of different games, speedups,
calibration rates, parity booleans).

Hardware is recorded without user or host names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "pokeralpha.bench/v1"


@dataclass
class BenchTask:
    id: str
    description: str
    game: str
    algorithm: str
    seeds: List[int]
    budget: str
    metric: str
    expected: str
    fn: Callable[[bool], dict]


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def _exploit_curve(game, solver, checkpoints):
    from .solvers.evaluation import exploitability
    out, done = [], 0
    t0 = time.perf_counter()
    for c in checkpoints:
        solver.train(c - done)
        done = c
        out.append({"iterations": c, "exploitability": exploitability(game, solver.average_strategy()),
                    "seconds": round(time.perf_counter() - t0, 3)})
    return out


def t_kuhn(quick):
    from .games.kuhn import KuhnPoker
    from .solvers.cfr_plus import CFRPlusSolver
    g = KuhnPoker()
    curve = _exploit_curve(g, CFRPlusSolver(g), [10, 100, 300 if quick else 1000])
    e = curve[-1]["exploitability"]
    return {"curve": curve, "final_exploitability": e, "pass": e < 2e-3}


def t_leduc(quick):
    from .games.leduc import LeducPoker
    from .solvers.cfr_plus import CFRPlusSolver
    g = LeducPoker()
    curve = _exploit_curve(g, CFRPlusSolver(g), [10, 50, 100 if quick else 300])
    e = curve[-1]["exploitability"]
    return {"curve": curve, "final_exploitability": e, "pass": e < (0.03 if quick else 0.01)}


def t_cfr_family(quick):
    from .games.leduc import LeducPoker
    from .research.cfr_variants import DiscountedCFRSolver
    from .solvers.cfr import CFRSolver
    from .solvers.cfr_plus import CFRPlusSolver
    from .solvers.evaluation import exploitability
    from .solvers.mccfr import MCCFRSolver
    g = LeducPoker()
    iters = 60 if quick else 200
    res = {}
    for name, s in (("cfr", CFRSolver(g)), ("cfr+", CFRPlusSolver(g)), ("dcfr", DiscountedCFRSolver.preset(g, "dcfr")),
                    ("lcfr", DiscountedCFRSolver.preset(g, "lcfr"))):
        t = time.perf_counter()
        s.train(iters)
        res[name] = {"iterations": iters, "exploitability": exploitability(g, s.average_strategy()),
                     "seconds": round(time.perf_counter() - t, 3)}
    es = []
    for seed in range(3 if quick else 5):
        m = MCCFRSolver(g, seed=seed)
        m.train(3000 if quick else 20000)
        es.append(exploitability(g, m.average_strategy()))
    res["es_mccfr"] = {"iterations": 3000 if quick else 20000, "exploitability_mean": float(np.mean(es)),
                       "exploitability_std": float(np.std(es, ddof=1)), "seeds": len(es)}
    ok = res["cfr+"]["exploitability"] < res["cfr"]["exploitability"] and res["dcfr"]["exploitability"] < res["cfr"]["exploitability"]
    return {"results": res, "pass": ok, "invariant_detail": "at equal iterations CFR+ and DCFR beat vanilla CFR on Leduc"}


def t_auction(quick):
    from .games.auction import AuctionDuel
    from .solvers.cfr_plus import CFRPlusSolver
    from .solvers.evaluation import exploitability
    g = AuctionDuel(n_values=4, n_bids=4)
    s = CFRPlusSolver(g)
    s.train(300 if quick else 2000)
    strat = s.average_strategy()
    e = exploitability(g, strat)
    # monotone bidding: expected bid non-decreasing in value (player 0)
    exp_bid = [sum(int(a) * p for a, p in strat[f"A{v}"].items()) if f"A{v}" in strat else None for v in range(4)]
    mono = all(x is not None for x in exp_bid) and all(b >= a - 1e-6 for a, b in zip(exp_bid, exp_bid[1:]))
    return {"exploitability": e, "expected_bid_by_value_p0": exp_bid, "monotone": mono, "pass": e < 0.01 and mono,
            "note": "pass requires low exploitability and monotone expected bids"}


def t_tournament(quick):
    from .agents.adapters import RandomAgent, ScriptedKuhnAgent, kuhn_policy_agent
    from .agents.tournament import round_robin
    from .games.kuhn import KuhnPoker
    agents = {"random": RandomAgent(), "tight": ScriptedKuhnAgent("tight"), "aggressive": ScriptedKuhnAgent("aggressive"),
              "cfr": kuhn_policy_agent(2000 if quick else 20000)}
    rr = round_robin(KuhnPoker(), agents, hands=300 if quick else 3000, seed=0)
    m = rr["matrix"]["cfr"]
    # equilibrium never loses in expectation: lower CI bound > -tolerance for every opponent
    ok = all(v["ci95"][1] > -0.02 and v["mean"] > -0.05 for v in m.values())
    return {"cfr_row": m, "pass": ok, "invariant_detail": "the CFR policy is not beaten (beyond noise) by any baseline"}


def t_strategy_artifact(quick):
    from .solver_config import RELEASE_STRATEGY
    from .solvers.strategy_artifact import load_artifact
    t = time.perf_counter()
    art = load_artifact(ROOT / RELEASE_STRATEGY)
    secs = time.perf_counter() - t
    sums = [sum(p.values()) for p in list(art.strategy.values())[:20000]]
    ok = len(art) == 144025 and max(abs(s - 1) for s in sums) < 1e-3
    return {"infosets": len(art), "load_seconds": round(secs, 3), "max_prob_sum_error": float(max(abs(s - 1) for s in sums)),
            "pass": ok, "note": "full-game exploitability is not computable; integrity + normalization only"}


def t_native_parity(quick):
    from .native import native_available
    if not native_available():
        return {"skipped": "native backend not installed", "pass": None}
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from native_helpers import TapeRng, make_tape  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {"skipped": f"tape helpers unavailable: {e}", "pass": None}
    from .native import NativeMCCFRSolver
    from .solver_config import RELEASE_CONFIG
    from .solvers.mccfr import MCCFRSolver
    n = 30 if quick else 200
    tape = make_tape(seed=7, length=n * 4000)
    py = MCCFRSolver(RELEASE_CONFIG.build_game(), seed=0)
    py.rng = TapeRng(tape)
    nat = NativeMCCFRSolver(RELEASE_CONFIG, seed=0, _tape=tape)
    pv = [py.iterate() for _ in range(n)]
    nv = [nat._core.iterate() for _ in range(n)]
    return {"iterations": n, "values_bit_identical": pv == nv, "pass": pv == nv}


def t_native_throughput(quick):
    from .native import NativeMCCFRSolver, native_available
    if not native_available():
        return {"skipped": "native backend not installed", "pass": None}
    from .solver_config import RELEASE_CONFIG
    s = NativeMCCFRSolver(RELEASE_CONFIG, seed=0)
    s.train(2000)
    n = 5000 if quick else 30000
    t = time.perf_counter()
    s.train(n)
    ips = n / (time.perf_counter() - t)
    return {"warm_iterations": n, "it_per_s": round(ips, 1), "pass": ips > 100,
            "note": "warm throughput after 2k iterations; machine dependent"}


def t_confidence(quick):
    path = ROOT / "results" / "validation" / "confidence_calibration_dataset.json"
    rows = [r for r in json.loads(path.read_text())["rows"] if r.get("l1_error") is not None]
    acc = [r for r in rows if r["visits"] >= 20 and r.get("seed_disagreement") is not None and r["seed_disagreement"] < 0.2
           and (r.get("mv_recent") is None or r["mv_recent"] < 0.1)]
    fc = float(np.mean([r["wrong_best"] for r in acc]))
    allfc = float(np.mean([r["wrong_best"] for r in rows]))
    return {"rows": len(rows), "coverage": len(acc) / len(rows), "false_confidence_accepted": fc,
            "false_confidence_all": allfc, "pass": fc < 0.05 and fc < allfc / 3}


def t_abstraction(quick):
    from .games.reduced_holdem import RiverSubgame
    from .research.refinement import RiverRefinementProblem, refine
    from .research.river_solver import from_subgame
    board = ["Qh", "9h", "5h", "2h", "Kc"]
    r0 = ["AhKd", "QsQd", "KsKd", "9s9d", "AsQc", "JhTd", "8h7d", "QcJc", "KhJs", "Th8c", "AdAc", "6h4c"]
    r1 = ["Ah3c", "JhJd", "QdJd", "Kd9c", "7h7d", "AsAc", "TcTd", "KsQs", "9c8c", "Jh2c", "4h3d", "AcKc"]
    g = RiverSubgame(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    prob = RiverRefinementProblem(from_subgame(g), iterations=120 if quick else 300)
    base = prob.uniform([2, 2])
    ad = refine(prob, base, 3, "br_gain")["final"]
    un = prob.solve(prob.uniform(ad.cells_per_player))
    raw = prob.solve([np.arange(prob.solver.n[0]), np.arange(prob.solver.n[1])])
    return {"raw": raw.exploitability, "baseline": prob.solve(base).exploitability, "adaptive_br_gain": ad.exploitability,
            "uniform_matched": un.exploitability, "infosets": ad.num_infosets,
            "pass": raw.exploitability <= ad.exploitability and ad.exploitability < prob.solve(base).exploitability,
            "note": "single fixed instance; the controlled multi-seed study is experiments/phase2_adaptive_abstraction.py"}


def t_river_parity(quick):
    from .games.reduced_holdem import RiverSubgame
    from .research.river_solver import from_subgame
    from .solvers.cfr_plus import CFRPlusSolver
    from .solvers.evaluation import exploitability
    g = RiverSubgame(["Ks", "7d", "2c", "Qh", "4s"], ["AsKd", "2s2d", "QcJc", "5h3h"], ["KcTc", "QdTd", "7s7h", "AhJh"],
                     pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    a = CFRPlusSolver(g)
    a.train(30)
    b = from_subgame(g).solve(30)
    d = abs(exploitability(g, a.average_strategy()) - b.exploitability)
    return {"abs_exploitability_diff": d, "pass": d < 1e-10}


TASKS = [
    BenchTask("kuhn_cfr_plus", "Kuhn equilibrium convergence", "Kuhn", "CFR+", [0], "1,000 iterations",
              "exact exploitability (chips)", "< 2e-3", t_kuhn),
    BenchTask("leduc_cfr_plus", "Leduc equilibrium convergence", "Leduc", "CFR+", [0], "300 iterations",
              "exact exploitability (chips)", "< 0.01", t_leduc),
    BenchTask("cfr_family_leduc", "CFR family at equal iterations", "Leduc", "CFR, CFR+, DCFR, LCFR, ES-MCCFR",
              [0, 1, 2, 3, 4], "200 iterations (MCCFR 20k)", "exact exploitability", "CFR+ and DCFR beat CFR", t_cfr_family),
    BenchTask("auction_equilibrium", "Zero-sum auction duel equilibrium", "AuctionDuel(4,4)", "CFR+", [0],
              "2,000 iterations", "exact exploitability + bid monotonicity", "< 0.01 and monotone", t_auction),
    BenchTask("agent_tournament", "Seat-swapped Kuhn round robin", "Kuhn", "CFR policy vs baselines", [0],
              "3,000 hand pairs per pairing", "mean utility with 95% CI", "CFR not beaten beyond noise", t_tournament),
    BenchTask("strategy_artifact", "Release artifact integrity", "HUNL abstract", "load + normalization", [],
              "one load", "infosets, load time, prob-sum error", "144,025 infosets, sums within 1e-3", t_strategy_artifact),
    BenchTask("native_parity", "Python vs native MCCFR on a shared random tape", "HUNL abstract", "ES-MCCFR",
              [7], "200 iterations", "iterate values", "bit-identical", t_native_parity),
    BenchTask("native_throughput", "Native MCCFR warm throughput", "HUNL abstract", "ES-MCCFR", [0],
              "30,000 iterations", "iterations/s", "> 100 it/s", t_native_throughput),
    BenchTask("confidence_calibration", "Confidence v2 rule on exact-game calibration data", "9 exact games",
              "gate v2", [], "8,880 rows", "coverage, false-confidence rate", "accepted false-confidence < 5% and < 1/3 of base",
              t_confidence),
    BenchTask("abstraction_refinement", "Error-guided abstraction refinement (single instance)", "river subgame",
              "CFR+ + br_gain refinement", [0], "3 splits", "exact exploitability", "raw <= adaptive < baseline", t_abstraction),
    BenchTask("river_solver_parity", "Vectorized river solver vs generic CFR+", "river subgame", "CFR+", [],
              "30 iterations", "exploitability difference", "< 1e-10", t_river_parity),
]


def hardware() -> dict:
    info = {"os": platform.system(), "os_release": platform.release(), "arch": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "cpu_count": os.cpu_count()}
    try:
        if sys.platform == "darwin":
            info["cpu_model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["compiler"] = subprocess.check_output(["clang", "--version"], text=True).splitlines()[0]
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            info["compiler"] = subprocess.check_output(["c++", "--version"], text=True).splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    from .native import native_available, native_version
    info["native_backend"] = native_version() if native_available() else None
    return info


def run(task_ids=None, quick=False) -> dict:
    tasks = [t for t in TASKS if task_ids is None or t.id in task_ids]
    results = []
    t_all = time.perf_counter()
    for t in tasks:
        t0 = time.perf_counter()
        try:
            res = t.fn(quick)
            status = "skipped" if res.get("pass") is None else ("pass" if res["pass"] else "FAIL")
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=5)}
            status = "error"
        results.append({"id": t.id, "description": t.description, "game": t.game, "algorithm": t.algorithm,
                        "seeds": t.seeds, "budget": t.budget, "metric": t.metric, "expected_invariant": t.expected,
                        "status": status, "seconds": round(time.perf_counter() - t0, 2), "output": res})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {"format": FORMAT, "mode": "quick" if quick else "full", "commit": commit, "hardware": hardware(),
            "seconds": round(time.perf_counter() - t_all, 1),
            "status_counts": {s: sum(r["status"] == s for r in results) for s in ("pass", "FAIL", "error", "skipped")},
            "note": "no aggregate score by design; metrics are not commensurable",
            "tasks": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m poker_alpha.bench")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for t in TASKS:
            print(f"{t.id:24s} {t.game:18s} {t.algorithm:30s} expect: {t.expected}")
        return 0
    doc = run(a.tasks.split(",") if a.tasks else None, a.quick)
    for r in doc["tasks"]:
        print(f"{r['status']:8s} {r['id']:24s} {r['seconds']:7.2f}s")
    print(json.dumps(doc["status_counts"]))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, default=float))
        print("wrote", a.out)
    return 0 if doc["status_counts"]["FAIL"] == 0 and doc["status_counts"]["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
