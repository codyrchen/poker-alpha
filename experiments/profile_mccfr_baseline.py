"""Python MCCFR baseline profile for the native-solver project (Phase 1).

Measures representative release-config (v2) Hold'em MCCFR throughput and
breaks the runtime down by component, so the native port has a defensible
"before" number. Writes:

* ``results/benchmarks/mccfr_python_baseline.json`` — machine, config,
  throughput at each milestone, node counters, memory;
* ``results/benchmarks/mccfr_python_profile.txt`` — cProfile top functions.

Usage::

    python experiments/profile_mccfr_baseline.py [--iterations 10000]
        [--profile-iterations 1000] [--seed 0]

Cold start (empty infoset table) and warm training (table populated by the
preceding iterations) are reported separately: MCCFR discovers most of its
infosets early, so early iterations both run slower (allocation) and are not
representative of long-run speed.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from poker_alpha.solver_config import RELEASE_CONFIG
from poker_alpha.solvers.mccfr import MCCFRSolver


class CountingGame:
    """Wraps a game, counting node-type visits (delegates everything else)."""

    def __init__(self, game):
        self._g = game
        self.counts = {"terminal": 0, "chance": 0, "decision": 0,
                       "legal_actions": 0, "next_state": 0, "infoset_key": 0}

    def __getattr__(self, name):
        return getattr(self._g, name)

    def is_terminal(self, s):
        r = self._g.is_terminal(s)
        if r:
            self.counts["terminal"] += 1
        return r

    def is_chance(self, s):
        r = self._g.is_chance(s)
        if r:
            self.counts["chance"] += 1
        return r

    def current_player(self, s):
        self.counts["decision"] += 1
        return self._g.current_player(s)

    def infoset_key(self, s):
        self.counts["infoset_key"] += 1
        return self._g.infoset_key(s)

    def legal_actions(self, s):
        self.counts["legal_actions"] += 1
        return self._g.legal_actions(s)

    def next_state(self, s, a):
        self.counts["next_state"] += 1
        return self._g.next_state(s, a)


def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return ru / (1 << 20) if sys.platform == "darwin" else ru / (1 << 10)


def timed_block(solver, iters: int) -> dict:
    t0 = time.perf_counter()
    for _ in range(iters):
        solver.iterate()
    dt = time.perf_counter() - t0
    return {"iterations": iters, "seconds": round(dt, 3),
            "it_per_s": round(iters / dt, 3),
            "ms_per_it": round(1000 * dt / iters, 3),
            "infosets_after": len(solver.infosets)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10_000,
                    help="total iterations for the timing run")
    ap.add_argument("--profile-iterations", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/benchmarks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RELEASE_CONFIG
    report: dict = {
        "format": "pokeralpha.mccfr_python_baseline/v1",
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "config_signature": cfg.signature(),
        "seed": args.seed,
        "milestones": [],
    }

    # -- throughput, measured in blocks so cold vs warm is visible ---------
    solver = cfg.build_solver(seed=args.seed)
    blocks = []
    remaining = args.iterations
    schedule = [1_000, 4_000, 5_000, 10_000, 30_000, 50_000]
    for size in schedule:
        take = min(size, remaining)
        if take <= 0:
            break
        blocks.append(timed_block(solver, take))
        remaining -= take
    report["milestones"] = blocks
    report["total_iterations"] = solver.iterations
    report["final_infosets"] = len(solver.infosets)
    report["rss_mb"] = round(rss_mb(), 1)

    # -- node counters on a short counted run ------------------------------
    counted = CountingGame(cfg.build_game())
    csolver = MCCFRSolver(counted, seed=args.seed)
    n_count = min(1_000, args.iterations)
    t0 = time.perf_counter()
    for _ in range(n_count):
        csolver.iterate()
    dt = time.perf_counter() - t0
    c = counted.counts
    report["counters"] = {
        "iterations": n_count,
        "seconds": round(dt, 3),
        **c,
        "decision_nodes_per_iteration": round(c["decision"] / n_count, 2),
        "infosets_discovered": len(csolver.infosets),
    }

    # -- cProfile breakdown ------------------------------------------------
    psolver = cfg.build_solver(seed=args.seed + 1)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(args.profile_iterations):
        psolver.iterate()
    prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf)
    stats.sort_stats("cumulative").print_stats(45)
    stats.sort_stats("tottime").print_stats(45)
    profile_txt = out_dir / "mccfr_python_profile.txt"
    profile_txt.write_text(buf.getvalue())

    # Extract the top self-time entries into the JSON for quick reading.
    rows = []
    for func, (cc, nc, tt, ct, _) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append((tt, ct, nc, f"{func[0]}:{func[1]}:{func[2]}"))
    rows.sort(reverse=True)
    total_tt = sum(r[0] for r in rows)
    report["profile_top_self_time"] = [
        {"function": name, "self_s": round(tt, 3),
         "self_pct": round(100 * tt / total_tt, 1),
         "cumulative_s": round(ct, 3), "calls": nc}
        for tt, ct, nc, name in rows[:25]
    ]
    report["profile_iterations"] = args.profile_iterations

    out = out_dir / "mccfr_python_baseline.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("machine", "config_signature", "milestones",
                       "final_infosets", "rss_mb", "counters")}, indent=1))
    print(f"\nwrote {out} and {profile_txt}")


if __name__ == "__main__":
    main()
