"""Phase 28: MCCFR backend benchmark for the locked solver config.

Measures, for the code tree at ``--root`` (so the same script benchmarks a
baseline checkout and the optimized tree on one machine):

* iterations/second and per-iteration latency (median / p95) after a warm-up;
* infoset touches per second (lookups of an information set during traversal);
* memory: process max RSS and a structural estimate of the infoset table;
* checkpoint save / load time (median / p95 over repetitions) and MB/s;
* an exact SHA-256 digest of the trained tables (bit-for-bit equivalence).

Usage::

    python experiments/phase28_benchmark.py --root . --out bench.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import tempfile
import time
from pathlib import Path


def exact_digest(solver) -> str:
    h = hashlib.sha256()
    for k in sorted(solver.infosets):
        n = solver.infosets[k]
        h.update(k.encode())
        h.update(",".join(n.actions).encode())
        h.update(n.regret_sum.tobytes())
        h.update(n.strategy_sum.tobytes())
    return h.hexdigest()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def table_bytes(solver) -> int:
    total = sys.getsizeof(solver.infosets)
    for k, n in solver.infosets.items():
        total += (sys.getsizeof(k) + sys.getsizeof(n) + n.regret_sum.nbytes
                  + n.strategy_sum.nbytes + 2 * 112 + sys.getsizeof(n.actions))
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--ckpt-reps", type=int, default=5)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root.resolve()))
    from poker_alpha.solver_config import PRIMARY_CONFIG
    from poker_alpha.solvers.serialize import load_checkpoint, save_checkpoint

    solver = PRIMARY_CONFIG.build_solver(seed=a.seed)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    solver.train(a.warmup)
    warm = time.perf_counter() - t0

    per = []
    t1 = time.perf_counter()
    for _ in range(a.iterations):
        s = time.perf_counter()
        solver.iterate()
        per.append(time.perf_counter() - s)
    measured = time.perf_counter() - t1
    digest = exact_digest(solver)

    # Infoset touches per iteration, counted on a copy-free extra segment
    # (the counting wrapper is excluded from the timings above).
    touches = [0]
    orig = solver._get_infoset

    def counted(key, actions):
        touches[0] += 1
        return orig(key, actions)

    solver._get_infoset = counted
    extra = 100
    for _ in range(extra):
        solver.iterate()
    del solver._get_infoset
    touches_per_iter = touches[0] / extra

    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ck = {"save": [], "load": []}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.npz"
        for _ in range(a.ckpt_reps):
            s = time.perf_counter()
            save_checkpoint(solver, path)
            ck["save"].append(time.perf_counter() - s)
            s = time.perf_counter()
            load_checkpoint(path, PRIMARY_CONFIG.build_game())
            ck["load"].append(time.perf_counter() - s)
        size = path.stat().st_size
    its = a.iterations / measured
    out = {
        "label": a.label,
        "config_signature": PRIMARY_CONFIG.signature(),
        "seed": a.seed,
        "warmup_iterations": a.warmup,
        "warmup_seconds": warm,
        "measured_iterations": a.iterations,
        "measured_seconds": measured,
        "iterations_per_second": its,
        "iteration_ms_median": 1e3 * pct(per, 0.5),
        "iteration_ms_p95": 1e3 * pct(per, 0.95),
        "infoset_touches_per_iteration": touches_per_iter,
        "infoset_touches_per_second": touches_per_iter * its,
        "infosets_final": len(solver.infosets),
        "table_bytes_estimate": table_bytes(solver),
        "bytes_per_infoset_estimate": table_bytes(solver) / max(len(solver.infosets), 1),
        "max_rss_mb_start": rss0 / 1024, "max_rss_mb_end": rss1 / 1024,
        "checkpoint_bytes": size,
        "checkpoint_save_s_median": pct(ck["save"], 0.5),
        "checkpoint_save_s_p95": pct(ck["save"], 0.95),
        "checkpoint_load_s_median": pct(ck["load"], 0.5),
        "checkpoint_load_s_p95": pct(ck["load"], 0.95),
        "checkpoint_save_mb_per_s": size / 1e6 / pct(ck["save"], 0.5),
        "checkpoint_load_mb_per_s": size / 1e6 / pct(ck["load"], 0.5),
        "exact_digest_after_warmup_plus_measured": digest,
    }
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
