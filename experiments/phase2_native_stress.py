"""Phase II Programs 21 + 23: native correctness stress and memory / scale limits.

  scale       one native run to --scale-iters, milestones: infosets, peak RSS,
              table bytes, segment throughput -> fit RSS vs infosets and
              infosets vs iterations; extrapolate (labelled) to 8/16/32 GB
  checkpoint  N continuous iterations == K x (N/K) with save/load between
              chunks (content hash of the exported state)
  parity      Python vs native on a shared random tape for --parity-iters
              iterations: iterate values and final tables bit-identical

    python experiments/phase2_native_stress.py --parts scale,checkpoint,parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from poker_alpha.native import NativeMCCFRSolver, native_version  # noqa: E402
from poker_alpha.solver_config import RELEASE_CONFIG  # noqa: E402

OUT = ROOT / "results" / "phase2"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def state_hash(solver):
    keys_u64, keys_str, off, tok, reg, strat, rng = solver._core.export_state()
    h = hashlib.sha256()
    for a in (np.asarray(keys_u64), np.asarray(off), np.asarray(tok), np.asarray(reg), np.asarray(strat), np.asarray(rng)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def scale(total):
    s = NativeMCCFRSolver(RELEASE_CONFIG, seed=11)
    marks = [m for m in (10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 700_000, 1_000_000, 1_500_000, 2_000_000)
             if m <= total]
    rows, done = [], 0
    base_rss = rss_mb()
    for m in marks:
        t = time.perf_counter()
        s.train(m - done)
        dt = time.perf_counter() - t
        met = s.metrics()
        rows.append({"iterations": m, "infosets": met["infosets"], "segment_it_per_s": (m - done) / dt,
                     "peak_rss_mb": rss_mb(), "approx_table_bytes": met["approx_table_bytes"],
                     "feature_cache_entries": met["feature_cache_entries"],
                     "river_board_cache_entries": met["river_board_cache_entries"]})
        done = m
        print(rows[-1], flush=True)
    inf = np.array([r["infosets"] for r in rows], float)
    rss = np.array([r["peak_rss_mb"] for r in rows], float)
    its = np.array([r["iterations"] for r in rows], float)
    slope, intercept = np.polyfit(inf, rss, 1)                     # MB per infoset
    # infosets vs iterations: power-law fit on the tail (growth is sublinear)
    b, loga = np.polyfit(np.log(its[-4:]), np.log(inf[-4:]), 1)
    ext = {}
    for gb in (8, 16, 32):
        need_inf = (gb * 1024 - intercept) / slope
        need_it = float(np.exp((np.log(need_inf) - loga) / b)) if b > 0 else None
        ext[f"{gb}GB"] = {"infosets": need_inf, "iterations_for_that_table_at_this_abstraction": need_it}
    return {"baseline_process_rss_mb": base_rss, "milestones": rows,
            "fit_rss_mb_per_infoset": slope, "fit_rss_intercept_mb": intercept,
            "fit_infosets_vs_iterations_exponent_tail": b,
            "bytes_per_infoset_table_only": rows[-1]["approx_table_bytes"] / rows[-1]["infosets"],
            "extrapolation": ext,
            "extrapolation_label": "EXTRAPOLATION from a linear RSS-vs-infosets fit and a 4-point power-law fit of "
                                   "infosets vs iterations; the abstraction's reachable key space is finite, so "
                                   "the iteration estimates are upper-bound-unsafe and likely never reached"}


def checkpoint(total, chunks):
    a = NativeMCCFRSolver(RELEASE_CONFIG, seed=5)
    t = time.perf_counter()
    a.train(total)
    t_cont = time.perf_counter() - t
    ha = state_hash(a)
    b = NativeMCCFRSolver(RELEASE_CONFIG, seed=5)
    step = total // chunks
    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        for k in range(chunks):
            b.train(step)
            p = Path(td) / f"ck{k}.npz"
            b.save_checkpoint(p)
            b = NativeMCCFRSolver.load_checkpoint(p, RELEASE_CONFIG)
    t_chunk = time.perf_counter() - t
    hb = state_hash(b)
    return {"iterations": total, "chunks": chunks, "continuous_hash": ha, "chunked_hash": hb, "identical": ha == hb,
            "continuous_seconds": t_cont, "chunked_with_save_load_seconds": t_chunk, "iterations_after": b.iterations}


def parity(n):
    from native_helpers import TapeRng, make_tape
    from poker_alpha.solvers.mccfr import MCCFRSolver
    tape = make_tape(seed=2026, length=n * 4000)
    py = MCCFRSolver(RELEASE_CONFIG.build_game(), seed=0)
    py.rng = TapeRng(tape)
    nat = NativeMCCFRSolver(RELEASE_CONFIG, seed=0, _tape=tape)
    t = time.perf_counter()
    mism = 0
    first = None
    for i in range(n):
        a = py.iterate()
        b = nat._core.iterate()
        if a != b:
            mism += 1
            if first is None:
                first = i
    secs = time.perf_counter() - t
    pa = py.average_strategy()
    na = nat.average_strategy()
    table_equal = set(pa) == set(na) and all(pa[k][x] == na[k][x] for k in pa for x in pa[k])
    return {"iterations": n, "iterate_value_mismatches": mism, "first_mismatch": first, "tape_position_python": py.rng.pos,
            "tape_position_native": nat._core.tape_pos(), "average_strategy_bit_identical": table_equal,
            "infosets": len(pa), "seconds": secs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="scale,checkpoint,parity")
    ap.add_argument("--scale-iters", type=int, default=1_000_000)
    ap.add_argument("--ckpt-iters", type=int, default=100_000)
    ap.add_argument("--parity-iters", type=int, default=10_000)
    a = ap.parse_args()
    parts = a.parts.split(",")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if "scale" in parts:
        doc = {"format": "pokeralpha.phase2.native_memory/v1", "native_backend": native_version(), "commit": commit,
               "config": RELEASE_CONFIG.signature(), **scale(a.scale_iters)}
        (OUT / "native_memory.json").write_text(json.dumps(doc, indent=1, default=float))
    if "checkpoint" in parts or "parity" in parts:
        doc = {"format": "pokeralpha.phase2.native_correctness_stress/v1", "commit": commit,
               "native_backend": native_version()}
        if "checkpoint" in parts:
            doc["checkpoint"] = checkpoint(a.ckpt_iters, 10)
            print(doc["checkpoint"], flush=True)
        if "parity" in parts:
            doc["shared_tape_parity"] = parity(a.parity_iters)
            print(doc["shared_tape_parity"], flush=True)
        prev = OUT / "native_correctness_stress.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            for k in ("checkpoint", "shared_tape_parity"):
                if k not in doc and k in old:
                    doc[k] = old[k]
        prev.write_text(json.dumps(doc, indent=1, default=float))


if __name__ == "__main__":
    main()
