"""Phase II Program 22: evaluator stress.

Exhaustive native-vs-Python parity over ALL 6-card (20,358,520) and ALL
7-card (133,784,560) hands, sharded across processes by the first card, plus
a suit-permutation metamorphic check (all 24 suit permutations must leave
the hand value unchanged) on a random sample.

    python experiments/phase2_evaluator_stress.py [--workers 8] [--sizes 6,7]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "phase2" / "evaluator_stress.json"


def pack(value) -> int:
    cat, *ties = value
    out = cat << 20
    for t, sh in zip(ties, (16, 12, 8, 4, 0)):
        out |= t << sh
    return out


def shard(args):
    size, first = args
    from poker_alpha.native import native_module
    from poker_alpha.poker.evaluator import evaluate_best_codes
    m = native_module()
    n = mism = 0
    example = None
    for rest in itertools.combinations(range(first + 1, 52), size - 1):
        h = [first, *rest]
        n += 1
        if m.debug_evaluate(h) != pack(evaluate_best_codes(h)):
            mism += 1
            if example is None:
                example = h
    return size, first, n, mism, example


def metamorphic(samples=200000, seed=0):
    from poker_alpha.native import native_module
    from poker_alpha.poker.evaluator import evaluate_best_codes
    m = native_module()
    rng = np.random.default_rng(seed)
    perms = list(itertools.permutations(range(4)))
    bad = 0
    for _ in range(samples):
        k = int(rng.integers(5, 8))
        h = rng.choice(52, k, replace=False).tolist()
        base = m.debug_evaluate(h)
        p = perms[int(rng.integers(len(perms)))]
        g = [p[c // 13] * 13 + c % 13 for c in h]
        if m.debug_evaluate(g) != base or pack(evaluate_best_codes(g)) != base:
            bad += 1
    return {"samples": samples, "permutations": "random one of 24 per sample", "failures": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sizes", default="6,7")
    a = ap.parse_args()
    doc = {"format": "pokeralpha.phase2.evaluator_stress/v1", "exhaustive": {}}
    for size in map(int, a.sizes.split(",")):
        t0 = time.time()
        with Pool(a.workers) as pool:
            res = pool.map(shard, [(size, f) for f in range(52 - size + 1)], chunksize=1)
        n = sum(r[2] for r in res)
        mism = sum(r[3] for r in res)
        doc["exhaustive"][f"{size}_card"] = {"hands": n, "mismatches": mism,
                                             "examples": [r[4] for r in res if r[4]][:5],
                                             "seconds": round(time.time() - t0, 1)}
        print(size, n, mism, round(time.time() - t0), "s", flush=True)
    t0 = time.time()
    doc["suit_permutation"] = metamorphic()
    doc["suit_permutation"]["seconds"] = round(time.time() - t0, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
