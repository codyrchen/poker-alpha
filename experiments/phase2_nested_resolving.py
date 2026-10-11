"""Phase II hypothesis H13: re-solve after off-tree bets vs translate.

    python experiments/phase2_nested_resolving.py --workers 8
"""

from __future__ import annotations

import argparse
import json
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

from poker_alpha.research.translation import evaluate_nested_resolving  # noqa: E402

OUT = ROOT / "results" / "phase2" / "nested_resolving.json"
FINE = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
COARSE = {"B_50_100": (0.5, 1.0), "C_33_75_150": (0.33, 0.75, 1.5)}


def job(args):
    name, board, b0, b1, seed, cname = args
    rng = np.random.default_rng(seed)
    r0, r1 = sample_range(board, b0, 24, rng), sample_range(board, b1, 24, rng)
    r = evaluate_nested_resolving(board, r0, r1, COARSE[cname], FINE, iters=800, resolve_iters=400, us=1)
    return {"name": name, "seed": seed, "coarse": cname, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=6100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    boards = list(SUBGAMES) + list(CONFIRM_BOARDS)
    jobs = [(n, b, b0, b1, 6100 + s, c) for n, b, b0, b1 in boards for s in range(3) for c in COARSE]
    jobs = [(n, b, b0, b1, a.seed_base + (sd - 6100), c) for n, b, b0, b1, sd, c in jobs]
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(job, jobs, chunksize=1)
    arms = ["translate:nearest", "translate:nearest_log", "translate:pseudo_harmonic", "resolve:unsafe", "resolve:safe"]
    summ = {}
    for c in COARSE:
        rs = [r for r in res if r["coarse"] == c]
        base = np.array([r["restricted_menu_equilibrium"] for r in rs])
        summ[c] = {a_: {"cost_mean": float(np.mean([r[a_] for r in rs]) - base.mean()),
                        "worse_than_pseudo_harmonic": int(sum(r[a_] > r["translate:pseudo_harmonic"] + 1e-6 for r in rs)),
                        "better_than_pseudo_harmonic": int(sum(r[a_] < r["translate:pseudo_harmonic"] - 1e-6 for r in rs))}
                   for a_ in arms}
        summ[c]["n"] = len(rs)
    doc = {"format": "pokeralpha.phase2.nested_resolving/v1",
           "question": "after an opponent's off-tree bet, does re-solving the response subtree beat translation?",
           "definition": "cost = opponent BR value minus vs the best coarse-menu-restricted strategy (chips, pot 10); "
                         "resolver = IP; blueprint elsewhere = pseudo-harmonic translation; 400 CFR+ iterations per subtree",
           "summary": summ, "instances": res, "seconds": round(time.time() - t0, 1),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (Path(a.out) if a.out else OUT).write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
