"""Phase II hypothesis H4: exact cost of action translation, by mapping and by
off-tree size (river subgames, 9 boards x 3 seeds).

    python experiments/phase2_translation.py --workers 6
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

from poker_alpha.research.translation import MAPPINGS, evaluate_mappings  # noqa: E402

OUT = ROOT / "results" / "phase2" / "translation.json"
FINE = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
COARSE = {"B_50_100": (0.5, 1.0), "C_33_75_150": (0.33, 0.75, 1.5)}
SINGLE_ADDS = (0.25, 0.75, 1.5, 2.0)


def job(args):
    name, board, b0, b1, seed, cname = args
    rng = np.random.default_rng(seed)
    r0, r1 = sample_range(board, b0, 24, rng), sample_range(board, b1, 24, rng)
    coarse = COARSE[cname]
    out = {"name": name, "seed": seed, "coarse": cname, "full_fine": evaluate_mappings(board, r0, r1, coarse, FINE, iters=800)}
    # one off-tree size at a time (opponent menu = coarse + {x})
    out["single"] = {}
    for x in SINGLE_ADDS:
        if x in coarse:
            continue
        fine = tuple(sorted(set(coarse) | {x}))
        out["single"][str(x)] = evaluate_mappings(board, r0, r1, coarse, fine, iters=800)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=900)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    boards = list(SUBGAMES) + list(CONFIRM_BOARDS)
    jobs = [(n, b, b0, b1, a.seed_base + s, c) for n, b, b0, b1 in boards for s in range(a.seeds) for c in COARSE]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    summ = {}
    for c in COARSE:
        rs = [r for r in res if r["coarse"] == c]
        s = {}
        for us in ("us=0", "us=1"):
            base = np.array([r["full_fine"][us]["restricted_menu_equilibrium"] for r in rs])
            s[us] = {"translation_cost_vs_restricted_equilibrium": {m: float(np.mean([r["full_fine"][us][m] for r in rs]) - base.mean())
                                                                    for m in MAPPINGS},
                     "restricted_equilibrium_br_value": float(base.mean()),
                     "cost_by_single_offtree_size": {x: {m: float(np.mean([r["single"][x][us][m] - r["single"][x][us]["restricted_menu_equilibrium"]
                                                                          for r in rs])) for m in MAPPINGS}
                                                     for x in rs[0]["single"]}}
        summ[c] = s
    doc = {"format": "pokeralpha.phase2.translation/v1",
           "question": "how much worst-case value does action translation cost, by mapping and by off-tree size?",
           "definition": "cost = opponent best-response value vs our translated strategy minus vs our best strategy restricted to the "
                         "coarse menu in the same fine game (chips, pot 10); us=0 OOP, us=1 IP",
           "fine_menu": FINE, "coarse_menus": COARSE, "summary": summ, "instances": res, "seconds": round(time.time() - t0, 1),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (Path(a.out) if a.out else OUT).write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps(summ, indent=1)[:4000])


if __name__ == "__main__":
    main()
