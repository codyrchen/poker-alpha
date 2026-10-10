"""Phase 18/19 (final-trust): restricted-LBR lower bounds for the release
candidates on the full Hold'em game.

Runs HoldemLBR (validated on exact games: Spearman 0.986, no upper
violations — results/validation/lbr_validation.json) against the 200k
release, native 300k and native 1M strategies, both seats, N hands each,
in parallel processes.

The number reported is a LOWER-BOUND ESTIMATE of real-game exploitability,
restricted to the abstract action menu — NOT exact exploitability. Use it
to RANK candidates; absolute values are floors.

Compute record: 6 runs x ~18 min, parallel -> ~20 min wall; output
results/validation/lbr_holdem.json; stop = all runs complete.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CANDIDATES = {
    "release_200k": "results/strategy/holdem_v2_seed0_200k.npz",
    "native_300k": "results/strategy/candidates/native_300k/holdem_v2_native_seed0_300k.npz",
    "native_1m": "results/strategy/candidates/native_1m/holdem_v2_native_seed0_1000k.npz",
}


def run_one(job):
    name, path, seat, hands, seed = job
    from poker_alpha.solver_config import V2_CONFIG
    from poker_alpha.solvers.strategy_artifact import load_artifact
    from poker_alpha.validation.lbr import HoldemLBR

    game = V2_CONFIG.build_game()
    art = load_artifact(ROOT / path, game)
    lbr = HoldemLBR(game, lambda k: art.lookup(k))
    t0 = time.time()
    res = lbr.run(seat, hands, seed=seed)
    return name, seat, res.to_dict(), round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=10_000)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    jobs = [(name, path, seat, args.hands, 1_000 + seat)
            for name, path in CANDIDATES.items() for seat in (0, 1)]
    out = {name: {} for name in CANDIDATES}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, seat, res, secs in ex.map(run_one, jobs):
            out[name][f"seat{seat}"] = res
            print(f"{name} seat{seat}: {res['bb_per_100']:+.0f} bb/100 "
                  f"(CI {res['ci95_bb_per_100'][0]:+.0f}..{res['ci95_bb_per_100'][1]:+.0f}) "
                  f"[{secs}s]", flush=True)
    for name, seats in out.items():
        est = (seats["seat0"]["mean_bb_per_hand"] + seats["seat1"]["mean_bb_per_hand"]) / 2
        import numpy as np

        se = float(np.hypot(seats["seat0"]["stderr_bb_per_hand"],
                            seats["seat1"]["stderr_bb_per_hand"]) / 2)
        seats["lbr_lower_bound_bb_per_100"] = round(100 * est, 1)
        seats["lbr_se_bb_per_100"] = round(100 * se, 1)
    doc = {"format": "pokeralpha.lbr_holdem/v1", "hands_per_seat": args.hands,
           "semantics": ("restricted LBR (abstract action menu, myopic values, "
                         "sampled equities): APPROXIMATE LOWER BOUND on real-game "
                         "exploitability; validated on exact reduced games "
                         "(lbr_validation.json). Use for ranking."),
           "candidates": out, "seconds": round(time.time() - t0, 1)}
    dest = ROOT / "results/validation/lbr_holdem.json"
    dest.write_text(json.dumps(doc, indent=1))
    for name, seats in out.items():
        print(f"{name}: LBR lower bound {seats['lbr_lower_bound_bb_per_100']:+.0f} "
              f"± {seats['lbr_se_bb_per_100']:.0f} bb/100")
    print("wrote", dest)


if __name__ == "__main__":
    main()
