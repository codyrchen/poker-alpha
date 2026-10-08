"""Phase 25: controlled MCCFR training probe (bucket encoder, fixed seed).

Trains external-sampling MCCFR on abstracted heads-up Hold'em and, at each
milestone, saves a checkpoint and records diagnostics that separate
*state-space discovery* (new information sets keep appearing, most seen
once) from *strategy learning* (policies at frequently visited information
sets settle):

    runtime, iterations/sec, infoset count, new infosets since last milestone,
    visit distribution (0 / 1 / 2-5 / 6-20 / >20 visits),
    L1 movement of the N most-visited infosets of the previous milestone,
    visit-weighted strategy entropy, checkpoint size, memory estimate, RSS.

"Visits" are the number of times an infoset was reached as the
*non-updating* player (external sampling accumulates the average strategy
only there); infosets reached only as the updating player show 0.

Exact exploitability of Hold'em is not computed.

Usage
-----
    python experiments/holdem_mccfr_validation.py --seed 0 \\
        --milestones 200,1000,5000 --ckpt-dir /tmp/ckpts --out run_seed0.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_alpha.abstraction import HoldemBucketEncoder  # noqa: E402
from poker_alpha.games import HoldemGame  # noqa: E402
from poker_alpha.solvers import MCCFRSolver  # noqa: E402
from poker_alpha.solvers.holdem_analysis import (infoset_visits,  # noqa: E402
                                                 solver_metrics)
from poker_alpha.solvers.serialize import save_checkpoint  # noqa: E402

BETS = {"b33": 0.33, "b75": 0.75, "b150": 1.5}


def make_encoder(name: str = "bucket"):
    from poker_alpha.solver_config import make_encoder as _make

    return _make(name)


ENCODER_NAMES = ("bucket", "transition", "transition_abstract", "compact",
                 "compact_exact")


def make_game(encoder: str = "bucket") -> HoldemGame:
    return HoldemGame(starting_stack=100.0, bet_fractions=dict(BETS),
                      encoder=make_encoder(encoder))


def visit_histogram(visits) -> dict:
    # strategy_sum adds a probability vector per visit, so totals are floats
    # like 0.9999999; round before bucketing.
    v = np.rint(np.array(list(visits.values())))
    n = max(len(v), 1)
    return {"0": float(np.mean(v == 0)) if len(v) else 0.0,
            "1": float(np.mean(v == 1)) if len(v) else 0.0,
            "2-5": float(np.mean((v >= 2) & (v <= 5))) if len(v) else 0.0,
            "6-20": float(np.mean((v >= 6) & (v <= 20))) if len(v) else 0.0,
            ">20": float(np.mean(v > 20)) if len(v) else 0.0,
            "n": n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--milestones", default="200,1000,5000")
    p.add_argument("--top-n", type=int, default=2000)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--encoder", choices=ENCODER_NAMES, default="bucket")
    p.add_argument("--locked-config", action="store_true",
                   help="train the locked HoldemSolverConfig (PRIMARY_CONFIG); "
                        "checkpoints carry its signature; overrides --encoder")
    p.add_argument("--legal-sizing-config", action="store_true",
                   help="train LEGAL_SIZING_CONFIG (HoldemSolverConfig v2: preflop raise-to "
                        "multiples, NLHE minimum bets/raises); checkpoints prefixed 'legal'")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue from this checkpoint (same game/config)")
    p.add_argument("--max-seconds", type=float, default=float("inf"),
                   help="stop at the first milestone reached after this budget")
    args = p.parse_args()
    milestones = [int(x) for x in args.milestones.split(",")]
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.legal_sizing_config:
        from poker_alpha.solver_config import LEGAL_SIZING_CONFIG

        game, tag = LEGAL_SIZING_CONFIG.build_game(), "legal"
    elif args.locked_config:
        from poker_alpha.solver_config import PRIMARY_CONFIG

        game, tag = PRIMARY_CONFIG.build_game(), "locked"
    else:
        game, tag = make_game(args.encoder), args.encoder
    if args.resume is not None:
        from poker_alpha.solvers.serialize import load_checkpoint

        solver = load_checkpoint(args.resume, game)
        milestones = [m for m in milestones if m > solver.iterations]
    else:
        solver = MCCFRSolver(game, seed=args.seed)
    start_it = solver.iterations
    prev_keys = 0
    prev_top = {}
    t_start = time.perf_counter()
    with open(args.out, "a" if args.resume else "w") as fh:
        for m in milestones:
            seg = time.perf_counter()
            solver.train(m - solver.iterations)
            seg = time.perf_counter() - seg
            visits = infoset_visits(solver)
            met = solver_metrics(solver)
            moves = []
            for key, old in prev_top.items():
                node = solver.infosets.get(key)
                if node is not None:
                    moves.append(float(np.abs(node.average_strategy() - old).sum()))
            top = sorted(visits, key=lambda k: (-visits[k], k))[:args.top_n]
            path = save_checkpoint(
                solver, args.ckpt_dir / f"{tag}_seed{args.seed}_it{m}.npz")
            row = {
                "seed": args.seed, "encoder": tag, "iterations": m,
                "config_signature": game.solver_config_signature(),
                "segment_seconds": seg,
                "total_seconds": time.perf_counter() - t_start,
                "iters_per_sec_segment": (m - (milestones[milestones.index(m) - 1]
                                               if milestones.index(m) else start_it)) / seg,
                "infosets": met.infosets,
                "new_infosets": met.infosets - prev_keys,
                "visit_fraction": visit_histogram(visits),
                "max_visits": met.visits_quantiles[2],
                "top_n": args.top_n,
                "top_n_prev_mean_l1": float(np.mean(moves)) if moves else None,
                "top_n_prev_median_l1": float(np.median(moves)) if moves else None,
                "entropy_bits_visit_weighted": met.mean_entropy_bits,
                "checkpoint_mb": os.path.getsize(path) / 1e6,
                "memory_mb_estimate": met.memory_bytes_estimate / 1e6,
                "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "checkpoint": str(path),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps(row), flush=True)
            prev_keys = met.infosets
            prev_top = {k: solver.infosets[k].average_strategy().copy() for k in top}
            if time.perf_counter() - t_start > args.max_seconds:
                print(f"stopping: time budget exhausted after {m} iterations")
                break


if __name__ == "__main__":
    main()
