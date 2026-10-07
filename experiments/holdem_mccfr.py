"""Experiment: external-sampling MCCFR on abstracted heads-up NL Hold'em.

Trains MCCFR with checkpoint/resume and reports *scalable convergence
proxies*. Exact exploitability is NOT computed: it requires traversing the
full Hold'em chance tree, which is infeasible. The proxies below are
diagnostics, not exploitability bounds:

* average-strategy L1 change between consecutive checkpoints (visit-weighted),
* L1 change over the most-visited infosets ("high-frequency stability"),
* head-to-head cross-play of each checkpoint against the previous one,
* optional seed-to-seed stability (L1 between two independently seeded runs).

Usage
-----
    python experiments/holdem_mccfr.py --iterations 2000 --seed 0 \\
        --encoder bucket --stack-bb 100 --checkpoint-every 500 \\
        --checkpoint results/checkpoints/holdem_bucket.npz

If ``--checkpoint`` exists it is resumed (exactly, including the RNG stream);
it is rewritten every ``--checkpoint-every`` iterations.

Outputs (under --outdir, default ./results):
    data/holdem_mccfr_<encoder>.csv   one row per checkpoint
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from poker_alpha.abstraction import (HoldemBucketEncoder, RawHoldemEncoder,
                                     ToyHoldemEncoder)
from poker_alpha.games import HoldemGame
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.holdem_analysis import (cross_play,
                                                 describe_strategy_at,
                                                 infoset_visits,
                                                 solver_metrics,
                                                 strategy_l1_change,
                                                 top_infoset_stability)
from poker_alpha.solvers.serialize import load_checkpoint, save_checkpoint

ENCODERS = {
    "raw": RawHoldemEncoder,
    "toy": ToyHoldemEncoder,
    "bucket": HoldemBucketEncoder,
}


def make_game(encoder: str, stack_bb: float, bets: str) -> HoldemGame:
    fractions = {f"b{int(b)}": int(b) / 100.0 for b in bets.split(",") if b}
    return HoldemGame(starting_stack=stack_bb, bet_fractions=fractions,
                      encoder=ENCODERS[encoder]())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=1000,
                   help="total iterations (including any resumed ones)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--encoder", choices=sorted(ENCODERS), default="bucket")
    p.add_argument("--stack-bb", type=float, default=100.0)
    p.add_argument("--bets", default="33,75,150",
                   help="comma-separated pot percentages for bets/raises")
    p.add_argument("--crossplay-hands", type=int, default=200)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--compare-seed", type=int, default=None,
                   help="also train a second seed and report seed-to-seed L1")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    game = make_game(args.encoder, args.stack_bb, args.bets)
    if args.checkpoint and args.checkpoint.exists():
        solver = load_checkpoint(args.checkpoint, game)
        print(f"resumed {args.checkpoint} at iteration {solver.iterations}")
    else:
        solver = MCCFRSolver(game, seed=args.seed)

    rows = []
    prev = solver.average_strategy() if solver.iterations else None
    while solver.iterations < args.iterations:
        chunk = min(args.checkpoint_every, args.iterations - solver.iterations)
        start = time.perf_counter()
        solver.train(chunk)
        elapsed = time.perf_counter() - start
        strat = solver.average_strategy()
        visits = infoset_visits(solver)
        m = solver_metrics(solver)
        row = {
            "iteration": solver.iterations,
            "iters_per_sec": chunk / elapsed,
            "infosets": m.infosets,
            "memory_mb_estimate": m.memory_bytes_estimate / 1e6,
            "mean_entropy_bits": m.mean_entropy_bits,
            "visits_median": m.visits_quantiles[0],
            "visits_p90": m.visits_quantiles[1],
            "visits_max": m.visits_quantiles[2],
            "singleton_fraction": m.singleton_fraction,
        }
        if prev is not None:
            row["l1_change"] = strategy_l1_change(prev, strat, visits)
            row["top_k_l1_change"] = top_infoset_stability(prev, strat, visits,
                                                          args.top_k)
            if args.crossplay_hands:
                cp = cross_play(game, strat, prev, args.crossplay_hands,
                                seed=args.seed + solver.iterations)
                row["crossplay_vs_prev_bb"] = cp.mean_bb_per_hand
                row["crossplay_se_bb"] = cp.std_error
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            path = save_checkpoint(solver, args.checkpoint)
            row["checkpoint_mb"] = os.path.getsize(path) / 1e6
        rows.append(row)
        print(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()))
        prev = strat

    if args.compare_seed is not None:
        other = MCCFRSolver(make_game(args.encoder, args.stack_bb, args.bets),
                            seed=args.compare_seed)
        other.train(solver.iterations)
        l1 = top_infoset_stability(other.average_strategy(),
                                   solver.average_strategy(),
                                   infoset_visits(solver), args.top_k)
        print(f"seed-to-seed top-{args.top_k} L1 ({args.seed} vs "
              f"{args.compare_seed}): {l1:.4f}")

    visits = infoset_visits(solver)
    strat = solver.average_strategy()
    for hole in (("As", "Ks"), ("7c", "2d")):
        print(f"\nBTN {args.stack_bb:g} BB {hole[0]}{hole[1]} preflop unopened")
        print(describe_strategy_at(game, strat, visits, "BTN", hole).format())

    if rows:
        out = args.outdir / "data"
        out.mkdir(parents=True, exist_ok=True)
        csv = out / f"holdem_mccfr_{args.encoder}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
