"""Phase 6 (final-trust): true-error calibration dataset on exact games.

Nine exact games (3 reduced preflop variants + 6 fixed-river subgames), each
with an exact CFR+ reference. Five native-MCCFR seeds are trained per game
with snapshots at 1k/3k/10k/30k/100k; at evaluation milestones 3k/10k/30k
every reference infoset becomes one dataset row:

  targets: strategy L1 error vs the reference; EV regret and wrong-best-
           action against exact counterfactual action values q(I,a) under
           the reference profile; action margin (best q − second q).
  signals: visits (seed 0); movement features — historical (1k→cur),
           medium (two-rungs-back→cur), recent (prev→cur), recent
           normalized per 10k iterations; seed disagreement (5 seeds,
           mean pairwise L1); action count; pot proxy; street family.
  v1 gate: decision of the production gate (visits + historical movement +
           seed disagreement, production thresholds).

Collision dispersion is undefined here (perfect-recall keys) and recorded
as NaN — its evaluation stays heuristic (documented in confidence docs).

Rows carry game ids so Phase 7 can do leave-one-game-out evaluation; no
signal is calibrated and tested on the same game.

Writes results/validation/confidence_calibration_dataset.json (compact).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase33_reduced_games import SUBGAMES, percentile_range  # noqa: E402
from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame  # noqa: E402
from poker_alpha.native.tabular import NativeTabularMCCFR, build_tabular_tree  # noqa: E402
from poker_alpha.solvers.action_values import (action_margin,  # noqa: E402
                                               cf_action_values, policy_regret)
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.decision.solver_gate import GateThresholds, KeyStats, gate  # noqa: E402

OUT = ROOT / "results/validation/confidence_calibration_dataset.json"

LADDER = [1_000, 3_000, 10_000, 30_000, 100_000]
EVAL_MILESTONES = [3_000, 10_000, 30_000]
SEEDS = [0, 1, 2, 3, 4]


def games():
    yield "preflop_akqjt_10bb", "preflop", ReducedPreflopGame()
    yield "preflop_akqj_20bb", "preflop", ReducedPreflopGame(
        ranks="AKQJ", suits="sh", stack=20.0, open_to=3.0)
    yield "preflop_akqjt_5bb_jam", "preflop", ReducedPreflopGame(
        ranks="AKQJT", suits="sh", stack=5.0, open_to=2.0)
    for name, board, b0, b1 in SUBGAMES:
        r0 = percentile_range(board, b0, 24)
        r1 = percentile_range(board, b1, 24)
        yield (f"river_{name.split()[0]}", "river",
               RiverSubgame(board, r0, r1, pot=10.0, stack=20.0,
                            bets=(0.5, 1.0), raise_cap=2, name=name))


def main() -> None:
    t0 = time.time()
    th = GateThresholds.calibrated()
    rows = []
    game_meta = {}
    for gid, family, game in games():
        t_g = time.time()
        ref_solver = CFRPlusSolver(game)
        ref_solver.train(1_000 if family == "preflop" else 1_500)
        ref = ref_solver.average_strategy()
        qvals = cf_action_values(game, ref)
        tree = build_tabular_tree(game)

        # Train 5 seeds with snapshots (policy + visit dicts) at the ladder.
        snaps = {}   # (seed, milestone) -> {key: {action: prob}}
        vis = {}     # (seed, milestone) -> {key: visits}
        for seed in SEEDS:
            solver = NativeTabularMCCFR(game, seed=seed, tree=tree)
            done = 0
            for m in LADDER:
                solver.train(m - done)
                done = m
                sv = solver.average_strategy_with_visits()
                snaps[(seed, m)] = {k: p for k, (p, _) in sv.items()}
                vis[(seed, m)] = {k: v for k, (_, v) in sv.items()}

        l1 = lambda a, b, acts: sum(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in acts)  # noqa: E731
        for m in EVAL_MILESTONES:
            i = LADDER.index(m)
            cur = snaps[(0, m)]
            hist_base = snaps[(0, LADDER[0])]
            med_base = snaps[(0, LADDER[max(0, i - 2)])]
            prev_base = snaps[(0, LADDER[i - 1])] if i > 0 else hist_base
            interval = m - (LADDER[i - 1] if i > 0 else 0)
            for key, info in qvals.items():
                acts = info["actions"]
                refp = ref.get(key)
                cand = cur.get(key)
                sd_pols = [snaps[(s, m)].get(key) for s in SEEDS]
                sd_pols = [p for p in sd_pols if p is not None]
                sd = (float(np.mean([l1(a, b, acts) for ix, a in enumerate(sd_pols)
                                     for b in sd_pols[ix + 1:]]))
                      if len(sd_pols) > 1 else math.nan)
                mv_hist = l1(cand, hist_base.get(key, {}), acts) if cand else math.nan
                mv_med = l1(cand, med_base.get(key, {}), acts) if cand else math.nan
                mv_rec = l1(cand, prev_base.get(key, {}), acts) if cand else math.nan
                visits = vis[(0, m)].get(key, 0.0)
                v1 = gate(KeyStats(visits, mv_hist, sd, math.nan), th)
                q = info["q"]
                rows.append({
                    "game": gid, "family": family, "milestone": m, "key": key,
                    "n_actions": len(acts),
                    "reach": round(info["reach"], 6),
                    "l1_error": round(l1(cand or {}, refp, acts) if refp else math.nan, 4),
                    "ev_regret": round(policy_regret(q, cand), 5),
                    "uniform_regret": round(policy_regret(q, None), 5),
                    "action_margin": round(action_margin(q), 5),
                    "wrong_best": int(bool(cand) and
                                      max(cand, key=cand.get) != max(q, key=q.get)),
                    "visits": visits,
                    "mv_hist": None if math.isnan(mv_hist) else round(mv_hist, 4),
                    "mv_med": None if math.isnan(mv_med) else round(mv_med, 4),
                    "mv_recent": None if math.isnan(mv_rec) else round(mv_rec, 4),
                    "mv_recent_per10k": None if math.isnan(mv_rec) else
                        round(mv_rec * 10_000 / interval, 4),
                    "seed_disagreement": None if math.isnan(sd) else round(sd, 4),
                    "v1_status": v1.status,
                })
        game_meta[gid] = {"family": family, "signature": game.signature(),
                          "ref_exploitability": None,
                          "infosets": len(qvals),
                          "seconds": round(time.time() - t_g, 1)}
        print(gid, "done", round(time.time() - t0), "s,", len(rows), "rows",
              flush=True)

    doc = {"format": "pokeralpha.confidence_calibration_dataset/v1",
           "ladder": LADDER, "eval_milestones": EVAL_MILESTONES,
           "seeds": SEEDS, "games": game_meta,
           "visits_source": "native seed 0 strategy_sum totals (production semantics)",
           "collision": "NaN by construction (perfect-recall keys)",
           "rows": rows}
    OUT.write_text(json.dumps(doc))
    print("wrote", OUT, "rows:", len(rows), "in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
