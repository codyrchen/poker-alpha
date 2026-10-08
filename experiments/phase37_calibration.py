"""Phase 37A: calibrate solver-confidence thresholds on exact games.

For exact reduced games (the Phase 33 reduced preflop game and two river
subgames) run external-sampling MCCFR with three seeds and record, per
information set of seed 0 at the final checkpoint:

* visits (non-updating-player visits, the same statistic stored for the
  Hold'em strategy),
* movement: L1 between the seed-0 average strategy at T/3 and at T,
* seed disagreement: mean pairwise L1 of the three seeds' average strategies,
* true error: L1 between seed 0's average strategy and the exact (CFR+)
  solution.

Then tabulates true error against each signal (binned) and picks the
smallest thresholds whose bins exceed a true-error level. Writes
``results/validation/solver_gate_calibration.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase33_reduced_games import SUBGAMES, percentile_range  # noqa: E402

from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame  # noqa: E402
from poker_alpha.solvers import MCCFRSolver  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402

OUT = ROOT / "results" / "validation" / "solver_gate_calibration.json"


def avg(node):
    return node.average_strategy()


def collect(game, ref_iters, T, seeds=(0, 1, 2)):
    ref_s = CFRPlusSolver(game)
    ref_s.train(ref_iters)
    ref = {k: n.average_strategy() for k, n in ref_s.infosets.items()}
    runs = {}
    mid = {}
    for sd in seeds:
        s = MCCFRSolver(game, seed=sd)
        s.train(T // 3)
        if sd == seeds[0]:
            mid = {k: n.average_strategy().copy() for k, n in s.infosets.items()
                   if n.strategy_sum.sum() > 0}
        s.train(T - T // 3)
        runs[sd] = s
    rows = []
    s0 = runs[seeds[0]]
    for k, n in s0.infosets.items():
        v = float(n.strategy_sum.sum())
        if v <= 0 or k not in ref:
            continue
        p0 = n.average_strategy()
        others = [runs[sd].infosets.get(k) for sd in seeds[1:]]
        pols = [p0] + [o.average_strategy() for o in others if o is not None and o.strategy_sum.sum() > 0]
        dis = np.mean([np.abs(a - b).sum() for i, a in enumerate(pols) for b in pols[i + 1:]]) \
            if len(pols) > 1 else np.nan
        mv = float(np.abs(p0 - mid[k]).sum()) if k in mid else np.nan
        rows.append((v, mv, float(dis), float(np.abs(p0 - ref[k]).sum())))
    return np.array(rows)


def binned(x, err, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 5:
            out.append({"bin": [lo, hi], "n": int(m.sum()),
                        "median_true_error": round(float(np.median(err[m])), 4),
                        "p90_true_error": round(float(np.quantile(err[m], 0.9)), 4),
                        "share_error_gt_0.5": round(float(np.mean(err[m] > 0.5)), 4)})
    return out


def bin_threshold(bins, level, direction):
    """Lowest bin edge from which every higher bin's median true error exceeds
    ``level`` ('above'), or highest edge below which every bin does ('below')."""
    if direction == "above":
        for i in range(len(bins)):
            if all(b["median_true_error"] > level for b in bins[i:]):
                return bins[i]["bin"][0]
        return None
    edge = None
    for b in bins:
        if b["median_true_error"] > level:
            edge = b["bin"][1]
        else:
            break
    return edge


def main():
    t = time.time()
    data = []
    games = [("reduced_preflop", ReducedPreflopGame(equity_samples=1500, seed=1), 1000)]
    for name, board, b0, b1 in SUBGAMES[:2]:
        games.append((name, RiverSubgame(board, percentile_range(board, b0, 24),
                                         percentile_range(board, b1, 24), pot=10, stack=20,
                                         bets=(0.5, 1.0), raise_cap=2, name=name), 800))
    per_game = {}
    for name, g, ref_iters in games:
        for T in (3000, 30000):
            rows = collect(g, ref_iters, T)
            data.append(rows)
            per_game[f"{name}@{T}"] = {"infosets": len(rows),
                                       "median_true_error": round(float(np.median(rows[:, 3])), 4)}
            print(name, T, per_game[f"{name}@{T}"], round(time.time() - t), flush=True)
    d = np.vstack(data)
    vis, mv, dis, err = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    ok = ~np.isnan(dis) & ~np.isnan(mv)
    vis, mv, dis, err = vis[ok], mv[ok], dis[ok], err[ok]
    corr = {"spearman_visits_vs_error": float(_spearman(vis, err)),
            "spearman_movement_vs_error": float(_spearman(mv, err)),
            "spearman_seed_disagreement_vs_error": float(_spearman(dis, err))}
    by_vis = binned(vis, err, [0, 5, 10, 20, 50, 100, 300, 1000, 1e9])
    by_mv = binned(mv, err, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 2.01])
    by_dis = binned(dis, err, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.01])
    doc = {
        "format": "pokeralpha.solver_gate_calibration/v1",
        "games": per_game, "pooled_infosets": int(len(err)),
        "correlations": corr,
        "by_visits": by_vis, "by_movement": by_mv, "by_seed_disagreement": by_dis,
        "thresholds": {
            "reject_seed_disagreement_at_or_above": bin_threshold(by_dis, 0.5, "above"),
            "low_conf_seed_disagreement_at_or_above": bin_threshold(by_dis, 0.25, "above"),
            "reject_movement_at_or_above": bin_threshold(by_mv, 0.5, "above"),
            "low_conf_movement_at_or_above": bin_threshold(by_mv, 0.25, "above"),
            "reject_visits_below": bin_threshold(by_vis, 0.5, "below"),
            "low_conf_visits_below": bin_threshold(by_vis, 0.25, "below"),
        },
        "threshold_rule": "lowest bin edge from which every bin's median true error exceeds 0.5 (reject) "
                          "or 0.25 (low confidence); visits: highest edge below which every bin does",
        "true_error": "L1 (0..2) between seed-0 MCCFR average strategy and the CFR+ solution",
        "seconds": round(time.time() - t)}
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["thresholds"], indent=1), json.dumps(corr))


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


if __name__ == "__main__":
    main()
