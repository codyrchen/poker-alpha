"""Phase 18 (final-trust): validate restricted LBR against exact best response.

On exact reduced games (reduced preflop + two river subgames), for MCCFR
strategies at several maturities plus the CFR+ reference, compare:

    exact exploitability          (best_response_value, both seats)
vs  restricted-LBR estimate      (SubgameLBR, both seats, sampled play)

Metrics: per-strategy bias (exact − LBR, expected >= 0: LBR is a lower
bound), pooled Spearman correlation, and ranking accuracy (does LBR order
strategies by true exploitability?). GO criterion for using LBR on full
Hold'em: rank correlation >= 0.8 and no significant upper violations
(LBR > exact + 3 SE).

Writes results/validation/lbr_validation.json.
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
from poker_alpha.native.tabular import NativeTabularMCCFR, build_tabular_tree  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import best_response_value, exploitability  # noqa: E402
from poker_alpha.validation.lbr import SubgameLBR  # noqa: E402

OUT = ROOT / "results/validation/lbr_validation.json"
HANDS = 6_000
MILESTONES = [300, 3_000, 30_000]


def games():
    yield "reduced_preflop", ReducedPreflopGame()
    for name, board, b0, b1 in SUBGAMES[:2]:
        r0 = percentile_range(board, b0, 24)
        r1 = percentile_range(board, b1, 24)
        yield (f"river_{name.split()[0]}",
               RiverSubgame(board, r0, r1, pot=10.0, stack=20.0,
                            bets=(0.5, 1.0), raise_cap=2, name=name))


def main() -> None:
    t0 = time.time()
    doc = {"format": "pokeralpha.lbr_validation/v1", "hands_per_seat": HANDS,
           "milestones": MILESTONES, "games": {}}
    pooled_exact, pooled_lbr = [], []
    for gid, game in games():
        tree = build_tabular_tree(game)
        strategies = {}
        solver = NativeTabularMCCFR(game, seed=0, tree=tree)
        done = 0
        for m in MILESTONES:
            solver.train(m - done)
            done = m
            strategies[f"mccfr_{m}"] = solver.average_strategy()
        ref = CFRPlusSolver(game)
        ref.train(1_000)
        strategies["cfr_plus_ref"] = ref.average_strategy()

        rows = {}
        for name, strat in strategies.items():
            exact = exploitability(game, strat)
            br0 = best_response_value(game, strat, 0)
            br1 = best_response_value(game, strat, 1)
            lbr = SubgameLBR(game, strat)
            r0 = lbr.run(0, HANDS, seed=11)
            r1 = lbr.run(1, HANDS, seed=12)
            est = (r0.mean_bb_per_hand + r1.mean_bb_per_hand) / 2.0
            se = float(np.hypot(r0.stderr_bb_per_hand, r1.stderr_bb_per_hand) / 2)
            rows[name] = {"exact_exploitability": round(exact, 5),
                          "exact_br0": round(br0, 5), "exact_br1": round(br1, 5),
                          "lbr_estimate": round(est, 5), "lbr_se": round(se, 5),
                          "lbr_seat0": round(r0.mean_bb_per_hand, 5),
                          "lbr_seat1": round(r1.mean_bb_per_hand, 5),
                          "bias_exact_minus_lbr": round(exact - est, 5),
                          "upper_violation": bool(est > exact + 3 * se),
                          "capture_ratio": round(est / exact, 3) if exact > 1e-9 else None}
            pooled_exact.append(exact)
            pooled_lbr.append(est)
            print(f"{gid} {name}: exact {exact:.4f} lbr {est:.4f} ± {se:.4f}",
                  flush=True)
        # ranking accuracy inside this game
        names = list(rows)
        ex_rank = np.argsort([rows[n]["exact_exploitability"] for n in names])
        lb_rank = np.argsort([rows[n]["lbr_estimate"] for n in names])
        rows["_ranking_identical"] = bool((ex_rank == lb_rank).all())
        doc["games"][gid] = rows

    from scipy.stats import spearmanr

    doc["pooled_spearman"] = float(spearmanr(pooled_exact, pooled_lbr).statistic)
    doc["upper_violations"] = int(sum(r.get("upper_violation", False)
                                      for g in doc["games"].values()
                                      for r in g.values() if isinstance(r, dict)))
    doc["go_criterion"] = "spearman >= 0.8 and no upper violations"
    doc["go"] = bool(doc["pooled_spearman"] >= 0.8 and doc["upper_violations"] == 0)
    doc["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(doc, indent=1))
    print("pooled spearman:", round(doc["pooled_spearman"], 3),
          "violations:", doc["upper_violations"], "GO:", doc["go"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
