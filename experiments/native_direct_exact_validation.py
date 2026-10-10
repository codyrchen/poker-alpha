"""Phase 3 (final-trust): DIRECT native exact-game validation.

Runs native MCCFR (through the tabular adapter — zero rule duplication, the
Python game defines every payoff and key) against exact CFR+ references and
Python MCCFR on:

* the reduced preflop game (AKQJT x 2 suits, Phase 33 configuration);
* the six fixed-board 52-card river subgames (Phase 33 configuration).

Metrics per milestone: exact exploitability, game-value error vs the CFR+
reference, reach-weighted strategy L1 to the reference. 3 seeds per sampler.

Writes results/validation/native_direct_exact_validation.json.

Compute record (>30 min policy): question = "does native MCCFR converge on
exact games at the same rate as the validated Python MCCFR?"; expected
duration ~45 min single process; output above; stop = all games evaluated.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame  # noqa: E402
from poker_alpha.native.tabular import NativeTabularMCCFR, build_tabular_tree  # noqa: E402
from poker_alpha.solvers import MCCFRSolver  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import expected_value, exploitability  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from phase33_reduced_games import SUBGAMES, distance, infoset_reach, percentile_range  # noqa: E402

OUT = ROOT / "results/validation/native_direct_exact_validation.json"

PY_MILESTONES = [1_000, 10_000, 30_000]
NATIVE_MILESTONES = [100, 1_000, 10_000, 100_000, 300_000]
SEEDS = [0, 1, 2]


def evaluate(game, ref, ref_value, weights, strat):
    return {"exploitability": exploitability(game, strat),
            "ev_error": abs(expected_value(game, strat) - ref_value),
            "weighted_l1": distance(game, ref, strat, weights)}


def run_game(game, ref_iters: int) -> dict:
    t0 = time.time()
    ref_solver = CFRPlusSolver(game)
    ref_solver.train(ref_iters)
    ref = ref_solver.average_strategy()
    value = expected_value(game, ref)
    weights = infoset_reach(game, ref)
    out = {"signature": game.signature(),
           "reference": {"algorithm": "CFR+", "iterations": ref_iters,
                         "game_value_p0": value,
                         "exploitability": exploitability(game, ref),
                         "seconds": round(time.time() - t0, 1)}}

    out["python_mccfr"] = {}
    for seed in SEEDS:
        solver = MCCFRSolver(game, seed=seed)
        curve, done = [], 0
        for it in PY_MILESTONES:
            solver.train(it - done)
            done = it
            curve.append({"iterations": it,
                          **evaluate(game, ref, value, weights,
                                     solver.average_strategy())})
        out["python_mccfr"][str(seed)] = curve

    tree = build_tabular_tree(game)
    out["native_mccfr"] = {}
    for seed in SEEDS:
        solver = NativeTabularMCCFR(game, seed=seed, tree=tree)
        curve, done = [], 0
        for it in NATIVE_MILESTONES:
            solver.train(it - done)
            done = it
            curve.append({"iterations": it,
                          **evaluate(game, ref, value, weights,
                                     solver.average_strategy())})
        out["native_mccfr"][str(seed)] = curve
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    t_start = time.time()
    doc = {"format": "pokeralpha.native_direct_exact_validation/v1",
           "adapter": ("tabular tree built from the Python game "
                       "(poker_alpha/native/tabular.py); native solver = "
                       "cpp/src/tabular.cpp sharing the Hold'em backend's "
                       "regret matching, sampling and fsum dot"),
           "py_milestones": PY_MILESTONES,
           "native_milestones": NATIVE_MILESTONES,
           "seeds": SEEDS,
           "games": {}}

    pf = ReducedPreflopGame()   # Phase 33 configuration (AKQJT, sh, 10bb)
    print("reduced preflop...", flush=True)
    doc["games"]["reduced_preflop"] = run_game(pf, ref_iters=1_000)
    print("reduced preflop done", round(time.time() - t_start), "s", flush=True)

    for name, board, b0, b1 in SUBGAMES:
        r0 = percentile_range(board, b0, 24)
        r1 = percentile_range(board, b1, 24)
        g = RiverSubgame(board, r0, r1, pot=10.0, stack=20.0,
                         bets=(0.5, 1.0), raise_cap=2, name=name)
        print("river", name, "...", flush=True)
        doc["games"][f"river::{name}"] = run_game(g, ref_iters=1_500)
        print("river", name, "done", round(time.time() - t_start), "s", flush=True)

    # Verdict: native final-milestone exploitability must be in family with
    # Python's at the same iteration count (ratio bounds, not bit equality —
    # different RNG streams).
    verdict = {}
    for gname, res in doc["games"].items():
        py = [c for s in res["python_mccfr"].values() for c in s
              if c["iterations"] == 30_000]
        nat = [c for s in res["native_mccfr"].values() for c in s
               if c["iterations"] == 10_000]
        nat_final = [s[-1] for s in res["native_mccfr"].values()]
        verdict[gname] = {
            "python_expl_at_30k": sorted(round(c["exploitability"], 5) for c in py),
            "native_expl_at_10k": sorted(round(c["exploitability"], 5) for c in nat),
            "native_expl_final": sorted(round(c["exploitability"], 5) for c in nat_final),
            "reference_expl": round(res["reference"]["exploitability"], 6),
        }
    doc["verdict"] = verdict
    doc["seconds_total"] = round(time.time() - t_start, 1)
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps(verdict, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
