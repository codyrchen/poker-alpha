"""Phase II Program 14: general-game stress test.

1. Complexity inspector over every exact game in the repository.
2. All CFR-family solvers on the new signaling game (cheap-talk and costly
   variants) with exact exploitability, and the cheap-talk theory oracle:
   the game value must equal the receiver's no-information value.
3. Records whether any solver-core modification was needed (none: the
   solvers are imported unchanged).

    python experiments/phase2_general_games.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase33_reduced_games import SUBGAMES, percentile_range  # noqa: E402

from poker_alpha.games.auction import AuctionDuel  # noqa: E402
from poker_alpha.games.inspect import inspect_game  # noqa: E402
from poker_alpha.games.kuhn import KuhnPoker  # noqa: E402
from poker_alpha.games.leduc import LeducPoker  # noqa: E402
from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame  # noqa: E402
from poker_alpha.games.signaling import SignalingGame  # noqa: E402
from poker_alpha.research.cfr_variants import DiscountedCFRSolver, OutcomeSamplingMCCFR  # noqa: E402
from poker_alpha.solvers import CFRSolver, MCCFRSolver  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import expected_value, exploitability  # noqa: E402

OUT = ROOT / "results" / "phase2" / "general_games.json"


def signaling_variants():
    # 3 types (weak / medium / strong), 3 messages, receiver guesses-ish actions
    payoff = ((2.0, -1.0, -1.0), (-1.0, 2.0, -1.0), (-1.0, -1.0, 2.0))
    prior = (0.5, 0.3, 0.2)
    return {
        "cheap_talk_3x3": SignalingGame(prior, payoff, None, 3),
        "costly_separating_3x3": SignalingGame(prior, payoff, ((0.0, 1.5, 3.0), (0.0, 0.0, 1.5), (0.0, 0.0, 0.0)), 3),
        "beer_quiche_like_2x2": SignalingGame((0.1, 0.9), ((1.0, -1.0), (-1.0, 1.0)), ((0.0, 0.6), (0.6, 0.0)), 2),
    }


def solvers_for(game):
    return {"cfr": lambda: CFRSolver(game), "cfr+": lambda: CFRPlusSolver(game),
            "dcfr": lambda: DiscountedCFRSolver.preset(game, "dcfr"), "lcfr": lambda: DiscountedCFRSolver.preset(game, "lcfr"),
            "es_mccfr": lambda: MCCFRSolver(game, seed=0), "os_mccfr": lambda: OutcomeSamplingMCCFR(game, seed=0)}


def main():
    t0 = time.time()
    n, board, b0, b1 = SUBGAMES[0]
    games = {"kuhn": KuhnPoker(), "leduc": LeducPoker(), "auction_5x5": AuctionDuel(5, 5),
             "reduced_preflop_AKQJ_sh_8bb": ReducedPreflopGame(ranks="AKQJ", suits="sh", stack=8.0, equity_samples=300),
             "river_subgame_6x6": RiverSubgame(board, percentile_range(board, b0, 6), percentile_range(board, b1, 6),
                                               pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)}
    games.update(signaling_variants())
    doc = {"format": "pokeralpha.phase2.general_games/v1", "complexity": {}, "signaling": {},
           "solver_core_modifications_required": "none: solvers imported unchanged from poker_alpha.solvers / research.cfr_variants"}
    for name, g in games.items():
        doc["complexity"][name] = inspect_game(g)
        print(name, {k: doc["complexity"][name][k] for k in ("nodes", "decision_nodes", "infosets", "max_depth", "ok")}, flush=True)
    for name, g in signaling_variants().items():
        res = {"no_information_value_receiver": g.no_information_value_receiver(), "solvers": {}}
        for sname, mk in solvers_for(g).items():
            s = mk()
            iters = 200000 if sname.endswith("mccfr") else 3000
            tt = time.perf_counter()
            s.train(iters)
            strat = s.average_strategy()
            res["solvers"][sname] = {"iterations": iters, "exploitability": exploitability(g, strat),
                                     "value_receiver": -expected_value(g, strat), "seconds": round(time.perf_counter() - tt, 2)}
        ref = CFRPlusSolver(g)
        ref.train(20000)
        strat = ref.average_strategy()
        res["reference_cfr_plus_20000"] = {"exploitability": exploitability(g, strat), "value_receiver": -expected_value(g, strat),
                                           "strategy": strat}
        if "cheap_talk" in name:
            res["cheap_talk_oracle"] = {"expected_value_receiver": g.no_information_value_receiver(),
                                        "solved_value_receiver": res["reference_cfr_plus_20000"]["value_receiver"],
                                        "abs_error": abs(g.no_information_value_receiver() - res["reference_cfr_plus_20000"]["value_receiver"])}
        else:
            # informativeness of messages: does the receiver's posterior depend on the message?
            sender = [strat[f"S{t}"] for t in range(g.n_types)]
            post = {}
            for m in range(g.n_messages):
                w = np.array([g.prior[t] * sender[t][f"m{m}"] for t in range(g.n_types)])
                post[f"m{m}"] = (w / w.sum()).round(4).tolist() if w.sum() > 1e-9 else None
            res["posterior_type_given_message"] = post
            res["value_gain_vs_no_information"] = res["reference_cfr_plus_20000"]["value_receiver"] - g.no_information_value_receiver()
        doc["signaling"][name] = res
        print(name, {k: (round(v["exploitability"], 5)) for k, v in res["solvers"].items()},
              "value", round(res["reference_cfr_plus_20000"]["value_receiver"], 4), "noinfo", round(g.no_information_value_receiver(), 4), flush=True)
    doc["seconds"] = round(time.time() - t0, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
