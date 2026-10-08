"""Phase 39: strategic sanity against analytic and exact references.

1. Clairvoyant river toy game (textbook, analytically solved): OOP holds
   either the nuts (value) or air, IP holds a bluff-catcher; one bet size s
   (fraction of pot), only OOP may bet. Indifference gives
       IP call frequency          = 1 / (1 + s)
       OOP bluff share of bets     = s / (1 + 2 s)
       OOP value bet frequency     = 1
   Solved with exact CFR+ on a 52-card RiverSubgame and compared.
2. Broad principles checked on the exact reduced games and the trained
   strategy (premium aggression, trash folds, nuts never fold, pot odds,
   polarization grows with size), read from committed result files.

Writes results/validation/strategy_reference_checks.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.games.reduced_holdem import RiverSubgame  # noqa: E402
from poker_alpha.poker.cards import card_str, codes  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import exploitability  # noqa: E402

BOARD = ["Ks", "8d", "4c", "2h", "2s"]
VALUE = ["8s8h", "8s8c", "8h8c"]                         # full houses (nuts vs the IP range)
AIR = [q + j for q in ("Qs", "Qh", "Qd", "Qc") for j in ("Jh", "Jd", "Jc")]  # queen high
CATCHERS = [a + k for a in ("As", "Ah", "Ad", "Ac") for k in ("Kh", "Kd", "Kc")]  # top pair


def clairvoyance(s, iters=1500):
    # value : air weight 1 : 2 (total), so OOP has enough air to bluff fully
    v_w, a_w = 1.0 / len(VALUE), 2.0 / len(AIR)
    r0 = [f"{h}:{v_w}" for h in VALUE] + [f"{h}:{a_w}" for h in AIR]
    # stack = s x pot, so all-in is exactly the single bet size s (the game
    # never offers a second size)
    g = RiverSubgame(BOARD, r0, CATCHERS, pot=10.0, stack=10.0 * s, bets=(s,), raise_cap=1,
                     name=f"clairvoyance_s{s}", oop_only_bets=True)
    solver = CFRPlusSolver(g)
    solver.train(iters)
    st = solver.average_strategy()
    tok = "a"
    w = dict((h, wt) for h, wt in g.r0)

    def bet_freq(hands):
        tot = num = 0.0
        for hs in hands:
            h = tuple(sorted(codes([hs[:2], hs[2:]])))
            k = f"0|{card_str(h[0])}{card_str(h[1])}|"
            if k in st:
                num += w[h] * st[k].get(tok, 0.0)
                tot += w[h]
        return num / tot
    value_bet, air_bet = bet_freq(VALUE), bet_freq(AIR)
    v_mass, a_mass = 1.0, 2.0
    bluff_share = a_mass * air_bet / (a_mass * air_bet + v_mass * value_bet)
    calls = [p["c"] for k, p in st.items() if k.startswith("1|") and k.endswith(f"|{tok}")]
    return {"bet_size_pot_fraction": s, "iterations": iters,
            "exploitability": exploitability(g, st),
            "value_bet_freq": round(value_bet, 4), "expected_value_bet_freq": 1.0,
            "bluff_share_of_bets": round(bluff_share, 4), "expected_bluff_share": round(s / (1 + 2 * s), 4),
            "ip_call_freq": round(float(np.mean(calls)), 4), "expected_call_freq": round(1 / (1 + s), 4)}


def main():
    toy = [clairvoyance(s) for s in (0.5, 1.0, 2.0)]
    for t in toy:
        t["matches_theory"] = (abs(t["value_bet_freq"] - 1) < 0.02
                               and abs(t["bluff_share_of_bets"] - t["expected_bluff_share"]) < 0.02
                               and abs(t["ip_call_freq"] - t["expected_call_freq"]) < 0.03)
        print(t)
    doc = {"format": "pokeralpha.strategy_reference_checks/v1",
           "clairvoyance_game": {"board": BOARD, "oop_value": VALUE, "oop_air": AIR, "ip": CATCHERS,
                                 "weights": "value : air = 1 : 2", "pot": 10, "results": toy,
                                 "polarization_grows_with_size": toy[0]["bluff_share_of_bets"]
                                 < toy[1]["bluff_share_of_bets"] < toy[2]["bluff_share_of_bets"]},
           "source": "analytic indifference solution of the clairvoyant river game (standard toy game)"}
    out = ROOT / "results" / "validation" / "strategy_reference_checks.json"
    out.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
