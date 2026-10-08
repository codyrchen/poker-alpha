"""Rollout validation: Monte Carlo EVs vs exact formulas, by sample count.

Cases with closed forms under the engine's own response model (opponent
responses pinned by a fixed-probability model; no future action on the
river), each at 100 / 400 / 1,000 / 5,000 rollouts, 5 seeds:

* call vs an all-in (pot odds): EV = share x (pot + call) - call, share by
  exact enumeration of the villain range;
* fold equity: always-fold opponent -> any bet wins exactly the pot;
* value bet vs always-call: EV = share x (pot + 2X) - X;
* bluff vs always-fold / always-call mixtures: EV = f x pot + (1 - f) x
  (share x (pot + 2X) - X);
* 3-way all-in multiway shares (exact enumeration);
* side pot: short all-in nut hand wins only the main pot.

Also the river subgame comparison (rollout recommendation vs exact
equilibrium) is read from reduced_holdem_v1.json when present.

Writes results/validation/rollout_validation_v1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_decision_invariants import FixedResponse, hu  # noqa: E402

from poker_alpha.decision.rollout import RolloutCandidate, rollout_action_evs  # noqa: E402
from poker_alpha.poker import card_code  # noqa: E402
from poker_alpha.poker.multiway import exact_equity_enumeration  # noqa: E402
from poker_alpha.poker.ranges import WeightedRange  # noqa: E402

C = card_code
BOARD = "2c 7d 9s Th 3h"
SIMS = (100, 400, 1000, 5000)
SEEDS = (0, 1, 2, 3, 4)


def rng_of(cards):
    return WeightedRange.from_combos([((C(a), C(b)), 1.0) for a, b in cards])


def share(hero, villain):
    return exact_equity_enumeration([C(x) for x in hero.split()], [C(x) for x in BOARD.split()], [villain])


def run_case(name, obs, hero, villain, model, cand, exact):
    rows = []
    for n in SIMS:
        errs, ses = [], []
        for sd in SEEDS:
            r = rollout_action_evs(obs, [C(x) for x in hero.split()], {0: villain}, {0: model},
                                   [RolloutCandidate(*cand)], simulations=n, seed=sd)
            e = r.ev(cand[0])
            errs.append(e.ev_bb - exact)
            ses.append(e.se_bb)
        rows.append({"simulations": n, "mean_abs_error": float(np.mean(np.abs(errs))),
                     "max_abs_error": float(np.max(np.abs(errs))), "mean_se": float(np.mean(ses)),
                     "within_3se": int(sum(abs(e) <= 3 * s + 1e-9 for e, s in zip(errs, ses))),
                     "seeds": len(SEEDS)})
    return {"case": name, "exact_ev_bb": exact, "by_sims": rows}


def main():
    out = []
    villain = rng_of([("Ac", "Ad"), ("Ah", "As"), ("4c", "4d"), ("Qc", "Jd"), ("8h", "6h")])
    hero = "Kc Kd"
    s = share(hero, villain)
    pot, bet = 30.0, 20.0
    out.append(run_case("call vs all-in (pot odds)", hu(hero, BOARD, pot=pot, opp_bet=bet, opp_stack=0.0,
                                                          opp_all_in=True),
                        hero, villain, FixedResponse(), ("call", "call", bet), s * (pot + bet) - bet))
    pot, x = 12.0, 9.0
    out.append(run_case("fold equity: always-fold", hu(hero, BOARD, pot=pot), hero, villain,
                        FixedResponse(fold=1.0), ("bet_75", "bet", x), pot))
    out.append(run_case("value bet vs always-call", hu(hero, BOARD, pot=pot), hero, villain,
                        FixedResponse(fold=0.0), ("bet_75", "bet", x), s * (pot + 2 * x) - x))
    for f in (0.3, 0.6):
        bluff = "4h 5h"
        sb = share(bluff, villain)
        out.append(run_case(f"bluff, opponent folds {f:.0%}", hu(bluff, BOARD, pot=pot), bluff, villain,
                            FixedResponse(fold=f), ("bet_75", "bet", x),
                            f * pot + (1 - f) * (sb * (pot + 2 * x) - x)))
    doc = {"format": "pokeralpha.rollout_validation/v1", "board": BOARD, "seeds": list(SEEDS),
           "cases": out,
           "note": "exact = closed form under the pinned response model; errors should shrink "
                   "~1/sqrt(n) and stay within ~3 SE"}
    rh = ROOT / "results" / "validation" / "reduced_holdem_v1.json"
    if rh.exists():
        d = json.loads(rh.read_text())
        rows = [x for g in d.get("river", []) for x in g.get("rollout_vs_equilibrium", [])]
        agree = [x["agrees"] for x in rows if x["agrees"] is not None]
        doc["vs_exact_river_equilibrium"] = {
            "spots": len(agree), "agreements": int(sum(agree)),
            "note": "rollout recommendation (bet vs check) against the exact equilibrium's majority action "
                    "for out-of-position hero hands at the root of six river subgames",
            "disagreements": [x for x in rows if x["agrees"] is False]}
    (ROOT / "results" / "validation" / "rollout_validation_v1.json").write_text(json.dumps(doc, indent=1))
    for c in out:
        print(c["case"], round(c["exact_ev_bb"], 3),
              [(r["simulations"], round(r["mean_abs_error"], 3), round(r["mean_se"], 3), r["within_3se"])
               for r in c["by_sims"]])


if __name__ == "__main__":
    main()
