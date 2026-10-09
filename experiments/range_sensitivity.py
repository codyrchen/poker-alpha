"""Range / opponent-model sensitivity of decisions (Phase 63).

Question: how much do equity, the recommended action and action EVs depend
on the assumed opponent range and behaviour model — and what does it cost
to act on the default assumptions if a different one is true?

For each spot, the DEFAULT analysis (inferred range, "regular" model) is
compared with variants: the other behaviour archetypes (nit,
calling_station, maniac: change both the Bayesian range update and the
rollout responses) and explicit tight / wide opponent ranges. Per variant:
hero equity, recommended action, flip vs default, and the misspecification
regret = EV(best action under the variant) - EV(default's recommendation
under the variant), both from the variant's rollouts (paired seeds).

All numbers are model-relative: there is no ground-truth opponent here.

    python experiments/range_sensitivity.py --sims 1500 --out results/validation/range_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision import DecisionConfig, recommend_action  # noqa: E402
from poker_alpha.pipeline import observe_manual  # noqa: E402
from poker_alpha.poker.ranges import WeightedRange  # noqa: E402

TIGHT = "99+,AJs+,KQs,AQo+"
WIDE = "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,A2o+,K8o+,Q9o+,J9o+,T9o"


def hu(**kw):
    d = {"num_seats": 2, "hero_seat": 0, "small_blind": 0.5, "big_blind": 1.0,
         "board": "", "actions": []}
    d.update(kw)
    return d


def seats(h_stack, h_bet, h_c, o_stack, o_bet, o_c):
    return [{"stack": h_stack, "bet": h_bet, "committed": h_c},
            {"stack": o_stack, "bet": o_bet, "committed": o_c}]


PRE = [{"street": "preflop", "seat": 0, "kind": "raise", "amount": 2.5},
       {"street": "preflop", "seat": 1, "kind": "call", "amount": 2.5}]
PRE_BB = [{"street": "preflop", "seat": 1, "kind": "raise", "amount": 2.5},
          {"street": "preflop", "seat": 0, "kind": "call", "amount": 2.5}]

SPOTS = {
    "BB facing 3x open, KJo": hu(hero_cards="Kc Jd", dealer=1, actor=0,
                                 seats=seats(99.0, 1.0, 1.0, 97.0, 3.0, 3.0), pot=4.0,
                                 actions=[{"street": "preflop", "seat": 1, "kind": "raise",
                                           "amount": 3.0}]),
    "BTN open decision, A5s": hu(hero_cards="As 5s", dealer=0, actor=0,
                                 seats=seats(99.5, 0.5, 0.5, 99.0, 1.0, 1.0), pot=1.5),
    "flop c-bet decision, top pair weak kicker": hu(
        hero_cards="Qs 6s", board="Qh 8c 3d", street="flop", dealer=0, actor=0,
        seats=seats(97.5, 0.0, 2.5, 97.5, 0.0, 2.5), pot=5.0, actions=PRE),
    "flop facing bet, flush draw": hu(
        hero_cards="Ah 4h", board="Kh 9h 2c", street="flop", dealer=1, actor=0,
        seats=seats(97.5, 0.0, 2.5, 94.0, 3.5, 6.0), pot=8.5,
        actions=PRE_BB + [{"street": "flop", "seat": 1, "kind": "bet", "amount": 3.5}]),
    "turn facing bet, middle pair": hu(
        hero_cards="9c 8c", board="Kd 9s 4h 2c", street="turn", dealer=1, actor=0,
        seats=seats(94.5, 0.0, 5.5, 88.0, 6.5, 12.0), pot=17.5,
        actions=PRE_BB + [{"street": "flop", "seat": 1, "kind": "bet", "amount": 3.0},
                          {"street": "flop", "seat": 0, "kind": "call", "amount": 3.0},
                          {"street": "turn", "seat": 1, "kind": "bet", "amount": 6.5}]),
    "river facing pot bet, bluff-catcher": hu(
        hero_cards="Jc Jd", board="Ks 8s 4d 3c 2h", street="river", dealer=1, actor=0,
        seats=seats(88.0, 0.0, 12.0, 64.0, 24.0, 36.0), pot=48.0,
        actions=PRE_BB + [{"street": "flop", "seat": 1, "kind": "bet", "amount": 3.5},
                          {"street": "flop", "seat": 0, "kind": "call", "amount": 3.5},
                          {"street": "turn", "seat": 1, "kind": "bet", "amount": 6.0},
                          {"street": "turn", "seat": 0, "kind": "call", "amount": 6.0},
                          {"street": "river", "seat": 1, "kind": "bet", "amount": 24.0}]),
    "river first to act, two pair": hu(
        hero_cards="Ad 8d", board="Ac 8h 5s Tc 2d", street="river", dealer=1, actor=0,
        seats=seats(90.0, 0.0, 10.0, 90.0, 0.0, 10.0), pot=20.0,
        actions=PRE_BB + [{"street": "flop", "seat": 1, "kind": "check"},
                          {"street": "flop", "seat": 0, "kind": "bet", "amount": 3.5},
                          {"street": "flop", "seat": 1, "kind": "call", "amount": 3.5},
                          {"street": "turn", "seat": 1, "kind": "check"},
                          {"street": "turn", "seat": 0, "kind": "check"}]),
}

VARIANTS = {
    "default (inferred, regular)": dict(),
    "model nit": dict(model="nit"),
    "model calling_station": dict(model="calling_station"),
    "model maniac": dict(model="maniac"),
    "range tight": dict(range=TIGHT),
    "range wide": dict(range=WIDE),
}


def analyse(state, variant, sims, seed):
    cfg = DecisionConfig(equity_simulations=4000, rollout_simulations=sims, seed=seed,
                         default_model=variant.get("model", "regular"))
    ranges = None
    if "range" in variant:
        ranges = {s: WeightedRange.from_string(variant["range"]) for s in state.opponents_in_hand}
    r = recommend_action(state, opponent_ranges=ranges, config=cfg)
    evs = {c.label: c.ev_bb for c in r.candidates if c.ev_bb is not None}
    return r, evs


def run(sims, seed, log):
    out = {}
    for name, d in SPOTS.items():
        st = observe_manual(d).state
        base, base_evs = analyse(st, VARIANTS["default (inferred, regular)"], sims, seed)
        rows = {}
        for vname, v in VARIANTS.items():
            r, evs = analyse(st, v, sims, seed)
            best = max(evs, key=evs.get) if evs else None
            regret = (evs[best] - evs.get(base.recommended, np.nan)) if best else None
            rows[vname] = {"equity": r.hero_equity, "recommended": r.recommended,
                           "method": r.method, "flip_vs_default": r.recommended != base.recommended,
                           "evs_bb": evs, "regret_of_default_rec_bb": regret,
                           "opponent_range_combos": [s.live_combos for s in r.opponent_ranges]}
        eq = [x["equity"] for x in rows.values() if x["equity"] is not None]
        out[name] = {"state": st.describe(), "variants": rows,
                     "equity_spread": float(max(eq) - min(eq)) if eq else None,
                     "flip_rate": float(np.mean([x["flip_vs_default"] for x in rows.values()])),
                     "max_regret_bb": float(np.nanmax([x["regret_of_default_rec_bb"] or 0.0
                                                      for x in rows.values()]))}
        log(f"{name}: default {base.recommended}; " + ", ".join(
            f"{k.split(' ', 1)[1] if ' ' in k else k}={x['recommended']}"
            f"({x['regret_of_default_rec_bb'] or 0:.2f})" for k, x in rows.items()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sims", type=int, default=1500)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", type=Path, default=ROOT / "results/validation/range_sensitivity.json")
    a = p.parse_args(argv)
    t0 = time.time()
    spots = run(a.sims, a.seed, print)
    flips = [s["flip_rate"] for s in spots.values()]
    res = {"format": "pokeralpha.range_sensitivity/v1", "sims": a.sims, "seed": a.seed,
           "note": "model-relative: no ground-truth opponent; regret = EV(best under variant) - "
                   "EV(default recommendation under variant), rollout EVs with paired seeds",
           "variants": {k: v for k, v in VARIANTS.items()},
           "summary": {"mean_flip_rate_over_variants": float(np.mean(flips)),
                       "spots_with_any_flip": int(sum(f > 0 for f in flips)),
                       "max_regret_bb": max(s["max_regret_bb"] for s in spots.values()),
                       "max_equity_spread": max(s["equity_spread"] or 0 for s in spots.values())},
           "spots": spots, "seconds": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res["summary"]))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
