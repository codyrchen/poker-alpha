"""Rollout depth study (Phase 61).

Question: how much do action EVs and recommendations change when rollouts
play later streets (depth="showdown") instead of resolving the current
street and checking down (depth="street", the fast default), and what does
it cost in latency? Same heads-up spots as the range-sensitivity study,
default opponent model, paired seeds.

    python experiments/rollout_depth_study.py --sims 1000 --out results/validation/rollout_depth.json
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
sys.path.insert(0, str(ROOT / "experiments"))

from range_sensitivity import SPOTS  # noqa: E402

from poker_alpha.decision import DecisionConfig, recommend_action  # noqa: E402
from poker_alpha.pipeline import observe_manual  # noqa: E402


def analyse(st, depth, sims, seed, model):
    cfg = DecisionConfig(equity_simulations=2000, rollout_simulations=sims, seed=seed,
                         rollout_depth=depth, default_model=model)
    t = time.perf_counter()
    r = recommend_action(st, config=cfg)
    dt = time.perf_counter() - t
    return r, {c.label: (c.ev_bb, c.ev_se_bb) for c in r.candidates if c.ev_bb is not None}, dt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sims", type=int, default=1000)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--models", default="regular,calling_station")
    p.add_argument("--out", type=Path, default=ROOT / "results/validation/rollout_depth.json")
    a = p.parse_args(argv)
    out = {}
    for model in a.models.split(","):
        for name, d in SPOTS.items():
            st = observe_manual(d).state
            fast, ef, tf = analyse(st, "street", a.sims, a.seed, model)
            deep, ed, td = analyse(st, "showdown", a.sims, a.seed, model)
            diffs = {k: ed[k][0] - ef[k][0] for k in ef if k in ed}
            # regret of the fast recommendation if the deep model is right
            best = max(ed, key=lambda k: ed[k][0])
            regret = ed[best][0] - ed[fast.recommended][0] if fast.recommended in ed else None
            out[f"{model} | {name}"] = {
                "fast": {"recommended": fast.recommended, "evs_bb": ef, "seconds": round(tf, 3)},
                "showdown": {"recommended": deep.recommended, "evs_bb": ed, "seconds": round(td, 3)},
                "ev_shift_bb": diffs, "max_abs_ev_shift_bb": max(abs(x) for x in diffs.values()),
                "flip": fast.recommended != deep.recommended,
                "regret_of_fast_rec_under_showdown_bb": regret}
            print(f"{model:15s} {name:45s} fast {fast.recommended:10s} deep {deep.recommended:10s} "
                  f"max|dEV| {out[f'{model} | {name}']['max_abs_ev_shift_bb']:.2f} "
                  f"regret {regret if regret is None else round(regret, 2)} "
                  f"t {tf:.2f}s/{td:.2f}s", flush=True)
    rows = list(out.values())
    summary = {"spots": len(rows), "flips": sum(r["flip"] for r in rows),
               "median_max_abs_ev_shift_bb": float(np.median([r["max_abs_ev_shift_bb"] for r in rows])),
               "max_regret_of_fast_bb": max((r["regret_of_fast_rec_under_showdown_bb"] or 0) for r in rows),
               "median_seconds_fast": float(np.median([r["fast"]["seconds"] for r in rows])),
               "median_seconds_showdown": float(np.median([r["showdown"]["seconds"] for r in rows]))}
    res = {"format": "pokeralpha.rollout_depth/v1", "sims": a.sims, "seed": a.seed,
           "note": "both depths are heuristic policies (behaviour models); neither is an "
                   "equilibrium. 'regret' is measured under the showdown-depth model.",
           "summary": summary, "spots": out}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
