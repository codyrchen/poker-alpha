"""Range / opponent-model sensitivity of recommendations.

For selected decisions, vary the opponent model (which drives both the
Bayesian range updates and the rollout responses) across the archetypes
regular / nit (tight-passive) / calling_station (loose-passive) / maniac
(loose-aggressive), plus a uniform "any two cards" range, with common
random numbers. Records recommended action, its EV and the spread of the
best action's EV across models: the model uncertainty a report should not
hide.

Writes results/validation/range_sensitivity_v1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision import DecisionConfig, recommend_action  # noqa: E402
from poker_alpha.holdem import ManualStateAdapter  # noqa: E402
from poker_alpha.opponent import ARCHETYPE_MODELS  # noqa: E402
from poker_alpha.poker.ranges import WeightedRange  # noqa: E402

SPOTS = {
    "HU flop top pair facing a bet": {
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "Kh Td", "board": "Ks 7d 2c", "pot": 10.5, "actor": 1,
        "seats": [{"stack": 92.5, "bet": 5.5, "committed": 8.0}, {"stack": 97.5, "bet": 0, "committed": 2.5}],
        "actions": [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
                    {"street": 0, "seat": 1, "kind": "call", "amount": 2.5},
                    {"street": 1, "seat": 1, "kind": "check"},
                    {"street": 1, "seat": 0, "kind": "bet", "amount": 5.5}]},
    "HU river bluff-catcher facing pot bet": {
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "Qh Qd", "board": "As 9d 4c 7h 2s", "pot": 46.0, "actor": 1,
        "seats": [{"stack": 67.0, "bet": 20.0, "committed": 33.0}, {"stack": 87.0, "bet": 0, "committed": 13.0}],
        "actions": [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
                    {"street": 0, "seat": 1, "kind": "call", "amount": 2.5},
                    {"street": 1, "seat": 1, "kind": "check"}, {"street": 1, "seat": 0, "kind": "bet", "amount": 3.5},
                    {"street": 1, "seat": 1, "kind": "call", "amount": 3.5},
                    {"street": 2, "seat": 1, "kind": "check"}, {"street": 2, "seat": 0, "kind": "bet", "amount": 7.0},
                    {"street": 2, "seat": 1, "kind": "call", "amount": 7.0},
                    {"street": 3, "seat": 1, "kind": "check"}, {"street": 3, "seat": 0, "kind": "bet", "amount": 20.0}]},
    "HU flop draw first to act": {
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "Jh Th", "board": "9h 8c 2h", "pot": 5.0, "actor": 1,
        "seats": [{"stack": 97.5, "bet": 0, "committed": 2.5}, {"stack": 97.5, "bet": 0, "committed": 2.5}],
        "actions": [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
                    {"street": 0, "seat": 1, "kind": "call", "amount": 2.5}]},
}


def main():
    out = {}
    for name, spot in SPOTS.items():
        obs = ManualStateAdapter.from_dict(spot)
        rows = {}
        variants = [(m, {"default_model": m}, None) for m in ARCHETYPE_MODELS] + \
                   [("any_two_cards", {}, {0: WeightedRange.uniform()})]
        for label, kw, ranges in variants:
            rep = recommend_action(obs, opponent_ranges=ranges, config=DecisionConfig(
                equity_simulations=2000, rollout_simulations=1500, seed=3, **kw))
            if rep.recommended is None:
                raise SystemExit(f"{name}: invalid spot {rep.warnings}")
            best = max((c for c in rep.candidates if c.ev_bb is not None), key=lambda c: c.ev_bb)
            rows[label] = {"recommended": rep.recommended, "best_ev_bb": round(best.ev_bb, 3),
                           "best_ev_se": round(best.ev_se_bb or 0, 3),
                           "equity": round(rep.hero_equity or 0, 4),
                           "evs": {c.label: (None if c.ev_bb is None else round(c.ev_bb, 3)) for c in rep.candidates},
                           "range_effective_combos": [round(r.effective_combos) for r in rep.opponent_ranges]}
        recs = {r["recommended"] for r in rows.values()}
        evs = [r["best_ev_bb"] for r in rows.values()]
        out[name] = {"by_model": rows, "distinct_recommendations": sorted(recs),
                     "recommendation_stable": len(recs) == 1,
                     "best_ev_spread_bb": round(max(evs) - min(evs), 3)}
        print(name, out[name]["distinct_recommendations"], out[name]["best_ev_spread_bb"], flush=True)
    doc = {"format": "pokeralpha.range_sensitivity/v1", "spots": out,
           "note": "model uncertainty: how recommendations and EVs move with the assumed opponent model"}
    (ROOT / "results" / "validation" / "range_sensitivity_v1.json").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
