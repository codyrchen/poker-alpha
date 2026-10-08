"""Phase 33: exact Hold'em-shaped validation games.

33A reduced preflop game: exact CFR and CFR+ (reference), external-sampling
    MCCFR (3 seeds) at log-spaced iterations; exact game value, best
    responses and exploitability.
33B six fixed-board 52-card river subgames with explicit range structures:
    exact CFR+, MCCFR, exact exploitability; rollout recommendation of the
    decision engine for selected hero hands; compact-strategy compatibility.
33C strategy distance of MCCFR to the CFR+ reference: reach-weighted L1,
    EV error and exploitability by iteration.

Writes results/validation/reduced_holdem_v1.json.

Usage: python experiments/phase33_reduced_games.py [--quick]
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

from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame  # noqa: E402
from poker_alpha.poker.cards import card_str, codes  # noqa: E402
from poker_alpha.poker.evaluator import evaluate_best_codes  # noqa: E402
from poker_alpha.solvers import CFRSolver, MCCFRSolver  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import (best_response_value,  # noqa: E402
                                            expected_value, exploitability)

OUT = ROOT / "results" / "validation" / "reduced_holdem_v1.json"


def infoset_reach(game, strategy):
    """Total reach probability (chance x both players) of every infoset under
    ``strategy`` — the weights for strategy distance."""
    w = {}

    def walk(s, p):
        if p <= 0 or game.is_terminal(s):
            return
        if game.is_chance(s):
            for q, c in game.chance_outcomes(s):
                walk(c, p * q)
            return
        k = game.infoset_key(s)
        w[k] = w.get(k, 0.0) + p
        probs = strategy.get(k)
        acts = game.legal_actions(s)
        for a in acts:
            walk(game.next_state(s, a), p * (probs[a] if probs else 1.0 / len(acts)))
    walk(game.root(), 1.0)
    return w


def distance(game, ref, strat, weights):
    tot = sum(weights.values())
    d = 0.0
    for k, wk in weights.items():
        r = ref.get(k)
        s = strat.get(k)
        if r is None:
            continue
        if s is None:
            s = {a: 1.0 / len(r) for a in r}
        d += wk * sum(abs(r[a] - s.get(a, 0.0)) for a in r)
    return d / tot if tot else 0.0


def evaluate(game, ref, ref_value, weights, strat):
    return {"exploitability": exploitability(game, strat),
            "ev_error": abs(expected_value(game, strat) - ref_value),
            "weighted_l1": distance(game, ref, strat, weights)}


def run_game(game, ref_iters, mccfr_iters, cfr_iters, seeds):
    t = time.time()
    ref_solver = CFRPlusSolver(game)
    ref_solver.train(ref_iters)
    ref = ref_solver.average_strategy()
    value = expected_value(game, ref)
    out = {"signature": game.signature(), "infosets": len(ref_solver.infosets),
           "reference": {"algorithm": "CFR+", "iterations": ref_iters,
                         "game_value_p0": value,
                         "exploitability": exploitability(game, ref),
                         "br_value_vs_p1_strategy": best_response_value(game, ref, 0),
                         "br_value_vs_p0_strategy": best_response_value(game, ref, 1),
                         "seconds": round(time.time() - t, 1)}}
    weights = infoset_reach(game, ref)
    # vanilla CFR curve
    cfr = CFRSolver(game)
    curve, done = [], 0
    for it in cfr_iters:
        cfr.train(it - done)
        done = it
        curve.append({"iterations": it, **evaluate(game, ref, value, weights, cfr.average_strategy())})
    out["cfr"] = curve
    cfrp = CFRPlusSolver(game)
    curve, done = [], 0
    for it in cfr_iters:
        cfrp.train(it - done)
        done = it
        curve.append({"iterations": it, **evaluate(game, ref, value, weights, cfrp.average_strategy())})
    out["cfr_plus"] = curve
    out["mccfr"] = {}
    for seed in seeds:
        s = MCCFRSolver(game, seed=seed)
        curve, done, t = [], 0, time.time()
        for it in mccfr_iters:
            s.train(it - done)
            done = it
            curve.append({"iterations": it, "seconds": round(time.time() - t, 1),
                          **evaluate(game, ref, value, weights, s.average_strategy())})
        out["mccfr"][str(seed)] = curve
        print(game.signature()[:40], "seed", seed, [(c["iterations"], round(c["exploitability"], 4)) for c in curve],
              flush=True)
    return out, ref


# -- river subgames ------------------------------------------------------------

def percentile_range(board, lo_hi_pairs, max_combos, step_seed=0):
    """Combos whose river strength percentile (among all combos) falls in any
    [lo, hi) band; deterministically thinned to at most ``max_combos``."""
    b = codes(board)
    combos = [(a, c) for a in range(52) for c in range(a + 1, 52) if a not in b and c not in b]
    vals = sorted(((evaluate_best_codes([a, c] + b), (a, c)) for a, c in combos))
    n = len(vals)
    chosen = []
    for lo, hi in lo_hi_pairs:
        band = [h for i, (_, h) in enumerate(vals) if lo <= i / n < hi]
        chosen.extend(band)
    if len(chosen) > max_combos:
        idx = np.linspace(0, len(chosen) - 1, max_combos).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx))]
    return [card_str(a) + card_str(c) for a, c in chosen]


SUBGAMES = [
    ("dry_high polarized vs bluff-catcher", ["Ks", "7d", "2c", "Qh", "4s"],
     [(0.0, 0.10), (0.90, 1.0)], [(0.45, 0.80)]),
    ("paired uncapped vs capped", ["Td", "Tc", "4s", "8h", "2d"],
     [(0.30, 0.60), (0.94, 1.0)], [(0.30, 0.80)]),
    ("four-straight strong vs strong", ["9s", "8d", "7c", "6h", "2s"],
     [(0.75, 1.0)], [(0.70, 1.0)]),
    ("four-flush thin value", ["Qh", "9h", "5h", "2h", "Kc"],
     [(0.60, 0.85)], [(0.40, 0.75)]),
    ("monotone bluff-heavy", ["As", "Js", "6s", "8d", "3c"],
     [(0.0, 0.30), (0.93, 1.0)], [(0.40, 0.85)]),
    ("low connected mixed", ["5d", "4c", "3h", "Ks", "9h"],
     [(0.10, 0.40), (0.80, 1.0)], [(0.25, 0.90)]),
]


def rollout_comparison(game, ref, hero_hands, rollouts):
    """Decision-engine rollout recommendation for OOP hero hands at the root
    vs the exact equilibrium mix (opponent range = the subgame's range1)."""
    from poker_alpha.decision import DecisionConfig, recommend_action
    from poker_alpha.holdem.adapters import ManualStateAdapter
    from poker_alpha.poker.ranges import WeightedRange

    opp = WeightedRange.from_combos([(h, w) for h, w in game.r1])
    out = []
    half = game.pot / 2
    for h in hero_hands:
        key = f"0|{card_str(h[0])}{card_str(h[1])}|"
        mix = ref.get(key)
        obs = ManualStateAdapter.from_dict({
            "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
            "hero_cards": [card_str(h[0]), card_str(h[1])],
            "board": [card_str(c) for c in game.board], "street": "river", "actor": 1,
            "seats": [{"stack": game.stack, "bet": 0, "committed": half},
                      {"stack": game.stack, "bet": 0, "committed": half}],
            "pot": game.pot, "actions": []})
        rep = recommend_action(obs, opponent_ranges={0: opp}, config=DecisionConfig(
            equity_simulations=1000, rollout_simulations=rollouts, seed=1))
        rec_kind = "check" if rep.recommended == "check" else ("bet" if rep.recommended else None)
        eq_bet = 1.0 - mix["c"] if mix else None
        out.append({"hero": card_str(h[0]) + card_str(h[1]), "equilibrium_mix": mix,
                    "equilibrium_bet_freq": eq_bet, "rollout_recommendation": rep.recommended,
                    "rollout_method": rep.method,
                    "agrees": None if eq_bet is None else (
                        (rec_kind == "bet" and eq_bet >= 0.5) or (rec_kind == "check" and eq_bet <= 0.5))})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    doc = {"format": "pokeralpha.reduced_holdem/v1", "machine": "4-vCPU container, CPython",
           "note": "exact = exact for the defined reduced game; reference = CFR+ average strategy"}
    q = a.quick
    pre = ReducedPreflopGame(equity_samples=300 if q else 4000)
    res, _ = run_game(pre, 200 if q else 2000,
                      [100, 1000, 3000] if q else [100, 300, 1000, 3000, 10000, 30000, 100000, 300000],
                      [10, 50] if q else [10, 30, 100, 300, 1000],
                      [0] if q else [0, 1, 2])
    doc["preflop"] = res
    doc["preflop"]["game"] = {"deck": "AKQJT x {s,h} (10 cards, 45 hands)", "stack_bb": 10,
                              "blinds": "0.5/1", "actions": "fold / call-or-limp / raise to 2.5 BB / all-in",
                              "showdown": "expected pot share from a seeded equity table (4,000 boards per pair)"}
    print("preflop done", round(time.time() - t0), flush=True)
    doc["river"] = []
    for name, board, b0, b1 in SUBGAMES[: 2 if q else None]:
        r0 = percentile_range(board, b0, 24)
        r1 = percentile_range(board, b1, 24)
        g = RiverSubgame(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2, name=name)
        res, ref = run_game(g, 150 if q else 1500,
                            [1000, 5000] if q else [300, 1000, 3000, 10000, 30000, 100000],
                            [10] if q else [10, 30, 100, 300], [0] if q else [0, 1, 2])
        heroes = [h for h, _ in g.r0][:: max(1, len(g.r0) // 6)][:6]
        res["rollout_vs_equilibrium"] = rollout_comparison(g, ref, heroes, 200 if q else 1000)
        res["compact_strategy_lookup"] = ("not compatible: subgame pot 10 / stack 20 / bets 50%, 100%, "
                                          "all-in, cap 2 differ from the locked 100 BB tree; no lookup made")
        res.update(name=name, board=board, range0=r0, range1=r1, pot=10, stack=20)
        doc["river"].append(res)
        print("river", name, "done", round(time.time() - t0), flush=True)
    doc["seconds"] = round(time.time() - t0)
    out = OUT if not q else OUT.with_name("reduced_holdem_quick.json")
    out.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
