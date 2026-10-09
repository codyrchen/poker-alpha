"""Phases 59-61: rollout response model vs exact river equilibria.

A family of exact river subgames (52-card deck, explicit ranges, one bet
size + all-in, at most one raise) is solved with CFR+. On each game:

responder side (IP facing an OOP bet of each size)
    equilibrium fold frequency over the IP range vs the response models'
    predicted fold frequency (current BehaviorModel "regular" on ABSOLUTE
    hand strength; alternatives below).

hero side (OOP root decision for every OOP hand)
    exact equilibrium action values Q(check), Q(bet), Q(all-in) (the hand's
    value of taking that action, then both players following the
    equilibrium) vs the rollout's recommendation:
        agreement   rollout best action == argmax Q
        EV regret   max Q - Q(rollout choice)          (BB, >= 0)
        EV bias     rollout EV(choice) - Q(choice)     (BB; + = optimistic)
    plus bet-size, bluff and value-bet bias vs the equilibrium mix.

Response models compared
    behavior        current default: BehaviorModel("regular"), absolute strength
    mdf_range       defend the top 1/(1+s) of the responder's own range
                    (range-relative strength), no raises
    mdf_calibrated  defend share a + b/(1+s), (a, b) least-squares fitted to the
                    equilibrium defence of the CALIBRATION games only

Games are split deterministically into calibration and held-out halves;
adoption decisions (Phase 60) use held-out results only.

    python experiments/rollout_response.py [--sims 600] [--iters 600]
Writes results/validation/rollout_response_v1.json (diagnosis) and is reused
by Phases 60-61.
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

from phase33_reduced_games import percentile_range  # noqa: E402

from poker_alpha.games.reduced_holdem import RiverSubgame  # noqa: E402
from poker_alpha.opponent.behavior import ARCHETYPE_MODELS  # noqa: E402
from poker_alpha.opponent.ranges import strength_vector  # noqa: E402
from poker_alpha.poker.cards import card_str  # noqa: E402
from poker_alpha.poker.ranges import COMBO_INDEX  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402

BOARDS = [
    ("dry_high", ["Ks", "7d", "2c", "Qh", "4s"]),
    ("paired", ["Td", "Tc", "4s", "8h", "2d"]),
    ("four_straight", ["9s", "8d", "7c", "6h", "2s"]),
    ("four_flush", ["Qh", "9h", "5h", "2h", "Kc"]),
    ("monotone", ["As", "Js", "6s", "8d", "3c"]),
    ("low_connected", ["5d", "4c", "3h", "Ks", "9h"]),
]
SHAPES = {   # OOP bands, IP bands (river strength percentile among all combos)
    "polar_vs_catchers": ([(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    "merged_vs_merged": ([(0.35, 0.95)], [(0.30, 0.90)]),
}
SIZES = (0.5, 1.0, 2.0)
SPRS = (1.0, 2.5, 5.0)
POT = 10.0


def build_games(range_n: int):
    games = []
    for bname, board in BOARDS:
        for shape, (b0, b1) in SHAPES.items():
            for size in SIZES:
                for spr in SPRS:
                    if size > spr:          # bet would be an all-in anyway
                        continue
                    games.append({"board_name": bname, "board": board, "shape": shape,
                                  "size": size, "spr": spr,
                                  "r0": percentile_range(board, b0, range_n),
                                  "r1": percentile_range(board, b1, range_n)})
    for i, g in enumerate(games):
        g["split"] = "calibration" if i % 2 == 0 else "held_out"
        g["id"] = f"{g['board_name']}|{g['shape']}|s{g['size']}|spr{g['spr']}"
    return games


def solve(g, iters):
    game = RiverSubgame(g["board"], g["r0"], g["r1"], pot=POT, stack=POT * g["spr"],
                        bets=(g["size"],), raise_cap=2, name=g["id"])
    s = CFRPlusSolver(game)
    s.train(iters)
    from poker_alpha.solvers.evaluation import exploitability

    strat = s.average_strategy()
    return game, strat, exploitability(game, strat)


def ev_under(game, strat, state):
    """P0's expected utility from ``state`` with both players on ``strat``."""
    if game.is_terminal(state):
        return game.utility(state)
    p = strat.get(game.infoset_key(state))
    acts = game.legal_actions(state)
    if p is None:
        p = {a: 1.0 / len(acts) for a in acts}
    return sum(p.get(a, 0.0) * ev_under(game, strat, game.next_state(state, a))
               for a in acts if p.get(a, 0.0) > 0)


def root_action_values(game, strat, hand):
    """Q(a) for P0 holding ``hand`` at the root (chips, P0 utility)."""
    deals = [(w, s) for w, s in game.chance_outcomes(game.root()) if s.holes[0] == hand]
    tot = sum(w for w, _ in deals)
    acts = game.legal_actions(deals[0][1])
    return {a: sum(w * ev_under(game, strat, game.next_state(s, a)) for w, s in deals) / tot
            for a in acts}


def range_relative(strengths, weights):
    """Percentile of each combo within its own (weighted) range."""
    order = np.argsort(strengths, kind="stable")
    cw = np.cumsum(weights[order])
    tot = cw[-1]
    rel = np.empty(len(strengths))
    rel[order] = (cw - weights[order] / 2) / tot
    return rel


def responder_side(game, strat, g, calib=None):
    """Equilibrium vs model fold frequency of IP facing the OOP bet / all-in."""
    sv = strength_vector(game.board)
    hands = [h for h, _ in game.r1]
    w = np.array([wt for _, wt in game.r1])
    s_abs = np.array([sv[COMBO_INDEX[tuple(sorted(h))]] for h in hands])
    rel = range_relative(s_abs, w)
    out = {}
    probe = next(s for _, s in game.chance_outcomes(game.root()))
    for a in game.legal_actions(probe):
        if a in ("c", "f"):
            continue
        st = game.next_state(probe, a)
        c, level, _, _ = game._state(st.history)
        size = (level - c[1]) / (game.pot + c[0] + c[1] - (level - c[1]) + 1e-9) \
            if False else (level) / game.pot
        eq_fold = []
        for h in hands:
            key = f"1|{card_str(h[0])}{card_str(h[1])}|{a}"
            p = strat.get(key)
            eq_fold.append(p.get("f", 0.0) if p else np.nan)
        eq_fold = np.array(eq_fold)
        ok = ~np.isnan(eq_fold)
        beh = ARCHETYPE_MODELS["regular"].probabilities(s_abs, True, size)["fold"]
        mdf = (rel < size / (1 + size)).astype(float)
        row = {"size_pot": round(size, 3),
               "required_call_equity": round(size / (1 + 2 * size), 3),
               "eq_fold": float(np.average(eq_fold[ok], weights=w[ok])),
               "behavior_fold": float(np.average(beh[ok], weights=w[ok])),
               "mdf_range_fold": float(np.average(mdf[ok], weights=w[ok])),
               "per_hand_mae": {"behavior": float(np.average(np.abs(beh - eq_fold)[ok],
                                                             weights=w[ok])),
                                "mdf_range": float(np.average(np.abs(mdf - eq_fold)[ok],
                                                              weights=w[ok]))}}
        if calib is not None:
            share = float(np.clip(calib[0] + calib[1] / (1 + size), 0, 1))
            cal = (rel < 1 - share).astype(float)
            row["mdf_calibrated_fold"] = float(np.average(cal[ok], weights=w[ok]))
            row["per_hand_mae"]["mdf_calibrated"] = float(
                np.average(np.abs(cal - eq_fold)[ok], weights=w[ok]))
        out[a] = row
    return out


def hero_side(game, strat, g, model, sims, seed=1, calib=None):
    """Rollout recommendation vs exact action values for every OOP hand."""
    from poker_alpha.decision.rollout import RolloutCandidate, rollout_action_evs
    from poker_alpha.holdem.adapters import ManualStateAdapter
    from poker_alpha.poker.ranges import WeightedRange

    opp = WeightedRange.from_combos([(h, w) for h, w in game.r1])
    half = game.pot / 2
    rows = []
    sv = strength_vector(game.board)
    w0 = np.array([wt for _, wt in game.r0])
    s0 = np.array([sv[COMBO_INDEX[tuple(sorted(h))]] for h, _ in game.r0])
    rel0 = range_relative(s0, w0)
    # IP range median strength: hands below it are "bluffs" for the OOP side
    w1 = np.array([wt for _, wt in game.r1])
    s1 = np.array([sv[COMBO_INDEX[tuple(sorted(h))]] for h, _ in game.r1])
    ip_median = float(np.median(np.repeat(s1, np.maximum(1, (w1 * 10).astype(int)))))
    for idx, (h, _) in enumerate(game.r0):
        q = root_action_values(game, strat, h)
        key = f"0|{card_str(h[0])}{card_str(h[1])}|"
        mix = strat.get(key, {})
        cands = [RolloutCandidate("c", "check", 0.0)]
        for a in q:
            if a.startswith("b"):
                cands.append(RolloutCandidate(a, "bet", g["size"] * game.pot))
            elif a == "a":
                cands.append(RolloutCandidate("a", "all_in", game.stack))
        obs = ManualStateAdapter.from_dict({
            "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
            "hero_cards": [card_str(h[0]), card_str(h[1])],
            "board": [card_str(c) for c in game.board], "street": "river", "actor": 1,
            "seats": [{"stack": game.stack, "bet": 0, "committed": half},
                      {"stack": game.stack, "bet": 0, "committed": half}],
            "pot": game.pot, "actions": []})
        kw = {}
        if model != "behavior":
            kw = {"response": model, "response_params": calib}
        res = rollout_action_evs(obs, [card_str(c) for c in h], {0: opp},
                                 {0: ARCHETYPE_MODELS["regular"]}, cands,
                                 simulations=sims, seed=seed, bootstrap=0, **kw)
        evs = {e.label: e.ev_bb for e in res.evs}
        choice = max(evs, key=evs.get)
        q_bb = {a: (v + half) for a, v in q.items()}          # vs folding now, BB=1 chip
        best = max(q_bb, key=q_bb.get)
        sizes = {"c": 0.0, **{a: (g["size"] if a.startswith("b") else game.stack / game.pot)
                              for a in q if a != "c"}}
        eq_size = sum(mix.get(a, 0.0) * sizes[a] for a in sizes)
        rows.append({"hand": card_str(h[0]) + card_str(h[1]),
                     "range_percentile": round(float(rel0[idx]), 3),
                     "is_bluff": bool(s0[idx] < ip_median),
                     "q_bb": q_bb, "eq_mix": mix, "rollout_ev_bb": evs,
                     "rollout_choice": choice, "eq_best": best,
                     "agrees": choice == best,
                     "regret_bb": q_bb[best] - q_bb[choice],
                     "ev_bias_bb": evs[choice] - q_bb[choice],
                     "rollout_size": sizes[choice], "eq_mean_size": eq_size,
                     "rollout_bets": choice != "c", "eq_bet_freq": 1.0 - mix.get("c", 0.0)})
    return rows


def fit_calibration(resp_rows):
    """Least squares: defend share = a + b / (1 + s) on calibration games."""
    X, y = [], []
    for r in resp_rows:
        X.append([1.0, 1.0 / (1.0 + r["size_pot"])])
        y.append(1.0 - r["eq_fold"])
    sol, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return [float(sol[0]), float(sol[1])]


def summarize(rows):
    if not rows:
        return {}
    a = np.array([r["agrees"] for r in rows], float)
    reg = np.array([r["regret_bb"] for r in rows])
    bias = np.array([r["ev_bias_bb"] for r in rows])
    rs = np.array([r["rollout_size"] for r in rows])
    es = np.array([r["eq_mean_size"] for r in rows])
    bl = [r for r in rows if r["is_bluff"]]
    va = [r for r in rows if not r["is_bluff"]]
    return {"decisions": len(rows), "agreement": float(a.mean()),
            "mean_regret_bb": float(reg.mean()), "max_regret_bb": float(reg.max()),
            "mean_ev_bias_bb": float(bias.mean()),
            "bet_size_bias_pot": float((rs - es).mean()),
            "bluff_bet_freq": {"rollout": float(np.mean([r["rollout_bets"] for r in bl]))
                               if bl else None,
                               "equilibrium": float(np.mean([r["eq_bet_freq"] for r in bl]))
                               if bl else None},
            "value_bet_freq": {"rollout": float(np.mean([r["rollout_bets"] for r in va]))
                               if va else None,
                               "equilibrium": float(np.mean([r["eq_bet_freq"] for r in va]))
                               if va else None}}


def breakdown(games, key):
    out = {}
    for g in games:
        out.setdefault(str(g[key]), []).extend(g["hero"])
    return {k: summarize(v) for k, v in out.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--sims", type=int, default=600)
    p.add_argument("--range", type=int, default=10)
    p.add_argument("--models", default="behavior")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "validation" / "rollout_response_v1.json")
    a = p.parse_args(argv)
    t0 = time.time()
    games = build_games(a.range)
    if a.limit:
        games = games[:a.limit]
    solved = []
    for g in games:
        game, strat, expl = solve(g, a.iters)
        g["exploitability_chips"] = expl
        g["responder"] = responder_side(game, strat, g)
        solved.append((g, game, strat))
        print(g["id"], "solved, exploitability %.4f" % expl, flush=True)
    calib = fit_calibration([r for g, _, _ in solved if g["split"] == "calibration"
                             for r in g["responder"].values()])
    models = a.models.split(",")
    for g, game, strat in solved:
        g["responder"] = responder_side(game, strat, g, calib)
        g["by_model"] = {}
        for m in models:
            g["by_model"][m] = hero_side(game, strat, g, m, a.sims, calib=calib)
        g["hero"] = g["by_model"][models[0]]
    out = {"format": "pokeralpha.rollout_response/v1", "method": __doc__.split("\n\n")[1],
           "settings": {"iters": a.iters, "sims": a.sims, "range": a.range, "pot": POT,
                        "sizes": SIZES, "sprs": SPRS, "models": models},
           "calibration_fit": {"defend_share": "a + b/(1+s)", "a_b": calib},
           "games": [{k: v for k, v in g.items() if k not in ("r0", "r1")} for g, _, _ in solved]}
    summary = {}
    for m in models:
        for split in ("calibration", "held_out"):
            rows = [r for g, _, _ in solved if g["split"] == split for r in g["by_model"][m]]
            summary[f"{m}|{split}"] = summarize(rows)
    out["summary"] = summary
    resp = [r for g, _, _ in solved for r in g["responder"].values()]
    out["responder_fold_bias"] = {
        "by_size": {str(s): {k: float(np.mean([r[k] - r["eq_fold"] for r in resp
                                               if abs(r["size_pot"] - s) < 1e-6]))
                             for k in ("behavior_fold", "mdf_range_fold", "mdf_calibrated_fold")
                             if all(k in r for r in resp)}
                    for s in sorted({r["size_pot"] for r in resp})},
        "mean_eq_fold": float(np.mean([r["eq_fold"] for r in resp])),
        "mean_behavior_fold": float(np.mean([r["behavior_fold"] for r in resp]))}
    gs = [g for g, _, _ in solved]
    out["breakdown"] = {k: breakdown(gs, k) for k in ("size", "spr", "shape", "board_name")}
    out["seconds"] = round(time.time() - t0, 1)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=1))
    print(json.dumps(out["responder_fold_bias"], indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
