"""Monte Carlo error calibration (Phase 62).

Question: are the equity and rollout-EV standard errors PokerAlpha reports
honest? For each spot and sample size n (100..5000) we repeat the estimate
with R independent seeds and measure

* bias and RMSE against a reference value,
* SE calibration: mean reported SE / empirical SD of the estimates (1 = honest,
  < 1 = over-confident),
* coverage of the nominal 95% interval  estimate +- 1.96 SE.

Equity references are EXACT (river/turn: full enumeration of opponent combos
and remaining cards) except preflop, whose reference is a 2,000,000-sample
estimate (its own SE is reported and is << the SEs being tested). Rollout-EV
references are 40,000-simulation estimates (no exact value exists for the
rollout policy); their SE is reported too.

    python experiments/mc_error_calibration.py --reps 200 --out results/validation/mc_error_calibration.json
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
from poker_alpha.poker.cards import codes  # noqa: E402
from poker_alpha.poker.multiway import exact_equity_enumeration, multiway_equity  # noqa: E402
from poker_alpha.poker.ranges import WeightedRange  # noqa: E402

SIZES = (100, 300, 1000, 2000, 5000)

EQUITY_SPOTS = {
    # name: (hero, board, [opponent range strings])
    "preflop AKs vs 15% range": ("As Ks", "", ["22+,A2s+,K9s+,QTs+,JTs,A9o+,KTo+,QJo"]),
    "flop TT vs overpair/draw range": ("Td Tc", "9s 8h 2d", ["JJ+,AKs,QJs,JTs,98s,98o"]),
    "turn draw vs top pair": ("Ah 5h", "Kh 9h 2c 7d", ["KQ,KJ,KT,K9s,99,22"]),
    "river bluff-catcher": ("Qc Qd", "Ks 8s 4d 3c 2h", ["AK,KQ,K8s,88,44,QJs,JTs,A5s"]),
    "turn 3-way": ("Jh Jd", "Ts 7c 3h 2s", ["TT+,AT,KT", "77-99,A7s,87s"]),
}

ROLLOUT_SPOTS = {
    "preflop SB open (100bb)": dict(hero_cards="As Ks", dealer=0, actor=0,
                                    seats=[{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                                           {"stack": 99.0, "bet": 1.0, "committed": 1.0}],
                                    pot=1.5),
    "flop facing a bet": dict(hero_cards="Td Tc", board="9s 8h 2d", dealer=1, actor=0,
                              street="flop",
                              seats=[{"stack": 94.0, "bet": 0.0, "committed": 3.0},
                                     {"stack": 91.0, "bet": 3.0, "committed": 6.0}], pot=9.0),
    "river first to act": dict(hero_cards="Qc Qd", board="Ks 8s 4d 3c 2h", dealer=1, actor=0,
                               street="river",
                               seats=[{"stack": 80.0, "bet": 0.0, "committed": 20.0},
                                      {"stack": 80.0, "bet": 0.0, "committed": 20.0}], pot=40.0),
}


def parse_cards(text):
    return codes(text.split() if isinstance(text, str) else text)


def _ranges(specs):
    return [WeightedRange.from_string(s) for s in specs]


def exact_turn_equity(hero, board, ranges):
    """Joint enumeration over opponent combos and the river card."""
    from itertools import product

    from poker_alpha.poker.ranges import COMBOS
    from poker_alpha.poker.evaluator import evaluate_best_codes

    h = parse_cards(hero)
    b = parse_cards(board)
    dead = set(h) | set(b)
    lists = []
    for r in ranges:
        w = r.weights
        lists.append([(int(COMBOS[i, 0]), int(COMBOS[i, 1]), float(w[i]))
                      for i in np.flatnonzero(w)
                      if int(COMBOS[i, 0]) not in dead and int(COMBOS[i, 1]) not in dead])
    num = den = 0.0
    for combo in product(*lists):
        cards = [c for a, b_, _ in combo for c in (a, b_)]
        if len(set(cards)) != len(cards):
            continue
        weight = float(np.prod([w for _, _, w in combo]))
        used = dead | set(cards)
        for river in range(52):
            if river in used:
                continue
            board5 = list(b) + [river]
            hv = evaluate_best_codes(list(h) + board5)
            ov = [evaluate_best_codes([a, b_] + board5) for a, b_, _ in combo]
            best = max([hv] + ov)
            share = (1.0 / (1 + sum(v == best for v in ov))) if hv == best else 0.0
            num += weight * share
            den += weight
    return num / den


def calibrate(estimates, ses, ref):
    est, se = np.asarray(estimates), np.asarray(ses)
    sd = float(est.std(ddof=1))
    return {"bias": float(est.mean() - ref), "rmse": float(np.sqrt(np.mean((est - ref) ** 2))),
            "empirical_sd": sd, "mean_reported_se": float(se.mean()),
            "se_ratio": float(se.mean() / sd) if sd > 0 else None,
            "coverage_95": float(np.mean(np.abs(est - ref) <= 1.96 * se))}


def run_equity(reps, sizes, log):
    out = {}
    for name, (hero, board, specs) in EQUITY_SPOTS.items():
        ranges = _ranges(specs)
        hero, board = hero.split(), board.split()
        t0 = time.time()
        nb = len(parse_cards(board))
        if nb == 5:
            ref, ref_se, kind = exact_equity_enumeration(hero, board, ranges), 0.0, "exact"
        elif nb == 4:
            ref, ref_se, kind = exact_turn_equity(hero, board, ranges), 0.0, "exact"
        else:
            big = multiway_equity(hero, board, ranges, simulations=2_000_000, seed=987654)
            ref, ref_se, kind = big.expected_share, big.std_error, "2M-sample estimate"
        rows = {}
        for n in sizes:
            e, s = [], []
            for r in range(reps):
                res = multiway_equity(hero, board, ranges, simulations=n, seed=10_000 + r)
                e.append(res.expected_share)
                s.append(res.std_error)
            rows[str(n)] = calibrate(e, s, ref)
        out[name] = {"reference": ref, "reference_se": ref_se, "reference_kind": kind,
                     "seconds": round(time.time() - t0, 1), "by_samples": rows}
        log(f"equity {name}: ref {ref:.4f} ({kind}) " + " ".join(
            f"n={n}:cov={rows[str(n)]['coverage_95']:.2f},ratio={rows[str(n)]['se_ratio']:.2f}"
            for n in sizes))
    return out


def run_rollout(reps, sizes, ref_sims, log):
    out = {}
    for name, spot in ROLLOUT_SPOTS.items():
        d = {"num_seats": 2, "hero_seat": 0, "small_blind": 0.5, "big_blind": 1.0,
             "board": "", "actions": []}
        d.update(spot)
        st = observe_manual(d).state
        t0 = time.time()

        def evs(n, seed):
            r = recommend_action(st, config=DecisionConfig(
                equity_simulations=100, rollout_simulations=n, seed=seed))
            return {c.label: (c.ev_bb, c.ev_se_bb) for c in r.candidates if c.ev_bb is not None}
        ref = evs(ref_sims, 424242)
        rows = {}
        for n in sizes:
            per = {lab: ([], []) for lab in ref}
            for r in range(reps):
                for lab, (e, s) in evs(n, 50_000 + r).items():
                    per[lab][0].append(e)
                    per[lab][1].append(s)
            rows[str(n)] = {lab: calibrate(e, s, ref[lab][0]) for lab, (e, s) in per.items()}
        out[name] = {"reference_sims": ref_sims,
                     "reference": {k: {"ev_bb": v[0], "se_bb": v[1]} for k, v in ref.items()},
                     "seconds": round(time.time() - t0, 1), "by_samples": rows}
        log(f"rollout {name}: " + " ".join(
            f"n={n}:cov={np.mean([v['coverage_95'] for v in rows[str(n)].values()]):.2f},"
            f"ratio={np.mean([v['se_ratio'] or 0 for v in rows[str(n)].values()]):.2f}"
            for n in sizes))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--rollout-reps", type=int, default=60)
    p.add_argument("--rollout-ref", type=int, default=40_000)
    p.add_argument("--sizes", default=",".join(map(str, SIZES)))
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "results/validation/mc_error_calibration.json")
    a = p.parse_args(argv)
    sizes = tuple(int(x) for x in a.sizes.split(","))
    t0 = time.time()
    res = {"format": "pokeralpha.mc_error_calibration/v1",
           "what": "bias, RMSE, SE calibration (mean reported SE / empirical SD) and "
                   "95% coverage of PokerAlpha's Monte Carlo estimates",
           "reps": a.reps, "rollout_reps": a.rollout_reps, "sizes": list(sizes),
           "equity": run_equity(a.reps, sizes, print)}
    if not a.skip_rollout:
        res["rollout_ev"] = run_rollout(a.rollout_reps, sizes, a.rollout_ref, print)
    res["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
