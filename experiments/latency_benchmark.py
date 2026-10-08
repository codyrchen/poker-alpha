"""Phase 40D: online latency of every pipeline component (median / p95).

Online inference only — training cost is reported separately (offline
section, read from result files). Run on an otherwise idle machine.

Writes results/validation/latency_benchmark.json.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision import DecisionConfig  # noqa: E402
from poker_alpha.decision.rollout import RolloutCandidate, rollout_action_evs  # noqa: E402
from poker_alpha.history import load_hands  # noqa: E402
from poker_alpha.opponent import ARCHETYPE_MODELS  # noqa: E402
from poker_alpha.opponent.ranges import update_range_for_action  # noqa: E402
from poker_alpha.pipeline import (analyze, load_solver, observe_hand_history,  # noqa: E402
                                  observe_manual)
from poker_alpha.platform_demo import manual_hu_spot  # noqa: E402
from poker_alpha.poker import card_code  # noqa: E402
from poker_alpha.poker.equity import estimate_equity  # noqa: E402
from poker_alpha.poker.evaluator import evaluate_best_codes  # noqa: E402
from poker_alpha.poker.multiway import multiway_equity  # noqa: E402
from poker_alpha.poker.ranges import WeightedRange  # noqa: E402

C = card_code


def bench(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return {"median_ms": round(1e3 * statistics.median(ts), 4),
            "p95_ms": round(1e3 * ts[min(len(ts) - 1, int(0.95 * (len(ts) - 1)))], 4), "reps": reps}


def main():
    rng = np.random.default_rng(0)
    hands = [list(rng.choice(52, 7, replace=False)) for _ in range(2000)]
    hero = [C("Ah"), C("Kd")]
    board = [C("Qs"), C("Jh"), C("4c")]
    unif = WeightedRange.uniform()
    out = {}
    out["hand_evaluation (7 cards, x2000)"] = bench(lambda: [evaluate_best_codes(h) for h in hands], 20)
    out["HU equity vs random (2,000 sims)"] = bench(lambda: estimate_equity(hero, board, simulations=2000, seed=1), 20)
    out["multiway equity, 2 opponents (2,000 sims)"] = bench(
        lambda: multiway_equity(hero, board, [unif, unif], simulations=2000, seed=1), 20)
    out["range update (Bayes, 1,326 combos)"] = bench(
        lambda: update_range_for_action(unif, "raise", facing_bet=True, board=board, dead=hero,
                                        model=ARCHETYPE_MODELS["regular"]), 30)
    solver = load_solver(ROOT / "results" / "strategy" / "holdem_v1_seed0.npz")
    obs = observe_manual(manual_hu_spot())
    out["solver lookup incl. gate"] = bench(lambda: solver.lookup(obs.state), 200)
    from poker_alpha.holdem.adapters import ManualStateAdapter
    st = ManualStateAdapter.from_dict({
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "Ah Kd", "board": "Qs Jh 4c", "pot": 5.0, "actor": 1,
        "seats": [{"stack": 97.5, "bet": 0, "committed": 2.5}, {"stack": 97.5, "bet": 0, "committed": 2.5}]})
    cands = [RolloutCandidate("check", "check", 0.0), RolloutCandidate("bet_75", "bet", 3.75),
             RolloutCandidate("all_in", "all_in", 97.5)]
    for n in (100, 400, 1000):
        out[f"rollout {n} (3 candidates)"] = bench(
            lambda n=n: rollout_action_evs(st, hero, {0: unif}, {0: ARCHETYPE_MODELS["regular"]}, cands,
                                           simulations=n, seed=1), 5 if n == 1000 else 10)
    try:
        from PIL import Image

        from poker_alpha.observer.fusion import StateTracker
        from poker_alpha.observer.pokernow import PokerNowStyleAdapter, default_layout
        from poker_alpha.observer.calibration import TableCalibration
        img = Image.open(ROOT / "tests" / "fixtures" / "table.png").convert("RGB")
        cal = TableCalibration.load(ROOT / "tests" / "fixtures" / "table_calibration.json")
        ad = PokerNowStyleAdapter(cal)
        frame = ad.read_frame(img)
        out["observer frame (synthetic fixture)"] = bench(lambda: ad.read_frame(img), 10)
        tr = StateTracker(cal, 0.5, 1.0)
        out["state fusion (one update + observed state)"] = bench(
            lambda: (tr.update(frame), tr.to_observed_state()), 30)
    except ImportError:
        out["observer"] = "Pillow not installed"
    for name, cfg in (("full DecisionReport, solver path (1,500 equity sims)", DecisionConfig(equity_simulations=1500)),
                      ("full DecisionReport, heuristic (1,500 equity sims)", DecisionConfig(equity_simulations=1500)),
                      ("full DecisionReport, rollout 400", DecisionConfig(equity_simulations=1500,
                                                                         rollout_simulations=400))):
        prov = solver if "solver path" in name else None
        out[name] = bench(lambda cfg=cfg, prov=prov: analyze(obs, replace(cfg), solver=prov), 8)
    hh = observe_hand_history(load_hands(ROOT / "tests" / "fixtures" / "hands" / "sample.json")[0])
    out["hand-history replay + DecisionReport (heuristic)"] = bench(
        lambda: analyze(observe_hand_history(load_hands(ROOT / "tests" / "fixtures" / "hands" / "sample.json")[0]),
                        DecisionConfig(equity_simulations=1500)), 8)
    del hh
    offline = {}
    for f in ("backend_benchmark.json", "holdem_training_v1_300k.json", "holdem_training_v2.json"):
        p = ROOT / "results" / "validation" / f
        if p.exists():
            d = json.loads(p.read_text())
            if f == "backend_benchmark.json":
                offline["mccfr_iterations_per_second_v1_single_process"] = d["optimized"]["iterations_per_second"]
            elif "convergence_proxies" in d:
                offline[f] = {s: {"iterations": r[-1]["iterations"], "infosets": r[-1]["infosets"]}
                              for s, r in d["convergence_proxies"].items()}
    doc = {"format": "pokeralpha.latency_benchmark/v1",
           "machine": {"python": sys.version.split()[0], "platform": platform.platform(),
                       "processor": platform.processor() or "unknown", "cpus": __import__("os").cpu_count(),
                       "backend": "pure Python + NumPy (no compiled extension)"},
           "online_inference": out, "offline_training": offline}
    (ROOT / "results" / "validation" / "latency_benchmark.json").write_text(json.dumps(doc, indent=1))
    for k, v in out.items():
        print(f"{k:55s} {v}")


if __name__ == "__main__":
    main()
