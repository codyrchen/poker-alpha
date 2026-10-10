"""Benchmark: Hold'em platform workloads (median / p95 latency).

Times, separately: 7-card hand evaluation, heads-up equity, multiway equity
(3- and 9-handed), range sampling, decision rollouts, screen recognition of
one frame, and end-to-end recommendation. Each workload is repeated
``--repeats`` times after one warm-up; median and p95 wall-clock latency are
reported with the hardware description. Seeds are fixed, so results (not
timings) are reproducible.

Usage
-----
    python experiments/holdem_benchmark.py --label baseline
    python experiments/holdem_benchmark.py --label optimized \\
        --compare results/data/holdem_benchmark_baseline.csv

Outputs (under --outdir, default ./results):
    data/holdem_benchmark_<label>.csv
    data/holdem_benchmark_comparison.csv   (with --compare)
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_alpha.decision import DecisionConfig, recommend_action  # noqa: E402
from poker_alpha.decision.rollout import (RolloutCandidate,  # noqa: E402
                                          rollout_action_evs)
from poker_alpha.holdem import ManualStateAdapter  # noqa: E402
from poker_alpha.opponent import ARCHETYPE_MODELS  # noqa: E402
from poker_alpha.poker import card_code, estimate_equity  # noqa: E402
from poker_alpha.poker.evaluator import evaluate_best_codes  # noqa: E402
from poker_alpha.poker.multiway import multiway_equity  # noqa: E402
from poker_alpha.poker.ranges import WeightedRange, preflop_strength  # noqa: E402


def hardware() -> str:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return (f"{cpu} x{os.cpu_count()} | {platform.system()} | Python "
            f"{platform.python_version()} | NumPy {np.__version__}")


def flop_state():
    return ManualStateAdapter.from_dict({
        "num_seats": 6, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": "As Ks", "board": "Qs Js 4h",
        "pot": 13.5, "actor": 0,
        "seats": [{"stack": 96.5}, {"stack": 99.5, "folded": True},
                  {"stack": 90.0}, {"stack": 100, "folded": True},
                  {"stack": 100, "folded": True}, {"stack": 100, "folded": True}],
        "actions": [{"street": 0, "seat": 0, "kind": "raise", "amount": 3.5},
                    {"street": 0, "seat": 2, "kind": "call", "amount": 2.5},
                    {"street": 1, "seat": 2, "kind": "check"}]})


def workloads(args):
    C = card_code
    hands = np.random.default_rng(0).integers(0, 52, size=(1000, 7))
    seven = [list(dict.fromkeys(int(x) for x in h)) for h in hands]
    seven = [h for h in seven if len(h) == 7][:500]
    wide = WeightedRange.from_string("22+,A2s+,K9s+,QTs+,JTs,ATo+,KJo+")
    obs = flop_state()
    hero = [C("As"), C("Ks")]
    cands = [RolloutCandidate("check", "check", 0.0),
             RolloutCandidate("bet_75", "bet", 10.125),
             RolloutCandidate("all_in", "all_in", 96.5)]
    reg = ARCHETYPE_MODELS["regular"]
    out = {
        "hand_evaluation_x500": lambda: [evaluate_best_codes(h) for h in seven],
        "headsup_equity_1000": lambda: estimate_equity(
            ["As", "Ks"], ["Qs", "Js", "4h"], simulations=1000, seed=1),
        "multiway_equity_3way_1000": lambda: multiway_equity(
            hero, [C("Qs"), C("Js"), C("4h")], [wide, wide],
            simulations=1000, seed=1),
        "multiway_equity_9way_300": lambda: multiway_equity(
            hero, (), [None] * 8, simulations=300, seed=1),
        "range_sampling_x1000": lambda: wide.sample(
            np.random.default_rng(1), 1000, exclude=hero),
        "decision_rollout_1000": lambda: rollout_action_evs(
            obs, hero, {2: wide}, {2: reg}, cands, simulations=1000, seed=1),
        "end_to_end_recommendation": lambda: recommend_action(
            obs, config=DecisionConfig(equity_simulations=1000,
                                       rollout_simulations=1000, seed=1)),
    }
    try:
        from poker_alpha.observer.pokernow import (PokerNowStyleAdapter,
                                                   default_layout)
        from PIL import Image

        fix = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
        img = Image.open(fix / "table.png").convert("RGB")
        ad = PokerNowStyleAdapter(default_layout(6, 0))
        out["screen_recognition_frame"] = lambda: ad.read_frame(img)
    except ImportError:
        pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="run")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--compare", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    args = p.parse_args()
    preflop_strength()  # one-time table, excluded from timings
    hw = hardware()
    rows = []
    for name, fn in workloads(args).items():
        fn()  # warm-up
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        arr = np.array(times) * 1000
        rows.append({"workload": name, "median_ms": float(np.median(arr)),
                     "p95_ms": float(np.percentile(arr, 95)),
                     "samples": args.repeats, "label": args.label,
                     "hardware": hw})
        print(f"{name:<30} median {rows[-1]['median_ms']:9.2f} ms   "
              f"p95 {rows[-1]['p95_ms']:9.2f} ms")
    out = args.outdir / "data"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"holdem_benchmark_{args.label}.csv", index=False)
    print(f"hardware: {hw}")
    if args.compare:
        base = pd.read_csv(args.compare).set_index("workload")
        cmp = df.set_index("workload")[["median_ms"]].join(
            base[["median_ms"]], rsuffix="_baseline")
        cmp["speedup"] = cmp["median_ms_baseline"] / cmp["median_ms"]
        cmp.to_csv(out / "holdem_benchmark_comparison.csv")
        print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
