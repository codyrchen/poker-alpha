"""Headline experiment: how many hands does it take to identify an opponent?

For each archetype, repeat a match where the hero plays the *static*
equilibrium strategy (so the outcome measures identification alone, with no
exploitation feedback loop) against that archetype, and record — at a grid of
hand counts — whether the Bayesian posterior's MAP estimate is correct and how
confident (1 − normalized entropy) it is. Averaging over repeats turns each
single noisy run into an accuracy curve.

Usage
-----
    python experiments/opponent_identification.py --repeats 50 --seed 42

Outputs (under --outdir, default ./results):
    data/opponent_identification.csv
    figures/opponent_identification_accuracy.png
    figures/opponent_identification_confidence.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _opponent_common import build_archetypes, solve_leduc_equilibrium

from poker_alpha.opponent import ArchetypeBelief, simulate_match
from poker_alpha.utils.plotting import save_convergence_plot
from poker_alpha.utils.seeds import set_seed

CHECKPOINTS = (5, 10, 25, 50, 100, 200, 400, 800)


def run_one(game, equilibrium, candidates, true_name, opponent, hands,
           rng):
    belief = ArchetypeBelief(candidates=candidates)
    checkpoint_set = set(CHECKPOINTS)
    out = []

    def on_hand(record):
        belief.update(record.observation)
        h = belief.hands_observed
        if h in checkpoint_set:
            post = belief.posterior()
            out.append({
                "hands": h,
                "correct": belief.map_estimate() == true_name,
                "confidence": belief.confidence(),
                "posterior_true": post[true_name],
            })

    simulate_match(game, equilibrium, opponent, hands=hands, rng=rng,
                   on_hand=on_hand)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--eq-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = set_seed(args.seed)
    game, equilibrium = solve_leduc_equilibrium(args.eq_iterations)
    candidates = build_archetypes(equilibrium)
    max_hands = max(CHECKPOINTS)

    rows = []
    for name, opponent in candidates.items():
        if name == "balanced":
            continue  # no deviation from equilibrium to detect
        print(f"{name}...")
        for rep in range(args.repeats):
            for point in run_one(game, equilibrium, candidates, name,
                                 opponent, max_hands, rng):
                rows.append({"archetype": name, "repeat": rep, **point})

    df = pd.DataFrame(rows)
    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "opponent_identification.csv"
    df.to_csv(csv_path, index=False)

    summary = (df.groupby(["archetype", "hands"])
                 .agg(accuracy=("correct", "mean"),
                      mean_confidence=("confidence", "mean"),
                      mean_posterior_true=("posterior_true", "mean"))
                 .reset_index())
    print(summary.to_string(index=False))

    archetypes = [a for a in candidates if a != "balanced"]
    acc_series = [(a, CHECKPOINTS,
                  [summary[(summary.archetype == a) & (summary.hands == h)]
                   ["accuracy"].iloc[0] for h in CHECKPOINTS])
                 for a in archetypes]
    conf_series = [(a, CHECKPOINTS,
                    [summary[(summary.archetype == a) & (summary.hands == h)]
                     ["mean_confidence"].iloc[0] for h in CHECKPOINTS])
                   for a in archetypes]

    fig1 = save_convergence_plot(
        acc_series, "hands observed", "P(MAP estimate correct)",
        f"Opponent identification accuracy ({args.repeats} repeats)",
        args.outdir / "figures" / "opponent_identification_accuracy.png",
        logx=True)
    fig2 = save_convergence_plot(
        conf_series, "hands observed", "posterior confidence (1 − norm. entropy)",
        f"Belief confidence over hands ({args.repeats} repeats)",
        args.outdir / "figures" / "opponent_identification_confidence.png",
        logx=True)

    print(f"\nwrote {csv_path}")
    print(f"wrote {fig1}")
    print(f"wrote {fig2}")


if __name__ == "__main__":
    main()
