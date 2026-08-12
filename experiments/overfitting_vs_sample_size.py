"""Headline experiment: does the exploitability guardrail prevent overfitting?

An adaptive agent that commits to a strategy after too little evidence risks
committing to a *wrong* read — early hands are dominated by sampling noise, not
signal (see ``opponent_identification.py``). This experiment isolates that
risk from the risk of *acting* on it:

For each archetype and a grid of "evidence sizes" (hands of belief-forming
observation), repeatedly (1) simulate that many hands with the hero playing
equilibrium and the belief updating, (2) freeze the resulting posterior into a
strategy two ways — **unguarded** (``max_exploitability=None``, capped only by
confidence) and **guarded** (capped to an exploitability budget) — and
(3) score each frozen strategy's *exact* EV against the true opponent via
``profile_value`` (no evaluation noise; only the belief-formation step is
stochastic). Equilibrium and oracle-best-response EVs are exact reference
lines.

The expected shape: at small sample sizes the unguarded strategy can commit
hard to a wrong read and lose to equilibrium (and does so with high variance);
the guardrail bounds how much a wrong small-sample commitment can cost.

Usage
-----
    python experiments/overfitting_vs_sample_size.py --repeats 10 --seed 42

Outputs (under --outdir, default ./results):
    data/overfitting_vs_sample_size.csv
    figures/overfitting_vs_sample_size_<archetype>.png (one per archetype)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _opponent_common import build_archetypes, seat_avg_ev, solve_leduc_equilibrium

from poker_alpha.opponent import ArchetypeBelief, exploitative_strategy, simulate_match
from poker_alpha.opponent.exploit import (
    blend,
    cap_lambda_by_exploitability,
    confidence_lambda,
    deviation_magnitude,
)
from poker_alpha.utils.plotting import save_convergence_plot
from poker_alpha.utils.seeds import set_seed

LEARN_GRID = (5, 20, 60, 150, 400, 1000)


def frozen_evs(game, equilibrium, candidates, true_name, opponent,
              learn_hands, epsilon, rng):
    """Simulate ``learn_hands`` of belief formation; return (guarded, unguarded)
    exact seat-averaged EVs of the resulting frozen strategies."""
    belief = ArchetypeBelief(candidates=candidates)
    simulate_match(game, equilibrium, opponent, hands=learn_hands, rng=rng,
                   on_hand=lambda r: belief.update(r.observation))

    estimate = belief.mixture_strategy()
    confidence = belief.confidence()
    exploit = exploitative_strategy(game, estimate, equilibrium)
    dev = deviation_magnitude(equilibrium, estimate)
    lam = confidence_lambda(confidence, dev)

    unguarded = blend(equilibrium, exploit, lam)
    lam_guarded = cap_lambda_by_exploitability(
        game, equilibrium, exploit, lam, epsilon)
    guarded = blend(equilibrium, exploit, lam_guarded)

    return (seat_avg_ev(game, guarded, opponent),
           seat_avg_ev(game, unguarded, opponent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--eq-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = set_seed(args.seed)
    game, equilibrium = solve_leduc_equilibrium(args.eq_iterations)
    candidates = build_archetypes(equilibrium)

    rows = []
    for name, opponent in candidates.items():
        if name == "balanced":
            continue
        print(f"{name}...")
        eq_ev = seat_avg_ev(game, equilibrium, opponent)
        oracle = exploitative_strategy(game, opponent, equilibrium)
        oracle_ev = seat_avg_ev(game, oracle, opponent)
        for learn_hands in LEARN_GRID:
            for rep in range(args.repeats):
                guarded_ev, unguarded_ev = frozen_evs(
                    game, equilibrium, candidates, name, opponent,
                    learn_hands, args.epsilon, rng)
                rows.append({
                    "archetype": name, "learn_hands": learn_hands,
                    "repeat": rep, "equilibrium_ev": eq_ev,
                    "oracle_ev": oracle_ev,
                    "guarded_ev": guarded_ev, "unguarded_ev": unguarded_ev,
                })

    df = pd.DataFrame(rows)
    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "overfitting_vs_sample_size.csv"
    df.to_csv(csv_path, index=False)

    summary = (df.groupby(["archetype", "learn_hands"])
                 .agg(equilibrium_ev=("equilibrium_ev", "first"),
                      oracle_ev=("oracle_ev", "first"),
                      guarded_mean=("guarded_ev", "mean"),
                      guarded_min=("guarded_ev", "min"),
                      unguarded_mean=("unguarded_ev", "mean"),
                      unguarded_min=("unguarded_ev", "min"))
                 .reset_index())
    print(summary.to_string(index=False))

    fig_dir = args.outdir / "figures"
    for name in summary["archetype"].unique():
        sub = summary[summary.archetype == name]
        series = [
            ("equilibrium", LEARN_GRID, sub["equilibrium_ev"].tolist()),
            ("oracle best response", LEARN_GRID, sub["oracle_ev"].tolist()),
            ("adaptive, guarded", LEARN_GRID, sub["guarded_mean"].tolist()),
            ("adaptive, unguarded", LEARN_GRID, sub["unguarded_mean"].tolist()),
        ]
        save_convergence_plot(
            series, "hands of belief-forming evidence",
            "exact hero EV (chips/hand, seat-averaged)",
            f"Overfitting vs sample size: {name}",
            fig_dir / f"overfitting_vs_sample_size_{name}.png", logx=True)

    print(f"\nwrote {csv_path}")
    print(f"wrote {fig_dir}/overfitting_vs_sample_size_<archetype>.png "
          f"({len(summary['archetype'].unique())} files)")


if __name__ == "__main__":
    main()
