"""Headline experiment: does risk-constrained adaptation actually pay?

Three heroes face each opponent archetype over a long Leduc match:

* **equilibrium** — static CFR+ average strategy. Unexploitable, leaves EV on
  the table against a flawed opponent.
* **oracle best response** — best-responds to the opponent's *true* strategy
  from hand 1. Not a realizable agent (it cheats by knowing the opponent
  exactly and unboundedly); it is the EV ceiling and the exploitability floor
  a real agent trades against.
* **adaptive** — the risk-constrained agent from ``opponent/exploit.py``:
  maintains a Bayesian posterior over the six archetypes (``ArchetypeBelief``),
  refreshes its blended strategy every ``--refresh`` hands from the current
  posterior mixture and confidence, capped to stay under an exploitability
  budget (``--epsilon``).

A second stage sweeps that budget itself against one archetype
(``--sweep-archetype``, default maniac): this is the project's central
question made quantitative — exactly how much equilibrium robustness
(exploitability) must be spent to capture how much realized EV, between the
epsilon=0 (equilibrium) and epsilon=infinity (oracle best response) endpoints.

Usage
-----
    python experiments/adaptation_vs_archetypes.py --hands 2000 --seed 42

Outputs (under --outdir, default ./results):
    data/adaptation_vs_archetypes.csv
    data/adaptation_epsilon_sweep.csv
    figures/adaptation_vs_archetypes.png
    figures/adaptation_epsilon_sweep.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _opponent_common import build_archetypes, seat_avg_ev, solve_leduc_equilibrium

from poker_alpha.opponent import (
    ArchetypeBelief,
    exploitative_strategy,
    simulate_match,
)
from poker_alpha.opponent.exploit import adaptive_strategy
from poker_alpha.risk import bootstrap_ci
from poker_alpha.solvers import exploitability
from poker_alpha.utils.plotting import save_convergence_plot, save_grouped_bar_chart
from poker_alpha.utils.seeds import set_seed


def make_adaptive_provider(game, equilibrium, candidates, refresh: int,
                           epsilon: float):
    """Return (provider, belief) where provider(hand_index) -> current strategy.

    Refreshes the blended strategy every ``refresh`` hands from the belief's
    posterior mixture; the belief itself updates every hand via ``on_hand``.
    """
    belief = ArchetypeBelief(candidates=candidates)
    state = {"strategy": equilibrium, "lambdas": [], "expl": []}

    def provider(hand_index: int):
        if hand_index % refresh == 0 and belief.hands_observed > 0:
            estimate = belief.mixture_strategy()
            decision = adaptive_strategy(
                game, equilibrium, estimate, belief.confidence(),
                max_exploitability=epsilon)
            state["strategy"] = decision.strategy
            state["lambdas"].append(decision.lam_applied)
        return state["strategy"]

    def on_hand(record):
        belief.update(record.observation)

    return provider, on_hand, state


def run_archetype(game, equilibrium, candidates, name: str, opponent: Dict,
                  hands: int, refresh: int, epsilon: float,
                  rng: np.random.Generator) -> list:
    rows = []

    # -- equilibrium hero --
    recs = simulate_match(game, equilibrium, opponent, hands=hands, rng=rng)
    chips = np.array([r.hero_utility for r in recs])
    lo, hi = bootstrap_ci(chips, seed=0)
    rows.append({"archetype": name, "mode": "equilibrium",
                 "chips_per_100": 100 * chips.mean(),
                 "ci_low": 100 * lo, "ci_high": 100 * hi,
                 "exploitability": exploitability(game, equilibrium),
                 "mean_lambda": 0.0})

    # -- oracle best response (knows the true opponent strategy exactly) --
    oracle = exploitative_strategy(game, opponent, equilibrium)
    recs = simulate_match(game, oracle, opponent, hands=hands, rng=rng)
    chips = np.array([r.hero_utility for r in recs])
    lo, hi = bootstrap_ci(chips, seed=0)
    rows.append({"archetype": name, "mode": "oracle_br",
                 "chips_per_100": 100 * chips.mean(),
                 "ci_low": 100 * lo, "ci_high": 100 * hi,
                 "exploitability": exploitability(game, oracle),
                 "mean_lambda": 1.0})

    # -- adaptive: learns the archetype from public actions only --
    provider, on_hand, state = make_adaptive_provider(
        game, equilibrium, candidates, refresh, epsilon)
    recs = simulate_match(game, provider, opponent, hands=hands, rng=rng,
                          on_hand=on_hand)
    chips = np.array([r.hero_utility for r in recs])
    lo, hi = bootstrap_ci(chips, seed=0)
    mean_lambda = float(np.mean(state["lambdas"])) if state["lambdas"] else 0.0
    rows.append({"archetype": name, "mode": "adaptive",
                 "chips_per_100": 100 * chips.mean(),
                 "ci_low": 100 * lo, "ci_high": 100 * hi,
                 "exploitability": exploitability(game, state["strategy"]),
                 "mean_lambda": mean_lambda})
    return rows


def run_epsilon_sweep(game, equilibrium, candidates, name: str,
                      opponent: Dict, hands: int, refresh: int,
                      epsilon_grid, rng: np.random.Generator) -> list:
    rows = []
    for epsilon in epsilon_grid:
        provider, on_hand, state = make_adaptive_provider(
            game, equilibrium, candidates, refresh, epsilon)
        recs = simulate_match(game, provider, opponent, hands=hands, rng=rng,
                              on_hand=on_hand)
        chips = np.array([r.hero_utility for r in recs])
        lo, hi = bootstrap_ci(chips, seed=0)
        mean_lambda = (float(np.mean(state["lambdas"]))
                       if state["lambdas"] else 0.0)
        rows.append({"archetype": name, "epsilon": epsilon,
                     "chips_per_100": 100 * chips.mean(),
                     "ci_low": 100 * lo, "ci_high": 100 * hi,
                     "exploitability": exploitability(game, state["strategy"]),
                     "mean_lambda": mean_lambda})
        print(f"  epsilon={epsilon:6.3f}  chips/100={100 * chips.mean():+7.2f}"
              f"  mean_lambda={mean_lambda:.4f}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--refresh", type=int, default=20,
                  help="hands between adaptive strategy recomputations")
    p.add_argument("--epsilon", type=float, default=0.1,
                  help="exploitability budget for the adaptive hero")
    p.add_argument("--eq-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--sweep-archetype", type=str, default="maniac")
    p.add_argument("--sweep-hands", type=int, default=2000)
    p.add_argument("--sweep-refresh", type=int, default=40)
    p.add_argument("--skip-sweep", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = set_seed(args.seed)
    game, equilibrium = solve_leduc_equilibrium(args.eq_iterations)
    candidates = build_archetypes(equilibrium)

    rows = []
    for name, opponent in candidates.items():
        print(f"{name}...")
        rows.extend(run_archetype(game, equilibrium, candidates, name,
                                  opponent, args.hands, args.refresh,
                                  args.epsilon, rng))

    df = pd.DataFrame(rows)
    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "adaptation_vs_archetypes.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    categories = list(candidates)
    modes = ["equilibrium", "oracle_br", "adaptive"]
    series = {m: [df[(df.archetype == c) & (df["mode"] == m)]
                  ["chips_per_100"].iloc[0] for c in categories]
              for m in modes}
    errors = {m: [(df[(df.archetype == c) & (df["mode"] == m)]["ci_low"].iloc[0],
                   df[(df.archetype == c) & (df["mode"] == m)]["ci_high"].iloc[0])
                  for c in categories]
              for m in modes}
    fig_path = save_grouped_bar_chart(
        categories, series, "chips / 100 hands (hero)",
        f"Equilibrium vs oracle best response vs adaptive ({args.hands} hands/match)",
        args.outdir / "figures" / "adaptation_vs_archetypes.png", errors=errors)

    print(f"\nwrote {csv_path}")
    print(f"wrote {fig_path}")

    if args.skip_sweep:
        return

    print(f"\nepsilon sweep vs {args.sweep_archetype}...")
    sweep_rows = run_epsilon_sweep(
        game, equilibrium, candidates, args.sweep_archetype,
        candidates[args.sweep_archetype], args.sweep_hands,
        args.sweep_refresh, (0.0, 0.02, 0.05, 0.1, 0.3, 1.0), rng)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_csv = data_dir / "adaptation_epsilon_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    eq_ev = df[(df.archetype == args.sweep_archetype)
              & (df["mode"] == "equilibrium")]["chips_per_100"].iloc[0]
    oracle_ev = df[(df.archetype == args.sweep_archetype)
                  & (df["mode"] == "oracle_br")]["chips_per_100"].iloc[0]
    epsilons = sweep_df["epsilon"].tolist()
    ev_series = [
        ("adaptive (budget epsilon)", epsilons, sweep_df["chips_per_100"].tolist()),
        ("equilibrium", epsilons, [eq_ev] * len(epsilons)),
        ("oracle best response", epsilons, [oracle_ev] * len(epsilons)),
    ]
    sweep_fig = save_convergence_plot(
        ev_series, "exploitability budget epsilon (chips)",
        "chips / 100 hands (hero)",
        f"Robustness-exploitation tradeoff vs {args.sweep_archetype}",
        args.outdir / "figures" / "adaptation_epsilon_sweep.png")

    print(f"wrote {sweep_csv}")
    print(f"wrote {sweep_fig}")


if __name__ == "__main__":
    main()
