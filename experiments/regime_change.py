"""Headline experiment: does the adaptive agent notice when the opponent changes?

``ArchetypeBelief`` accumulates log-likelihood over *every* hand it has ever
seen, with no decay — a well-founded design for a stationary opponent, but an
explicit assumption this experiment tests. The opponent here plays one
archetype for the first half of a long match and a second, distinct archetype
for the second half; the hero runs the same adaptive pipeline as
``adaptation_vs_archetypes.py`` (periodic belief-driven refresh, exploitability
budget). We track, every ``--refresh`` hands: the belief's MAP estimate,
posterior mass on the *currently true* archetype, confidence, applied lambda,
and realized chips in that window.

If accumulated evidence never decays, old (pre-switch) hands should keep
outvoting new (post-switch) ones for a long time — the posterior should react
to the regime change far more slowly than it identified the *first* archetype
from a cold start (measured in ``opponent_identification.py``). This is
measured directly, not assumed.

Usage
-----
    python experiments/regime_change.py --seed 42

Outputs (under --outdir, default ./results):
    data/regime_change.csv
    figures/regime_change_confidence.png
    figures/regime_change_chips.png
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
from poker_alpha.opponent.exploit import adaptive_strategy
from poker_alpha.utils.plotting import save_convergence_plot
from poker_alpha.utils.seeds import set_seed


def run_regime_change(game, equilibrium, candidates, before: str, after: str,
                      hands_per_regime: int, refresh: int, epsilon: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    belief = ArchetypeBelief(candidates=candidates)
    state = {"strategy": equilibrium}
    window_chips = {"acc": 0.0, "n": 0}
    rows = []

    def opponent_provider(hand_index: int):
        return candidates[before] if hand_index < hands_per_regime else candidates[after]

    def hero_provider(hand_index: int):
        if hand_index % refresh == 0 and belief.hands_observed > 0:
            estimate = belief.mixture_strategy()
            decision = adaptive_strategy(
                game, equilibrium, estimate, belief.confidence(),
                max_exploitability=epsilon)
            state["strategy"] = decision.strategy
            true_now = before if hand_index < hands_per_regime else after
            post = belief.posterior()
            rows.append({
                "hand": hand_index,
                "regime": "before" if hand_index < hands_per_regime else "after",
                "true_archetype": true_now,
                "map_estimate": belief.map_estimate(),
                "correct": belief.map_estimate() == true_now,
                "confidence": belief.confidence(),
                "posterior_true": post[true_now],
                "lambda_applied": decision.lam_applied,
                "window_chips_per_100": (100 * window_chips["acc"] / window_chips["n"]
                                         if window_chips["n"] else 0.0),
            })
            window_chips["acc"] = 0.0
            window_chips["n"] = 0
        return state["strategy"]

    def on_hand(record):
        belief.update(record.observation)
        window_chips["acc"] += record.hero_utility
        window_chips["n"] += 1

    total_hands = 2 * hands_per_regime
    simulate_match(game, hero_provider, opponent_provider, hands=total_hands,
                   rng=rng, on_hand=on_hand)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=str, default="maniac")
    p.add_argument("--after", type=str, default="nit")
    p.add_argument("--hands-per-regime", type=int, default=2000)
    p.add_argument("--refresh", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eq-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = set_seed(args.seed)
    game, equilibrium = solve_leduc_equilibrium(args.eq_iterations)
    candidates = build_archetypes(equilibrium)

    frames = []
    for rep in range(args.repeats):
        print(f"repeat {rep}...")
        df = run_regime_change(game, equilibrium, candidates, args.before,
                               args.after, args.hands_per_regime,
                               args.refresh, args.epsilon, rng)
        df["repeat"] = rep
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    data_dir = args.outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "regime_change.csv"
    df.to_csv(csv_path, index=False)

    # First hand (after the switch) at which each repeat's MAP estimate
    # flips to the new true archetype, and stays flipped through the run.
    after_rows = df[df.regime == "after"]
    flip_hands = []
    for rep in range(args.repeats):
        sub = after_rows[after_rows.repeat == rep].sort_values("hand")
        correct_from = sub[sub.correct]
        flip_hands.append(int(correct_from["hand"].iloc[0]) - args.hands_per_regime
                          if len(correct_from) else None)
    print(f"hands after switch until MAP estimate reads '{args.after}': "
         f"{flip_hands}")

    summary = (df.groupby(["regime", "hand"])
                 .agg(mean_confidence=("confidence", "mean"),
                      mean_posterior_true=("posterior_true", "mean"),
                      accuracy=("correct", "mean"),
                      mean_lambda=("lambda_applied", "mean"),
                      mean_window_chips=("window_chips_per_100", "mean"))
                 .reset_index().sort_values("hand"))
    print(summary.to_string(index=False))

    hands = summary["hand"].tolist()
    conf_series = [
        ("posterior mass on true archetype", hands, summary["mean_posterior_true"].tolist()),
        ("belief confidence", hands, summary["mean_confidence"].tolist()),
        ("MAP-estimate accuracy", hands, summary["accuracy"].tolist()),
    ]
    fig1 = save_convergence_plot(
        conf_series, "hand index (switch at "
        f"{args.hands_per_regime})", "value",
        f"Regime change: {args.before} -> {args.after} at hand {args.hands_per_regime}",
        args.outdir / "figures" / "regime_change_confidence.png")

    chips_series = [("adaptive hero, chips/100 in window", hands,
                     summary["mean_window_chips"].tolist())]
    fig2 = save_convergence_plot(
        chips_series, "hand index", "chips / 100 hands (windowed)",
        f"Realized EV around the regime change ({args.before} -> {args.after})",
        args.outdir / "figures" / "regime_change_chips.png")

    print(f"\nwrote {csv_path}")
    print(f"wrote {fig1}")
    print(f"wrote {fig2}")


if __name__ == "__main__":
    main()
