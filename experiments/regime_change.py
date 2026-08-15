"""Headline experiment: recency-aware belief vs the stationary baseline.

``regime_change.py`` (phase 15) found that ``ArchetypeBelief``'s stationary
posterior (decay=1.0: every hand ever seen counts equally forever) never
recovers from a mid-match archetype switch within any practical hand count.
``ArchetypeBelief`` now takes a ``decay`` parameter (exponential forgetting of
old evidence; see ``opponent/beliefs.py``); this experiment measures whether
that actually fixes the problem, and what it costs.

Two parts:

1. **Switching comparison** — one opponent archetype for the first half of a
   long match, a different one for the second half. A single hand sequence
   (hero fixed at equilibrium, so the sequence is identical either way) feeds
   *both* a decay=1.0 belief and a decay=``--decay`` belief in parallel, so
   the comparison is paired, not just repeated. Tracked every ``--id-refresh``
   hands: MAP estimate, posterior mass on the *true current* archetype,
   confidence. Every ``--ev-every`` hands (a coarser grid — this needs a
   best-response computation): the *exact* EV (``profile_value``, no
   simulation noise) of each belief's current guarded adaptive strategy
   against the true current opponent — "EV lost" is the gap between this and
   the oracle/equilibrium reference lines.

2. **Stationary control (false-switch sensitivity)** — the opponent's
   archetype never changes. A decaying belief has no old evidence protecting
   it from noise the way the stationary baseline's ever-growing evidence pool
   does, so it can misidentify a *stationary* opponent transiently just from
   an unlucky run of hands. Measured across a decay sweep and two archetypes
   (maniac: easy to identify; bluff_heavy: the hardest archetype in
   ``opponent_identification.py``) as the post-warmup fraction of checkpoints
   with a wrong MAP estimate and the count of correct-to-wrong flip episodes.

Usage
-----
    python experiments/regime_change.py --seed 42

Outputs (under --outdir, default ./results):
    data/regime_change_switching.csv        per-checkpoint belief state
    data/regime_change_switching_ev.csv     per-checkpoint exact EV
    data/regime_change_control.csv          stationary-control raw checkpoints
    data/regime_change_control_summary.csv  flip rates per (archetype, decay)
    figures/regime_change_posterior_true.png
    figures/regime_change_confidence.png
    figures/regime_change_ev.png
    figures/regime_change_control_flip_rate.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _opponent_common import build_archetypes, seat_avg_ev, solve_leduc_equilibrium

from poker_alpha.opponent import ArchetypeBelief, simulate_match
from poker_alpha.opponent.exploit import (
    blend,
    cap_lambda_by_exploitability,
    confidence_lambda,
    deviation_magnitude,
    exploitative_strategy,
)
from poker_alpha.utils.plotting import save_convergence_plot, save_grouped_bar_chart
from poker_alpha.utils.seeds import set_seed

BELIEF_LABELS = ("baseline", "recency")


def frozen_ev(game, equilibrium, belief: ArchetypeBelief, opponent,
             epsilon: float) -> float:
    """Exact seat-averaged EV of ``belief``'s current guarded adaptive
    strategy against ``opponent`` (no simulation noise; see
    overfitting_vs_sample_size.py for the same pattern)."""
    estimate = belief.mixture_strategy()
    confidence = belief.confidence()
    exploit = exploitative_strategy(game, estimate, equilibrium)
    dev = deviation_magnitude(equilibrium, estimate)
    lam = confidence_lambda(confidence, dev)
    lam_guarded = cap_lambda_by_exploitability(
        game, equilibrium, exploit, lam, epsilon)
    guarded = blend(equilibrium, exploit, lam_guarded)
    return seat_avg_ev(game, guarded, opponent)


def run_switching(game, equilibrium, candidates, before: str, after: str,
                  hands_per_regime: int, id_refresh: int, ev_every: int,
                  epsilon: float, decay: float, repeats: int,
                  seed: int) -> "tuple[pd.DataFrame, pd.DataFrame]":
    total_hands = 2 * hands_per_regime
    id_rows: List[dict] = []
    ev_rows: List[dict] = []

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        beliefs = {
            "baseline": ArchetypeBelief(candidates=candidates, decay=1.0),
            "recency": ArchetypeBelief(candidates=candidates, decay=decay),
        }

        def opponent_provider(hand_index: int):
            return candidates[before] if hand_index < hands_per_regime else candidates[after]

        def on_hand(record):
            for belief in beliefs.values():
                belief.update(record.observation)
            h = record.hand_index + 1
            true_now = before if record.hand_index < hands_per_regime else after
            regime = "before" if record.hand_index < hands_per_regime else "after"
            if h % id_refresh == 0:
                for label, belief in beliefs.items():
                    post = belief.posterior()
                    id_rows.append({
                        "repeat": rep, "hand": h, "regime": regime,
                        "true_archetype": true_now, "belief": label,
                        "map_estimate": belief.map_estimate(),
                        "correct": belief.map_estimate() == true_now,
                        "confidence": belief.confidence(),
                        "posterior_true": post[true_now],
                    })
            if h % ev_every == 0:
                opponent_now = candidates[true_now]
                for label, belief in beliefs.items():
                    ev = frozen_ev(game, equilibrium, belief, opponent_now, epsilon)
                    ev_rows.append({
                        "repeat": rep, "hand": h, "regime": regime,
                        "true_archetype": true_now, "belief": label, "ev": ev,
                    })

        simulate_match(game, equilibrium, opponent_provider, hands=total_hands,
                       rng=rng, on_hand=on_hand)
        print(f"  switching repeat {rep} done")

    return pd.DataFrame(id_rows), pd.DataFrame(ev_rows)


def sustained_flip_delay(sub: pd.DataFrame, hands_per_regime: int) -> Optional[int]:
    """Hands after the switch until ``correct`` turns True and stays True for
    the rest of the (sorted-by-hand) group; None if it never does."""
    sub = sub.sort_values("hand")
    correct = sub["correct"].to_numpy()
    hands = sub["hand"].to_numpy()
    if len(correct) == 0:
        return None
    rev = correct[::-1].astype(int)
    mask = np.cumprod(rev)[::-1].astype(bool)
    if not mask.any():
        return None
    return int(hands[np.argmax(mask)]) - hands_per_regime


def run_stationary_control(game, equilibrium, candidates, archetype: str,
                           hands: int, refresh: int, decays: Sequence[float],
                           repeats: int, seed: int) -> pd.DataFrame:
    opponent = candidates[archetype]
    rows: List[dict] = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        beliefs = {d: ArchetypeBelief(candidates=candidates, decay=d)
                  for d in decays}

        def on_hand(record):
            h = record.hand_index + 1
            for belief in beliefs.values():
                belief.update(record.observation)
            if h % refresh == 0:
                for d, belief in beliefs.items():
                    rows.append({
                        "repeat": rep, "hand": h, "decay": d,
                        "archetype": archetype,
                        "correct": belief.map_estimate() == archetype,
                        "confidence": belief.confidence(),
                    })

        simulate_match(game, equilibrium, opponent, hands=hands, rng=rng,
                       on_hand=on_hand)
    return pd.DataFrame(rows)


def count_flip_episodes(correct: pd.Series) -> int:
    arr = correct.to_numpy()
    return int(np.sum((arr[:-1]) & (~arr[1:]))) if len(arr) > 1 else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=str, default="maniac")
    p.add_argument("--after", type=str, default="nit")
    p.add_argument("--hands-per-regime", type=int, default=2000)
    p.add_argument("--id-refresh", type=int, default=20)
    p.add_argument("--ev-every", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--decay", type=float, default=0.995,
                   help="recency-aware forgetting factor "
                        "(effective horizon ~= 1/(1-decay) hands)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--control-hands", type=int, default=3000)
    p.add_argument("--control-refresh", type=int, default=20)
    p.add_argument("--control-repeats", type=int, default=30)
    p.add_argument("--control-decays", type=float, nargs="+",
                   default=[1.0, 0.999, 0.995, 0.99, 0.98])
    p.add_argument("--control-archetypes", type=str, nargs="+",
                   default=["maniac", "bluff_heavy"])
    p.add_argument("--control-warmup", type=int, default=100,
                   help="ignore checkpoints before this many hands "
                        "(still-learning, not yet a 'false switch')")
    p.add_argument("--eq-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    game, equilibrium = solve_leduc_equilibrium(args.eq_iterations)
    candidates = build_archetypes(equilibrium)

    data_dir = args.outdir / "data"
    fig_dir = args.outdir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. switching comparison ---------------------------------------------
    print(f"switching: {args.before} -> {args.after} at hand "
         f"{args.hands_per_regime}, baseline (decay=1.0) vs recency "
         f"(decay={args.decay})...")
    id_df, ev_df = run_switching(
        game, equilibrium, candidates, args.before, args.after,
        args.hands_per_regime, args.id_refresh, args.ev_every, args.epsilon,
        args.decay, args.repeats, args.seed)
    id_df.to_csv(data_dir / "regime_change_switching.csv", index=False)
    ev_df.to_csv(data_dir / "regime_change_switching_ev.csv", index=False)

    print("\ndetection delay (hands after switch until MAP estimate reads "
         f"'{args.after}' and stays correct through the rest of the match):")
    for label in BELIEF_LABELS:
        delays = []
        for rep in range(args.repeats):
            sub = id_df[(id_df.belief == label) & (id_df.repeat == rep)
                       & (id_df.regime == "after")]
            delays.append(sustained_flip_delay(sub, args.hands_per_regime))
        print(f"  {label:>9}: {delays}")

    id_summary = (id_df.groupby(["belief", "regime", "hand"])
                       .agg(mean_confidence=("confidence", "mean"),
                            mean_posterior_true=("posterior_true", "mean"),
                            accuracy=("correct", "mean"))
                       .reset_index().sort_values("hand"))
    ev_summary = (ev_df.groupby(["belief", "regime", "hand"])
                       .agg(mean_ev=("ev", "mean")).reset_index()
                       .sort_values("hand"))

    hands_id = sorted(id_df["hand"].unique())
    post_series = [(label, hands_id,
                    [id_summary[(id_summary.belief == label)
                                & (id_summary.hand == h)]
                     ["mean_posterior_true"].iloc[0] for h in hands_id])
                  for label in BELIEF_LABELS]
    conf_series = [(label, hands_id,
                    [id_summary[(id_summary.belief == label)
                                & (id_summary.hand == h)]
                     ["mean_confidence"].iloc[0] for h in hands_id])
                  for label in BELIEF_LABELS]
    fig1 = save_convergence_plot(
        post_series, f"hand index (switch at {args.hands_per_regime})",
        "posterior mass on true archetype",
        f"Regime change {args.before} -> {args.after}: baseline vs recency "
        f"(decay={args.decay})",
        fig_dir / "regime_change_posterior_true.png")
    fig2 = save_convergence_plot(
        conf_series, f"hand index (switch at {args.hands_per_regime})",
        "belief confidence",
        f"Confidence stays high through the switch (baseline) vs recovers (recency)",
        fig_dir / "regime_change_confidence.png")

    # Exact EV reference lines (equilibrium, oracle) per regime, aligned to
    # the same ev-checkpoint hands so they render as step functions.
    hands_ev = sorted(ev_df["hand"].unique())
    eq_ev = {name: seat_avg_ev(game, equilibrium, candidates[name])
            for name in (args.before, args.after)}
    oracle_ev = {name: seat_avg_ev(
                    game, exploitative_strategy(game, candidates[name], equilibrium),
                    candidates[name])
                for name in (args.before, args.after)}

    def ref_line(table):
        return [table[args.before if h <= args.hands_per_regime else args.after]
                for h in hands_ev]

    ev_series = [
        (label, hands_ev,
         [ev_summary[(ev_summary.belief == label) & (ev_summary.hand == h)]
          ["mean_ev"].iloc[0] for h in hands_ev])
        for label in BELIEF_LABELS
    ] + [
        ("equilibrium", hands_ev, ref_line(eq_ev)),
        ("oracle best response", hands_ev, ref_line(oracle_ev)),
    ]
    fig3 = save_convergence_plot(
        ev_series, f"hand index (switch at {args.hands_per_regime})",
        "exact hero EV (chips/hand, seat-averaged)",
        f"EV lost after the switch: baseline vs recency (decay={args.decay})",
        fig_dir / "regime_change_ev.png")

    print(f"\nwrote {data_dir / 'regime_change_switching.csv'}")
    print(f"wrote {data_dir / 'regime_change_switching_ev.csv'}")
    print(f"wrote {fig1}\nwrote {fig2}\nwrote {fig3}")

    # -- 2. stationary control: false-switch / noise sensitivity ------------
    print(f"\nstationary control: decays={args.control_decays}, "
         f"archetypes={args.control_archetypes}...")
    control_frames = []
    for archetype in args.control_archetypes:
        print(f"  {archetype}...")
        control_frames.append(run_stationary_control(
            game, equilibrium, candidates, archetype, args.control_hands,
            args.control_refresh, args.control_decays, args.control_repeats,
            args.seed))
    control_df = pd.concat(control_frames, ignore_index=True)
    control_df.to_csv(data_dir / "regime_change_control.csv", index=False)

    warm = control_df[control_df.hand >= args.control_warmup]
    flip_summary = warm.groupby(["archetype", "decay"]).apply(
        lambda g: pd.Series({
            "flip_rate": float((~g["correct"]).mean()),
            "mean_flip_episodes_per_repeat": float(
                g.groupby("repeat")["correct"]
                 .apply(count_flip_episodes).mean()),
            "mean_confidence": float(g["confidence"].mean()),
        }),
        include_groups=False,
    )
    flip_summary = flip_summary.reset_index()
    print(flip_summary.to_string(index=False))
    flip_summary.to_csv(data_dir / "regime_change_control_summary.csv",
                        index=False)

    decay_labels = [str(d) for d in args.control_decays]
    bar_series = {
        arch: [flip_summary[(flip_summary.archetype == arch)
                            & (flip_summary.decay == d)]["flip_rate"].iloc[0]
              for d in args.control_decays]
        for arch in args.control_archetypes
    }
    fig4 = save_grouped_bar_chart(
        decay_labels, bar_series, "P(MAP estimate wrong) post-warmup",
        f"False-switch rate on a stationary opponent vs decay "
        f"({args.control_repeats} repeats)",
        fig_dir / "regime_change_control_flip_rate.png")

    print(f"\nwrote {data_dir / 'regime_change_control.csv'}")
    print(f"wrote {data_dir / 'regime_change_control_summary.csv'}")
    print(f"wrote {fig4}")


if __name__ == "__main__":
    main()
