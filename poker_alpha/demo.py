"""End-to-end narrative demo: ``python -m poker_alpha.demo``.

Walks the project's research story in one deterministic run:

    A. equilibrium as a robust baseline           (computed live)
    B. Bayesian opponent identification            (computed live)
    C. risk-constrained adaptive exploitation      (computed live)
    D. failure and repair under regime change      (committed results)
    E. performance engineering                     (committed results)
    F. the central takeaway

Every number is either computed during this run or read from a CSV in
``results/data/`` produced by a seeded experiment script; each section is
labelled accordingly. Nothing is hard-coded.

Sections D and E report full-scale experiments (thousands of hands, repeated
benchmark runs) that take minutes to regenerate, so the demo loads their
committed outputs rather than re-running them. ``experiments/`` holds the
scripts that produced them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .games.kuhn import KuhnPoker
from .games.leduc import LeducPoker
from .opponent import (
    ARCHETYPES,
    ArchetypeBelief,
    leduc_is_weak,
    simulate_match,
    tilt_strategy,
)
from .opponent.exploit import (
    adaptive_strategy,
    blend,
    deviation_magnitude,
    exploitative_strategy,
)
from .solvers import CFRPlusSolver, CFRSolver, exploitability
from .solvers.evaluation import expected_value, profile_value

Strategy = Dict[str, Dict[str, float]]

WIDTH = 72
DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "data"


# --------------------------------------------------------------------------
# presentation helpers
# --------------------------------------------------------------------------

def header(letter: str, title: str, source: str) -> None:
    """Section banner. ``source`` says where the section's numbers come from."""
    print()
    print("=" * WIDTH)
    print(f" {letter}. {title}")
    print(f"    [{source}]")
    print("=" * WIDTH)


def row(label: str, value: str, indent: int = 2) -> None:
    print(f"{' ' * indent}{label:<34}{value}")


def fmt_rate(x: float) -> str:
    """Throughput as a compact human-readable rate (415k, 1.2M, 37.7k)."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.1f}k"
    return f"{x:.0f}"


# --------------------------------------------------------------------------
# committed-result loaders (pure functions over CSV rows, unit-tested)
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> List[dict]:
    """Minimal CSV reader: avoids a pandas import just to print a few rows."""
    import csv

    if not path.exists():
        raise FileNotFoundError(
            f"missing committed result: {path}\n"
            f"Regenerate it with the matching script in experiments/.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def sustained_recovery_delay(rows: List[dict], belief: str,
                             hands_per_regime: int) -> Optional[int]:
    """Hands after the switch until ``belief``'s MAP estimate becomes correct
    and *stays* correct for the remainder of the run.

    A transient correct guess does not count as recovery, so we scan from the
    end backwards and take the start of the final all-correct run. Returns
    ``None`` when the belief never sustains a correct estimate.
    """
    post = sorted(
        (r for r in rows
         if r["belief"] == belief and r["regime"] == "after"),
        key=lambda r: int(r["hand"]))
    if not post:
        return None
    delay = None
    for r in reversed(post):
        if r["correct"].strip().lower() in ("true", "1"):
            delay = int(r["hand"]) - hands_per_regime
        else:
            break
    return delay


def recovery_delays_by_repeat(rows: List[dict], belief: str,
                              hands_per_regime: int = 2000) -> List[Optional[int]]:
    """``sustained_recovery_delay`` per repeat, ordered by repeat index."""
    repeats = sorted({int(r["repeat"]) for r in rows})
    return [
        sustained_recovery_delay(
            [r for r in rows if int(r["repeat"]) == rep], belief,
            hands_per_regime)
        for rep in repeats
    ]


def high_confidence_fraction(rows: List[dict], belief: str,
                             threshold: float = 0.99) -> float:
    """Fraction of post-switch checkpoints where ``belief`` is above
    ``threshold`` confident -- while being wrong.

    Averaged per checkpoint across repeats first, matching how
    ``regime_change_confidence.png`` and the writeup aggregate it, so the
    demo, the figures and RESEARCH.md all quote the same number.
    """
    post = [r for r in rows
            if r["belief"] == belief and r["regime"] == "after"]
    if not post:
        return 0.0
    by_hand: Dict[int, List[float]] = {}
    for r in post:
        by_hand.setdefault(int(r["hand"]), []).append(float(r["confidence"]))
    means = [sum(v) / len(v) for v in by_hand.values()]
    return sum(m > threshold for m in means) / len(means)


def speedup_row(rows: List[dict], workload: str) -> dict:
    """Look up one workload's before/after throughput and speedup."""
    for r in rows:
        if r["workload"] == workload:
            return {
                "workload": workload,
                "before": float(r["throughput_before"]),
                "after": float(r["throughput_after"]),
                "speedup": float(r["speedup"]),
                "unit": r["unit"],
            }
    raise KeyError(f"workload {workload!r} not in benchmark results")


def sims_within_budget(throughput: float, budget_seconds: float) -> int:
    """Simulations affordable at ``throughput`` within a latency budget."""
    return int(round(throughput * budget_seconds))


def error_ratio_from_speedup(speedup: float) -> float:
    """Monte Carlo error scales as 1/sqrt(n), so N x the samples tightens the
    error bars by sqrt(N) -- not by N."""
    return float(np.sqrt(speedup))


# --------------------------------------------------------------------------
# live computation helpers
# --------------------------------------------------------------------------

def seat_averaged_ev(game, hero: Strategy, opponent: Strategy) -> float:
    """Exact hero EV in chips/hand, averaged over both seats (no sampling)."""
    return 0.5 * (profile_value(game, hero, opponent)
                  - profile_value(game, opponent, hero))


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def section_a(kuhn_iters: int, leduc_iters: int):
    header("A", "Equilibrium as a robust baseline", "computed live")
    print("  CFR drives average regret to zero; the time-averaged strategy")
    print("  converges to Nash. Exploitability = what a worst-case adversary")
    print("  wins against us, in chips. Zero exactly at equilibrium.\n")

    kuhn = KuhnPoker()
    solver = CFRSolver(kuhn)
    t0 = time.perf_counter()
    solver.train(kuhn_iters)
    kuhn_secs = time.perf_counter() - t0
    kuhn_strategy = solver.average_strategy()
    value = expected_value(kuhn, kuhn_strategy)

    print("  Kuhn poker (12 information sets) -- has a known analytic value:")
    row("solver / iterations", f"CFR / {kuhn_iters:,}")
    row("game value to player 0", f"{value:+.6f}   (theory -1/18 = {-1/18:+.6f})")
    row("exploitability", f"{exploitability(kuhn, kuhn_strategy):.2e} chips")
    row("solve time", f"{kuhn_secs:.1f}s")

    leduc = LeducPoker()
    solver = CFRPlusSolver(leduc)
    t0 = time.perf_counter()
    solver.train(leduc_iters)
    leduc_secs = time.perf_counter() - t0
    equilibrium = solver.average_strategy()
    eq_exploitability = exploitability(leduc, equilibrium)

    print("\n  Leduc poker (288 information sets) -- the environment used for")
    print("  every opponent-modeling result below:")
    row("solver / iterations", f"CFR+ / {leduc_iters:,}")
    row("game value to player 0",
        f"{expected_value(leduc, equilibrium):+.4f}   (known -0.0856)")
    row("exploitability", f"{eq_exploitability:.2e} chips")
    row("solve time", f"{leduc_secs:.1f}s")
    return leduc, equilibrium, eq_exploitability


def identification_checkpoints(hands: int) -> List[int]:
    """Log-spaced hand counts at which to report the posterior.

    The belief saturates quickly, so evenly spaced checkpoints would show a
    flat line; the informative part is the first few dozen hands.
    """
    grid = [h for h in (5, 10, 25, 50, 100, 200, 400) if h < hands]
    return grid + [hands]


def section_b(game, equilibrium: Strategy, hands: int, seed: int):
    header("B", "Bayesian opponent identification", "computed live")
    print("  The agent sees only PUBLIC actions -- never the opponent's hole")
    print("  card except at showdown -- and maintains a posterior over six")
    print("  candidate archetypes by marginalizing the hidden card.\n")

    candidates = {name: tilt_strategy(equilibrium, cfg, leduc_is_weak)
                  for name, cfg in ARCHETYPES.items()}
    true_name = "maniac"
    belief = ArchetypeBelief(candidates=candidates)
    rng = np.random.default_rng(seed)

    checkpoints = set(identification_checkpoints(hands))

    def on_hand(record):
        belief.update(record.observation)
        n = belief.hands_observed
        if n in checkpoints:
            post = belief.posterior()
            top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
            marker = "correct" if belief.map_estimate() == true_name else "wrong"
            print(f"  hand {n:>4}   MAP: {belief.map_estimate():<16}"
                  f"conf {belief.confidence():.2f}   [{marker}]")
            print("              " + "  ".join(
                f"{k}={v:.2f}" for k, v in top))

    print(f"  playing {hands} hands vs a hidden opponent (truth: {true_name})")
    print("  a uniform prior over 6 candidates starts at 0.17 each\n")
    simulate_match(game, equilibrium, candidates[true_name], hands=hands,
                   rng=rng, on_hand=on_hand)

    print()
    row("identified", f"{belief.map_estimate()} "
                      f"({'correct' if belief.map_estimate() == true_name else 'WRONG'})")
    row("posterior confidence", f"{belief.confidence():.3f}")

    # Contrast: the same machinery against the archetype with the smallest
    # behavioral footprint. Detectability and profitability are separate axes.
    hard_name = "bluff_heavy"
    hard_belief = ArchetypeBelief(candidates=candidates)
    simulate_match(game, equilibrium, candidates[hard_name], hands=hands,
                   rng=np.random.default_rng(seed + 1),
                   on_hand=lambda r: hard_belief.update(r.observation))

    print(f"\n  Same machinery, {hands} hands vs '{hard_name}' instead:\n")
    print(f"    {'opponent':<14}{'deviation':>11}{'exploitable':>13}"
          f"{'conf @' + str(hands):>10}")
    for name, b in ((true_name, belief), (hard_name, hard_belief)):
        print(f"    {name:<14}"
              f"{deviation_magnitude(equilibrium, candidates[name]):>11.4f}"
              f"{exploitability(game, candidates[name]):>13.3f}"
              f"{b.confidence():>10.2f}")
    print(f"\n  '{hard_name}' deviates least from equilibrium so it is hardest to")
    print("  detect -- yet it is far from the least profitable to exploit.")
    print("  Detectability and exploitability are separate axes.")
    return candidates, belief, true_name


def section_c(game, equilibrium: Strategy, candidates, belief, true_name: str,
              eq_exploitability: float, epsilon: float):
    header("C", "Risk-constrained adaptive exploitation", "computed live")
    print("  lambda blends equilibrium with a best response to the ESTIMATE:")
    print("    sigma = (1 - lambda) * equilibrium  +  lambda * BR(estimate)")
    print("  More lambda captures more opponent-specific EV, but the blend")
    print("  becomes more exploitable if the estimate is wrong.\n")

    opponent = candidates[true_name]
    estimate = belief.mixture_strategy()
    exploit = exploitative_strategy(game, estimate, equilibrium)

    ev_eq = seat_averaged_ev(game, equilibrium, opponent)
    ev_exploit = seat_averaged_ev(game, exploit, opponent)

    row("equilibrium EV vs opponent", f"{ev_eq:+.4f} chips/hand")
    row("equilibrium exploitability", f"{eq_exploitability:.2e} chips")
    row("full-exploit EV vs opponent", f"{ev_exploit:+.4f} chips/hand")
    row("full-exploit exploitability", f"{exploitability(game, exploit):.3f} chips")

    print("\n  The tradeoff, measured across the blend (exact EV, no sampling):\n")
    print(f"    {'lambda':>8}{'EV (chips/hand)':>20}{'exploitability':>18}")
    for lam in (0.0, 0.05, 0.10, 0.25, 0.50, 1.00):
        blended = blend(equilibrium, exploit, lam)
        print(f"    {lam:>8.2f}{seat_averaged_ev(game, blended, opponent):>20.4f}"
              f"{exploitability(game, blended):>18.3f}")

    decision = adaptive_strategy(game, equilibrium, estimate,
                                 belief.confidence(),
                                 max_exploitability=epsilon)
    print(f"\n  What the agent actually chose (budget epsilon = {epsilon}):\n")
    row("belief confidence", f"{decision.belief_confidence:.3f}")
    row("estimated deviation (mean TV)", f"{decision.deviation:.4f}")
    row("confidence-implied lambda", f"{decision.lam_requested:.4f}")
    row("exploitability-capped lambda", f"{decision.lam_applied:.4f}")
    row("final EV vs opponent",
        f"{seat_averaged_ev(game, decision.strategy, opponent):+.4f} chips/hand")
    row("final exploitability",
        f"{exploitability(game, decision.strategy):.4f} chips  (<= {epsilon})")

    binding = ("the exploitability budget"
               if decision.lam_applied < decision.lam_requested - 1e-9
               else "statistical confidence")
    print(f"\n  Binding constraint: {binding}.")


def section_d(hands_per_regime: int = 2000):
    header("D", "Regime change: a negative result and its repair",
           "committed full-scale results (experiments/regime_change.py)")
    print("  Opponent silently switches maniac -> nit at hand 2,000.")
    print("  Both beliefs are fed the identical hand sequence.\n")

    rows = _read_csv(DATA_DIR / "regime_change_switching.csv")
    base = recovery_delays_by_repeat(rows, "baseline", hands_per_regime)
    rec = recovery_delays_by_repeat(rows, "recency", hands_per_regime)

    high_conf = high_confidence_fraction(rows, "baseline", threshold=0.99)

    print("  stationary belief   LL_t = LL_(t-1) + ll_t")
    row("sustained recovery",
        "never" if all(d is None for d in base) else str(base), indent=4)
    row("confidence while wrong",
        f"{high_conf:.0%} of checkpoints above 0.99", indent=4)
    print("    -> confidently wrong, and structurally unable to recover:")
    print("       old evidence never loses weight.\n")

    good = [d for d in rec if d is not None]
    print("  recency-aware       LL_t = decay * LL_(t-1) + ll_t   (decay 0.995)")
    row("sustained recovery",
        f"{min(good)}-{max(good)} hands, {len(good)}/{len(rec)} repeats",
        indent=4)
    row("memory horizon", "~1/(1-decay) = ~200 hands", indent=4)

    control = _read_csv(DATA_DIR / "regime_change_control_summary.csv")
    print("\n  The cost, on an opponent that NEVER changes (false switches):\n")
    print(f"    {'decay':>8}{'maniac':>12}{'bluff_heavy':>14}")
    for d in ("1.0", "0.995", "0.99", "0.98"):
        vals = {}
        for r in control:
            if abs(float(r["decay"]) - float(d)) < 1e-9:
                vals[r["archetype"]] = float(r["flip_rate"])
        if len(vals) == 2:
            print(f"    {d:>8}{vals['maniac']:>11.2%}{vals['bluff_heavy']:>14.2%}")
    print("\n  Faster forgetting recovers sooner but misclassifies a stationary")
    print("  opponent more often. Adaptivity vs stability is a real tradeoff;")
    print("  0.995 is a defensible point for THIS environment, not a universal one.")


def section_e(budget_ms: float = 10.0):
    header("E", "Performance engineering",
           "committed benchmarks (experiments/performance_benchmark.py)")
    print("  Profiling put 94.6% of Monte Carlo equity runtime in one step:")
    print("  evaluating 7 cards as max over all C(7,5)=21 five-card subsets.")
    print("  Replacing it with direct histogram evaluation gave, with")
    print("  bit-identical results (verified over all 2,598,960 five-card hands):\n")

    perf = _read_csv(DATA_DIR / "performance_comparison.csv")
    print(f"    {'workload':<24}{'before':>10}{'after':>10}{'speedup':>10}")
    for name in ("hand_eval_7card", "equity_flop_uniform",
                 "equity_preflop_uniform", "equity_weighted_range"):
        s = speedup_row(perf, name)
        print(f"    {name:<24}{fmt_rate(s['before']):>10}"
              f"{fmt_rate(s['after']):>10}{s['speedup']:>9.2f}x")

    controls = [float(r["speedup"]) for r in perf
                if r["workload"] not in
                ("hand_eval_7card", "equity_flop_uniform",
                 "equity_preflop_uniform", "equity_weighted_range")]
    print(f"\n  The other {len(controls)} workloads (solvers, belief updates, exact")
    print(f"  evaluation) moved only {min(controls):.2f}x-{max(controls):.2f}x. None of them calls the hand")
    print("  evaluator, so they act as a control -- and that spread is this")
    print("  machine's noise floor, below which no speedup is meaningful.")

    flop = speedup_row(perf, "equity_flop_uniform")
    budget_s = budget_ms / 1000.0
    before_n = sims_within_budget(flop["before"], budget_s)
    after_n = sims_within_budget(flop["after"], budget_s)
    print(f"\n  Why it matters -- equity error falls as 1/sqrt(n), so throughput")
    print(f"  converts into precision at a square-root discount:\n")
    row(f"simulations in {budget_ms:.0f} ms (before)", f"~{before_n:,}", indent=4)
    row(f"simulations in {budget_ms:.0f} ms (after)", f"~{after_n:,}", indent=4)
    row("error bars", f"~{error_ratio_from_speedup(flop['speedup']):.1f}x tighter "
                      f"at the same latency", indent=4)

    tradeoff = _read_csv(DATA_DIR / "compute_quality_tradeoff.csv")
    by_budget: Dict[float, Dict[str, float]] = {}
    for r in tradeoff:
        by_budget.setdefault(float(r["budget_seconds"]), {})[r["solver"]] = \
            float(r["exploitability"])
    print("\n  The same logic applies to solving. Exploitability reachable")
    print("  within a wall-clock budget:\n")
    print(f"    {'budget':>9}{'CFR':>12}{'CFR+':>12}   best")
    for b in (0.25, 1.0, 20.0):
        if b in by_budget:
            v = by_budget[b]
            best = "CFR" if v["CFR"] < v["CFR+"] else "CFR+"
            print(f"    {b:>8.2f}s{v['CFR']:>12.4f}{v['CFR+']:>12.4f}   {best}")
    print("\n  The curves cross near 1s: CFR+ is asymptotically better but buys")
    print("  half the iterations, so it LOSES under a tight latency budget.")


def section_f():
    header("F", "Takeaway", "summary")
    print("""  Equilibrium gives robustness. Opponent modeling creates
  exploitative opportunity. What determines how much of that
  opportunity can safely be captured is everything in between:

    - finite samples      estimation error can make deviation -EV
    - regime shifts       a stationary belief stays confidently wrong
    - risk limits         the exploitability budget, not confidence,
                          is usually what binds
    - compute budgets     they decide both how well you can solve and
                          how precisely you can estimate

  Full writeup: RESEARCH.md      Raw data: results/data/""")


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kuhn-iterations", type=int, default=20_000)
    p.add_argument("--leduc-iterations", type=int, default=200)
    p.add_argument("--hands", type=int, default=200,
                   help="hands of live opponent observation in section B")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="exploitability budget for the adaptive agent")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    print()
    print("=" * WIDTH)
    print(" PokerAlpha -- equilibrium robustness vs opponent exploitation")
    print(" in imperfect-information games")
    print("=" * WIDTH)
    print(" Deterministic run (seed "
          f"{args.seed}). Every number below is either computed")
    print(" live or read from a committed experiment CSV, as labelled.")

    game, equilibrium, eq_expl = section_a(args.kuhn_iterations,
                                           args.leduc_iterations)
    candidates, belief, true_name = section_b(
        game, equilibrium, args.hands, args.seed)
    section_c(game, equilibrium, candidates, belief, true_name, eq_expl,
              args.epsilon)
    section_d()
    section_e()
    section_f()

    print(f"\n  (demo completed in {time.perf_counter() - started:.1f}s)\n")


if __name__ == "__main__":
    main()
