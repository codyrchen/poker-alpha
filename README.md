# PokerAlpha

Game-theoretic poker solver and adaptive exploitation engine — a quantitative
research project studying the tradeoff between **equilibrium robustness** and
**opponent-specific exploitation** in imperfect-information games.

> **Core research question:** How much equilibrium robustness should a poker
> agent sacrifice in order to exploit statistically detected weaknesses in an
> opponent?

This is an offline simulation and research project. It does not connect to,
automate, or interact with any real poker platform.

## Status

Phase 1 of the [development plan](TODO.md) is complete and verified:

- **Kuhn Poker** implemented as an extensive-form game (states, chance nodes,
  information sets, utilities).
- **Leduc Poker** — public card, two betting rounds with raises and a raise
  cap, pot-based fold/showdown utilities; converges to the known game value.
- **Vanilla CFR** with regret matching, cumulative/average strategies.
- **CFR+** with regret clipping, alternating updates, and linear averaging.
- **External-sampling MCCFR** with seeded, deterministic sampling.
- **Exploitability evaluation** via a true imperfect-information best response.
- **Convergence experiments** with reproducible results.

Current measured results (`python experiments/kuhn_convergence.py --iterations 100000 --seed 42`):

| quantity | measured | theory |
| --- | --- | --- |
| game value (player 0) | −0.055554 | −1/18 ≈ −0.055556 |
| exploitability after 100k iterations | 6.3 × 10⁻⁴ chips | → 0 at Nash |
| King facing bet: call | 100% | 100% (dominant) |
| Jack facing bet: fold | 100% | 100% (dominant) |
| Queen facing bet: call | 0.336 | 1/3 |

![CFR convergence on Kuhn Poker](results/figures/kuhn_convergence.png)

Solver comparison on Kuhn
(`python experiments/cfr_comparison.py --iterations 100000 --seed 42`),
measured on the same machine, exploitability in chips:

| algorithm | game | iterations | runtime | it/s | final exploitability |
| --- | --- | --- | --- | --- | --- |
| CFR | Kuhn | 100k | 27.9 s | 3,587 | 6.3 × 10⁻⁴ |
| CFR+ | Kuhn | 100k | 53.6 s | 1,865 | 2.1 × 10⁻⁶ |
| MCCFR (external sampling) | Kuhn | 100k | 12.3 s | 8,143 | 2.3 × 10⁻³ |

Two readings worth internalizing:

- **CFR+ converges near O(1/T)** versus CFR's O(1/√T) — ~300× lower
  exploitability here — at ~1.9× cost per iteration (an iteration is two
  alternating traversals, one per player).
- **MCCFR iterations are 2.3× cheaper than CFR's even on Kuhn**, but its
  sampled convergence is noisier, and on a ~50-node tree cheap iterations
  can't win: CFR+ dominates per unit wall-clock. Sampling pays off as the game
  grows — that scaling is what the Leduc experiment measures.

![CFR vs CFR+ vs MCCFR convergence](results/figures/cfr_comparison.png)
![CFR vs CFR+ vs MCCFR by wall-clock](results/figures/cfr_comparison_time.png)

An implementation note worth knowing for interviews: CFR+'s regret clip must be
applied to an information set's **total** regret per iteration, not per visited
history. An information set spans several histories (in Kuhn, "hold the Jack"
spans two opponent cards), and clipping between visits biases the update — with
that bug, CFR+ degraded to CFR's O(1/√T) rate; buffering the per-iteration
deltas and clipping once restored O(1/T). The commit history preserves both
measurements.

Upcoming phases (see [TODO.md](TODO.md)): CFR+, MCCFR, Leduc poker, a hand
evaluator and Monte Carlo equity engine, an abstracted heads-up Hold'em,
Bayesian opponent modeling, the equilibrium–exploitation tradeoff experiments,
and risk analytics.

## The math so far (no CFR background assumed)

**Extensive-form games & information sets.** Poker is a sequential game where
players cannot see each other's cards. All game states that look identical to
the acting player (same own cards, same public history) form an *information
set*; a strategy must choose the same action distribution everywhere inside one
— that is what "not seeing the opponent's cards" means formally.

**Regret.** After playing, compare what you earned with what you *would* have
earned by always picking some action `a` at an information set. That difference
is the regret for `a`. *Counterfactual* regret weights this by the probability
that the other players and chance even let you reach that information set.

**Regret matching.** Next iteration, play each action with probability
proportional to its accumulated *positive* regret:

```
σ(a) = max(R(a), 0) / Σ_b max(R(b), 0)      (uniform if all regrets ≤ 0)
```

**Why CFR converges.** Regret matching guarantees average regret grows like
O(√T), so average regret per iteration → 0. In two-player zero-sum games, a
profile in which neither player has positive average regret is a Nash
equilibrium — and the guarantee applies to the **time-averaged** strategy, which
is why the solver tracks cumulative strategy sums and reports the average (the
current iterate oscillates; the average converges).

**Exploitability** measures the distance from equilibrium: how much a
best-responding adversary could win against your strategy, averaged over both
seats. Zero exactly at Nash. Our best response respects information sets (it
does not peek at hidden cards), computed by resolving information sets
deepest-first with counterfactual reach weights.

## Running

```bash
pip install -e ".[dev]"     # or: pip install -r requirements.txt
pytest                       # 17 tests
python experiments/kuhn_convergence.py --iterations 100000 --seed 42
```

Experiments take command-line arguments (`--iterations`, `--seed`, `--outdir`)
and write raw CSV data to `results/data/` and figures to `results/figures/`.
All randomness is seeded for reproducibility.

## Repository structure

```
poker_alpha/
  games/      extensive-form game definitions (base interface, Kuhn; Leduc & Hold'em later)
  solvers/    CFR (+ CFR+/MCCFR later) and strategy evaluation (EV, best response, exploitability)
  poker/      card engine, hand evaluator, equity simulation (later phases)
  opponent/   archetypes, statistics, Bayesian modeling, exploitation (later phases)
  risk/       bankroll, Kelly, risk metrics (later phases)
  utils/      seeding, plotting
experiments/  runnable, parameterized experiments
tests/        pytest suite
results/      generated data and figures (never hand-entered)
```

## Design principles

- Solvers depend only on a minimal `Game` interface, so one CFR implementation
  trains on every game in the project.
- Math is kept visible: `regret_matching(regrets: np.ndarray)` is a function,
  not a framework.
- All reported numbers come from actual runs; nothing in this README or the
  research notes is hand-entered.
- Results carry uncertainty wherever sampling is involved (later phases add
  bootstrap confidence intervals for match results).

## License

MIT
