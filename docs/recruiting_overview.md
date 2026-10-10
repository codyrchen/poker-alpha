# PokerAlpha — technical overview (pre-interview briefing)

A 2-page orientation for a technical reviewer. Everything here is backed by
a committed result file; the authoritative status document is
[solver_status_final.md](solver_status_final.md).

## Problem

Make good decisions in a game you cannot fully observe and cannot fully
solve. Heads-up no-limit Hold'em has ~10¹⁶⁰ states; tractable solving
requires an abstraction, the practical abstraction sacrifices perfect
recall (and with it CFR's equilibrium guarantee), and the training
algorithm is stochastic. The project's central question evolved from
*"how do I compute a strategy?"* to *"how do I know, per decision, whether
to trust the strategy I computed?"*

## Algorithms

- **CFR / CFR+ / external-sampling MCCFR** over a minimal `Game` interface
  (Kuhn, Leduc, reduced Hold'em games, abstracted HU NLHE).
- **Exact evaluation** (best response, exploitability) on every game small
  enough to enumerate — the ground truth everything else is held against.
- **Restricted Local Best Response**: a validated lower-bound estimator of
  full-game exploitability (range tracking through the strategy, myopic
  value scoring), checked against exact best response on reduced games
  (Spearman 0.986, no upper violations).
- **Confidence/abstention**: a per-infoset gate on visits, cross-seed
  disagreement and recent policy movement, calibrated against *exact
  EV regret* on nine exactly solved games with leave-one-game-out
  evaluation.

## Architecture

- `poker_alpha/` — Python library: games, solvers, card engine, ranges and
  opponent models, decision engine, optional read-only screen observer,
  Streamlit UI.
- `cpp/` — C++17 native MCCFR backend (pybind11 + scikit-build-core). The
  entire hot path is native (rules, dealing, evaluator, encoder, regret
  matching, traversal); Python crosses the boundary once per training
  chunk with the GIL released and owns configuration, file formats and
  analysis. The Python solver remains the reference implementation.
- Artifacts: versioned, checksummed strategy + confidence files; the
  confidence table SHA-binds to its strategy so mismatched pairs refuse to
  load.

## Hardest engineering decisions

1. **Bitwise cross-language equivalence as the correctness contract.** The
   C++ backend is proven identical to Python on a shared random tape —
   which required exactly-rounded summation (an fsum port), identical cdf
   sampling semantics, and disabling FMA contraction after the parity
   suite caught a last-ulp divergence in betting arithmetic.
2. **Packed 64-bit infoset keys** that render byte-identically to the
   Python encoder's canonical strings — including faithfully reproducing a
   quirk where the Python key *merges* two semantically distinct fields,
   because the released artifacts were trained under that partition.
3. **Validating the fast solver on exact games without rewriting them in
   C++**: a tabular-tree adapter flattens any enumerable Python game into
   arrays (payoffs and keys produced by the Python game itself), so native
   MCCFR is measured directly against exact CFR+ with zero rule
   duplication.
4. **Replacing a plausible-but-wrong trust signal.** The v1 confidence
   gate's "policy movement since an early checkpoint" turned out to
   measure training distance, not instability — it rejected 71% of a
   mature strategy's infosets while correlating *less* with disagreement
   as training progressed. The fix was evidence-first: build a dataset
   with exact EV-regret targets, show movement carries no signal at any
   horizon, change only the movement definition, and verify held-out.

## Measured results

| result | number | source |
| --- | --- | --- |
| native training speedup | 38.1× warm (837 vs 22 it/s) | `results/benchmarks/native_mccfr_v1.json` |
| cross-language parity | bitwise over 1,000 iterations | `tests/test_native_mccfr_parity.py` |
| evaluator | all 2,598,960 five-card hands exact | `tests/test_native_evaluator.py` |
| direct exact validation | exploitability ratio 1.04 vs Python on 7 games | `results/validation/native_direct_exact_validation.json` |
| release training | 3 seeds × 2M iterations | `results/validation/native_training_2m.json` |
| confidence v2 | 1.7× coverage at equal accepted EV regret (held-out) | `results/validation/confidence_signal_quality.json` |
| release selection | +50..+117 bb/100 vs previous release in duplicate cross-play | `results/validation/final_release_comparison.json` |
| exploitability floor | restricted-LBR ≥ ~100 bb/100 for every candidate | `results/validation/lbr_holdem.json` |

## Validation and limitations

Validated: rules, evaluator, solver math (exact games), native/Python
equivalence, checkpoint/artifact integrity, confidence calibration
(held-out). Measured-with-caveats: flop/turn abstraction error (3.7/5.1 bb
median on exact subgames → those streets are capped at low confidence),
rollout fallback biases. Experimental: the trained strategy itself —
imperfect recall means no equilibrium claim, and the LBR floor is stated
up front. Blocked: real-screen recognition accuracy awaits independent
annotated frames. The observer is read-only by design.

## Five strong discussion topics

1. Why external-sampling MCCFR for Hold'em, and what the random-tape
   parity harness buys over statistical testing.
2. The FMA/`-ffp-contract` incident: reproducibility vs compiler
   optimization, and why the parity suite was designed to catch it.
3. Policy L1 vs decision EV: why ~94% of "alarming" preflop seed
   disagreement is harmless, how that was measured, and what the 7
   consequential jam-defense states have in common.
4. Designing an abstention rule you can falsify: exact-game regret
   targets, leave-one-game-out, the risk-coverage frame, and the maturity
   extrapolation caveat.
5. Where the next real gain is: the measured abstraction floor (LBR +
   exact subgames) says better buckets, not more iterations.
