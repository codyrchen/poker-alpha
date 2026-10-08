# Release status (end of Phase 31)

Machine-readable: [`results/validation/final_platform_validation.json`](../results/validation/final_platform_validation.json)
(regenerate with `python experiments/final_validation.py ...`). Evidence for
each line is in [validation.md](validation.md) and the result files it
cites.

**What PokerAlpha is not:** a solver for no-limit Hold'em, a GTO or Nash
strategy for real Hold'em, a profitable bot, or a validated PokerNow
reader. The trained strategy comes from an **imperfect-recall** abstraction
and has **no equilibrium guarantee**; its exploitability is unknown.

## Component status

| status | component | evidence |
| --- | --- | --- |
| VALIDATED | Kuhn / Leduc CFR research core | exact exploitability; canonical digests pinned and unchanged through Phases 26-31 |
| VALIDATED | Hold'em rules engine (2-9 seats) | chip conservation, side / split pots, reopening rules under property tests |
| VALIDATED | hand evaluator and equity | evaluator checked on all 2,598,960 five-card hands; seeded equity with SE |
| VALIDATED | locked solver config, config-bound checkpoints and artifacts | signature over encoder, feature tables, bets, stack, raise cap, sampling, reference range; mismatches rejected; frozen regression corpus and canonical suite |
| VALIDATED | optimized MCCFR backend | 1.99x iterations/s (15.2 -> 30.1), bit-identical exact digests; no compiled backend |
| VALIDATED | unified pipeline, source priority, coded rejections, uncertainty breakdown | all four input sources reach `DecisionReport` through `pipeline.analyze` in tests |
| PARTIALLY VALIDATED | compact abstraction (primary) | only candidate passing all Phase-27 criteria (>= 5 visits: 50.8% vs 0.31% for bucket at 5k); imperfect recall (874 / 1,508 colliding keys) |
| PARTIALLY VALIDATED | decision engine, rollout and heuristic methods | closed-form invariants pass; EVs rest on heuristic response models and a check-down assumption |
| PARTIALLY VALIDATED | screen observer on synthetic images | synthetic fixtures only; read-only |
| EXPERIMENTAL | trained abstract HU strategy (`results/strategy/holdem_v1_seed0.npz`) | 3 seeds x 300k iterations; 10 / 10 sanity checks per seed; seeds even in cross-play; 300k beats 100k by ~84 bb/100 in the abstract game; canonical-spot policies still moving; exploitability not computed |
| EXPERIMENTAL | range and opponent modelling | Bayesian beliefs under heuristic priors; no ground-truth accuracy |
| BLOCKED | real PokerNow recognition accuracy | **BLOCKED ON REAL FIXTURES**: 0 annotated screenshots; nothing measured, nothing fabricated; see [observer.md](observer.md) |

## Performance (4-vCPU container)

| | value |
| --- | --- |
| offline: MCCFR training, locked config | ~23-30 it/s per process; 300k iterations ~3.5 h per seed; checkpoint ~28 MB |
| online: strategy artifact load | ~300 ms once |
| online: observe (manual / simulation / hand history) | < 1 ms |
| online: observe (synthetic screenshot, 3 frames) | ~240 ms median |
| online: analyze, solver or heuristic (1,500 equity sims) | ~70-100 ms median |
| online: analyze, 400 rollouts | ~150-230 ms median |

Online numbers are in `results/validation/final_benchmark.json`; solver
lookups are dictionary reads, and equity and rollout sampling dominate
online time.

## Readiness

| use | recommendation |
| --- | --- |
| research on abstraction, MCCFR behaviour and opponent modelling | **ready**: measured, reproducible, documented limits |
| post-hand analysis (hand histories, manual spots) | **ready with caveats**: every report shows its source, rejections and uncertainty; rollout EVs are model-dependent |
| live assistance in private / play-money / test games where permitted | **usable with caution**: read-only; screen recognition is unvalidated on the real client, so verify the recognized state; never in games that forbid assistance |
| trusted solver-quality recommendations | **not ready**: abstract imperfect-recall strategy, not converged, exploitability unknown |

## Unblocking next steps

1. Real PokerNow fixtures: annotated screenshots in `tests/fixtures/pokernow/`
   (steps in [observer.md](observer.md)); then run `experiments/observer_validation.py`.
2. An exploitability estimate inside the abstract game (e.g. local best
   response over the compact abstraction) before calling any strategy strong.
3. A perfect-recall abstraction coarse enough to be revisited (the
   transition encoder failed only because the exact betting history
   dominates), or more compute, if equilibrium guarantees matter.
