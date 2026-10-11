# HiddenInfoBench (Phase II, Program 12)

`python -m poker_alpha.bench` runs eleven tasks and writes one
`pokeralpha.bench/v1` document. There is **no aggregate score**: the
metrics (exploitability in different games, throughput, calibration
rates, parity booleans) are not commensurable.

```bash
python -m poker_alpha.bench --list
python -m poker_alpha.bench --quick                    # about 1 min
python -m poker_alpha.bench --out bench.json           # full
```

| task | game | metric | expected invariant |
|---|---|---|---|
| kuhn_cfr_plus | Kuhn | exploitability | < 2e-3 at 1,000 iterations |
| leduc_cfr_plus | Leduc | exploitability | < 0.01 at 300 iterations |
| cfr_family_leduc | Leduc | exploitability, 5 algorithms | CFR+ and DCFR beat CFR |
| auction_equilibrium | auction duel | exploitability + bid monotonicity | < 0.01 and monotone |
| agent_tournament | Kuhn | seat-swapped utilities | equilibrium not beaten beyond noise |
| strategy_artifact | HUNL release | integrity, normalization | 144,025 infosets |
| native_parity | HUNL | shared-tape iterate values | bit-identical |
| native_throughput | HUNL | it/s | > 100 |
| confidence_calibration | 9 exact games | false-confidence rate | < 5 % and below 1/3 of base |
| abstraction_refinement | river subgame | exploitability | raw ≤ adaptive < baseline |
| river_solver_parity | river subgame | vectorized vs generic | < 1e-10 |

Each task records its game, algorithm, seeds, budget, metric and expected
invariant. Hardware is recorded without user or host names: OS, arch, CPU
model, Python, numpy, compiler and native backend version. Optional
components (native backend, test helpers) are reported as `skipped` when
absent, never as passed.
