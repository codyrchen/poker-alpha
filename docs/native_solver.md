# Native MCCFR training backend

A C++17 implementation of the release-v2 Hold'em external-sampling MCCFR
trainer, exposed to Python through pybind11. It is **optional and training
only**: strategy lookup, the DecisionEngine, the observer and the UI never
need it, and the Python `MCCFRSolver` remains the reference implementation.

## Install

```bash
pip install -e .          # PokerAlpha, Python-only (unchanged)
pip install ./cpp         # adds the native backend (builds from source)
```

Build dependencies (`pybind11`, `scikit-build-core`, CMake, Ninja) are
Python build requirements resolved by pip; no Homebrew, no sudo. Apple
clang / GCC with C++17 is required. Verify with:

```bash
python -m poker_alpha.doctor        # "native MCCFR backend: OK ..."
python -c "from poker_alpha.native import native_available; print(native_available())"
```

## Use

```python
from poker_alpha.native import NativeMCCFRSolver, build_solver
from poker_alpha.solver_config import RELEASE_CONFIG

solver = NativeMCCFRSolver(RELEASE_CONFIG, seed=0)
solver.train(100_000)                      # GIL released; Ctrl+C safe
solver.save_checkpoint("ckpt.npz")         # exact-resume native checkpoint
solver.export_strategy("strategy.npz")     # standard strategy artifact

solver, backend = build_solver(RELEASE_CONFIG, seed=0, backend="auto")
```

Training scripts:

```bash
# one seed, with milestones, logs and checkpoints:
python experiments/holdem_mccfr_validation.py --v2-config --backend native \
    --seed 0 --milestones 100000,200000,300000 --ckpt-dir ckpts --out seed0.jsonl

# resume:
python experiments/holdem_mccfr_validation.py --v2-config --backend native \
    --seed 0 --milestones 400000 --resume ckpts/v2_native_seed0_it300000.npz \
    --ckpt-dir ckpts --out seed0.jsonl

# three seeds in parallel processes:
python experiments/train_native_multiseed.py --seeds 0,1,2 \
    --milestones 100000,200000,300000 --out-dir results/native_training

# validation / benchmark:
python experiments/validate_native_solver.py
python experiments/benchmark_native_solver.py
```

`--backend` is `python` (default — historical behavior and formats are
untouched), `native` (require the extension) or `auto`. The active backend
is always printed; nothing falls back silently.

## What is native

Hold'em rules, chance dealing, hand evaluation, the compact-v2 encoder
(including exact river percentiles, nuts detection, textures, blockers),
infoset storage (packed uint64 keys → fixed-size nodes), regret matching,
sampling, the full traversal and the training loop. Python crosses the
boundary once per `train()` chunk. See `docs/native_solver_design.md` and
`docs/native_solver_architecture_audit.md`.

## Correctness

* All 2,598,960 five-card hands + 1M random seven-card hands: evaluator
  parity with the Python reference.
* 20k+ random reachable states: identical mechanics, legal-action lists,
  bit-equal chip arithmetic, identical infoset-key strings.
* Shared random tape: bitwise-identical traces, values, regret/strategy
  sums and average strategies through 1,000 iterations.
* Exact checkpoint resume; corrupt/mismatched checkpoints rejected.
* ASan+UBSan: clean on the standalone self-test (`cpp/tests/native_selftest.cpp`),
  including forced hash-table rehashes during training.

Exact reduced-game convergence (Phases 25/26 of the original validation)
transfers to the native backend **transitively**: the native traversal is
bitwise-equal to the Python MCCFR on shared randomness, and the Python
MCCFR's convergence on exact reduced games is measured in
`docs/solver_validation.md` / `results/validation/reduced_holdem_v1.json`.
The backend intentionally implements only the locked Hold'em configs, not a
generic game interface.

## Reproducibility

* Production RNG: xoshiro256** (`xoshiro256**/v1`), splitmix64-seeded;
  state is 4×uint64, stored in checkpoints. Same seed + same binary
  semantics → bit-identical runs; `train(a)+save+load+train(b)` is
  bit-identical to `train(a+b)`.
* The native and NumPy RNG streams are **not** bit-identical; a Python
  checkpoint therefore cannot continue its exact random stream natively.
  (All historical resumable checkpoints are absent from the repo anyway —
  see `results/strategy/candidates/v2_extension/MANIFEST.json`.)
* Cross-platform (macOS arm64 / Linux x86_64) bit identity is not claimed;
  the compiler flags forbid FMA contraction and fast-math, which removes
  the known sources of divergence, and CI runs the same parity suite on
  Linux. Artifact metadata records the backend and RNG.

## Checkpoint format

`pokeralpha.native_checkpoint/v1` — uncompressed `.npz` written by Python
(`allow_pickle=False`): config/game/encoder signatures, backend schema,
RNG name + state, iterations, packed uint64 keys and their canonical
rendered strings, per-infoset action tokens, float64 regret/strategy sums,
content SHA-256. Loading validates everything; a truncated, bit-flipped,
or wrong-config file raises `NativeBackendError`. Writes are atomic
(temp + rename).

Strategy artifacts exported by the native solver use the standard
`pokeralpha.strategy_artifact/v2` format with extra provenance
(`backend=native`, backend schema/version, RNG) and load everywhere the
Python-trained ones do.

## Third-party code (license audit, Phase 93)

| component | origin | license |
|---|---|---|
| pybind11 | build dependency (header-only) | BSD-3-Clause |
| scikit-build-core, CMake, Ninja | build-time only, nothing shipped | Apache-2.0 / BSD |
| xoshiro256** + splitmix64 | algorithms by Blackman & Vigna, reimplemented in `cpp/src/rng.hpp` | public domain (CC0 reference) |
| exact summation | Shewchuk's expansion algorithm (as in CPython's `math.fsum`), reimplemented in `cpp/src/fsum.hpp` | algorithm; no code copied |

No third-party evaluator or hash-map library is used; everything else is
C++17 standard library. All compatible with the repository's MIT license.

## Limitations

* Supports the locked `HoldemSolverConfig` compact-encoder configs
  (`compact`, `compact_river_pct10/20`, v1 and v2 games) — not ad-hoc
  encoders, Kuhn/Leduc, or `history="exact"` variants.
* Training-only: no exploitability computation; everything in
  `docs/release_status.md` about abstraction error and experimental status
  applies unchanged to natively trained strategies.
