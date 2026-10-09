# Native solver — design (Phase 2)

Goal: a C++17 training backend for release-v2 Hold'em external-sampling MCCFR
that preserves the Python reference semantics and crosses the Python/C++
boundary only at coarse granularity.

## Boundary

**Native (C++):** card codes, hand evaluator, Hold'em state + rules, chance
dealing, card features (made class, draws, nuts, blockers, texture, river
percentile), compact-v2 encoding, infoset table, regret matching, action
sampling, the full MCCFR traversal, the training loop, RNG, random-tape mode,
trace mode.

**Python:** configuration (`HoldemSolverConfig` stays the single source of
truth), file formats (strategy artifact + native checkpoint `.npz` written by
Python from arrays the native solver hands over in one call), analysis,
DecisionEngine, UI, observer, experiments, CI glue.

**Crossings per `train(n)` call: 1** (plus one per progress/checkpoint
interval, configurable). The GIL is released for the whole chunk.

## Python API

```python
from poker_alpha.native import native_available, NativeMCCFRSolver

solver = NativeMCCFRSolver(RELEASE_CONFIG, seed=0)   # validates config support
solver.train(100_000, progress_every=10_000, progress=callback)
solver.iterations          # int
solver.metrics()           # dict: infosets, nodes visited, approx bytes, ...
solver.average_strategy()  # {key_str: {action: prob}} (export-sized, not hot)
solver.export_strategy(path, meta=...)   # standard strategy artifact (Python writes)
solver.save_checkpoint(path)             # native checkpoint .npz (Python writes)
NativeMCCFRSolver.load_checkpoint(path, config)      # exact resume
```

The pybind module `poker_alpha._native` exposes a lower-level
`NativeSolverCore`; `poker_alpha/native/backend.py` wraps it with config
plumbing, file I/O, and signature checks. If the extension is missing,
`native_available()` is False and nothing else in PokerAlpha changes.

## Config transport

Python passes a flat struct: stack, blinds, raise cap, ordered postflop
fractions (name, value), ordered preflop multiples (name, value),
enforce_min_raise, river percentile buckets, encoder options, plus the three
signature strings (config/game/encoder) which native stores verbatim for
checkpoint stamping. Native validates it supports the combination (compact
encoder, abstract history, texture, river blockers) and raises otherwise —
no silent fallback. v1 configs (no preflop multiples / no min-raise) are also
supported by the same code paths.

## State

POD `HoldemState` (~160 bytes), passed by value in the traversal:

* `holes[2][2]`, `board[5]`, `board_n` (uint8)
* `street` (uint8), `to_act` (uint8), `folded` (int8), `all_in` (bool)
* `total[2]`, `street_paid[2]` (double) — maintained incrementally by the
  same arithmetic sequence as the Python replay fold (bit-equal)
* `n_raises` (uint8), `min_inc` (double)
* betting-context fields: `facing_cls[2]`, `own_prior[2]`,
  `street_last_aggr`, `last_aggr_prev` (int8), `spr_bucket` (uint8, computed
  at street start)
* action history for debug/trace: `hist[24]` tokens + per-street offsets
  (kept always; 26 bytes, enables canonical reconstruction of the Python
  `streets` strings)

## Actions

`enum class ActionId : uint8_t { Fold, CheckCall, AllIn, B33, B75, B150,
X200, X250, X350 }` — generated from the config at solver construction, not
hardcoded: the config's ordered token lists define the mapping
`ActionId <-> token string`, and the legal-action generator emits them in
exactly the Python order (`f?, c, sized..., a`). `MAX_ACTIONS = 6`
(f + c + 3 sized + a), asserted against the config at construction.

## Infoset key

Packed `uint64` of the exact compact-v2 information partition (field layout
in `docs/native_solver_architecture_audit.md` §6). Properties:

* bijective packing of enumerated fields → no collisions by construction;
* `render_key(key) -> std::string` reproduces the canonical Python key
  byte-for-byte (used for export, checkpoints, debugging, parity tests);
* `parse_key(str) -> uint64` for importing Python-written checkpoints.

Parity tests assert `render(native_key(state)) == python_infoset_key(state)`
over large random corpora — which simultaneously proves no extra information
(same string ⇒ same key) and no lost information (different string ⇒
different key), because the map string↔u64 is bijective.

## Infoset storage

```cpp
struct Node {
    std::array<double, MAX_ACTIONS> regret_sum{};
    std::array<double, MAX_ACTIONS> strategy_sum{};
    uint8_t num_actions;
    std::array<ActionId, MAX_ACTIONS> actions{};
};
std::unordered_map<uint64_t, Node> table;   // reserve() on construction
```

`std::unordered_map` first; alternatives only if profiling demands (Phase 39).
Node references are **never held across recursive calls that may insert**
(rehash invalidates references): the traversal re-looks-up or indexes by key
after children return. Phase 76 forces rehashes under ASan to verify.

## Evaluator

Histogram evaluator transcribed from `evaluate_best_codes`, returning a
packed `uint32` (`category << 20 | tiebreakers`, 4 bits each) that preserves
the Python tuple ordering exactly. Validated exhaustively on all 2,598,960
five-card hands and on ≥1M random 7-card hands against Python.

## Card features

Direct ports of `made_hand_class`, `draw_type`, `_flush_blockers`,
`texture_code`, `_is_current_nuts` (exact enumeration), `river_percentile`
(per-board cache with the same rank-pair dedup). Features cached per
`(sorted hole, sorted board)` in a native hash map, mirroring Python's
`lru_cache` semantics (unbounded within a training run; measured in the
memory audit).

## MCCFR

Transcription of `MCCFRSolver._traverse`: terminal → signed utility; chance
→ sample one successor; opponent node → accumulate `strategy_sum += σ`,
sample one action from the cdf of σ (cumsum ÷ last, upper_bound — same
semantics as Python's searchsorted right); own node → recurse all actions,
`node_value = Σ σ_i v_i` computed by **Neumaier-compensated summation in
index order** (Python uses `math.fsum`, which is exactly rounded; for ≤6
terms the compensated sum is bitwise-equal to fsum in all tested cases and
the parity suite verifies trajectories end-to-end), `regret += v − node_value`.
No fast-math; no reassociation.

## RNG

* **Production:** xoshiro256** seeded via splitmix64 from the user seed
  (public-domain reference implementation, transcribed). Uniform doubles via
  `(x >> 11) * 2^-53`; bounded ints via rejection sampling (unbiased,
  deterministic). State = 4×uint64, serialized in checkpoints. Documented
  version `native_rng=xoshiro256**/v1`.
* **Random tape (testing only):** both backends consume a shared pre-generated
  tape. Tape entries are uniform doubles; dealing uses
  `idx = floor(u * remaining)` over the sorted live-card list with removal,
  opponent sampling uses the cdf rule above. A Python tape game/solver
  subclass (test helper) implements the identical consumption, giving exact
  cross-backend trajectory identity without reimplementing numpy.
* Cross-backend bit identity of *production* streams: **not claimed**.

## Checkpoints

Native checkpoint = `.npz` written by Python (arrays come from one pybind
call): format `pokeralpha.native_checkpoint/v1`, config/game/encoder
signatures, backend + RNG version, iterations, sorted rendered keys, action
tokens, float64 regret/strategy sums, RNG state, content SHA-256. Load
validates everything and refuses mismatches. Resume is exact:
`train(a); save; load; train(b) == train(a+b)` bitwise.

Strategy export reuses `strategy_artifact.export_solver`'s format via a
shim object, so artifacts are byte-compatible with the existing loader and
DecisionEngine; metadata additionally records `backend=native`, RNG type,
and backend schema version.

## Interruptibility / progress

`train(n)` runs in chunks of `min(progress_every, 2000)` iterations inside
C++; between chunks it re-acquires nothing but checks
`PyErr_CheckSignals()` via a pybind `gil_scoped_acquire` only every chunk —
Ctrl+C aborts cleanly at a chunk boundary, raising `KeyboardInterrupt` with
the solver still in a consistent state (iterations = completed chunks).

## Fallback and selection

`backend="python" | "native" | "auto"` in training scripts; auto = native if
importable and the config is supported, else Python, always printing the
active backend. `MCCFRSolver` (Python) is untouched and remains the
reference. DecisionEngine never needs the extension.

## Build

C++17, pybind11, scikit-build-core + CMake, all as Python build
dependencies (`pip install -e ".[native]"` or building the `poker-alpha-native`
component — see Phase 3 doc/README). `-O3 -DNDEBUG -Wall -Wextra -Wpedantic`;
no `-ffast-math`; optional ASan/UBSan presets for development.
