# Native solver — Phase 0 architecture audit

Audit of the Python release-v2 Hold'em MCCFR implementation before the C++
port. Everything below was verified against the repository at commit
`b398c4e` (branch `claude/live-observer`); nothing is assumed from memory.

## 1. What is being ported

The release training stack is:

```
HoldemSolverConfig (V2_CONFIG = RELEASE_CONFIG)        poker_alpha/solver_config.py
  encoder            = "compact_river_pct20"
  bet_fractions      = b33=0.33, b75=0.75, b150=1.5    (postflop pot fractions)
  preflop_raise_multiples = x200=2.0, x250=2.5, x350=3.5 (raise-to multiples)
  enforce_min_raise  = True (NLHE minimum bet/raise)
  starting_stack     = 100.0 BB, blinds 0.5 / 1.0
  raise_cap          = 3 per street
  sampling           = external-sampling-mccfr, averaging = uniform
  signature          = HoldemSolverConfig:v2:733e52f1d1014e2e7973

MCCFRSolver(game, seed)                                 poker_alpha/solvers/mccfr.py
  iterate() = _traverse(root, 0); _traverse(root, 1); iterations += 1
```

## 2. Python call graph of one traversal

```
MCCFRSolver._traverse(state, update_player)
├── game.is_terminal(state)            HoldemGame
│     └── _betting_closed(state)       -> _tokens(street string)  [lru_cache]
├── game.utility(state)                terminal only
│     └── evaluate_best(7 cards) x2    poker/evaluator.py (showdown only)
├── game.is_chance(state)
│     └── _betting_closed(state)
├── game.sample_chance(state, rng)     numpy Generator.choice without replacement
├── game.current_player(state)
│     └── _replay(state)               memoized full betting replay by `streets`
├── game.infoset_key(state)
│     └── CompactHoldemEncoder.encode  abstraction/holdem_v2.py
│           ├── card_features(hole, board)      features.py [lru_cache 1M]
│           │     ├── made_hand_class -> evaluate_best_codes
│           │     ├── draw_type, _flush_blockers, texture_code
│           │     ├── _is_current_nuts (strength>=5 only; ~1,081 evals, cached)
│           │     └── river_percentile (river only; cached per board)
│           └── betting_context(game, state)    betting_history.py [memoized]
│                 └── full token replay + game.legal_actions(state)
├── game.legal_actions(state)          memoized by `streets`
│     ├── _replay(state)
│     └── _min_increment(streets)      memoized full replay
├── game.next_state(state, action)
│     └── _replay(state) + dataclass copy
├── InfoSet.current_strategy()         regret_matching (numpy)
├── MCCFRSolver._sample(strategy)      cumsum + searchsorted + rng.random()
└── strategy_dot(strategy, values)     math.fsum of elementwise product
```

Measured on the release config (see `results/benchmarks/mccfr_python_baseline.json`):
roughly **327 decision nodes, 240 terminals, 120 chance nodes per iteration**
(2 traversals), each decision node performing an infoset-key encode, a
legal-actions lookup, and (updating player) one `next_state` per action.

## 3. State representation (Python)

`HoldemState` — frozen dataclass (`games/holdem.py`):

| field     | type                                   | notes |
|-----------|----------------------------------------|-------|
| `holes`   | `((int,int),(int,int))` or `None`      | `None` = pre-deal root chance node |
| `board`   | `Tuple[int, ...]` (0/3/4/5 cards)      | card codes 0..51, deal order kept |
| `streets` | `Tuple[str, ...]`                      | per-street action strings, e.g. `("cb200c","b33c","","")` |
| `contrib` | `(float, float)`                       | cumulative chips per player (cache; authoritative source is replay) |
| `folded`  | int (−1 none)                          | |
| `all_in`  | bool                                   | set when any `a` is played |

Everything else (street_paid, totals, player to act, raise count, min
increment) is **derived by replaying `streets` from the blinds**, memoized per
`streets` tuple. The replay is a left fold over action tokens; the native port
can maintain the same fold state incrementally — the arithmetic sequence is
identical, so doubles stay bit-equal.

Card codes: `rank = code % 13` (0=2 .. 12=A), `suit = code // 13` (c,d,h,s).

## 4. Game mechanics that must be preserved exactly

* Blinds: SB=0.5 (player 0, button, acts first preflop), BB=1.0 (player 1,
  acts first postflop).
* Tokens: `f`, `c`, `a`, postflop `b33|b75|b150`, preflop `x200|x250|x350`.
* `raise_add` (`games/holdem.py`):
  * `x` tokens: `level = my_street_paid + owe; add = mult*level − my_street_paid`
  * `b` tokens: `add = owe + frac*(pot_now + owe)` (pot-after-call fraction)
* Legal actions (`_legal_uncached`): `f` iff `owe > 1e-9`; always `c`; raise
  menu iff `n_raises < raise_cap and my_stack > owe + 1e-9 and opp_stack >
  1e-9`; each sized raise offered iff `add − owe >= min_inc − 1e-9` (NLHE
  minimum) **and** `add < my_stack − 1e-9` (strictly; `a` covers the top);
  `a` appended last. Order: `[f?] c sized-raises... a`.
* Min increment (`_min_increment`): reset to `BIG_BLIND` at each street
  start; every aggressive token (`b`, `x`, `a`) with `add − owe >= inc − 1e-9`
  sets `inc = add − owe`.
* Betting closed: after a fold; after a `c` that is not the street's first
  action (`len(tokens) >= 2`); when `all_in` and contributions equal.
* Chance: root (`holes is None`), and whenever betting is closed with
  `len(board) < 5` (covers street deals and the all-in runout; each deal also
  appends an empty street string).
* Terminal: `folded != −1`, or `len(board) == 5` and betting closed.
* Utility (player 0): fold → `−contrib[0]` / `+contrib[1]`; showdown compares
  `evaluate_best(holes[p] + board)`; win takes the opponent's contribution,
  tie is 0.

Max decision actions = `f c + 3 sized + a` = **6** (`MAX_ACTIONS = 6`).
Max aggressive actions per street = raise_cap = 3.

## 5. Hand evaluator

`poker/evaluator.py::evaluate_best_codes` — direct histogram evaluation of
5–7 cards returning a lexicographic tuple `(category, tiebreakers...)`,
category 0 (high card) .. 8 (straight flush); wheel straight has high rank 3.
Already exhaustively pinned against `evaluate_five` in Python. The native
evaluator will reproduce the same ordering with a packed `uint32`
(`category << 20 | r1 << 16 | ... `); within a category the tuple length is
constant, so zero-padding preserves order and equality.

## 6. Compact v2 encoder (`compact_river_pct20`) — key anatomy

Key string: `"{street}|{player}|{card_part}|{betting_key}"`.

`card_part`:
* preflop — the 169-class name (`AA`, `AKs`, `72o`) from `preflop_class`.
  (The MC-derived preflop *tier* is NOT in the key — no Monte Carlo anywhere
  in the hot path.)
* flop/turn — `"{strength}{draw}{nut}|{texture}"` where
  * `strength` 0..7 = `MADE_STRENGTH[made_hand_class(hole, board)]`,
  * `draw` 0..3 = `DRAW_CLASS[draw_type]` if `strength < 5` else 0,
  * `nut` = 2 if exact current nuts (enumeration), 1 if strength>=5, else 0,
  * `texture` = `texture_code` v1: `[up][rtf][cd]` (12 codes).
* river — `"p{bucket}{draw}{nut}b{blocker}|{texture}"` with
  * `bucket = min(int(pct*20), 19)`; `pct` = exact share of all 1,081
    two-card holdings of the 47 non-board cards beaten (ties half) — hero's
    own cards are *not* removed;
  * `draw` = 0 on the river; `blocker` 0/1/2 from `_flush_blockers`.

`betting_key` (`BettingContext.key()`):
`"i{initiative[0]}r{raises}f{facing}a{own_prior[0]}s{spr}|{legal}"` —
initiative own/opp/none from the last aggressor on *earlier* streets; raises
0..3 this street; facing ∈ none/small/medium/large/over/allin (fraction
`(add−owe)/pot_after_call` with edges 0.5/0.9/1.25 ± 1e-9); own_prior ∈
none/check/call/aggr/fold (first letters n/c/c/a/f — note check and call
share `c`); spr bucket of `eff/pot` at street start with edges (1,3,8);
`legal` = `".".join(legal_actions)` verbatim.

All components are deterministic pure functions of `(hole, board, streets)` +
immutable tree parameters. All are computable natively without Python
callbacks. All betting-context fields are maintainable incrementally by the
same left fold as the replay.

### Native key packing

Every field is small and enumerable; a packed 64-bit key covers it:

| field | values | bits |
|-------|--------|------|
| street | 0..3 | 2 |
| player | 0..1 | 1 |
| preflop class | 0..168 | 8 (preflop only) |
| strength / river pct bucket | 0..7 / 0..19 | 5 |
| draw | 0..3 | 2 |
| nut | 0..2 | 2 |
| blocker | 0..2 | 2 |
| texture | 13 codes incl. "pre" | 4 |
| initiative | 3 | 2 |
| raises | 0..3 | 2 |
| facing | 6 | 3 |
| own_prior | 4 seen (n/c/a/f) | 2 |
| spr | 0..3 | 2 |
| legal menu | subset mask of 9 tokens | 9 |

≈ 40 bits worst case → `uint64` key with room to spare. A native key renders
back to the canonical Python string (for export, debugging, and 1:1 parity
tests). Collision risk is zero by construction (bijective packing), but the
parity suite still checks both directions on a large state corpus.

## 7. RNG usage (Python)

`np.random.default_rng(seed)` (PCG64). Consumption points per traversal:

1. Root deal: `rng.choice(52, size=4, replace=False)`.
2. Street deal: `rng.choice(len(live), size=need, replace=False)` over the
   sorted live-card list; picked order preserved.
3. Opponent action: `cdf = cumsum(strategy); cdf /= cdf[-1];
   searchsorted(cdf, rng.random(), side="right")`.

Bit-identical cross-backend RNG is **not** a goal (numpy's
choice-without-replacement internals are not worth reimplementing); the
parity strategy is a deterministic *random tape* consumed by both backends
(Phase 17), plus statistical tests for the production native RNG.

## 8. Checkpoints and artifacts

* Resumable checkpoint (`solvers/serialize.py`): `.npz`, format v2, keys
  sorted by UTF-8 bytes, per-infoset action lists, float64 regret/strategy
  sums, JSON of the numpy RNG state, and game/encoder/config signatures that
  must match on load. **No resumable checkpoints exist in the repo** —
  `results/strategy/candidates/v2_extension/MANIFEST.json` records 44 of them
  by SHA-256 only ("container only", 1.3 GB); the files are gone.
* Strategy artifact (`solvers/strategy_artifact.py`): compressed `.npz`,
  format `pokeralpha.strategy_artifact/v2`, average strategy + visits only,
  content SHA-256, full config JSON, generation commit. This is the
  compatibility surface the native backend must produce (via Python-side
  export of natively trained tables).

## 9. Existing candidates (verified, Phase 58 pre-work)

| iterations, seed | strategy artifact | confidence table | resumable checkpoint |
|---|---|---|---|
| 100k seed 0 | yes (v1 fmt) | yes | no |
| 200k seed 0 | yes (release) | yes (seeds 0-2 @200k) | no |
| 300k seed 0 | yes (candidate) | **no** | no |
| 200k seed 1, 2 | yes | n/a | no |
| 100k/300k seeds 1-2 | no | no | no |

Consequence: Python→native checkpoint import (Phase 19/59) is **moot for the
existing 200k/300k lineage** — there is no training state to import. The
clean path is fresh native training (Plan B), which also gives three fully
comparable seeds under one backend.

## 10. Historical throughput

* `docs/release_status.md`: ~9 it/s on the previous 4-core container.
* This machine (M3 Pro, 12 cores): ~26 it/s cold at iteration 0-200; see
  `results/benchmarks/mccfr_python_baseline.json` for the warm numbers that
  are the actual baseline for the ≥5x adoption target.

## 11. Expected native-port complexity

| component | port difficulty | notes |
|---|---|---|
| game mechanics | moderate | fold over tokens; incremental state; epsilon comparisons copied verbatim |
| evaluator | low | histogram evaluator → packed uint32; exhaustive 5-card parity test |
| card features | moderate | made-hand classes, draws, nuts enumeration, blockers, texture |
| river percentile | low | per-board cache, rank-pair dedup identical to Python |
| betting context | moderate | incremental per-player facing/own_prior/initiative |
| MCCFR traversal | low | direct transcription; fsum-equivalent dot (6 values) |
| infoset table | low | `unordered_map<uint64, Node>`, fixed arrays of 6 |
| RNG/tape | moderate | xoshiro/PCG + tape harness on both sides |
| pybind + export | low | coarse train() call; arrays out; Python writes npz |

Main risks: (1) unordered_map reference invalidation across recursive
insertion (Phase 75/76 tests), (2) any divergence in epsilon-guarded float
comparisons, (3) encoder edge cases (board_plays, quads-on-board, wheel,
nut-flush blocker) — all covered by corpus parity tests.
