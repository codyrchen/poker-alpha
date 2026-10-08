# Abstraction and heads-up solving

## Sampled chance and checkpoints

`Game.sample_chance(state, rng)` is part of the base interface; the default
samples `chance_outcomes()` with the exact RNG call MCCFR used before, so
Kuhn/Leduc MCCFR results are unchanged (pinned digests). `HoldemGame`
overrides it to deal without enumerating, which lets external-sampling MCCFR
train on Hold'em. Full CFR/CFR+ and exact exploitability still require
enumeration and therefore do not apply to Hold'em.

Checkpoints (`solvers/serialize.py`) are pickle-free `.npz` archives with a
format version, solver type, game signature, encoder signature, iteration
count, sorted infoset keys, actions in solver order, regret and strategy
sums and — for MCCFR — the complete NumPy bit-generator state as JSON.
`train(a) + save + load + train(b)` is bit-identical to `train(a + b)`.
Mismatched game or encoder signatures raise `CheckpointError`.

## Information-state encoders

`HoldemGame(encoder=...)` delegates `infoset_key` to an
`InformationStateEncoder` (`encode(game, state)`, `signature()`):

| encoder | key | recall |
| --- | --- | --- |
| `RawHoldemEncoder` (default) | exact hole cards, board, history — byte-identical to the original key | perfect |
| `ToyHoldemEncoder` | pair/suited/offsuit+broadway preflop, made-hand category postflop | **imperfect** — demo only |
| `HoldemBucketEncoder` | 169-class preflop label, then per street `e{equity bucket}{texture}s{SPR bucket}`, every earlier street's bucket, exact action history | perfect within the abstraction |

**PERFECT-RECALL WARNING.** An encoder that forgets information a player
had (e.g. the toy encoder drops the preflop class once the flop is dealt)
yields an imperfect-recall abstraction: CFR may still run, but its output is
not an equilibrium of anything and must not be presented as one.

**Measurement correction (Phase 25).** Earlier docs compared infoset counts
from *separate* MCCFR runs (raw 49,050, toy 23,940, bucket 77,872). Those
runs sampled different trajectories and the bucket run even used a
different bet menu (33/75/150% vs 50/100/200%), so they did **not** measure
compression. The encoder-independent measurement on one frozen corpus is in
[validation.md](validation.md): every encoder is a function of the raw
information state (0 invariant violations), the bucket encoder has perfect
recall (0 violations in the collision audit), and its postflop compression
is small because keys carry the full bucket history.

## Phase 26 scalable encoders (`abstraction/holdem_v2.py`)

Built on cheap, exact, cached card features (`abstraction/features.py`:
made-hand class on a 0..7 strength ladder, draw class, current-nuts flag,
nut-flush blockers, texture) and an abstract betting context
(`abstraction/betting_history.py`: street, position, initiative, raise
count, size class of the bet faced, own prior aggression, SPR bucket **and
the legal-action code**, so one key never spans two action menus).

| encoder | key | recall |
| --- | --- | --- |
| `TransitionHoldemEncoder("exact")` | preflop class / flop state / turn and river *transitions* / exact history | perfect (0 audit violations) |
| `TransitionHoldemEncoder("abstract")` | same cards, abstract betting context | imperfect |
| `CompactHoldemEncoder()` | street, position, current strength/draw/nut (+blocker on river), texture, betting context | **IMPERFECT RECALL — NO STANDARD CFR EQUILIBRIUM GUARANTEE** |
| `CompactHoldemEncoder("exact")` | current card state, exact history | imperfect |

Results (same frozen corpus, identical seeds/budgets, full numbers in
[validation.md](validation.md#8-phase-26-scalable-abstractions) and
`results/validation/abstraction_v2.json`): only the compact encoder
materially improves revisitation — 60,747 infosets after 5,000 iterations
vs 1.65M for the Phase-25 bucket encoder, 50.8% of infosets with >= 5 visits
(bucket 0.31%), all 14 canonical spots visited. The perfect-recall
transition encoder does **not** help (1.51M infosets): the exact betting
history, not the card state, dominates the key space. The price is
imperfect recall: 874 of 1,508 colliding compact keys merge states whose
earlier observations differ.

## Locked solver configuration (Phase 27, `poker_alpha/solver_config.py`)

**PRIMARY_SOLVER_ENCODER = `CompactHoldemEncoder` (abstract betting
context). IMPERFECT RECALL — NO STANDARD CFR EQUILIBRIUM GUARANTEE.**
It was the only candidate to pass every selection criterion (revisitation,
not-mostly-discovery, canonical coverage across seeds, no legal-action
mixing, within-key equity coherence); the scorecard is
`results/validation/solver_abstraction_selection.json`.

`HoldemSolverConfig` (v1) pins: encoder `compact`; card buckets = the
deterministic features of `abstraction/features.py` (a digest of the
strength/draw tables is part of the signature); bet menu 33/75/150% pot +
all-in; 100 BB stacks, 0.5/1 blinds; raise cap 3 per street; external-
sampling MCCFR; reference range "uniform random hand", used only for offline
quality metrics. `PRIMARY_CONFIG.signature()` (`HoldemSolverConfig:v1:<hash>`)
is written into checkpoints (format 2, field `solver_config`) and checked on
load: a checkpoint never loads into a game built from a different config or
from no config. Format-1 checkpoints still load into config-less games.

Regression fixtures under `tests/fixtures/solver_v1/`: a frozen corpus of
seeded random-policy decision states with expected keys and legal actions,
and the canonical-state suite. `tests/test_solver_config.py` fails if any
key changes; changing the abstraction means a new config version and new
fixtures, deliberately.

## Trained strategy (Phase 29)

Three seeds of the locked config were trained to 100k and then 300k
iterations (see [validation.md](validation.md#11-phase-29-training-the-locked-config)
and section 12); the seed-0 300k average strategy is exported as
`results/strategy/holdem_v1_seed0.npz`
(`pokeralpha.strategy_artifact/v1`, loaded by
`SolverStrategyProvider.from_artifact`, which rejects it with
`CONFIG_MISMATCH` under any other config). It is an abstract strategy for
heads-up 100 BB play with a 33/75/150% + all-in menu, trained on an
imperfect-recall abstraction: not an equilibrium, not GTO, exploitability
unknown.

## Release configuration v2 (Phases 35-36)

**RELEASE_CONFIG = `V2_CONFIG` (`HoldemSolverConfig:v2:733e52f1d1014e2e7973`).**
Still the compact encoder and still **IMPERFECT RECALL — no standard CFR
equilibrium guarantee**, with two evidence-backed changes
([solver_validation.md](solver_validation.md)):

* legal NLHE sizing — preflop raises to 2 / 2.5 / 3.5x the current bet
  (tokens `x200` / `x250` / `x350`), postflop 33 / 75 / 150% pot, and no
  bet or raise below the NLHE minimum (v1 offered a 1.66 BB "open");
* river cards bucketed by exact strength percentile against all holdings
  (20 buckets) instead of the 0..7 made-hand rung — exact river-subgame
  exploitability of the compact strategy 9.24 -> 0.57 BB and 2.24 -> 0.28 BB.

The v2 signature also records `averaging=uniform` (simple external-sampling
averaging) and the action-abstraction description. Release artifact:
`results/strategy/holdem_v2_seed0.npz` (seed 0, 100k iterations, 2.2 MB)
plus `holdem_v2_seed0_confidence.npz` for the solver-use gate. v1
(`PRIMARY_CONFIG`, artifact `holdem_v1_seed0.npz`, 300k) stays the locked
reference; strategy files load under whichever known config their
signature names (`solver_config.config_for_signature`).

## Card abstraction (`abstraction/cards.py`)

* 169 preflop classes, exhaustively tested over all 1,326 combos (13 pairs ×
  6, 78 suited × 4, 78 offsuit × 12).
* Suit canonicalization over all 24 suit permutations (flop order ignored,
  street order kept).
* Typed `BoardTexture` (pairedness, suitedness, connectedness, straight /
  flush possible, high-card class) and `HandFeatures` (made-hand category,
  deterministic suit-invariant equity vs a random hand, exact current
  strength vs all combos, nut status, draw type, flush and straight
  blockers, SPR, position, effective stack).
* Equity is seeded from a SHA-256 of the canonical cards, so isomorphic
  inputs give identical buckets on every platform.

Uniform equity buckets are a simple, inspectable first choice — not
potential-aware, not distribution-aware, not optimal.

**Cost of perfect recall.** Carrying the 169-class preflop label and every
earlier street bucket in each key multiplies the key space street by street:
on a fixed action line the current flop bucket label alone takes ~69
values, but the full flop key (×169 preflop classes) left 1,471 of 2,000
sampled flop states distinct; turn 1,897/2,000; river 1,992/2,000. In
practice the abstraction is barely coarser than raw postflop, which is why
MCCFR training is dominated by state-space discovery (see
[validation.md](validation.md)). Coarser perfect-recall designs, or an
explicitly *imperfect-recall* postflop abstraction (standard in large
solvers, but without equilibrium guarantees), are the options; neither is
implemented yet.

## Betting abstraction (`abstraction/betting.py`)

`ActionAbstraction` turns a `BettingContext` (pot, amount to call, hero
stack, largest callable opponent stack, street commitment, min raise, big
blind) into a menu of `check/call/fold/bet_X/raise_X/all_in`, lifting sizes
below the legal minimum, collapsing sizes at or above the all-in ceiling and
de-duplicating labels that resolve to the same wager. Observed wagers map
back with the pseudo-harmonic translation of Ganzfried & Sandholm (2013).
Bets are `f × pot`; raises are pot-relative (`call`, then `f × (pot + call)`).
Keep the abstract action *sequence* in information-set keys — keying by
"which menu was legal" destroys perfect recall.

## Running the heads-up experiment

```bash
python experiments/holdem_mccfr.py --iterations 2000 --seed 0 --encoder bucket \
    --stack-bb 100 --checkpoint-every 500 --checkpoint results/checkpoints/hu.npz
```

Reported per checkpoint: iterations/s, infosets, memory estimate,
checkpoint size, visit-weighted entropy, visit histogram, L1 change vs the
previous checkpoint, top-K infoset stability, seeded cross-play vs the
previous checkpoint; optional seed-to-seed L1 (`--compare-seed`). **Exact
exploitability is not computed and not approximated.**

Smoke-run measurements (`results/data/holdem_mccfr_bucket.csv`, bucket
encoder, bets 33/75/150%, this container): ~1.7–1.9 iterations/s, 77,872
infosets and ~34 MB after 200 iterations, >99% of infosets visited once,
checkpoint-to-checkpoint L1 falling 0.0056 → 0.0023, top-200 L1 0.106 →
0.033, seed-to-seed top-200 L1 0.33. These numbers say the run is **far from
converged**; producing a usable heads-up strategy needs orders of magnitude
more iterations (and likely a faster implementation).

`solvers.holdem_analysis.describe_strategy_at(game, strategy, visits, "BTN",
("As", "Ks"))` prints the policy at a readable spot, flagging unvisited
information sets.
