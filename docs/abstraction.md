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
