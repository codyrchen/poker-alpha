# Solver validation (Phases 32-37)

Is the MCCFR implementation correct, and are the suspicious preflop outputs
of the locked strategy (`HoldemSolverConfig:v1`, compact encoder, imperfect
recall) caused by bugs or by abstraction and training? Every number here is
from a committed result file:

* `results/validation/preflop_audit_v1.json` — 169-class audit (32A),
  heatmaps in `results/figures/phase32/`;
* `results/validation/solver_quality_v1.json` — rules, geometry, utility,
  MCCFR trace, averaging windows, action EVs, recall, jams (32B/32C).

## Phase 32 — solver trustworthiness

### 32A Full 169-class preflop audit

10 situations x 169 classes x 9 checkpoints (seeds 0/1/2 at 10k / 100k /
300k): key, visits, average and current strategy, regret and strategy sums,
movement and seed disagreement (`experiments/phase32_preflop_audit.py`).

| situation | median visits @300k | seed disagreement (mean L1, max 2) | 100k->300k movement (L1) | mean jam freq. | classes jamming > 50% |
| --- | --- | --- | --- | --- | --- |
| BTN first action | 1,363 | 0.90 | 0.40 | 5.1% | 0 |
| BB vs limp | 1,364 | 0.86 | 0.42 | 9.9% | 4 |
| BB vs open 33% (to 1.66 BB) | 1,364 | 0.95 | 0.44 | 16.8% | 14 |
| BB vs open 75% (to 2.5 BB) | 1,364 | 0.90 | 0.43 | 13.3% | 10 |
| BB vs open 150% (to 4 BB) | 1,364 | 0.84 | 0.46 | 13.2% | 4 |
| BB vs all-in | 1,364 | 0.38 | 0.21 | — | — |
| BTN limp vs raise 33/75/150% | 119 | 1.13-1.17 | 0.29-0.33 | 18-20% | 4-10 |
| BTN open 75% vs 3-bet 75% | 688 | 1.07 | 0.38 | 11.0% | 6 |

**Main finding: the preflop strategy is dominated by noise.** Every
first-action infoset has ~1,360 visits at 300k, yet independently seeded
runs disagree by L1 ~0.9 on the same class, and each run still moves ~0.4
between 100k and 300k. 1,107 of 1,690 class-situation states exceed L1 0.8
seed disagreement. Clear-cut hands are stable (72o first in: fold 93%, seed
L1 0.13; facing a jam, L1 0.38); close decisions are not.

### 32B AA limping (BTN first, 61% limp averaged over seeds)

| hypothesis | test | result |
| --- | --- | --- |
| rules bug | HU order, blinds, pot after limp-check (2.0), all-in and call amounts (100), 3-bet effective stack | **ruled out** (all pass; `tests/test_solver_trust.py`) |
| utility bug | SB fold -0.5, BB fold to open +1, raise-call-checkdown +2.5, all-in-call +/-100, split 0 | **ruled out** |
| MCCFR update bug | instrumented updates at the AA infoset: node value = strategy . child values, regret delta sums to 0 under the strategy, strategy sum untouched on traverser visits, fold child = -0.5; first-action infosets get exactly one average-strategy visit per iteration | **ruled out** in Hold'em; exact-game convergence in Phase 33 |
| action abstraction | concrete sizes: BTN open 33% = **to 1.66 BB** (below the 2 BB legal minimum), raise over a limp 33% = to 1.66 BB; 81% of preflop `b33` offers and ~30% of postflop `b33` bets/raises are **below NLHE minimums** | **contributes** (sizes are not legal NLHE; the game is internally consistent but mis-shaped preflop) |
| averaging effect | full average vs windows: seed 0 limp 62% (full), 49% (100k-300k window), current 55%; the first 10k iterations hold only ~3% of AA's visits | **not the cause** (recent windows and current strategies also limp 40-82%) |
| imperfect recall | the AA first-action key is the exact class (no merge). After a limp the flop keys AA reaches merge it with other holdings and lines (e.g. one top-pair+ limped-pot key holds 16 different preflop classes; others mix 4 preflop lines) | **contributes** to the continuation values, not to the preflop key itself |
| undertraining / sampling noise | EVs of forced first actions under the trained 300k profile (10,000 paired deals): raise 75% beats limp by **+1.12 BB (SE 0.46)** (seed 0) and **+0.96 (0.46)** (seed 1); jam is -3.9 / -2.8 BB; individual sampled child values range over +/-100 BB | **primary cause**: limping is not a best response to the profile, but the ~1 BB gap is small against the per-sample variance of ~1,300 regret updates |
| equilibrium feature | — | **unresolved**: no exact solution of this abstract game exists to compare |

**Diagnosis: UNDERTRAINING (sampling noise at 100 BB) + ACTION ABSTRACTION
+ IMPERFECT-RECALL EFFECT. Not a rule, utility, MCCFR or averaging bug.**

### 32C Preflop all-ins

* Highest where the open is tiny: BB vs the 1.66 BB "33%" open jams with 14
  classes > 50% (16.8% mean).
* EVs under the profile (8,000 paired deals): BB 99 vs open33 — every raise
  incl. jam beats calling by ~4-6 BB, jam within noise of raising; AJo vs
  open75 — jam best (+5.3 vs call, SE 0.58); A5o vs open33 — every action
  within ~1 SE (jams 77% of the time: noise); 72o vs open33 — jam -11 BB, and
  72o rarely jams. Jams are roughly EV-maximizing against a BTN that folds
  to a jam with ~66% of classes.
* **Imperfect-recall collision found:** the BTN's information set facing a
  jam is the same after a 33%, 75% or 150% open (`b33a` / `b75a` / `b150a`
  map to one key: the betting context keeps "facing all-in", not the size
  of the BTN's own open).
* Utility of all-in lines is exact (32B). Diagnosis: **pot-relative opening
  sizes + near-indifference noise + response collision; not a utility bug.**
