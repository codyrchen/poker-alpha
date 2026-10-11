# Observer ML scaffold (optional, data-gated)

`python -m poker_alpha.observer.ml train-cards <dataset_root>` prepares
for replacing the weakest recognizer component (card rank/suit) with a
trained classifier — ONCE enough real labeled data exists.

Hard gates (in code, tested — a COVERAGE check, not a count threshold):

* ≥ 3 tuning sessions and ≥ 1 independent validation session;
* every rank seen ≥ 5× and every suit ≥ 10× in tuning;
* validation covers ≥ 8 distinct ranks and ≥ 3 suits (so per-class
  failure is detectable);
* no severe imbalance (top rank ≤ 50% of tuning labels);
* a 200-card overall floor — a floor, not a sufficiency guarantee.

Any unmet requirement makes the command exit with
`insufficient real labeled data: …` listing all of them, writing
nothing. It never trains on synthetic fixtures and claims success.
The held-out split is never touched by training or model selection;
final comparison happens once, via
`poker_alpha.observer.dataset baseline`.

Model + registry:

* architecture: TWO heads — a 13-way rank classifier and a 4-way suit
  classifier — rather than one 52-class model. For small calibrated
  crops this is the defensible factorization: suit is color/shape, rank
  is a glyph, and per-class support stays meaningful at realistic
  dataset sizes;
* artifact: small, inspectable numpy softmax heads on normalized
  calibrated card crops (`card_classifier.npz`);
* metadata (`pokeralpha.model/v1`): model type, dataset root, split
  hash, config, git commit, timestamp, per-head validation metrics with per-class support, held-out
  status, artifact SHA-256, and an explicit claims-scope line. No opaque
  `model.pt` without provenance.

Current status: **BLOCKED on data** — the sufficiency gate fires because
only one annotated real frame exists. The trainer's mechanics are unit
tested on synthetic arrays (explicitly not an accuracy claim).

The product does not depend on any of this: Demo, Manual, Hand history
and Study work with recognition entirely out of the loop.
