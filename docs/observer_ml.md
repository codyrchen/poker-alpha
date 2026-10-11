# Observer ML scaffold (optional, data-gated)

`python -m poker_alpha.observer.ml train-cards <dataset_root>` prepares
for replacing the weakest recognizer component (card rank/suit) with a
trained classifier — ONCE enough real labeled data exists.

Hard gates (in code, tested):

* ≥ 200 labeled hero cards across ≥ 4 train/validation sessions,
  otherwise the command exits with
  `insufficient real labeled data: …` and writes nothing. It never
  trains on synthetic fixtures and claims success.
* The held-out split is never touched by training or model selection;
  final comparison happens once, via
  `poker_alpha.observer.dataset baseline`.

Model + registry:

* artifact: small, inspectable numpy softmax on normalized calibrated
  card crops (`card_classifier.npz`);
* metadata (`pokeralpha.model/v1`): model type, dataset root, split
  hash, config, git commit, timestamp, validation accuracy, held-out
  status, artifact SHA-256, and an explicit claims-scope line. No opaque
  `model.pt` without provenance.

Current status: **BLOCKED on data** — the sufficiency gate fires because
only one annotated real frame exists. The trainer's mechanics are unit
tested on synthetic arrays (explicitly not an accuracy claim).

The product does not depend on any of this: Demo, Manual, Hand history
and Study work with recognition entirely out of the loop.
