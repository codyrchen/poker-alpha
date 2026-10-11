# Decision Receipts (pokeralpha.receipt/v1)

Every analysis can produce a reproducibility record:

state fingerprint · full state · engine commit · strategy artifact
SHA-256 (when the solver was in play) · decision config (simulations,
seed, abstraction signature, rollout model) · method · gate verdict +
reasons · infoset key + training visits · action-mapping quality ·
recommendation + mix + confidence.

`receipt_id` (e.g. `PA-B748E5A4`) is a stable short hash of the
normalized content — a **reproducibility ID**, deliberately not called a
cryptographic proof.

**Verification:** `verify_receipt(receipt)` re-runs the analysis from
the recorded inputs (possible because every stochastic component records
its seed) and compares recommendation, mix and gate verdict; an
artifact-SHA mismatch is reported instead of silently producing
different numbers. Exposed as `POST /receipt/verify`; `/analyze`
attaches a receipt by default.

**Privacy:** no filesystem paths, usernames, screenshots or history
identifiers — tested (`test_receipt_privacy`).
