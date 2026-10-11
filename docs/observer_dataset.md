# Observer dataset pipeline

Everything needed to collect, split and evaluate a REAL vision dataset —
without fabricating any data. Current status is unchanged and stated
plainly: **REAL HELD-OUT VALIDATION: BLOCKED** — exactly one annotated
real frame exists (`tests/fixtures/pokernow`, role `tuning`, the frame
the recognizers were fitted on). Collect sessions with
`docs/real_pokernow_test_checklist.md` and the observer test-session
recorder, then:

    python -m poker_alpha.observer.dataset manifest  <root>
    python -m poker_alpha.observer.dataset split     <root> --seed 0 \
        --fractions 0.6,0.2,0.2 [--lock]
    python -m poker_alpha.observer.dataset check     <root>
    python -m poker_alpha.observer.dataset coverage  <root>
    python -m poker_alpha.observer.dataset baseline  <root> --out report.json

* **Splits are per SESSION**, never per frame — adjacent frames are
  near-duplicates and frame-level splits would leak. `--lock` freezes
  the held-out session set in `heldout.lock.json`; later splits cannot
  move those sessions and `check` flags any tampering as a LOCK
  VIOLATION.
* **`check`** also catches byte-identical images appearing in more than
  one split and per-frame role drift inside a session. Exit code 1 on
  any problem.
* **`coverage`** reports ranks, suits, streets, pot/stack/bet buckets,
  dealer seats, hero side, fold/all-in observations and image
  resolutions, with explicit missing-coverage lists. Coverage of labels
  says nothing about accuracy, and the report says so.
* **`baseline`** runs the EXISTING recognizer over every split with each
  session's own calibration and scores it against the human annotations
  (per-field accuracy, exact hero pair, amount errors, confidence
  buckets). A split with no real data reports `BLOCKED` — numbers are
  never invented.

Privacy: sessions and screenshots stay local; the recorder's
privacy-safe fixture export is unchanged (docs/observer_sessions.md).

Confidence calibration: the baseline's confidence buckets become
reliability/risk-coverage curves once real held-out data exists; until
then no precision/coverage numbers are published anywhere.

## Collection targets (for annotators)

These are COLLECTION TARGETS that make the recognition problems
well-posed — not validated sample-complexity guarantees. Aim for:

* **≥ 6 separate sessions**, recorded on different days/table instances
  (splits are per session; more sessions = better splits);
* within the tuning+validation pool: **every rank ≥ 5×, every suit
  ≥ 10×** among labeled hero cards — deal until coverage fills rather
  than stopping at a frame count;
* **all four streets** represented, including fold and all-in states;
* **varied pots/stacks/bet sizes** (the coverage report's buckets should
  all be non-empty) and at least two window sizes/resolutions;
* annotate completely: a frame with unknown hero cards still helps
  coverage of pots/stacks, but only fully labeled cards feed the card
  classifier.

Run `coverage` after every session batch and collect toward the gaps it
lists, rather than accumulating more of what is already covered.
