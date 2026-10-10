# Native 2M candidates (final-trust project)

Seed 0 of this set was **promoted to the release**: identical copies are
`results/strategy/holdem_v2_native_seed0_2m.npz` (artifact, SHA f1860de7…)
and `holdem_v2_native_seed0_2m_confidence.npz` (the `_confidence_v2` file
here, SHA ef5c3223…). Seeds 1-2 are the comparability companions used for
seed-disagreement data. `holdem_v2_native_seed0_2000k_confidence.npz`
(no `_v2` suffix) is the schema-1 construction kept only for the
milestone-trend comparison (gate_acceptance_v2_native_2000k.json).

Training checkpoints (10k…2M per seed, ~45 MB each) are NOT committed; they
live in results/native_training/seed{0,1,2}/ on the training machine, with
resume commands in the JSONL logs. Evidence:
results/validation/native_training_2m.json, final_release_comparison.json.
