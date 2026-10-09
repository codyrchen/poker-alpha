# Strategy artifacts: metadata, integrity and rejection (Phase 55)

Code: `poker_alpha/solvers/strategy_artifact.py`; tests:
`tests/test_artifact_integrity.py`, `tests/test_phase29_tools.py`.

## What every artifact carries

| item | v2 artifact (new exports) | committed v1 artifacts |
| --- | --- | --- |
| solver config version + signature | `config_sig`, `config_json.config_version` | `config_sig` + manifest `config` |
| full config (encoder, action abstraction, stack, blinds, raise cap, averaging, sampler) | `config_json` (its hash must equal the signature) | `<name>.manifest.json` |
| game signature (stack, blinds, bet menu, preflop sizes, raise cap, min-raise rule) | `game_signature` | same |
| encoder version | `encoder_sig` | same |
| seed, iterations, sampler | `meta` | `meta` + manifest |
| generation commit | `commit` (read from `.git`, no subprocess) | not recorded at generation time: manifest `generation_commit: null` + `first_committed_in` |
| checksum | `content_sha256` over keys / offsets / actions / probabilities / visits | manifest `sha256` of the file (matches `release_candidate.json`) |

The committed `.npz` files were **not rewritten** (a format migration must
never risk existing artifacts); `experiments/write_artifact_manifests.py`
wrote sidecar manifests next to them instead.

## Rejection (no silent fallback)

`load_artifact` raises `StrategyArtifactError` for: a file that cannot be
read (truncated, bit flips in the compressed data), an unknown format, a v2
content checksum mismatch (tampered probabilities), a `config_json` that
does not hash to the stored signature, a sidecar manifest whose SHA-256 the
file no longer matches, and — when used with a game — any solver-config,
game (stack / bet menu / raise cap / preflop sizes / min-raise rule) or
encoder signature mismatch.

`SolverStrategyProvider.from_artifact` turns those into a `LookupMiss`
(`CONFIG_MISMATCH` / `INCOMPATIBLE_CHECKPOINT`) that the decision report
shows in its source cascade; recommendations then come from rollouts /
heuristics with that reason visible — never from the mismatched strategy.
Blinds are fixed at 0.5 / 1 BB in the abstract game (amounts are in big
blinds), so a different small-blind ratio or an ante is refused per
decision ("blind structure differs from the abstraction").

## Determinism and resume audit (Phase 70)

What is checked, and where:

| property | test |
| --- | --- |
| CFR / CFR+ / MCCFR resume bit-identical (Kuhn, Leduc, default Hold'em), RNG state restored (negative control: resetting the RNG changes the result) | `tests/test_checkpoint.py` |
| MCCFR resume bit-identical under the **release v2 config** | `tests/test_determinism_audit.py` |
| MCCFR digest, a full decision report and an observer state are identical under different `PYTHONHASHSEED` values (no dependence on set/dict iteration order) | `tests/test_determinism_audit.py` |
| decision reports (recommendation, EVs to 4 dp) stable across runs | `tests/test_golden_e2e.py` |
| observer session replay reproduces the recorded tracker sequence | `tests/test_session_replay.py` |
| experiment CLI checkpoints and resumes | `tests/test_holdem_experiment.py` |

Training runs in `experiments/holdem_mccfr_validation.py` checkpoint at
milestones; a resumed run continues the same RNG stream, so an interrupted
run that is resumed gives the same strategy as an uninterrupted one.
