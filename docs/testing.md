# Tests (Phase 79)

```bash
pytest -q                          # everything (~5 min on 4 cores; slow tests included)
POKERALPHA_SKIP_SLOW=1 pytest -q   # fast inner loop (what CI's core job runs)
pytest -q -m vision                # observer tests that need Pillow (+ mss / scipy)
POKERALPHA_UPDATE_GOLDEN=1 pytest tests/test_golden_e2e.py   # only after an intended change
```

Markers (registered in `tests/conftest.py`): `slow` (expensive regression,
skipped with `POKERALPHA_SKIP_SLOW=1`), `vision` (needs the `[vision]`
extra). Tests that need an optional package call `pytest.importorskip`, so a
core-only install skips them cleanly instead of erroring (verified in a
clean Python 3.11 venv, core + `[dev]` only: 760 passed, 12 skipped — all Streamlit; clean Python 3.12 venv with `[dev,vision,ui,ocr]`, headless: 773 passed, 0 skipped).

CI (`.github/workflows/tests.yml`): **core** — Python 3.11 and 3.12, `[dev]`
only, non-slow suite; **vision + UI extras** — Python 3.12 with
`[dev,vision,ui]`, the full non-slow suite (no hand-maintained file list:
a new test file can not be forgotten), headless (no display, no Tesseract).

## Layout

| area | files |
| --- | --- |
| research core (Kuhn / Leduc, CFR family, exact exploitability — results must stay unchanged) | `test_kuhn`, `test_leduc`, `test_cfr`, `test_cfr_plus`, `test_mccfr`, `test_checkpoint`, `test_reproducibility`, `test_opponent`, `test_risk`, `test_statistics` |
| cards, evaluator, equity, ranges | `test_cards`, `test_evaluator`, `test_equity`, `test_multiway_equity`, `test_ranges` |
| Hold'em rules and solver game | `test_holdem_engine`, `test_pots`, `test_holdem`, `test_sampled_chance`, `test_reduced_games`, `test_legal_sizing`, `test_betting_abstraction`, `test_card_abstraction`, `test_abstraction_v2`, `test_abstraction_validation`, `test_encoder_seam`, `test_solver_config`, `test_holdem_experiment` |
| solver trust, artifacts, determinism | `test_solver_gate`, `test_solver_trust`, `test_artifact_integrity`, `test_determinism_audit`, `test_performance_equivalence`, `test_phase29_tools` |
| decision engine | `test_decision_engine`, `test_decision_invariants`, `test_decision_faults`, `test_rollout`, `test_pipeline`, `test_golden_e2e`, `test_definition_of_done`, `test_observed_state` |
| history, sessions, replay | `test_replay`, `test_session`, `test_validation`, `test_demo`, `test_demos` |
| observer: recognition and fusion | `test_observer_core`, `test_observer_pokernow`, `test_pokernow_real` (tuning frame — regression only), `test_observer_fusion`, `test_observer_state_machine` (+ `observer_timelines.py`), `test_observer_failures`, `test_observer_robustness` |
| observer: live, geometry, calibration | `test_live_observer`, `test_capture_geometry`, `test_calibration_versioning` |
| observer: test sessions and data | `test_observer_session`, `test_session_replay`, `test_annotation_tool`, `test_annotation_validation`, `test_fixture_export`, `test_observer_fixture_validation`, `test_observer_debug` (+ `observer_helpers.py`) |
| UI, privacy, tooling | `test_ui`, `test_privacy`, `test_doctor_launcher` |

## Rules

* No test may count the real tuning frame (or a transformation of it) as
  validation; `test_pokernow_real` and the golden tuning-frame case are
  regression guards.
* Never skip, xfail or loosen a failing test to get green; fix the cause.
* Golden files change only with an intended behaviour change, reviewed in
  the diff.
