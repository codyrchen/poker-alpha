# Observer test sessions: record, replay, annotate, export

The workflow for collecting real observer data with little manual effort,
then turning it into fixtures. Everything is local and read-only: nothing
clicks, types or controls a browser, and nothing is uploaded.

```
Live screen -> Observer Test Session   (record: Phase 41)
   ~/pokeralpha_sessions/<id>/
python -m poker_alpha.observer.session_replay <session>   (replay: Phase 42)
Streamlit -> Annotate session          (ground truth: Phase 43)
python experiments/export_observer_fixtures.py ...   (redacted fixtures: Phase 44)
python experiments/observer_validation.py --fixture-dir ...   (metrics: Phases 45-46)
```

## 1. Recording (Observer Test Session)

Streamlit -> Input -> **Live screen** -> Mode **Observer Test Session
(diagnostic)**. Set up capture and calibration as in live mode
(`docs/observer.md`), then **Start session**.

Defaults: screen observation on, decision computation **off**, frames kept
only when something meaningful happens, local files only.

Controls: Start session · Pause capture · Resume capture · Stop session
(capture stops; you can still mark/save the current frame and add notes) ·
Finish session (writes the final manifest) · Mark current frame (keeps it,
with the optional note) · Save current frame · Add note · Reset tracker.

### Which frames are kept

| policy | keeps a frame when |
| --- | --- |
| manual / marked only | you press Mark or Save |
| every N seconds | N seconds passed since the last interval sample |
| state change | the fused table state changed (cards, board, pot, stacks, bets, dealer, actor, seat status, hand number) |
| warning / error | the set of step errors / warnings / tracker flags *changed* (a persisting warning is kept once, not every frame) |
| low-confidence transition | the critical confidence crossed the threshold (default 0.5) in either direction |
| new hand | the tracker started a new hand |
| street transition | the number of board cards changed |
| hybrid (**default**) | any of: state change, warning / error, new hand, street |

Automatic policies also keep the first recognized frame (`first_frame`) as a
baseline. Marked frames are always kept. A frame kept for several reasons is
stored once with all reasons.

### Limits (safe defaults)

| limit | default | when reached |
| --- | --- | --- |
| session duration | 4 h | recording stops, reason logged |
| kept frames | 2,000 | recording stops |
| disk written | 2,000 MB | recording stops |

A 1-FPS session at the default policy keeps roughly one frame per table
change (a few per hand), each a lossless PNG of the capture rectangle plus a
few KB of JSON.

### Session directory

```
~/pokeralpha_sessions/20261008_235959/
  .gitignore            "*"  (the folder can never be committed, wherever it is)
  session.json          status, policy, limits, capture, calibration id + checksum,
                        notes, counters, software (python, platform, commit)
  calibration.json      calibration at start; calibrations/<sha12>.json if changed
  manifest.json         kept samples (manifest.jsonl is streamed while recording)
  frames/<frame>.png
  observations/<frame>.json    raw readings + confidences per field, table bbox,
                               recognizer provenance (classes and parameters)
  observations/stream.jsonl    compact raw readings of EVERY processed frame (no
                               image, ~2-5 KB each) + tracker reset rows: replay input
  tracked_states/<frame>.json  fused snapshot, field confidences, hand number,
                               critical confidence, validation issues, tracker warnings
  diagnostics/<frame>.json     reasons, monitor, capture rect, captured size,
                               pixels per point (Retina: 2.0), timings (capture /
                               recognition / tracker), observer FPS, step error /
                               warning, events, decision flag
  events/events.jsonl   every tracker event (bet, fold, board, new_hand, ...) and
                        session event (start, pause, note, mark, capture_error, limit)
  annotations/          ground truth written later (Phase 43)
```

Sample ids are the observer's frame number (`000042`). Timestamps are both
wall clock (ISO) and monotonic seconds since the session started.
`pokeralpha_sessions/` and `pokeralpha_captures/` are also listed in the
repository `.gitignore`.

Code: `poker_alpha/observer/session.py` (`TestSessionRecorder`, no
Streamlit dependency), UI in `poker_alpha/ui/live_panel.py`, tests in
`tests/test_observer_session.py` (synthetic frames, fake screen).

## 2. Replay

```bash
python -m poker_alpha.observer.session_replay ~/pokeralpha_sessions/<id>            # stored
python -m poker_alpha.observer.session_replay <session> --mode recompute --write-report a.json
python -m poker_alpha.observer.session_replay <session> --compare-calibration new_cal.json
python -m poker_alpha.observer.session_replay <session> --mode recompute --compare-report a.json
```

| mode | input | answers |
| --- | --- | --- |
| `stored` (default) | `observations/stream.jsonl`: raw readings of **every** processed frame, plus a reset row whenever the tracker was rebuilt (start, Reset tracker, recalibration, blinds change) | Does the current tracker reproduce exactly what the live session concluded? Compares the fused state, hand number and critical confidence at every kept sample and the full event list. `reproduced: True` = deterministic and unchanged. |
| `recompute` | kept frame images | Does current recognition read the same values? Field-by-field value changes and confidence deltas vs the stored readings, then tracker-state, event and decision differences over the kept frames. |
| `--compare-calibration FILE` | kept frame images | Same, with a recompute under the session calibration as the baseline and FILE as the candidate. |
| `--compare-report OLD.json` | a report written earlier (e.g. by another commit) | Per-sample state / decision / confidence differences between the two code versions — no Git checkout juggling. |

`--no-decisions` skips decision computation (by default replay computes a
quick decision — 400 equity samples, no rollouts, no solver — for samples
whose critical check passes, to report decision-source changes).
`--blinds SB BB` overrides the blinds recorded in the stream.

Recompute mode only has images for kept frames, so its tracker states
describe that subsequence; both sides of a comparison replay the same
subsequence. A crashed session (no `manifest.json`, truncated last line)
still replays from the streamed `manifest.jsonl`.

Code: `poker_alpha/observer/session_replay.py`; tests:
`tests/test_session_replay.py`.
