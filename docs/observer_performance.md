# Live observer performance (Phase 50)

Measured with `experiments/live_observer_perf.py` on the container CPU with
a mocked screen source (no display: real `mss` capture time is not measured;
on macOS a full-window grab typically adds a few ms to a few tens of ms).
Numbers are machine-dependent; the shape matters more than the values.

```
python experiments/live_observer_perf.py profile --frames 120   # -> results/validation/live_observer_perf.json
python experiments/live_observer_perf.py soak --minutes 30 --fps 3   # -> results/validation/live_observer_soak.json
```

## Per-step latency (ms, 120 frames)

| stage | synthetic 1280x800 median / p99 | PokerNow tuning frame at 2x (timing only) median / p99 |
| --- | --- | --- |
| table detection | 8.9 / 10.4 | 27.8 / 35.7 |
| OCR | 8.0 / 10.1 | 12.0 / 15.7 |
| card recognition | 11.6 / 15.6 | 12.8 / 15.9 |
| overlay rendering | 7.5 / 9.0 | 7.5 / 10.4 |
| tracker + gate | 0.2 / 0.3 | 0.2 / 0.2 |
| **step + overlay** | **38.5 / 47.5** | **66.7 / 81.9** |

The PokerNow column re-uses the single tuning frame for timing only; it says
nothing about recognition accuracy.

Decision (only when the observed state changes, cached per state):
96 ms (2,000 equity samples, no rollout), 192 ms (+400 rollouts).
Streamlit full-page rerun in AppTest: ~210 ms (upper bound; the live page
reruns only its fragment).

## Frame budget

Share of the frame interval used by step + overlay + one UI rerun:

| FPS | synthetic median / p99 | PokerNow 2x median / p99 |
| --- | --- | --- |
| 0.5 | 0.12 / 0.13 | 0.14 / 0.15 |
| 1 | 0.25 / 0.26 | 0.28 / 0.29 |
| 2 | 0.50 / 0.52 | 0.55 / 0.58 |
| 3 | 0.75 / 0.77 | 0.83 / 0.88 |

All keep up at p99; 3 FPS leaves little headroom on a slow machine, so the
default stays at 1 FPS.

## Soak (30 min, 3 FPS, recording on)

5,394 steps, 1 late step, 221 simulated hands, 1% capture errors and 1%
blank frames injected. RSS rises to ~320 MB in the first 90 s (template and
font caches, first recorded frames) and then stays flat: +4 MB over the
remaining 28 min. Retained frames reached the 2,000-frame limit at 15.5 min;
disk then stopped growing (46 MB). The Python object count grows by ~5k/h:
that is the tracker's event history, which is bounded (trimmed back to
2,000 once it exceeds 4,000 events) — the 1-hour soak (Phase 71) checks the
plateau.
