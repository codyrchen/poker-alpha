# PokerAlpha — demo scripts

Two scripts: the **product demo** (the shipped surfaces, 3–5 min) and the
original **research demo** (solver/validation story, 5 min). Neither needs
a live PokerNow game or any network beyond localhost.

## Product demo (3–5 minutes)

Setup once: `.venv/bin/python -m poker_alpha.api` in one terminal,
`cd apps/web && npm run dev` in another.

1. **Homepage** (http://localhost:3000, 20 s) — "High-performance
   imperfect-information game solving." Point at the two columns: what it
   is (C++17, 38.1×, 2M iterations, exact validation) and what it is NOT
   (not GTO — the engine says so itself).
2. **Analyzer** (/analyze, 40 s) — demo spot 1: a solver-ACCEPTED river
   mix. Read the row tag out loud: HIGHEST FREQUENCY, not "recommended" —
   it is a distribution. Switch to the gate-rejected preflop spot: the
   amber "Solver recommendation withheld" panel with the real reason
   ("independently trained runs disagree here") and the rollout fallback.
3. **Strategy explorer** (/strategy, 40 s) — the 169-hand matrix, live
   from the artifact. The honest picture IS the demo: most preflop cells
   are dimmed because the gate rejects them, and PokerAlpha shows no
   numbers there. Click a lit cell for the full mix + training visits.
4. **Session review** (Streamlit → Study, 60 s) —
   `streamlit run poker_alpha/ui/app.py`, Mode → Study. The bundled demo
   session: summary counts by gate state, ESTIMATED decision regret only
   where a valid same-method comparison exists ("EV regret unavailable"
   elsewhere, with the reason), review categories, click a row to get the
   full Play-Mode rendering of that decision.
5. **Trainer** (Study → Trainer, 30 s) — answer a spot, show the grading
   language: "Common in solver mix — 83%", never right/wrong; uncertain
   states are review-only.
6. **Desktop overlay** (30 s) — `cd apps/desktop && npm start`: the
   always-on-top compact summary, expanded state, click-through toggle,
   and the engine sidecar in the logs. The packaged app bundles the engine (standalone, unsigned).
7. **The engineering story** (30 s) — /benchmarks: 38.1× native backend
   with bit-exact parity; API answers a full analysis in ~0.1 s warm.
8. **Close on abstention** (20 s) — "Every surface obeys the same rule:
   when the evidence is weak, PokerAlpha says so instead of inventing a
   number. That discipline is the product."

## Research demo (5 minutes)

A live walkthrough using only offline, committed functionality. No PokerNow
game, no network, no setup beyond the quick start. Commands assume the repo
root with `.venv` activated and `pip install -e ".[dev]" && pip install ./cpp`
done once.

## 0. One-line framing (15 s)

> "PokerAlpha solves imperfect-information games, accelerates training 38×
> with a bitwise-verified C++ backend, and — the interesting part —
> calibrates *when not to trust its own strategy* against exactly solved
> games."

## 1. The research story, end to end (60 s)

```bash
python -m poker_alpha.demo
```

Deterministic, ~12 s. Narrates the original research: equilibrium baseline,
Bayesian opponent identification, exploitation under an exploitability
budget, the regime-change failure and its fix, and the performance work.
Point out: every printed number is computed live or read from a committed
CSV, and it says which.

## 2. The 38× native backend (60 s)

```bash
python experiments/validate_native_solver.py
```

~6 s, six checks: game/encoder parity, evaluator parity, feature parity,
**bitwise random-tape MCCFR parity**, exact checkpoint resume, artifact
round trip. Then show the headline:

- open `results/figures/native_speedup.png` (837 vs 22 it/s, 38.1×);
- mention: entire hot path in C++17, one pybind crossing per training
  chunk, GIL released, `-ffp-contract=off` because the parity suite caught
  FMA changing results in the last ulp.

If asked for a live taste (~30 s): a quick native training burst —

```bash
python - <<'EOF'
import time
from poker_alpha.native import NativeMCCFRSolver
from poker_alpha.solver_config import RELEASE_CONFIG
s = NativeMCCFRSolver(RELEASE_CONFIG, seed=0); s.train(5000)
t0 = time.perf_counter(); s.train(20000)
print(f"{20000/(time.perf_counter()-t0):,.0f} it/s warm")
EOF
```

## 3. A real Hold'em decision (45 s)

```bash
python -m poker_alpha.holdem_demo
```

A 6-max hand → `DecisionReport`: action frequencies, EV ± SE, and —
the point to dwell on — **provenance**: which path produced the advice
(solver / rollout / heuristic) and every warning attached to it.

## 4. The abstention layer (60 s)

Open `results/figures/confidence_risk_coverage.png`:

> "Three independently trained seeds disagree a lot on paper. We measured
> action EVs and ~94% of that disagreement is mixing between near-equal
> actions. So the gate is calibrated against *exact-game EV regret* —
> 8,880 infosets from nine exactly solved games, held-out — and the v2
> rule gets ~1.7× the coverage of v1 at the same accepted risk. Flop and
> turn stay capped at low confidence because their abstraction error is
> measured, and no amount of stable training erases it."

Show the live gate if time allows:

```bash
python -m poker_alpha.platform_demo
```

(watch for `SOLVER_ACCEPT` / `SOLVER_REJECT` reason codes in the output).

## 5. Reproducibility and honesty (45 s)

- `python experiments/make_readme_figures.py` — the README figures rebuild
  from committed JSONs; nothing is hand-entered.
- `docs/solver_status_final.md` — one authoritative status file: what is
  PASS, what is measured-with-caveats, what is experimental.
- The honest headline: a validated lower bound says the strategy is still
  ≥ ~100 bb/100 exploitable in the full game — stated in the README's
  "At a glance" table, not buried.

## 6. Optional: the observer UI (30 s, only if asked)

```bash
./scripts/run_live_observer.sh
```

Read-only screen observer (calibration overlay, test-session recorder).
Say explicitly: **validated on synthetic frames only — real PokerNow
accuracy is blocked on independent held-out data**, and nothing in the
system clicks or acts.

## Close (15 s)

> "Everything you just saw traces to a committed result file, the C++ and
> Python solvers are provably the same algorithm, and the system knows —
> quantitatively — when to abstain. The limitations are measured, which is
> what makes the rest of the numbers trustworthy."
