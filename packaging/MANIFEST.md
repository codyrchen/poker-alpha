# Packaging manifest — PokerAlpha local development packages

Both packages are UNSIGNED local builds for development/preview. No Apple
developer certificate is used; distribution would require signing and
notarization (BLOCKED on credentials).

## Frozen engine (PyInstaller, onedir)

Build:

    cd packaging && ../.venv/bin/pyinstaller --noconfirm pokeralpha_engine.spec

Output: `packaging/dist/pokeralpha-engine/` — **227 MB**

Included (inference-time only):

| item | why |
|---|---|
| `pokeralpha-engine` binary + Python runtime | runs `poker_alpha.api` |
| `results/strategy/holdem_v2_native_seed0_2m.npz` | the release strategy |
| `results/strategy/holdem_v2_native_seed0_2m_confidence.npz` | SolverConfidence v2 table (gate) |
| `poker_alpha/data/*.json` | preflop priors + bundled demo session |
| numpy / scipy / pandas / fastapi / uvicorn | runtime deps |

Excluded on purpose: matplotlib, streamlit, PIL/mss/pytesseract (vision),
training checkpoints, raw sessions/screenshots, tests, tooling.

Verified: `./dist/pokeralpha-engine/pokeralpha-engine --port 8941` serves
`/health` (solver_loaded: true) and a solver-ACCEPTED `/analyze` using the
bundled artifact. First launch of the unsigned binary took ~26 s on macOS
(Gatekeeper verification of the unsigned onedir); warm launches are a few
seconds.

## Desktop shell (electron-builder, --dir)

Build:

    cd apps/desktop && npm run pack

Output: `apps/desktop/release/mac-arm64/PokerAlpha.app` — **233 MB**
(essentially the Electron framework; the app payload is <1 MB).
Signing explicitly disabled (`identity: null`).

The shell finds its engine in this order:
1. an engine already running on 127.0.0.1:8765 (attach, never duplicate);
2. `POKERALPHA_ENGINE_BIN` → the frozen engine above;
3. the repo's `.venv` Python running `python -m poker_alpha.api`.

## Not packaged

- a single double-clickable .app embedding the frozen engine (needs code
  signing to be distributable, and an installer story — out of scope for
  the unsigned preview);
- Windows/Linux builds (no machines/credentials here).
