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

## Standalone desktop app (electron-builder, --dir, engine embedded)

Build (frozen engine FIRST, then the shell):

    cd packaging && ../.venv/bin/pyinstaller --noconfirm pokeralpha_engine.spec
    cd ../apps/desktop && npm run pack

Output: `apps/desktop/release/mac-arm64/PokerAlpha.app` — **485 MB**
(Electron framework ~233 MB + the frozen engine ~227 MB under
`Contents/Resources/engine/` via electron-builder extraResources).
Signing explicitly disabled (`identity: null`): the app is a fully
STANDALONE local build, but unsigned — Gatekeeper will warn, and
trustworthy public distribution still requires signing/notarization.

Engine resolution at runtime:
1. an engine already running on 127.0.0.1:8765 (attach, never duplicate);
2. the BUNDLED engine at `<Resources>/engine/pokeralpha-engine`
   (packaged app; no environment variables, no repo, no Python needed);
3. `POKERALPHA_ENGINE_BIN` (development override);
4. the repo's `.venv` Python running `python -m poker_alpha.api`.

Standalone verification (performed on this build): the .app copied
OUTSIDE the repository, with no venv, no env vars and no pre-running
API — bundled engine launches on a free port, /health reports the solver
loaded, a solver-ACCEPTED demo analysis renders in the overlay, and quit
terminates the owned engine (0 processes left).

## Not packaged

- signed/notarized builds (Apple credentials — external blocker);
- Windows/Linux builds (no machines/credentials here).
