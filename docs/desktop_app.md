# Desktop overlay (`apps/desktop`) — standalone local build (unsigned)

Electron + TypeScript. A transparent, frameless, draggable,
always-on-top window rendering the engine's compact summary: top action
frequencies, the Solver gate badge, decision confidence; expanded state
adds the full mix, demo/manual inputs and the EV provenance note.

    cd apps/desktop
    npm install
    npm start          # build + launch
    npm run smoke      # headless: engine launch -> health -> shutdown
    npm run pack       # unsigned .app in release/ (electron-builder --dir)

## Engine integration (sidecar)

Resolution order in `src/sidecar.ts`:

1. attach to an engine already on `127.0.0.1:8765` (never duplicates);
2. the bundled frozen engine at `<Resources>/engine/pokeralpha-engine`
   (packaged app — no environment variables needed);
3. `POKERALPHA_ENGINE_BIN` (development override);
4. the repo's `.venv` Python running `python -m poker_alpha.api --port 0`
   (free port read from stdout).

Health (`/health` + API version) is checked before the overlay renders;
the owned process gets SIGTERM on quit (SIGKILL after 3 s).

## Failure states

Engine unavailable, API incompatible, invalid manual JSON and analysis
errors all render as plain-language panels with a log view (last 80
sidecar lines) — no stack traces at the user.

## Overlay behavior & boundaries

* Compact (300×190) / expanded (330×460) / hidden; `Cmd/Ctrl+Shift+O`
  toggles visibility and ALWAYS restores interaction.
* Optional click-through mode (`setIgnoreMouseEvents`) so the table stays
  usable; the bordered window remains visible and the shortcut restores
  interaction — there is **no stealth**: the app stays in the Dock.
* Analysis only: no clicking, betting, browser or client control, ever.
* Real-time use belongs in private / play-money / test environments where
  assistance is permitted. The live screen observer is NOT integrated in
  this preview (its real-table recognition accuracy is not yet
  independently validated — docs/observer_dataset.md); inputs are demo
  spots and manual states.

Packaging status: **standalone** unsigned `.app` (485 MB) with the frozen
engine embedded under `Contents/Resources/engine/` — verified outside the
repository with no Python, no env vars and no pre-running API (see
`packaging/MANIFEST.md`). Unsigned means Gatekeeper warns on first launch;
signed/notarized *distribution* is BLOCKED on Apple credentials, but the
standalone local bundle itself is not.
