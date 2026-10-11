# Product architecture

```
poker-alpha/
├── poker_alpha/            Python research + inference engine (authoritative)
│   ├── analysis/           session review, regret semantics, explorer,
│   │                       what-if, trainer
│   ├── api/                local product API (pokeralpha.api/v1, FastAPI)
│   ├── decision/           solver gate + rollout + heuristic cascade
│   ├── history/ session/   hand histories, recorded sessions
│   ├── observer/           screen recognition + dataset/ML tooling
│   └── ui/                 Streamlit app: Play / Study / Developer
├── cpp/                    native MCCFR training backend (bit-exact, 74.2x)
├── apps/
│   ├── web/                Next.js product frontend (static-exportable)
│   └── desktop/            Electron overlay preview + Python sidecar
├── packaging/              PyInstaller spec for the frozen engine
├── docs/  tests/  results/
```

## Rules that keep it coherent

1. **The Python engine is authoritative.** Frontends never reimplement
   poker logic; they render `pokeralpha.api/v1` responses (or, in
   Streamlit, the view models in `poker_alpha/ui/viewmodel.py`, which the
   API itself uses — one source of semantics).
2. **Provenance travels with every number.** Responses carry
   `solver_state` (gate), `source` (solver/rollout/heuristic),
   `ev_is_decision_basis`, gate reason codes and an EV provenance note;
   the UIs are required to respect them (tests pin this).
3. **Abstention is a first-class state.** Rejected lookup → no numbers,
   reasons + fallback instead, on every surface.
4. **Local-first.** The API binds 127.0.0.1 and refuses anything else;
   no accounts, no telemetry, no uploads. See docs/local_api.md and
   docs/privacy.md.

## Data formats (versioned)

| format | file |
|---|---|
| `pokeralpha.observed/v1` | manual/observed table state |
| `pokeralpha.hand/v1` | hand histories |
| `pokeralpha.api/v1` | API request/response contract |
| `pokeralpha.session_review/v1` | study exports |
| `pokeralpha.screenshot_annotation/v1` | observer ground truth |
| `pokeralpha.observer_dataset/v1` | dataset manifests/splits/reports |
| `pokeralpha.model/v1` | ML model provenance metadata |
