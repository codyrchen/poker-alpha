# Local product API (`pokeralpha.api/v1`)

    python -m poker_alpha.api              # 127.0.0.1:8765
    python -m poker_alpha.api --port 0     # free port, announced as
                                           # POKERALPHA_API_PORT=<n> on stdout

FastAPI app in `poker_alpha/api/server.py`; typed contract in
`poker_alpha/api/schemas.py`. The release strategy artifact loads once per
process (first request ~1.6 s cold; warm `/analyze` ≈ 0.10 s at default
sampling).

## Endpoints

| endpoint | purpose |
|---|---|
| `GET /health` | liveness + `solver_loaded` |
| `GET /version` | API + app version |
| `GET /release/info` | artifact path/SHA-256, iterations, config signature, honest disclaimer |
| `GET /demo-spots` | the five verified demo states |
| `POST /analyze` | one `pokeralpha.observed/v1` state → full decision response |
| `POST /what-if` | single-variable counterfactual comparison |
| `POST /session/review` | `pokeralpha.hand/v1` → `pokeralpha.session_review/v1` (≤ 500 hands) |
| `GET /strategy/preflop[?node=…]` | live 169-cell matrix from the artifact |
| `GET /strategy/preflop/{hand}` | one canonical hand class |

Response semantics mirror Play Mode: `solver_state`
(ACCEPTED / LOW CONFIDENCE / REJECTED / OFF), `recommended_tag`
(HIGHEST FREQUENCY for solver mixes), `ev_is_decision_basis`, abstention
and refusal objects, EV provenance note for solver mixes.

## Security / privacy model (v0.1)

Threat assumptions: single-user machine, local frontends, no hostile
network peers (the server is unreachable off-machine).

* Binds **127.0.0.1 only**; `python -m poker_alpha.api` refuses any other
  host.
* CORS allows only explicit local dev origins (`localhost:3000`,
  `localhost:8501`, and the literal `null` origin of the Electron
  `file://` renderer). The loopback bind is the boundary; CORS is
  defense-in-depth for browser tabs.
* No training endpoints, no filesystem paths in any request, no shell,
  no analytics/telemetry, no remote uploads. Request bodies are poker
  states and hand histories only.
* No authentication — acceptable ONLY because of the loopback bind.
  **Public deployment would require auth, rate limiting, request size
  limits and input hardening that v0.1 does not claim.**
