# Web app (`apps/web`)

Next.js 15 + TypeScript (strict), static-exportable (`output: 'export'`),
no Tailwind/UI kit — one hand-written CSS file with the product's design
rules (dark, one accent `#2a78d6`, hairlines, semantic status colors
only, no gradients/marketing).

    cd apps/web
    npm install
    npm run dev        # http://localhost:3000
    npm run test       # vitest (semantics of the bundled data)
    npm run typecheck  # tsc --noEmit
    npm run build      # static export in out/ (~1.5 MB)

## Routes

`/` (what it is / what it is NOT) · `/analyze` · `/study` · `/strategy`
· `/trainer` · `/research` · `/benchmarks` · `/download`.

## Engine connection model

Every interactive page probes `http://127.0.0.1:8765/health` and shows an
explicit badge:

* **LOCAL ENGINE CONNECTED** — requests go to the local API;
* **ENGINE OFFLINE — SHOWING PRE-COMPUTED OUTPUT** — the page falls back
  to `src/data/demo.json`, which is recorded output of the real engine
  (5 analyzed demo spots, 3 preflop matrices, release info) and is
  labeled as pre-computed. Live computation is never faked.

Regenerate the bundled data by running the API and re-executing the
snippet in the git history of the commit that added it (or any script
hitting `/demo-spots`, `/analyze`, `/strategy/preflop`, `/release/info`).

Session uploads on `/study` go to the local API and nowhere else.

Screenshots (headless Chrome, real pages): `docs/assets/web_*.png`.
