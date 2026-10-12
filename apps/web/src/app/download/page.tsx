import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Run locally",
  description: "Run the PokerAlpha engine and app on your own machine.",
};

export default function DownloadPage() {
  return (
    <main>
      <h1>Download</h1>
      <p className="lead">
        PokerAlpha is local-first: the engine, the web app and the desktop
        overlay all run on your machine. There is no hosted service and
        nothing to sign up for.
      </p>

      <h2>Desktop overlay — standalone local build (unsigned)</h2>
      <p>
        An Electron shell with a compact, always-on-top strategy overlay and
        the <b>engine bundled inside the app</b> — no Python, no environment
        variables. Built from source it is fully standalone (~540 MB), but
        <b> unsigned</b>: macOS Gatekeeper will warn on first launch, and
        there is no notarized installer yet (that requires Apple signing
        credentials).
      </p>
      <pre className="mono panel" style={{ whiteSpace: "pre-wrap" }}>
        {`git clone https://github.com/codyrchen/poker-alpha
cd poker-alpha
python -m venv .venv && .venv/bin/pip install -e ".[ui,vision,api]" pyinstaller
# freeze the engine, then build the standalone app
(cd packaging && ../.venv/bin/pyinstaller --noconfirm pokeralpha_engine.spec)
cd apps/desktop && npm install && npm run pack   # release/mac-arm64/PokerAlpha.app
npm start                                        # or run from source`}
      </pre>

      <h2>Local engine + web app</h2>
      <pre className="mono panel" style={{ whiteSpace: "pre-wrap" }}>
        {`# engine (binds 127.0.0.1 only)
.venv/bin/python -m poker_alpha.api

# full Streamlit app: Play / Study / Developer
.venv/bin/streamlit run poker_alpha/ui/app.py

# this website, locally
cd apps/web && npm install && npm run dev`}
      </pre>

      <h2>Three deployment shapes</h2>
      <ul style={{ marginLeft: "1.2rem" }}>
        <li>
          <b>Public static site</b> (if this page is hosted): research,
          benchmarks and pre-computed demo outputs only — it cannot reach a
          local engine from a public origin.
        </li>
        <li>
          <b>Local web app + local engine</b>: the full analyzer, strategy
          explorer and study upload against 127.0.0.1.
        </li>
        <li>
          <b>Desktop app</b>: the engine bundled inside, no setup.
        </li>
      </ul>

      <h2>Boundaries</h2>
      <ul style={{ marginLeft: "1.2rem" }}>
        <li>Analysis only: no automated clicking, betting or client control.</li>
        <li>
          Real-time use only in private / play-money / test environments where
          assistance is permitted; otherwise use post-session review.
        </li>
        <li>Hands, screenshots and sessions stay on your machine by default.</li>
      </ul>
    </main>
  );
}
