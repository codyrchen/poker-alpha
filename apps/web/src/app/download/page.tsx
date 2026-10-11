export default function DownloadPage() {
  return (
    <main>
      <h1>Download</h1>
      <p className="lead">
        PokerAlpha is local-first: the engine, the web app and the desktop
        overlay all run on your machine. There is no hosted service and
        nothing to sign up for.
      </p>

      <h2>Desktop overlay — development preview</h2>
      <p>
        An Electron shell with a compact, always-on-top strategy overlay. It
        is currently a <b>local development preview</b>: unsigned, built from
        source. There is no notarized installer yet (that requires Apple
        signing credentials).
      </p>
      <pre className="mono panel" style={{ whiteSpace: "pre-wrap" }}>
        {`git clone https://github.com/codyrchen/poker-alpha
cd poker-alpha
python -m venv .venv && .venv/bin/pip install -e ".[ui,vision,api]"
cd apps/desktop && npm install && npm start`}
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
