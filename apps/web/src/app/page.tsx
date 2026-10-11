import Link from "next/link";
import demo from "@/data/demo.json";

export default function Home() {
  const rel = (demo as any).release;
  return (
    <main>
      <h1>High-performance imperfect-information game solving</h1>
      <p className="lead" style={{ marginTop: "0.6rem" }}>
        PokerAlpha is a research engine for heads-up no-limit Hold&nbsp;em:
        external-sampling MCCFR with a bit-exact C++17 backend, direct exact
        validation against known solutions, and a confidence system that makes
        the solver <b>refuse to answer</b> when its own evidence is weak.
      </p>

      <div style={{ display: "flex", gap: "0.7rem", margin: "1.4rem 0", flexWrap: "wrap" }}>
        <Link className="btn primary" href="/analyze">
          Try the analyzer
        </Link>
        <Link className="btn" href="/strategy">
          Explore the strategy
        </Link>
        <a className="btn" href="https://github.com/codyrchen/poker-alpha">
          GitHub
        </a>
        <Link className="btn" href="/download">
          Desktop (local preview)
        </Link>
      </div>

      <hr className="hair" />

      <div className="grid2">
        <div>
          <h2>What it is</h2>
          <table className="data">
            <tbody>
              <tr>
                <td>Training backend</td>
                <td className="mono">C++17, bit-exact vs the Python reference</td>
              </tr>
              <tr>
                <td>Measured speedup</td>
                <td className="mono">38.1× (committed benchmark)</td>
              </tr>
              <tr>
                <td>Release strategy</td>
                <td className="mono">{rel.iterations.toLocaleString()} MCCFR iterations</td>
              </tr>
              <tr>
                <td>Validation</td>
                <td className="mono">exact CFR on Kuhn/Leduc + exact HU subgames</td>
              </tr>
              <tr>
                <td>Abstention</td>
                <td className="mono">per-state confidence gate (visits, seeds, movement)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          <h2>What it is not</h2>
          <p className="muted">
            Not GTO. The release strategy is an MCCFR average strategy on an
            imperfect-recall abstraction — measurably exploitable in the full
            game, and the engine says so. The product&rsquo;s defining feature is
            that it distinguishes what is <b>known</b>, <b>estimated</b>,{" "}
            <b>uncertain</b>, and <b>unsupported</b> — and withholds advice in
            the last two cases instead of bluffing confidence.
          </p>
        </div>
      </div>

      <h2>Surfaces</h2>
      <table className="data">
        <thead>
          <tr>
            <th>Surface</th>
            <th>What it does</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Link href="/analyze">Analyzer</Link>
            </td>
            <td>One decision at a time: distribution, gate state, why</td>
          </tr>
          <tr>
            <td>
              <Link href="/study">Study</Link>
            </td>
            <td>Session review with strict regret semantics and patterns</td>
          </tr>
          <tr>
            <td>
              <Link href="/strategy">Strategy explorer</Link>
            </td>
            <td>Live 169-hand preflop matrix from the trained artifact</td>
          </tr>
          <tr>
            <td>
              <Link href="/trainer">Trainer</Link>
            </td>
            <td>Quiz spots graded by solver-mix frequency, never right/wrong</td>
          </tr>
          <tr>
            <td>
              <Link href="/download">Desktop overlay</Link>
            </td>
            <td>Compact always-on-top summary (local development preview)</td>
          </tr>
        </tbody>
      </table>

      <p className="muted small" style={{ marginTop: "1.6rem" }}>
        Everything runs locally against a 127.0.0.1 Python engine; this site
        carries no accounts, no analytics and no uploads.
      </p>
    </main>
  );
}
