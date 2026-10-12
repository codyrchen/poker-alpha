import type { Metadata } from "next";
/* eslint-disable @next/next/no-img-element */

export const metadata: Metadata = {
  title: "Benchmarks",
  description: "Native training throughput and confidence-gate calibration figures.",
};

export default function BenchmarksPage() {
  return (
    <main>
      <h1>Benchmarks</h1>
      <p className="lead">
        Committed, reproducible measurements — the figures below are generated
        by scripts in the repository from recorded results, not illustrations.
      </p>

      <div className="grid2" style={{ marginTop: "1.4rem" }}>
        <figure className="fig">
          <img src="/figures/native_speedup.png" alt="Native C++ backend speedup vs Python reference" />
          <figcaption>
            Native C++17 training backend vs the Python reference:{" "}
            <b>74.2×</b> warm training throughput on the release configuration, with
            bit-exact regret/strategy tapes.
          </figcaption>
        </figure>
        <figure className="fig">
          <img src="/figures/confidence_risk_coverage.png" alt="Confidence gate risk/coverage" />
          <figcaption>
            Solver confidence gate risk/coverage: tightening the gate trades
            coverage for lower disagreement risk on held-out checkpoints.
          </figcaption>
        </figure>
      </div>

      <h2>Local product performance</h2>
      <table className="data">
        <tbody>
          <tr>
            <td>API cold start → /health</td>
            <td className="mono">~1.6 s</td>
          </tr>
          <tr>
            <td>Warm /analyze (2000 equity / 1000 rollout sims)</td>
            <td className="mono">~0.10 s</td>
          </tr>
          <tr>
            <td>Full 169-hand preflop matrix (169 artifact lookups)</td>
            <td className="mono">~1.0 s</td>
          </tr>
        </tbody>
      </table>
      <p className="muted small">
        Measured on an Apple-silicon laptop against the committed 2M release
        artifact; your numbers will vary with hardware and settings.
      </p>

      <p>
        Reproduce everything:{" "}
        <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/native_solver_benchmark.md">
          benchmark methodology
        </a>{" "}
        ·{" "}
        <a href="https://github.com/codyrchen/poker-alpha/tree/main/results">
          recorded results
        </a>
      </p>
    </main>
  );
}
