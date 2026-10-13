import type { Metadata } from "next";
// One-page project overview. Print-friendly.

import Link from "next/link";
import claims from "@/data/lab/claims.json";

export const metadata: Metadata = {
  title: "One-page overview",
  description: "PokerAlpha on one page: what was built, the hardest problems and the headline numbers.",
};

const C = (claims as any).payload.claims;

export default function OverviewPage() {
  return (
    <main>
      <h1>PokerAlpha — one-page overview</h1>
      <p className="lead" style={{ maxWidth: "46rem" }}>
        An end-to-end research platform for decision-making under hidden
        information, built independently by Cody Chen: a custom Hold&rsquo;em
        engine, game-theoretic solvers, a {C.native_speedup.value}× C++
        training backend, an exact-game validation suite, a calibrated
        uncertainty layer, and a local analysis product.
      </p>

      <h2>What was built (all implemented in this project)</h2>
      <ul style={{ marginLeft: "1.2rem", maxWidth: "46rem" }}>
        <li><b>Game engine</b> — full HU/6-max Hold&rsquo;em rules: side pots, min-raise reopening, all-in semantics; property-fuzzed and exhaustively checked on reduced variants.</li>
        <li><b>Solvers</b> — CFR, CFR+, external-sampling MCCFR on a generic extensive-form Game interface (also runs Kuhn, Leduc, and a non-poker auction game).</li>
        <li><b>Native backend</b> — the complete training hot path in C++17/pybind11; bit-exact against Python on a shared random tape.</li>
        <li><b>Validation system</b> — nine-rung ladder from exact Kuhn equilibria to an honest restricted-LBR exploitability floor.</li>
        <li><b>Confidence system</b> — per-state accept/low/reject gate calibrated on {C.calibration_rows.value.toLocaleString()} exact EV-regret targets; the product refuses rather than guess.</li>
        <li><b>Analysis engine</b> — weighted ranges, Monte-Carlo rollouts with paired uncertainty, session review with strict regret semantics, counterfactual and sensitivity analysis, reproducibility receipts.</li>
        <li><b>Product</b> — local FastAPI engine, Next.js web app, standalone Electron desktop app with the engine bundled inside (unsigned local build).</li>
      </ul>

      <h2>Hardest problems</h2>
      <ol style={{ marginLeft: "1.2rem", maxWidth: "46rem" }}>
        <li>Keeping a 74×-faster C++ training backend <i>provably the same algorithm</i> — down to disabling FMA contraction after the parity suite caught a last-ulp divergence.</li>
        <li>Discovering that policy movement does NOT predict strategy error, and rebuilding the confidence gate around signals that do (visits, cross-seed disagreement).</li>
        <li>Honest regret semantics: rollout EVs share common random numbers, so uncertainty must be the paired-difference SE; off-menu bet sizes are re-evaluated exactly rather than nearest-matched.</li>
      </ol>

      <h2>Headline numbers (traceable on /evidence)</h2>
      <div className="metric-grid">
        {[
          [`${C.native_speedup.value}×`, "C++ training speedup, measured"],
          [`${(C.release_iterations.value / 1e6).toFixed(0)}M`, "training iterations, 3 independent seeds"],
          [C.evaluator_hands.value.toLocaleString(), "5-card hands checked for evaluator parity"],
          [C.calibration_rows.value.toLocaleString(), "exact EV-regret calibration states"],
          [`${C.coverage_gain.value}×`, "confidence coverage at equal accepted risk"],
          [`${C.test_count.value}`, "tests at this commit"],
        ].map(([n, what]) => (
          <div key={what as string} className="metric">
            <div className="n">{n}</div>
            <div className="what">{what}</div>
          </div>
        ))}
      </div>

      <h2>Architecture in one line</h2>
      <p className="mono small" style={{ maxWidth: "46rem" }}>
        state → validation → modelled ranges → abstraction lookup →
        confidence gate → solver | rollout | refuse → decision report (+
        receipt)
      </p>

      <h2>Research contribution</h2>
      <p style={{ maxWidth: "46rem" }}>
        The project&rsquo;s central result is a <b>calibrated abstention
        layer</b> for solver output: confidence signals scored against exact
        EV-regret ground truth from solvable games, including the negative
        finding that checkpoint movement is not a trust signal. Supporting
        studies: measured abstraction error via exact subgames, multi-seed
        policy-vs-value disagreement analysis, adversarial failure search,
        and an opponent-adaptation study showing stationary Bayesian models
        fail under regime change.
      </p>

      <h2>Stack</h2>
      <p className="small muted" style={{ maxWidth: "46rem" }}>
        Implemented here: engine, solvers, abstraction, ranges, gate,
        rollouts, replay, receipts, dataset tooling, benchmark harnesses.
        Foundational tools: Python, NumPy, C++17, pybind11, CMake, FastAPI,
        Next.js/React, Electron, Streamlit, pytest, GitHub Actions.
      </p>

      <p>
        <Link className="btn primary" href="/tour">90-second tour</Link>{" "}
        <Link className="btn" href="/lab">Research Lab</Link>{" "}
        <a className="btn" href="https://github.com/codyrchen/poker-alpha">Source</a>
      </p>
      <p className="muted small">
        PokerAlpha is an independent research and engineering project by
        Cody Chen. The Hold&rsquo;em strategy is experimental — an abstract
        MCCFR strategy, measurably exploitable, never claimed to be GTO.
      </p>
    </main>
  );
}
