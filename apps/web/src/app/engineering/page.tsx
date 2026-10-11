/* eslint-disable @next/next/no-img-element */
// Engineering: the 74× story, the bit-exact story, the test matrix, and
// where each piece lives in the source tree.

import Link from "next/link";
import claims from "@/data/lab/claims.json";

const C = (claims as any).payload.claims;

export default function EngineeringPage() {
  return (
    <main>
      <h1>Engineering</h1>
      <p className="lead" style={{ maxWidth: "46rem" }}>
        The interesting part wasn&rsquo;t just the solver — it was keeping a
        {` ${C.native_speedup.value}×`} rewrite provably identical, and
        learning which trust signals were real.
      </p>

      <h2>Problem → decision → result: the {C.native_speedup.value}× backend</h2>
      <div className="pipeline">
        <div className="pipe-step">
          <b>PROBLEM</b>
          <p>
            Python trained at ~{C.python_it_per_s.value} it/s; a 2M-iteration
            experiment took a day. Profiling put ~60% of the time in hand
            evaluation and river percentile tables.
          </p>
        </div>
        <div className="pipe-step">
          <b>FAILED FIRST ATTEMPT</b>
          <p>
            Moving only helper functions native was measured and found
            insufficient — crossing the language boundary per node ate the
            gains. A real lesson, kept.
          </p>
        </div>
        <div className="pipe-step">
          <b>DECISION</b>
          <p>
            Move the ENTIRE hot path to C++17: rules, chance sampling,
            evaluator, feature tables, infoset encoding, regret matching,
            and the recursive traversal. Python crosses the pybind11
            boundary once per training chunk, GIL released. Infosets are
            packed 64-bit keys over fixed-size native nodes.
          </p>
        </div>
        <div className="pipe-step">
          <b>RESULT</b>
          <p>
            {C.python_it_per_s.value} → {C.native_it_per_s.value} it/s warm
            ({C.native_speedup.value}×, median of 3 trials). 2M-iteration ×
            3-seed training became a ~50-minute laptop job — which is what
            made the multi-seed uncertainty analysis affordable at all.
          </p>
        </div>
      </div>
      <figure className="fig" style={{ maxWidth: "560px" }}>
        <img src="/figures/native_speedup.png" alt="Measured throughput: Python reference vs native C++ MCCFR backend" />
        <figcaption>
          Committed benchmark; regenerate with experiments/make_readme_figures.py.
        </figcaption>
      </figure>

      <h2>Bit-exact across two languages</h2>
      <p style={{ maxWidth: "46rem" }}>
        Faster isn&rsquo;t useful if it silently becomes a different
        algorithm. Both backends consume a <b>shared random tape</b>; the
        parity suite requires identical trajectories, regret sums and
        average strategies — bitwise — across 1,000 training iterations,
        plus exact evaluator parity on all{" "}
        {C.evaluator_hands.value.toLocaleString()} five-card hands and 1M
        random 7-card hands, and bit-identical checkpoint resume.
      </p>
      <p style={{ maxWidth: "46rem" }}>
        It caught real bugs: clang&rsquo;s <b>FMA contraction</b> fused a
        multiply-add in betting arithmetic and changed results in the last
        ulp — harmless-looking, but through MCCFR&rsquo;s sampling it forked
        entire training trajectories. The build now requires{" "}
        <span className="mono">-ffp-contract=off</span>, and a
        platform-independent <span className="mono">fsum</span> dot product
        replaced BLAS dispatch for the same reason.
      </p>

      <h2>What breaks what: the test matrix</h2>
      <table className="data" style={{ maxWidth: "680px" }}>
        <tbody>
          {[
            ["Poker rules", "property/fuzz over engine-generated legal states + exhaustive reduced-variant enumeration"],
            ["Hand evaluator", "exhaustive 5-, 6- and 7-card parity (all 133,784,560 seven-card hands)"],
            ["Native backend", "shared-random-tape bitwise parity, ASan/UBSan clean"],
            ["Checkpoints", "train(a)+save+load+train(b) ≡ train(a+b), bitwise"],
            ["Confidence gate", "calibrated + held-out against exact-game EV regret"],
            ["Regret semantics", "paired-SE and exact-size invariants on every review"],
            ["API", "contract tests + fuzzed direct-vs-JSON parity"],
            ["Web", `vitest incl. browser-CFR-vs-Python fixture; strict tsc; static build`],
            ["Desktop", "sidecar smoke + packaged standalone smoke outside the repo"],
            ["Observer", "REAL held-out validation still BLOCKED on independent data — stated, not hidden"],
          ].map(([what, how]) => (
            <tr key={what as string}>
              <td style={{ whiteSpace: "nowrap" }}>{what}</td>
              <td className="small muted">{how}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small">
        {C.test_count.value} tests collected at this commit (counted by
        pytest when this page&rsquo;s data was generated).
      </p>

      <h2>Code map</h2>
      <table className="data" style={{ maxWidth: "680px" }}>
        <tbody>
          {[
            ["Hold'em engine", "poker_alpha/holdem/"],
            ["CFR / CFR+ / MCCFR", "poker_alpha/solvers/"],
            ["C++ training backend", "cpp/"],
            ["Abstraction + encoders", "poker_alpha/abstraction/"],
            ["Confidence gate", "poker_alpha/decision/solver_gate.py"],
            ["Ranges & opponent models", "poker_alpha/opponent/"],
            ["Analysis / study / red team", "poker_alpha/analysis/"],
            ["Agents & tournaments", "poker_alpha/agents/"],
            ["Local API", "poker_alpha/api/"],
            ["This site", "apps/web/ (data: scripts/build_web_research_data.py)"],
            ["Desktop overlay", "apps/desktop/"],
          ].map(([what, where]) => (
            <tr key={what as string}>
              <td>{what}</td>
              <td className="mono small">
                <a href={`https://github.com/codyrchen/poker-alpha/tree/main/${(where as string).split(" ")[0]}`}>
                  {where}
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>The local API, in one request</h2>
      <pre className="mono panel small" style={{ whiteSpace: "pre-wrap", maxWidth: "680px" }}>
        {`POST http://127.0.0.1:8765/analyze
{"state": {…pokeralpha.observed/v1…}}
→ { "recommended_tag": "HIGHEST FREQUENCY",
    "solver_state": "ACCEPTED",
    "actions": [{"label": "check", "frequency": 0.83}, …],
    "receipt": {"receipt_id": "PA-…", "strategy_artifact_sha256": "…"} }`}
      </pre>
      <p className="muted small">
        Binds 127.0.0.1 only; warm analysis ~0.1 s; every response can carry
        a reproducibility receipt.
      </p>

      <p>
        <Link className="btn" href="/evidence">Evidence &amp; validation ladder</Link>{" "}
        <Link className="btn" href="/benchmarks">Benchmarks</Link>{" "}
        <a className="btn" href="https://github.com/codyrchen/poker-alpha/blob/main/docs/native_solver.md">
          Native solver deep dive
        </a>
      </p>
    </main>
  );
}
