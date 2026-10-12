import type { Metadata } from "next";
/* eslint-disable @next/next/no-img-element */
// Engineering: the native speedup story, the bit-exact story, the test matrix, and
// where each piece lives in the source tree.

import Link from "next/link";
import claims from "@/data/lab/claims.json";

export const metadata: Metadata = {
  title: "Engineering",
  description: "How the C++ training backend was built, profiled and kept bit-identical to the Python reference.",
};

const C = (claims as any).payload.claims;

export default function EngineeringPage() {
  return (
    <main>
      <h1>Engineering</h1>
      <p className="lead" style={{ maxWidth: "46rem" }}>
        The interesting part was not just the solver. It was keeping a C++ training backend
        provably identical to the Python reference while making it {C.native_speedup.value}× faster,
        and learning which trust signals were real.
      </p>

      <h2>Problem → decision → result: the {C.native_speedup.value}× backend</h2>
      <div className="pipeline">
        <div className="pipe-step">
          <b>PROBLEM</b>
          <p>
            Python trained at ~{C.python_it_per_s.value} it/s, so one 2M-iteration run took more
            than a day. Profiling put ~60% of the time in hand evaluation and river percentile
            tables.
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
          <b>PORT RESULT</b>
          <p>
            ~{Math.round(C.python_it_per_s.value)} → {Math.round(C.native_it_per_s_phase1.value)} it/s warm
            ({C.native_speedup_phase1.value}×, median of 3 trials; historical benchmark). Multi-seed,
            2M-iteration training became practical on a laptop, which is what made the uncertainty
            analysis affordable at all.
          </p>
        </div>
        <div className="pipe-step">
          <b>PROFILE AGAIN</b>
          <p>
            A sampling profile of the native trainer found 69.5% of CPU rebuilding river percentile
            tables. Deduplicating that work, reusing evaluator board counts and a flat
            open-addressing infoset index gave a further 1.96× in an A/B test between native
            builds, with an identical strategy digest.
          </p>
        </div>
        <div className="pipe-step">
          <b>CURRENT</b>
          <p>
            {C.python_it_per_s.value} → {Math.round(C.native_it_per_s.value).toLocaleString("en-US")} it/s
            warm ({C.native_speedup.value}×, fresh benchmark, median of 3 trials). This is MCCFR
            training throughput on the release configuration, not end-user analysis latency.
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
        average strategies, bitwise. CI runs 1,000 iterations; a stress run held for{" "}
        {C.parity_iterations.value.toLocaleString("en-US")}. The evaluator matches on every five-,
        six- and seven-card hand ({C.evaluator_hands_7card.value.toLocaleString("en-US")} seven-card
        hands), and checkpoint resume is bit-identical (100k continuous equals 10 × 10k).
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
