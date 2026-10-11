import Link from "next/link";
import claims from "@/data/lab/claims.json";
import oneHand from "@/data/lab/one_hand.json";
import { MixBar } from "@/components/Lab";

const C = (claims as any).payload.claims;
const H = (oneHand as any).payload;

export default function Home() {
  return (
    <main>
      <section className="hero">
        <h1>I built an end-to-end poker AI research platform.</h1>
        <p className="sub">
          A custom Hold&nbsp;em engine, game-theoretic solvers that learn by
          self-play, a {C.native_speedup.value}× C++ training backend proven
          bit-identical to the Python reference, an exact-game validation
          system, a calibrated uncertainty layer that refuses weak answers —
          and the web and desktop product around it.
        </p>
        <div style={{ display: "flex", gap: "0.7rem", margin: "1.5rem 0", flexWrap: "wrap" }}>
          <Link className="btn primary" href="/tour">90-second tour</Link>
          <Link className="btn" href="/learn">Watch it learn</Link>
          <Link className="btn" href="/arena">Play against it</Link>
          <Link className="btn" href="/lab">Research Lab</Link>
          <a className="btn" href="https://github.com/codyrchen/poker-alpha">Source code</a>
        </div>
        <p className="muted small">
          Not an LLM wrapper, not a poker-site bot, not claimed GTO — the
          solver, engine, backend and validation are all implemented here,
          and the limitations are measured rather than hidden.
        </p>
      </section>

      {/* hero visual: a real decision through the real pipeline */}
      <section>
        <h2>One real decision, end to end</h2>
        <div className="pipeline" style={{ maxWidth: "46rem" }}>
          <div className="pipe-step">
            <b>HAND STATE</b>
            <p className="mono small">
              A♥K♦ on Q♠7♦2♣T♥3♥ · pot {H.state.pot}bb · checked to the hero
              on the river
            </p>
          </div>
          <div className="pipe-step">
            <b>OPPONENT BELIEF</b>
            <p>
              modelled range after three passive streets:{" "}
              {H.range_stages.at(-1).entropy_bits} bits, ≈
              {H.range_stages.at(-1).effective_combos} effective combos{" "}
              <span className="muted small">(a belief model, not their cards)</span>
            </p>
          </div>
          <div className="pipe-step">
            <b>TRUST CHECK</b>
            <p>
              gate verdict <span className="badge good">ACCEPTED</span> —{" "}
              {Math.round(H.visits).toLocaleString()} training visits, low
              cross-seed disagreement
            </p>
          </div>
          <div className="pipe-step">
            <b>LEARNED STRATEGY</b>
            <div style={{ maxWidth: "380px" }}>
              <MixBar mix={H.mix} height={16} />
              <p className="mono small" style={{ marginTop: "0.3rem" }}>
                {Object.entries(H.mix as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([a, p]) => `${a} ${(p * 100).toFixed(0)}%`)
                  .join(" · ")}
              </p>
            </div>
          </div>
        </div>
        <p className="small">
          <span className="badge mute">PRECOMPUTED</span>{" "}
          <span className="muted">
            real engine output, stamped with the generating commit —
          </span>{" "}
          <Link href="/replay">walk through the whole hand</Link>
          <span className="muted">, or recompute it live with the </span>
          <Link href="/analyze">local analyzer</Link>.
        </p>
      </section>

      <section>
        <h2>What I built</h2>
        <div className="pipeline">
          {[
            ["GAME ENGINE", "Implemented the rules and state transitions to simulate Hold'em exactly — side pots, min-raise reopening, all-ins.", "poker_alpha/holdem"],
            ["SOLVER", "Implemented game-theoretic learning (CFR, CFR+, external-sampling MCCFR) that learns strategy through self-play.", "poker_alpha/solvers"],
            ["NATIVE TRAINER", `Moved the complete training hot path into C++17 — measured ${C.native_speedup.value}× and proven bit-identical.`, "cpp"],
            ["VALIDATION SYSTEM", "Checked everything against games where the correct answer can be computed exactly.", "docs/solver_validation.md"],
            ["CONFIDENCE SYSTEM", "Measured when the learned strategy should NOT be trusted, and made the product refuse there.", "poker_alpha/decision/solver_gate.py"],
            ["ANALYSIS ENGINE", "Ranges, rollouts with paired uncertainty, replay, session review, counterfactuals, reproducibility receipts.", "poker_alpha/analysis"],
            ["PRODUCT", "Local API + this web app + a standalone desktop overlay with the engine bundled inside.", "apps"],
          ].map(([name, text, path]) => (
            <div key={name as string} className="pipe-step">
              <b>{name}</b>
              <p>
                {text}{" "}
                <a className="small" href={`https://github.com/codyrchen/poker-alpha/tree/main/${path}`}>
                  source
                </a>
              </p>
            </div>
          ))}
        </div>
        <p className="muted small" style={{ maxWidth: "44rem" }}>
          Foundational tools used (not claimed as custom): Python, NumPy,
          C++17, pybind11, CMake, FastAPI, Next.js/React, Electron,
          Streamlit, pytest, GitHub Actions.
        </p>
      </section>

      <section>
        <h2>Numbers that mean something</h2>
        <div className="metric-grid">
          {[
            [`${C.native_speedup.value}×`, "faster training — made million-iteration experiments practical", "native_speedup"],
            [`${(C.release_iterations.value / 1e6).toFixed(0)}M`, "iterations behind the release strategy, ×3 independent seeds", "release_iterations"],
            [C.evaluator_hands.value.toLocaleString(), "five-card hands — every one checked for evaluator parity", "evaluator_hands"],
            [C.calibration_rows.value.toLocaleString(), "states with EXACT regret targets calibrating the trust gate", "calibration_rows"],
            [`${C.coverage_gain.value}×`, "more accepted decisions at equal measured risk (confidence v2)", "coverage_gain"],
            [`${C.test_count.value}`, "tests at this commit — counted, not quoted", "test_count"],
          ].map(([n, what]) => (
            <div key={what as string} className="metric">
              <div className="n">{n}</div>
              <div className="what">{what}</div>
              <Link href="/evidence" className="small">evidence →</Link>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>Refusing is a feature</h2>
        <div className="grid2">
          <div>
            <h3 style={{ marginTop: 0 }}>Trusted state</h3>
            <p className="muted small">river value spot, 71k visits, seeds agree</p>
            <MixBar mix={H.mix} height={14} />
            <p className="mono small">check 83% · bet mixes behind it</p>
          </div>
          <div>
            <h3 style={{ marginTop: 0 }}>Untrusted state</h3>
            <p className="muted small">preflop open, independently trained seeds disagree</p>
            <p>
              <span className="badge mute">REJECTED</span>{" "}
              <span className="small muted">
                HIGH_SEED_DISAGREEMENT → strategy withheld; a clearly-labeled
                rollout answers instead
              </span>
            </p>
          </div>
        </div>
        <p className="small">
          <Link href="/trust">The full Trust Map →</Link>
        </p>
      </section>

      <section>
        <h2>Why this is hard</h2>
        <p style={{ maxWidth: "46rem" }}>
          <b>Hidden information</b> — the decisive facts are unknowable.{" "}
          <b>Scale</b> — ~10<sup>160</sup> states force lossy compression
          whose error must itself be measured. <b>Randomization</b> — correct
          play is a probability distribution, so there is no single answer to
          supervise against. <b>Stochastic training</b> — independent runs
          disagree, and telling harmless mixing from real uncertainty is a
          research problem. <b>Performance</b> — none of it is testable
          without a 38× faster loop. <b>Trust</b> — a confident-looking
          policy table says nothing about reliability.
        </p>
      </section>

      <section>
        <h2>Explore by who you are</h2>
        <table className="data" style={{ maxWidth: "620px" }}>
          <tbody>
            {[
              ["Recruiter", "/tour", "the 90-second tour"],
              ["Poker player", "/arena", "play it, then see your fingerprint"],
              ["Student", "/learn", "watch CFR converge live"],
              ["Engineer", "/engineering", "the 38× + bit-exact story"],
              ["Quant / researcher", "/lab", "the research lab + evidence"],
              ["AI researcher", "/agents", "agent benchmarks + LLM harness"],
            ].map(([who, href, what]) => (
              <tr key={who as string}>
                <td>{who}</td>
                <td>
                  <Link href={href as string}>{what}</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <p className="muted small" style={{ marginTop: "2rem", maxWidth: "46rem" }}>
        PokerAlpha is an independent research and engineering project by Cody
        Chen. The goal became larger than writing a poker solver: build the
        algorithm, make it fast, determine when it fails, and turn it into
        something people can inspect.
      </p>
    </main>
  );
}
