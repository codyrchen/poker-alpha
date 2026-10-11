// Evidence: every headline number, its meaning, method and source.

import claims from "@/data/lab/claims.json";
import { ProvenanceFooter, Question } from "@/components/Lab";

const env = claims as any;
const C = env.payload.claims;

const MEANING: Record<string, [string, string]> = {
  native_speedup: [
    "Native C++17 training backend vs the Python reference",
    "same algorithm, bit-exact on a shared random tape; warm-throughput median of 3 trials",
  ],
  release_iterations: [
    "Training scale of the promoted strategy",
    "external-sampling MCCFR on the locked v2 abstraction; exact checkpoint resume",
  ],
  training_seeds: [
    "Independent training runs",
    "training uncertainty is measured across seeds instead of assumed away",
  ],
  evaluator_hands: [
    "Exhaustive hand-evaluator parity",
    "every possible 5-, 6- and 7-card hand compared between Python and C++ (133,784,560 seven-card hands)",
  ],
  calibration_rows: [
    "Confidence calibration states",
    "infosets from nine exactly solved reduced games with TRUE EV-regret targets, leave-one-game-out",
  ],
  coverage_gain: [
    "Coverage gained by confidence v2 at equal accepted risk",
    "the promoted rule (seed disagreement < 0.2, recent movement < 0.1, visits ≥ 20) vs the v1 gate, held-out games",
  ],
  test_count: [
    "Tests collected at this commit",
    "counted by pytest --collect-only when the site data was generated — never a stale number",
  ],
  stored_infosets: [
    "Information sets in the release strategy",
    "the strategy artifact's key table",
  ],
  native_it_per_s: ["Native training throughput", "warm median"],
  python_it_per_s: ["Python reference throughput", "warm median"],
};

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toLocaleString();
  return String(v);
}

export default function EvidencePage() {
  return (
    <main>
      <h1>Evidence</h1>
      <Question>
        Every headline number on this site, with what it means, how it was
        measured, and where the artifact lives. The site reads these from a
        machine-checked claims registry — numbers cannot silently go stale.
      </Question>
      <table className="data">
        <thead>
          <tr>
            <th>claim</th>
            <th>value</th>
            <th>meaning &amp; method</th>
            <th>source</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(C).map(([key, c]: [string, any]) => (
            <tr key={key}>
              <td className="mono small">{key}</td>
              <td className="mono" style={{ whiteSpace: "nowrap" }}>
                {fmt(c.value)}
                {c.units === "x" ? "×" : ` ${c.units}`}
              </td>
              <td className="small" style={{ maxWidth: "22rem" }}>
                {MEANING[key]?.[0] ?? ""}
                <div className="muted">{c.qualifier ?? MEANING[key]?.[1] ?? ""}</div>
              </td>
              <td className="small">
                <a href={`https://github.com/codyrchen/poker-alpha/blob/main/${c.source.split(" ")[0]}`}>
                  {c.source.split("/").pop()}
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>The validation ladder</h2>
      <p className="muted small" style={{ maxWidth: "44rem" }}>
        How do you know a solver is working when the full game cannot be
        solved? Each rung checks something the previous rung cannot:
      </p>
      <ol style={{ marginLeft: "1.4rem", maxWidth: "46rem" }}>
        {[
          ["Kuhn poker", "exact CFR reaches the known equilibrium (game value −1/18)"],
          ["Leduc poker", "larger exact game; convergence + known-solution checks"],
          ["Exact reduced Hold'em subgames", "measures the ABSTRACTION's error directly, not just the solver's"],
          ["Python ↔ C++ random-tape parity", "10,000 iterations bitwise identical: same trajectories, regrets, strategies (1,000 in CI)"],
          ["Exhaustive evaluator parity", "all 2,598,960 five-, 20,358,520 six- and 133,784,560 seven-card hands"],
          ["Checkpoint identity", "train(a)+save+load+train(b) bit-identical to train(a+b)"],
          ["Three independent seeds", "cross-run disagreement measured, feeding the gate"],
          ["Confidence calibration", "8,880 exact EV-regret targets; leave-one-game-out"],
          ["Restricted LBR", "an honest lower bound: ≥ ~100 bb/100 exploitability remains in the full game"],
        ].map(([name, what], i) => (
          <li key={i} style={{ marginBottom: "0.4rem" }}>
            <b>{name}</b>
            <span className="muted small"> — {what}</span>
          </li>
        ))}
      </ol>
      <p className="small">
        Deep dives:{" "}
        <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/solver_validation.md">
          validation methodology
        </a>{" "}
        ·{" "}
        <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/native_solver.md">
          native parity
        </a>{" "}
        ·{" "}
        <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/solver_confidence_v2.md">
          confidence v2
        </a>
      </p>

      <ProvenanceFooter
        envelope={env}
        file="claims.json"
        method="registry built by scripts/build_web_research_data.py from the committed result files; test count from pytest --collect-only at generation time"
        limitation="values describe the recorded benchmark machine and committed artifacts; performance numbers vary with hardware"
      />
    </main>
  );
}
