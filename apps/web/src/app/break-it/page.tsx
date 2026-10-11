// Try to break PokerAlpha: real stress scenarios + the automated red
// team's discoveries. Finding weaknesses is the point.

import Link from "next/link";
import breakit from "@/data/lab/break_it.json";
import redteam from "@/data/lab/redteam.json";
import { ProvenanceFooter, Question } from "@/components/Lab";

const B = (breakit as any).payload;
const R = (redteam as any).payload;

export default function BreakItPage() {
  return (
    <main>
      <h1>Try to break PokerAlpha</h1>
      <Question>
        What happens when the state is hostile — off-tree sizes, distrusted
        regions, impossible inputs? Every scenario below is a REAL engine
        run, and the interesting part is which safeguard fires.
      </Question>

      <h2>Stress scenarios</h2>
      <table className="data">
        <thead>
          <tr>
            <th>scenario</th>
            <th>what should happen</th>
            <th>what actually happened</th>
          </tr>
        </thead>
        <tbody>
          {B.scenarios.map((s: any) => (
            <tr key={s.title}>
              <td style={{ maxWidth: "16rem" }}>{s.title}</td>
              <td className="muted" style={{ maxWidth: "15rem" }}>{s.expected}</td>
              <td className="small">
                {typeof s.outcome === "string" ? (
                  <>
                    <span className="badge good">{s.outcome}</span>
                    <div className="muted">{s.detail}</div>
                  </>
                ) : (
                  <>
                    <span className={`badge ${s.outcome.gate === "ACCEPTED" ? "good" : s.outcome.gate === "LOW CONFIDENCE" ? "warn" : "mute"}`}>
                      {s.outcome.refusal?.length ? "REFUSED" : s.outcome.gate}
                    </span>{" "}
                    <span className="mono">{s.outcome.method}</span>
                    <div className="muted">
                      {[...(s.outcome.reasons ?? []), ...(s.outcome.refusal ?? [])].join(", ") || "no flags"}
                    </div>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>The states that break PokerAlpha (automated red team)</h2>
      <p style={{ maxWidth: "46rem" }}>
        A search over {R.n_states_sampled} randomly generated LEGAL states
        (engine self-play snapshots, seed {R.seed}) scored on separate
        fragility signals. Gate verdicts across the sample:{" "}
        {Object.entries(R.summary.gate_counts as Record<string, number>)
          .map(([k, v]) => `${k} ${v}`)
          .join(" · ")}
        .
      </p>
      <h3>Structural failure clusters</h3>
      <table className="data" style={{ maxWidth: "720px" }}>
        <thead>
          <tr>
            <th>cluster</th>
            <th>states</th>
            <th>dominant signals</th>
          </tr>
        </thead>
        <tbody>
          {R.clusters.slice(0, 6).map((c: any, i: number) => (
            <tr key={i}>
              <td>
                {c.street} · {c.spr} · {c.bet}
              </td>
              <td className="mono">{c.count}</td>
              <td className="small muted">
                {c.signals.gate_rejected} rejected ·{" "}
                {(c.signals.top_gate_reasons as [string, number][])
                  .map(([r, n]) => `${r}×${n}`)
                  .join(", ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Descriptive groupings, not causal diagnoses. Extreme values found by
        random search are DISCOVERY results — the project&rsquo;s discipline
        requires replication with independent seeds before any becomes a
        claim.
      </p>

      <h3>Rule-generated research backlog</h3>
      <ul style={{ marginLeft: "1.2rem" }}>
        {(R.backlog ?? []).slice(0, 5).map((b: any, i: number) => (
          <li key={i} className="small">
            <b>{b.candidate}</b> in {b.cluster}{" "}
            <span className="muted">(rule: {b.rule})</span>
          </li>
        ))}
      </ul>

      <h2>Known failure modes (measured, not hidden)</h2>
      <table className="data">
        <tbody>
          {[
            ["Flop/turn abstraction error", "exact subgames show the encoder merges strategically different hands (lifted strategies 1.4–9.6 BB/hand exploitable vs ~0.02 for raw keys)", "permanent LOW-CONFIDENCE cap on those streets"],
            ["Preflop seed disagreement", "independently seeded 2M runs genuinely disagree at many preflop nodes", "the gate rejects; rollout fallback answers instead"],
            ["Off-tree bet sizes", "observed sizes are translated to the nearest trained size (log-space)", "flagged OFF_TREE_TRANSLATION and downgraded"],
            ["Rollout model bias", "the fallback's opponent model is simple (one response round, check-down)", "EVs carry SEs; model-sensitivity analysis provided"],
            ["Full-game exploitability", "restricted local best response shows ≥ ~100 bb/100 exploitability remains", "stated on every surface; never called GTO"],
            ["Real-screen recognition", "no independent held-out validation data exists yet", "live observer stays out of the product; BLOCKED label"],
          ].map(([what, why, mitigation]) => (
            <tr key={what as string}>
              <td style={{ maxWidth: "12rem" }}>{what}</td>
              <td className="muted small" style={{ maxWidth: "20rem" }}>{why}</td>
              <td className="small">{mitigation}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p>
        <Link className="btn" href="/trust">See where the gate trusts itself</Link>{" "}
        <Link className="btn" href="/evidence">Evidence for every claim</Link>
      </p>

      <ProvenanceFooter
        envelope={redteam as any}
        file="redteam.json"
        method={`random legal self-play states scored on separate objectives (gate, seed disagreement, visits, bet-translation distance, ±10% local sensitivity, archetype sensitivity); ${R.n_states_scored} states scored in ${R.wall_seconds}s`}
        limitation={R.discipline}
      />
    </main>
  );
}
