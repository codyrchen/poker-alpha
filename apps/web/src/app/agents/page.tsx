// Agent Arena: reproducible tournaments between algorithmic agents.

import Link from "next/link";
import data from "@/data/lab/tournament.json";
import { ProvenanceFooter, Question } from "@/components/Lab";

const env = data as any;
const P = env.payload;
const NAMES = Object.keys(P.agents);

export default function AgentsPage() {
  return (
    <main>
      <h1>Agent Arena</h1>
      <Question>
        Who beats whom — and by how much, with what uncertainty? A
        seat-swapped, seed-paired round-robin between five agents on Kuhn
        poker: {P.hands_per_pairing.toLocaleString()} paired hands per
        matchup, 95% confidence intervals.
      </Question>

      <div style={{ overflowX: "auto" }}>
        <table className="data" style={{ minWidth: "640px" }}>
          <thead>
            <tr>
              <th>row vs column (chips/hand to row)</th>
              {NAMES.map((n) => (
                <th key={n}>{n.split(" (")[0]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {NAMES.map((a) => (
              <tr key={a}>
                <td>{a}</td>
                {NAMES.map((b) => {
                  if (a === b)
                    return (
                      <td key={b} className="muted">
                        —
                      </td>
                    );
                  const m = P.matrix[a][b];
                  const sig = m.ci95[0] > 0 || m.ci95[1] < 0;
                  return (
                    <td key={b} className="mono small"
                        title={`95% CI [${m.ci95[0]}, ${m.ci95[1]}], ${m.hands} paired hands`}>
                      <span style={{ fontWeight: sig ? 700 : 400 }}>
                        {m.mean > 0 ? "+" : ""}
                        {m.mean.toFixed(3)}
                      </span>
                      <span className="muted"> ±{(1.96 * m.se).toFixed(3)}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Bold = the 95% CI excludes zero. The oracle&rsquo;s edge over CFR
        (+{P.matrix["Best response to CFR (oracle)"]["CFR (20k iterations)"].mean.toFixed(4)}
        /hand) is consistent with CFR&rsquo;s measured exploitability of{" "}
        {P.cfr_exploitability} — two independent computations of the same
        quantity agreeing is itself a correctness check.
      </p>

      <h2>Agent profiles (no marketing names)</h2>
      <table className="data">
        <tbody>
          {NAMES.map((n) => {
            const a = P.agents[n];
            return (
              <tr key={n}>
                <td style={{ minWidth: "11rem" }}>{n}</td>
                <td className="small">
                  <div>{a.algorithm}</div>
                  <div className="muted">
                    sees: {a.information} · adaptive: {a.adaptive} ·{" "}
                    {a.limitations}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        The &ldquo;best response&rdquo; agent is a diagnostic, not a fair
        player — it is constructed from its opponent&rsquo;s full strategy
        and represents the exploitability ceiling.
      </p>

      <p>
        <Link className="btn" href="/arena">Play against the CFR agent yourself</Link>{" "}
        <Link className="btn" href="/games">The games these agents run on</Link>
      </p>

      <ProvenanceFooter
        envelope={env}
        file="tournament.json"
        method={`round-robin via poker_alpha.agents.round_robin: seat-swapped pairs share chance seeds (deal luck pairs out), fixed tournament seed ${P.seed}; ${P.units}`}
        limitation="Kuhn poker only on this page; per-hand utilities are game chips, not money, and say nothing about full Hold'em strength"
      />
    </main>
  );
}
