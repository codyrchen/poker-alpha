"use client";

// Solver Microscope: the same information set across training
// checkpoints (300k / 1M / 2M iterations) and three independent seeds.

import { useState } from "react";
import data from "@/data/lab/microscope.json";
import { MixBar, MixLegend, ProvenanceFooter, Question } from "@/components/Lab";

const env = data as any;
const P = env.payload;

const BUCKET_TEXT: Record<string, string> = {
  low: "LOW cross-seed disagreement — three independent runs land on nearly the same mix",
  medium: "MEDIAN disagreement — the typical case",
  high: "HIGH disagreement — independently seeded runs genuinely differ here (one reason the gate rejects)",
};

export default function MicroscopePage() {
  const [sel, setSel] = useState(P.picks[0].key);
  const pick = P.picks.find((p: any) => p.key === sel)!;
  const series = P.series[sel] as Record<string, { mix: Record<string, number>; visits: number }>;

  return (
    <main>
      <h1>Solver Microscope</h1>
      <Question>
        How did the policy at one information set evolve — and do three
        independently seeded runs agree on it? Nine real artifacts: three
        training milestones × three seeds, no invented intermediate history.
      </Question>

      <label className="small muted" htmlFor="ms-key">
        information set (deterministically selected: well-visited, mixed,
        spanning low/median/high cross-seed disagreement)
      </label>
      <select id="ms-key" value={sel} onChange={(e) => setSel(e.target.value)}
              style={{ maxWidth: "36rem", display: "block", margin: "0.4rem 0" }}>
        {P.picks.map((p: any) => (
          <option key={p.key} value={p.key}>
            [{p.bucket} disagreement] {p.key}
          </option>
        ))}
      </select>
      <p className="small muted">
        {BUCKET_TEXT[pick.bucket]} · seed-disagreement signal{" "}
        <span className="mono">{pick.seed_disagreement}</span> · entropy{" "}
        <span className="mono">{pick.entropy} bits</span>
      </p>
      <MixLegend labels={Object.keys(Object.values(series)[0]?.mix ?? {})} />

      <table className="data" style={{ maxWidth: "720px", marginTop: "0.5rem" }}>
        <thead>
          <tr>
            <th>milestone</th>
            <th>seed 0</th>
            <th>seed 1</th>
            <th>seed 2</th>
          </tr>
        </thead>
        <tbody>
          {P.milestones.map((m: string) => (
            <tr key={m}>
              <td className="mono">{m}</td>
              {[0, 1, 2].map((s) => {
                const cell = series[`${m}/seed${s}`];
                return (
                  <td key={s} style={{ width: "28%" }}>
                    {cell ? (
                      <>
                        <MixBar mix={cell.mix} />
                        <span className="small muted mono">
                          {Math.round(cell.visits ?? 0).toLocaleString()} visits
                        </span>
                      </>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Why this matters</h2>
      <p style={{ maxWidth: "46rem" }}>
        Cross-seed disagreement is one of the signals the confidence gate is
        calibrated on. The project&rsquo;s calibration study (8,880 infosets
        with exact EV-regret targets) found that <i>movement between
        checkpoints</i> predicts true regret at no horizon, while visits and
        cross-seed disagreement are the real signals — which is exactly the
        difference between the &ldquo;low&rdquo; and &ldquo;high&rdquo;
        rows you can flip through above.
      </p>

      <ProvenanceFooter
        envelope={env}
        file="microscope.json"
        method={P.selection}
        limitation="12 hand-inspectable infosets out of 144,025; milestone artifacts exist at 300k/1M/2M only — nothing between them is shown or interpolated"
      />
    </main>
  );
}
