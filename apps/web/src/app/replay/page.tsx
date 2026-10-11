"use client";

// One hand through the whole system: state → modelled belief → abstract
// infoset → gate → strategy, with the opponent range evolving by street.

import { useState } from "react";
import data from "@/data/lab/one_hand.json";
import { MixBar, ProvenanceFooter, Question } from "@/components/Lab";
import { Cards } from "@/components/Decision";

const env = data as any;
const P = env.payload;

const GRID = (() => {
  const R = "AKQJT98765432";
  const rows: string[][] = [];
  for (let i = 0; i < 13; i++) {
    const row: string[] = [];
    for (let j = 0; j < 13; j++) {
      if (i === j) row.push(R[i] + R[j]);
      else if (i < j) row.push(R[i] + R[j] + "s");
      else row.push(R[j] + R[i] + "o");
    }
    rows.push(row);
  }
  return rows;
})();

function RangeMatrix({ matrix }: { matrix: number[][] }) {
  const peak = Math.max(...matrix.flat(), 1e-9);
  return (
    <table
      style={{ borderCollapse: "separate", borderSpacing: 1, fontSize: "0.5rem", lineHeight: 1.6, textAlign: "center" }}
      role="img"
      aria-label="13 by 13 modelled opponent range; darker cells carry more probability"
    >
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            {row.map((w, j) => (
              <td
                key={j}
                title={`${GRID[i][j]}: ${(w * 100).toFixed(2)}%`}
                style={{
                  background: `rgba(42,120,214,${0.06 + 0.8 * (w / peak)})`,
                  width: "1.55rem",
                }}
              >
                {GRID[i][j]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ReplayPage() {
  const [stage, setStage] = useState(P.range_stages.length - 1);
  const s = P.range_stages[stage];

  return (
    <main>
      <h1>One hand through the whole system</h1>
      <Question>
        What does PokerAlpha actually do with a hand? One real river
        decision, followed end to end: the state, the modelled opponent
        belief, the abstract information set, the confidence gate, and the
        learned mixed strategy.
      </Question>

      <h2>The spot</h2>
      <div style={{ display: "flex", gap: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
        <Cards text={P.state.hero_cards} />
        <span className="muted">|</span>
        <Cards text={P.state.board} small />
        <span className="muted small mono">
          pot {P.state.pot}bb · checked to the hero on the river
        </span>
      </div>

      <h2>1 · The modelled opponent belief (street by street)</h2>
      <p className="muted small" style={{ maxWidth: "44rem" }}>
        <b>MODELLED BELIEF — NOT GROUND TRUTH.</b> A position/line prior
        updated by Bayes&rsquo; rule after each observed action, under a
        behavior archetype. This is the same computation the decision engine
        uses (same function, truncated per street).
      </p>
      <div style={{ display: "flex", gap: "0.5rem", margin: "0.6rem 0", flexWrap: "wrap" }}>
        {P.range_stages.map((st: any, i: number) => (
          <button key={st.name} className={i === stage ? "primary" : ""} onClick={() => setStage(i)}>
            {st.name}
          </button>
        ))}
      </div>
      <div style={{ overflowX: "auto" }}>
        <RangeMatrix matrix={s.matrix} />
      </div>
      <p className="small muted">
        {s.name}: {s.entropy_bits} bits · ≈{s.effective_combos} effective
        combos — each passive action narrows the modelled range.
      </p>

      <h2>2 · The abstract information set</h2>
      <dl className="kv" style={{ maxWidth: "46rem" }}>
        <dt>infoset key</dt>
        <dd>{P.infoset}</dd>
        <dt>training visits</dt>
        <dd>{Math.round(P.visits).toLocaleString()}</dd>
      </dl>
      <p className="muted small" style={{ maxWidth: "44rem" }}>
        Full Hold&nbsp;em has ~10<sup>160</sup> states, so the engine maps
        the real state into a compressed key — street, position, a
        river-strength percentile bucket, and the shape of the betting
        history. Strategically different hands CAN share a key; that risk is
        measured, not ignored (see the gate below).
      </p>

      <h2>3 · The confidence gate</h2>
      <p>
        <span className="badge good">{P.gate.confidence === "SOLVER_ACCEPT" ? "ACCEPTED" : P.gate.confidence}</span>{" "}
        <span className="muted small">
          this key has {Math.round(P.visits).toLocaleString()} training
          visits and low cross-seed disagreement, so the trained strategy is
          used directly. On most preflop nodes the same gate REJECTS and the
          engine falls back — refusing is a feature.
        </span>
      </p>

      <h2>4 · The learned strategy</h2>
      <div style={{ maxWidth: "500px" }}>
        <MixBar mix={P.mix} height={18} />
      </div>
      <p className="mono small">
        {Object.entries(P.mix as Record<string, number>)
          .sort((a, b) => b[1] - a[1])
          .map(([a, p]) => `${a} ${(p * 100).toFixed(0)}%`)
          .join(" · ")}
      </p>
      <p className="muted small" style={{ maxWidth: "44rem" }}>
        A DISTRIBUTION, not a single answer: the solver checks{" "}
        {(P.mix[P.recommended] * 100).toFixed(0)}% here and mixes bets at
        several sizes. In hidden-information games, deliberately mixing is
        what makes a strategy hard to exploit. Hero equity under the
        modelled range: {(P.equity * 100).toFixed(0)}% ±
        {(P.equity_se * 100).toFixed(0)}%.
      </p>

      <h2>5 · The receipt</h2>
      <p className="muted small" style={{ maxWidth: "44rem" }}>
        Every analysis can carry a reproducibility record — the state
        fingerprint, strategy artifact SHA, configuration and seeds — with a
        stable ID. This one is{" "}
        <span className="mono">{P.receipt.receipt_id}</span>; feeding the
        receipt back to the engine re-runs the analysis and confirms the
        identical result.
      </p>
      <details>
        <summary className="small">full decision path</summary>
        <table className="data" style={{ maxWidth: "720px", marginTop: "0.5rem" }}>
          <tbody>
            {P.decision_path.map((r: any) => (
              <tr key={r.label}>
                <td style={{ whiteSpace: "nowrap" }}>{r.label}</td>
                <td className="small mono" style={{ wordBreak: "break-all" }}>
                  {r.value}
                  {r.note && <div className="muted" style={{ fontFamily: "inherit" }}>{r.note}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <ProvenanceFooter
        envelope={env}
        file="one_hand.json"
        method={`full pipeline run at ${env.generation_args.equity_simulations} equity samples; range stages computed by the engine's own infer_opponent_range per street`}
        limitation="one deterministic demo hand; the modelled range depends on the behavior archetype (regular) and is a belief model, never the opponent's actual cards"
      />
    </main>
  );
}
