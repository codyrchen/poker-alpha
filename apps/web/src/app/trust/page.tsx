"use client";

// Solver Trust Map: where does the trained strategy trust itself?

import { useMemo, useState } from "react";
import data from "@/data/lab/trust_map.json";
import { MixBar, ProvenanceFooter, Question, actionColor } from "@/components/Lab";

const env = data as any;
const P = env.payload;
const NODES = Object.keys(P.nodes);

type Mode = "strategy" | "trust" | "visits" | "disagreement";

const TRUST_BG: Record<string, string> = {
  ACCEPTED: "rgba(46,158,91,0.5)",
  "LOW CONFIDENCE": "rgba(194,136,42,0.45)",
  REJECTED: "rgba(138,143,150,0.3)",
  UNSUPPORTED: "rgba(194,80,71,0.25)",
};

function heat(v: number | null, max: number, hue = "42,120,214"): string {
  if (v === null || !isFinite(v)) return "transparent";
  const a = Math.min(Math.max(v / max, 0), 1);
  return `rgba(${hue},${0.08 + 0.72 * a})`;
}

export default function TrustPage() {
  const [node, setNode] = useState(NODES[0]);
  const [mode, setMode] = useState<Mode>("trust");
  const [sel, setSel] = useState<any | null>(null);

  const cells = P.nodes[node].cells as any[][];
  const maxVisits = useMemo(
    () => Math.max(...cells.flat().map((c) => c.visits ?? 0), 1),
    [cells],
  );
  const maxSd = useMemo(
    () =>
      Math.max(
        ...cells.flat().map((c) => c.signals?.seed_disagreement ?? 0),
        0.1,
      ),
    [cells],
  );

  return (
    <main>
      <h1>Solver Trust Map</h1>
      <Question>
        Where does the trained strategy trust itself? All 169 starting-hand
        classes at a fixed heads-up node, with the confidence gate&rsquo;s
        verdict and its raw evidence — training visits and cross-seed
        disagreement — for every cell.
      </Question>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: "0.8rem 0" }}>
        <select value={node} onChange={(e) => { setNode(e.target.value); setSel(null); }}
                style={{ maxWidth: "22rem" }} aria-label="preflop node">
          {NODES.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        {(["trust", "strategy", "visits", "disagreement"] as Mode[]).map((m) => (
          <button key={m} className={mode === m ? "primary" : ""} onClick={() => setMode(m)}>
            {m}
          </button>
        ))}
      </div>
      <p className="small muted">
        {Object.entries(P.nodes[node].stats as Record<string, number>)
          .sort()
          .map(([k, v]) => `${k}: ${v}`)
          .join(" · ")}{" "}
        — trust state is an evidence gate, <b>not</b> a probability of
        correctness.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="heatmap" aria-label={`trust map, ${mode} view`}>
          <tbody>
            {cells.map((row, i) => (
              <tr key={i}>
                {row.map((c) => {
                  let bg = "transparent";
                  if (mode === "trust") bg = TRUST_BG[c.status] ?? "transparent";
                  else if (mode === "strategy")
                    bg = c.dominant ? `${actionColor(c.dominant)}55` : "transparent";
                  else if (mode === "visits") bg = heat(c.visits, maxVisits);
                  else bg = heat(c.signals?.seed_disagreement ?? null, maxSd, "194,80,71");
                  return (
                    <td
                      key={c.hand}
                      tabIndex={0}
                      style={{ background: bg, opacity: c.status === "REJECTED" && mode === "strategy" ? 0.35 : 1 }}
                      onClick={() => setSel(c)}
                      onKeyDown={(e) => e.key === "Enter" && setSel(c)}
                      title={`${c.hand}: ${c.status}${c.dominant ? ` · ${c.dominant}` : ""}`}
                      className={sel?.hand === c.hand ? "sel" : ""}
                    >
                      {c.hand}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sel && (
        <div className="panel" style={{ maxWidth: "32rem" }}>
          <b>{sel.hand}</b>{" "}
          <span className={`badge ${sel.status === "ACCEPTED" ? "good" : sel.status === "LOW CONFIDENCE" ? "warn" : "mute"}`}>
            {sel.status}
          </span>
          {sel.mix?.length > 0 && (
            <>
              <div style={{ margin: "0.5rem 0 0.3rem" }}>
                <MixBar mix={sel.mix} />
              </div>
              <div className="small mono">
                {sel.mix.map(([a, p]: [string, number]) => `${a} ${(p * 100).toFixed(0)}%`).join(" · ")}
              </div>
            </>
          )}
          <dl className="kv" style={{ marginTop: "0.5rem" }}>
            {sel.visits != null && (
              <>
                <dt>training visits</dt>
                <dd>{Math.round(sel.visits).toLocaleString()}</dd>
              </>
            )}
            {Object.entries(sel.signals ?? {}).map(([k, v]) => (
              v != null && (
                <span key={k} style={{ display: "contents" }}>
                  <dt>{k}</dt>
                  <dd>{String(v)}</dd>
                </span>
              )
            ))}
            {sel.reasons?.length > 0 && (
              <>
                <dt>gate reasons</dt>
                <dd>{sel.reasons.join(", ")}</dd>
              </>
            )}
          </dl>
          {sel.status === "REJECTED" && (
            <p className="muted small">
              No numbers are shown for rejected cells in the product — the
              honest answer is &ldquo;not trusted&rdquo;, and the live
              pipeline falls back to a rollout there.
            </p>
          )}
        </div>
      )}

      <ProvenanceFooter
        envelope={env}
        file="trust_map.json"
        method="169 canonical classes × 3 fixed heads-up nodes, each a live lookup against the promoted artifact; gate verdicts and signals come from the real SolverConfidence v2 table"
        limitation="cells sample canonical classes at fixed nodes — NOT the natural frequency of hands in play; a mostly-rejected map is the honest picture of a strict preflop gate, not a rendering choice"
      />
    </main>
  );
}
