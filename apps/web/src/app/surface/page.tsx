"use client";

// Decision Surface: where does the recommended strategy change?

import { useState } from "react";
import data from "@/data/lab/surface.json";
import { MixBar, MixLegend, ProvenanceFooter, Question, actionColor } from "@/components/Lab";

const env = data as any;
const P = env.payload;

const GATE_COLOR: Record<string, string> = {
  ACCEPTED: "rgba(46,158,91,0.45)",
  "LOW CONFIDENCE": "rgba(194,136,42,0.45)",
  REJECTED: "rgba(138,143,150,0.35)",
  OFF: "rgba(138,143,150,0.25)",
};

export default function SurfacePage() {
  const [mode, setMode] = useState<"action" | "gate">("action");
  const [sel, setSel] = useState<[number, number] | null>(null);

  const cell = sel ? P.cells[sel[0]][sel[1]] : null;

  return (
    <main>
      <h1>Decision Surface</h1>
      <Question>
        Where does the recommended strategy change? One real river spot,
        swept over {P.x_values.length}×{P.y_values.length} combinations of
        the opponent&rsquo;s bet size and the effective stack — every cell a
        full engine analysis of a re-validated state. No interpolation.
      </Question>
      <div style={{ display: "flex", gap: "0.5rem", margin: "0.8rem 0" }}>
        <button className={mode === "action" ? "primary" : ""} onClick={() => setMode("action")}>
          Color by top action
        </button>
        <button className={mode === "gate" ? "primary" : ""} onClick={() => setMode("gate")}>
          Color by solver gate
        </button>
      </div>
      {mode === "action" ? (
        <MixLegend labels={["call", "fold", "raise", "all_in", "check"]} />
      ) : (
        <span className="small muted">
          green = ACCEPTED · amber = LOW CONFIDENCE · gray = REJECTED (gate
          state is an evidence gate, not a probability of correctness)
        </span>
      )}
      <div style={{ overflowX: "auto", marginTop: "0.6rem" }}>
        <table className="heatmap" role="grid"
               aria-label="decision surface: top action by bet size and effective stack">
          <tbody>
            {P.cells.map((row: any[], yi: number) => (
              <tr key={yi}>
                <th scope="row">{P.y_values[yi]}bb</th>
                {row.map((c: any, xi: number) => (
                  <td
                    key={xi}
                    tabIndex={0}
                    onClick={() => setSel([yi, xi])}
                    onKeyDown={(e) => e.key === "Enter" && setSel([yi, xi])}
                    className={sel && sel[0] === yi && sel[1] === xi ? "sel" : ""}
                    title={
                      c.valid
                        ? `bet ${P.x_values[xi]}bb, stack ${P.y_values[yi]}bb → ${c.top} (${c.gate}, ${c.method})`
                        : `invalid: ${c.reason}`
                    }
                    style={{
                      background: !c.valid
                        ? "transparent"
                        : mode === "action"
                          ? `${actionColor(c.top)}55`
                          : GATE_COLOR[c.gate] ?? "transparent",
                      opacity: c.valid ? 1 : 0.25,
                    }}
                  >
                    {c.valid ? c.top?.split("_")[0] : "—"}
                  </td>
                ))}
              </tr>
            ))}
            <tr>
              <th />
              {P.x_values.map((x: number) => (
                <th key={x} scope="col">
                  {x}bb
                </th>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted small">
        rows: effective stack · columns: opponent bet size · base spot: 9♥9♣
        on Q♠7♦2♣T♥3♥ facing a river bet
      </p>

      {cell && cell.valid && (
        <div className="panel" style={{ maxWidth: "32rem" }}>
          <b>
            bet {P.x_values[sel![1]]}bb · stack {P.y_values[sel![0]]}bb
          </b>{" "}
          <span className="badge mute">{cell.gate}</span>{" "}
          <span className="badge mute">{cell.method}</span>
          <div style={{ margin: "0.6rem 0 0.3rem" }}>
            <MixBar mix={cell.mix} />
          </div>
          <div className="small mono">
            {Object.entries(cell.mix as Record<string, number>)
              .map(([a, p]) => `${a} ${(p * 100).toFixed(0)}%`)
              .join(" · ")}
          </div>
          <p className="muted small" style={{ marginTop: "0.4rem" }}>
            confidence {String(cell.confidence).toUpperCase()}
            {cell.warnings ? ` · ${cell.warnings} warnings` : ""}
          </p>
        </div>
      )}
      {cell && !cell.valid && (
        <p className="badge warn">invalid combination: {cell.reason}</p>
      )}

      <ProvenanceFooter
        envelope={env}
        file="surface.json"
        method={`${P.x_values.length}×${P.y_values.length} grid over villain_bet × effective_stack; each cell re-validated (duplicate cards / impossible stacks rejected) and analysed by the real engine at ${env.generation_args.equity_simulations} equity / ${env.generation_args.rollout_simulations} rollout samples, seed ${env.seed}`}
        limitation="one base spot; discrete tested points only — cells are not interpolated and boundaries between cells are not resolved below the grid resolution"
      />
    </main>
  );
}
