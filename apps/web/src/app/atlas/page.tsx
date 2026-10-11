"use client";

// Policy Atlas: 6,000 real infosets from the release strategy, projected
// to 2D by deterministic PCA over their action distributions.

import { useEffect, useRef, useState } from "react";
import data from "@/data/lab/atlas.json";
import complexity from "@/data/lab/complexity.json";
import { ProvenanceFooter, Question, actionColor } from "@/components/Lab";

const env = data as any;
const P = env.payload;
const C = (complexity as any).payload;

type Mode = "top" | "street" | "gate" | "entropy";

const STREET_COLORS = ["#2a78d6", "#2e9e5b", "#c2882a", "#8a4fbf"];
const STREET_NAMES = ["preflop", "flop", "turn", "river"];
const GATE_COLORS: Record<string, string> = {
  ACCEPT: "#2e9e5b",
  LOW: "#c2882a",
  "LOW (street cap)": "#a06f2a",
  REJECT: "#8a8f96",
  NO_DATA: "#4a4f55",
};

export default function AtlasPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>("top");
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const W = 860;
    const H = 560;
    canvas.width = W * 2;
    canvas.height = H * 2;
    const g = canvas.getContext("2d")!;
    g.scale(2, 2);
    g.clearRect(0, 0, W, H);
    const xs = P.x as number[];
    const ys = P.y as number[];
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    for (let i = 0; i < xs.length; i++) {
      const px = 14 + ((xs[i] - x0) / (x1 - x0)) * (W - 28);
      const py = 14 + ((ys[i] - y0) / (y1 - y0)) * (H - 28);
      let color = "#8a8f96";
      if (mode === "top") color = actionColor(P.top[i]);
      else if (mode === "street") color = STREET_COLORS[P.street[i]];
      else if (mode === "gate") color = GATE_COLORS[P.gate[i]] ?? "#4a4f55";
      else {
        const e = Math.min(P.entropy[i] / 2, 1);
        color = `rgba(42,120,214,${0.15 + 0.8 * e})`;
      }
      g.fillStyle = color;
      g.globalAlpha = mode === "entropy" ? 1 : 0.65;
      g.fillRect(px - 1.4, py - 1.4, 2.8, 2.8);
    }
    g.globalAlpha = 1;
  }, [mode]);

  const legend =
    mode === "top"
      ? ["f (fold)", "c (check/call)", "x… / b… (raise/bet)", "a (all-in)"]
      : mode === "street"
        ? STREET_NAMES
        : mode === "gate"
          ? Object.keys(GATE_COLORS).filter((k) => k !== "NO_DATA")
          : ["darker = more mixed (higher entropy)"];

  return (
    <main>
      <h1>Policy Atlas</h1>
      <Question>
        What kinds of policies did the solver learn? {P.n.toLocaleString()}{" "}
        real information sets from the 2M-iteration release strategy,
        projected to 2D (deterministic PCA over their action
        distributions). Each point is an actual stored infoset.
      </Question>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: "0.8rem 0" }}>
        {(["top", "street", "gate", "entropy"] as Mode[]).map((m) => (
          <button key={m} className={mode === m ? "primary" : ""} onClick={() => setMode(m)}>
            {m === "top" ? "top action" : m === "gate" ? "table gate" : m}
          </button>
        ))}
      </div>
      <p className="small muted">
        {legend.map((l, i) => (
          <span key={l} style={{ marginRight: "1rem" }}>
            <span
              aria-hidden
              style={{
                display: "inline-block", width: 10, height: 10, borderRadius: 2,
                background:
                  mode === "top" ? actionColor(l[0])
                  : mode === "street" ? STREET_COLORS[i]
                  : mode === "gate" ? GATE_COLORS[l]
                  : "#2a78d6",
                marginRight: 4,
              }}
            />
            {l}
          </span>
        ))}
      </p>
      <div className="atlas-wrap">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`policy atlas: ${P.n} infosets projected by PCA; colored by ${mode}; clusters correspond to policy families like pure folds, pure calls, and mixed raise distributions`}
        />
      </div>
      <p className="muted small">
        Distance here is similarity of ACTION DISTRIBUTIONS under a fixed
        action basis — not poker-state similarity. The tight clusters are
        near-pure policies (always-fold, always-call); the connecting cloud
        is genuine mixing.
      </p>

      <h2>How mixed is the learned strategy?</h2>
      <table className="data" style={{ maxWidth: "640px" }}>
        <thead>
          <tr>
            <th>population</th>
            <th>near-pure (≥95%)</th>
            <th>moderately mixed</th>
            <th>highly mixed (&lt;70%)</th>
            <th>mean entropy</th>
          </tr>
        </thead>
        <tbody>
          {[["all infosets", C.overall], ...Object.entries(C.by_street)].map(
            ([name, b]: any) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="mono">{b.near_pure_pct.toFixed(1)}%</td>
                <td className="mono">{b.moderately_mixed_pct.toFixed(1)}%</td>
                <td className="mono">{b.highly_mixed_pct.toFixed(1)}%</td>
                <td className="mono">{b.mean_entropy_bits.toFixed(2)} bits</td>
              </tr>
            ),
          )}
        </tbody>
      </table>
      <p className="muted small">{C.note}.</p>

      <ProvenanceFooter
        envelope={env}
        file="atlas.json"
        method={`${P.sample}; projection: ${P.explained}; gate states from the real SolverConfidence v2 table with default thresholds (the live lookup additionally caps flop/turn at LOW)`}
        limitation="a 6,000-point deterministic sample of 144,025 stored infosets, weighted by storage — not by how often real hands reach them"
      />
    </main>
  );
}
