"use client";

// Shared Research Lab scaffolding: every lab page leads with its
// QUESTION, renders real data, and ends with provenance (lab-note mode:
// question / method / result / limitation / reproduce).

import Link from "next/link";
import React from "react";

export function ProvenanceFooter({
  envelope,
  file,
  method,
  limitation,
}: {
  envelope: { generated_from_commit?: string; strategy_sha256?: string; seed?: number };
  file: string;
  method: string;
  limitation: string;
}) {
  return (
    <footer className="labnote">
      <dl className="kv">
        <dt>Method</dt>
        <dd style={{ fontFamily: "inherit" }}>{method}</dd>
        <dt>Limitation</dt>
        <dd style={{ fontFamily: "inherit" }}>{limitation}</dd>
        <dt>Data</dt>
        <dd>
          <span className="badge mute">PRECOMPUTED</span>{" "}
          <a
            href={`https://github.com/codyrchen/poker-alpha/blob/main/apps/web/src/data/lab/${file}`}
          >
            {file}
          </a>
        </dd>
        <dt>Reproduce</dt>
        <dd>python scripts/build_web_research_data.py</dd>
        {envelope.generated_from_commit && (
          <>
            <dt>Commit</dt>
            <dd>{envelope.generated_from_commit.slice(0, 12)}</dd>
          </>
        )}
        {envelope.strategy_sha256 && (
          <>
            <dt>Strategy</dt>
            <dd>sha256:{envelope.strategy_sha256.slice(0, 12)}…</dd>
          </>
        )}
      </dl>
      <p className="small" style={{ marginTop: "0.6rem" }}>
        <Link href="/lab">← Research Lab</Link>
      </p>
    </footer>
  );
}

export function Question({ children }: { children: React.ReactNode }) {
  return <p className="lead question">{children}</p>;
}

const MIX_COLORS: [string, string][] = [
  ["all_in", "#8a4fbf"],
  ["a", "#8a4fbf"],
  ["raise", "#2a78d6"],
  ["bet", "#2a78d6"],
  ["x", "#2a78d6"],
  ["b", "#2a78d6"],
  ["call", "#2e9e5b"],
  ["check", "#2e9e5b"],
  ["c", "#2e9e5b"],
  ["fold", "#8a8f96"],
  ["f", "#8a8f96"],
  ["p", "#8a8f96"],
];

export function actionColor(label: string): string {
  for (const [k, v] of MIX_COLORS) if (label === k || label.startsWith(k + "_") || label.startsWith(k)) return v;
  return "#8a8f96";
}

/** Horizontal stacked mix bar (one strategy distribution). */
export function MixBar({
  mix,
  height = 14,
}: {
  mix: Record<string, number> | [string, number][];
  height?: number;
}) {
  const entries = Array.isArray(mix) ? mix : Object.entries(mix);
  return (
    <div
      className="mixbar"
      style={{ height }}
      role="img"
      aria-label={entries.map(([a, p]) => `${a} ${(p * 100).toFixed(0)}%`).join(", ")}
      title={entries.map(([a, p]) => `${a} ${(p * 100).toFixed(1)}%`).join("  ")}
    >
      {entries.map(([a, p]) => (
        <span
          key={a}
          style={{ width: `${p * 100}%`, background: actionColor(a) }}
        />
      ))}
    </div>
  );
}

export function MixLegend({ labels }: { labels: string[] }) {
  return (
    <span className="small muted">
      {labels.map((l) => (
        <span key={l} style={{ marginRight: "0.9rem" }}>
          <span
            aria-hidden
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: actionColor(l),
              marginRight: 4,
            }}
          />
          {l}
        </span>
      ))}
    </span>
  );
}

/** Minimal SVG line chart (optionally log-y). Real points only. */
export function LineChart({
  points,
  width = 560,
  height = 200,
  logY = false,
  yLabel,
  xLabel,
}: {
  points: { x: number; y: number }[];
  width?: number;
  height?: number;
  logY?: boolean;
  yLabel?: string;
  xLabel?: string;
}) {
  if (!points.length) return null;
  const xs = points.map((p) => Math.log10(Math.max(p.x, 1)));
  const ys = points.map((p) => (logY ? Math.log10(Math.max(p.y, 1e-9)) : p.y));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const px = (x: number) => 46 + ((x - x0) / (x1 - x0 || 1)) * (width - 60);
  const py = (y: number) => height - 26 - ((y - y0) / (y1 - y0 || 1)) * (height - 44);
  const path = points
    .map((p, i) => `${i ? "L" : "M"}${px(xs[i]).toFixed(1)},${py(ys[i]).toFixed(1)}`)
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="chart"
      role="img"
      aria-label={`${yLabel ?? "y"} versus ${xLabel ?? "x"}: ${points.length} measured points from ${points[0].y.toExponential(1)} down to ${points[points.length - 1].y.toExponential(1)}`}
    >
      <line x1="46" y1={height - 26} x2={width - 12} y2={height - 26} stroke="var(--hairline)" />
      <line x1="46" y1="12" x2="46" y2={height - 26} stroke="var(--hairline)" />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" />
      {points.map((p, i) => (
        <circle key={i} cx={px(xs[i])} cy={py(ys[i])} r="3" fill="var(--accent)">
          <title>{`${xLabel ?? "x"}=${p.x}  ${yLabel ?? "y"}=${p.y}`}</title>
        </circle>
      ))}
      {yLabel && (
        <text x="8" y="18" fill="var(--muted)" fontSize="10">
          {yLabel}
          {logY ? " (log)" : ""}
        </text>
      )}
      {xLabel && (
        <text x={width - 12} y={height - 8} fill="var(--muted)" fontSize="10" textAnchor="end">
          {xLabel} (log)
        </text>
      )}
    </svg>
  );
}
