"use client";

import { useEffect, useState } from "react";
import demo from "@/data/demo.json";
import { MatrixCell, preflopMatrix, PreflopMatrixResponse, pct, solverBadgeClass } from "@/lib/api";
import { ConnectionBadge, OfflineNote, useConnection } from "@/components/Connection";

const COLORS: [string, string][] = [
  ["all_in", "#8a4fbf"],
  ["raise", "#2a78d6"],
  ["bet", "#2a78d6"],
  ["call", "#2e9e5b"],
  ["check", "#2e9e5b"],
  ["fold", "#8a8f96"],
];

function color(label: string | null): string {
  for (const [k, v] of COLORS) if (label?.startsWith(k)) return v;
  return "#8a8f96";
}

export default function StrategyPage() {
  const { state: conn, base } = useConnection();
  const bundled = (demo as any).preflop as Record<string, PreflopMatrixResponse>;
  const nodes = Object.keys(bundled);
  const [node, setNode] = useState(nodes[0]);
  const [data, setData] = useState<PreflopMatrixResponse>(bundled[nodes[0]]);
  const [sel, setSel] = useState<MatrixCell | null>(null);

  useEffect(() => {
    let live = true;
    setData(bundled[node]);
    setSel(null);
    if (conn === "connected") {
      preflopMatrix(base, node)
        .then((d) => live && setData(d))
        .catch(() => {});
    }
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node, conn, base]);

  return (
    <main>
      <h1>Strategy explorer</h1>
      <p className="lead">
        The 169 canonical hands at representable heads-up preflop nodes. Every
        frequency is a lookup against the promoted 2M-iteration artifact —
        nothing is hard-coded. Dimmed cells are states the confidence gate
        rejects: PokerAlpha shows no numbers there.
      </p>
      <p>
        <ConnectionBadge state={conn} />
      </p>
      <OfflineNote state={conn} />

      <select value={node} onChange={(e) => setNode(e.target.value)} style={{ maxWidth: "26rem", margin: "0.8rem 0" }}>
        {nodes.map((n) => (
          <option key={n}>{n}</option>
        ))}
      </select>
      <p className="muted small">
        {Object.entries(data.stats)
          .sort()
          .map(([k, v]) => `${k}: ${v}`)
          .join(" · ")}
      </p>

      <div className="matrix">
        <table>
          <tbody>
            {data.cells.map((row, i) => (
              <tr key={i}>
                {row.map((c) => {
                  const usable = c.status === "ACCEPTED" || c.status === "LOW CONFIDENCE";
                  const col = color(c.dominant);
                  return (
                    <td
                      key={c.hand}
                      className={`${usable ? "" : "dim"} ${sel?.hand === c.hand ? "sel" : ""}`}
                      style={
                        usable
                          ? {
                              background: `${col}22`,
                              borderLeftColor: col,
                              opacity: c.status === "ACCEPTED" ? 1 : 0.55,
                            }
                          : undefined
                      }
                      onClick={() => setSel(c)}
                    >
                      <b>{c.hand}</b>
                      <br />
                      <span style={{ fontSize: "0.6rem" }}>
                        {usable ? `${c.dominant} ${pct(c.dominant_freq)}` : "—"}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sel && (
        <div className="panel" style={{ maxWidth: "30rem" }}>
          <b>{sel.hand}</b> <span className="muted small">queried as {sel.combo}</span>{" "}
          <span className={solverBadgeClass(sel.status)}>{sel.status}</span>
          {sel.mix.length > 0 && (
            <table className="data" style={{ marginTop: "0.6rem" }}>
              <tbody>
                {sel.mix.map(([a, p]) => (
                  <tr key={a}>
                    <td>{a}</td>
                    <td className="mono">{(p * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {sel.visits != null && (
            <p className="muted small">training visits: {Math.round(sel.visits).toLocaleString()}</p>
          )}
          {sel.notes.map((n) => (
            <p key={n} className="muted small">
              {n}
            </p>
          ))}
          {sel.reasons.length > 0 && (
            <p className="muted small">gate: {sel.reasons.join(", ")}</p>
          )}
        </div>
      )}

      <p className="muted small" style={{ marginTop: "1rem" }}>
        The trained strategy covers specific heads-up trees at ~100bb. A
        rejected or unsupported cell does not get interpolated numbers — the
        honest answer there is &ldquo;not trusted&rdquo;, and that is what the
        product shows. This is not continuous GTO coverage.
      </p>
    </main>
  );
}
