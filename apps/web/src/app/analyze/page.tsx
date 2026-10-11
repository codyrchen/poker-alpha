"use client";

import { useMemo, useState } from "react";
import demo from "@/data/demo.json";
import { analyze, AnalyzeResponse } from "@/lib/api";
import { ConnectionBadge, OfflineNote, useConnection } from "@/components/Connection";
import { DecisionView } from "@/components/Decision";

type Spot = { name: string; state: any; result: AnalyzeResponse };

export default function AnalyzePage() {
  const { state: conn, base } = useConnection();
  const spots = (demo as any).demo_spots as Spot[];
  const [spotIdx, setSpotIdx] = useState(0);
  const [mode, setMode] = useState<"demo" | "json">("demo");
  const [json, setJson] = useState(() =>
    JSON.stringify(spots[0].state, null, 1),
  );
  const [live, setLive] = useState<AnalyzeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const spot = spots[spotIdx];
  const bundled = useMemo(() => spot.result, [spot]);

  async function run(stateObj: any) {
    setBusy(true);
    setError(null);
    try {
      setLive(await analyze(base, stateObj));
    } catch (e: any) {
      setError(String(e?.message ?? e));
      setLive(null);
    } finally {
      setBusy(false);
    }
  }

  const shown = live ?? (mode === "demo" ? bundled : null);
  const shownState =
    live != null
      ? mode === "demo"
        ? spot.state
        : safeParse(json)
      : mode === "demo"
        ? spot.state
        : null;

  return (
    <main>
      <h1>Analyzer</h1>
      <p className="lead">
        One decision, the full picture: the strategy distribution, whether the
        trained solver is trusted here, and what the numbers are based on.
      </p>
      <p>
        <ConnectionBadge state={conn} />
      </p>
      <OfflineNote state={conn} />

      <div style={{ display: "flex", gap: "0.6rem", margin: "1rem 0" }}>
        <button
          className={mode === "demo" ? "primary" : ""}
          onClick={() => {
            setMode("demo");
            setLive(null);
          }}
        >
          Demo spots
        </button>
        <button
          className={mode === "json" ? "primary" : ""}
          onClick={() => {
            setMode("json");
            setLive(null);
          }}
          disabled={conn !== "connected"}
          title={conn !== "connected" ? "needs the local engine" : ""}
        >
          Your own state (JSON)
        </button>
      </div>

      {mode === "demo" ? (
        <>
          <select
            value={spotIdx}
            onChange={(e) => {
              setSpotIdx(Number(e.target.value));
              setLive(null);
            }}
            style={{ maxWidth: "34rem" }}
          >
            {spots.map((s, i) => (
              <option key={s.name} value={i}>
                {s.name}
              </option>
            ))}
          </select>
          {conn === "connected" && (
            <p style={{ marginTop: "0.6rem" }}>
              <button className="primary" disabled={busy} onClick={() => run(spot.state)}>
                {busy ? "Analyzing…" : "Re-analyze live"}
              </button>{" "}
              <span className="muted small">
                {live ? "live result from your local engine" : "bundled result shown; click to recompute locally"}
              </span>
            </p>
          )}
        </>
      ) : (
        <>
          <textarea
            className="mono"
            rows={14}
            value={json}
            onChange={(e) => setJson(e.target.value)}
          />
          <p style={{ marginTop: "0.6rem" }}>
            <button
              className="primary"
              disabled={busy || conn !== "connected"}
              onClick={() => {
                const st = safeParse(json);
                if (!st) setError("invalid JSON");
                else run(st);
              }}
            >
              {busy ? "Analyzing…" : "Analyze"}
            </button>
          </p>
        </>
      )}

      {error && <p className="badge err">{error}</p>}
      <hr className="hair" />
      {shown ? (
        <DecisionView r={shown} state={shownState} />
      ) : (
        <p className="muted">Enter a state and analyze it with the local engine.</p>
      )}
    </main>
  );
}

function safeParse(s: string): any | null {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}
