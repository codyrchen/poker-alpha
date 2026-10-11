"use client";

// Web trainer: shows a bundled demo spot, takes an answer, reveals the
// engine's distribution and grades by frequency — never binary right/wrong.
// With a local engine connected, grading uses a fresh live analysis.

import { useMemo, useState } from "react";
import demo from "@/data/demo.json";
import { analyze, AnalyzeResponse, pct } from "@/lib/api";
import { ConnectionBadge, OfflineNote, useConnection } from "@/components/Connection";
import { DecisionView, Cards } from "@/components/Decision";

type Spot = { name: string; state: any; result: AnalyzeResponse };

function grade(r: AnalyzeResponse, chosen: string): { head: string; cls: string } {
  if (r.source === "solver" && r.solver_state === "LOW CONFIDENCE")
    return { head: "PokerAlpha uncertain — review only", cls: "mute" };
  if (r.source === "solver" && r.solver_state === "ACCEPTED") {
    const a = r.actions.find((x) => x.label === chosen);
    const f = a?.frequency ?? 0;
    if (f >= 0.25) return { head: `Common in solver mix — ${pct(f)}`, cls: "good" };
    if (f >= 0.05) return { head: `Mixed option — ${pct(f)}`, cls: "warn" };
    if (f > 0) return { head: `Rare in solver mix — ${f < 0.005 ? "<1%" : pct(f)}`, cls: "warn" };
    return { head: "Not in the trained mix", cls: "mute" };
  }
  // rollout fallback: same-method EV comparison when both EVs exist
  const a = r.actions.find((x) => x.label === chosen);
  const best = r.actions.find((x) => x.label === r.recommended);
  if (r.ev_is_decision_basis && a?.ev_bb != null && best?.ev_bb != null) {
    const reg = best.ev_bb - a.ev_bb;
    return {
      head:
        Math.abs(reg) < 1e-9
          ? "Matches the rollout's best action"
          : `Estimated decision regret ${reg >= 0 ? "+" : ""}${reg.toFixed(2)} BB`,
      cls: "warn",
    };
  }
  return { head: "Not scored — no valid same-method comparison", cls: "mute" };
}

export default function TrainerPage() {
  const { state: conn, base } = useConnection();
  const spots = (demo as any).demo_spots as Spot[];
  const [i, setI] = useState(0);
  const [answer, setAnswer] = useState<string | null>(null);
  const [live, setLive] = useState<AnalyzeResponse | null>(null);
  const [tally, setTally] = useState<Record<string, number>>({});

  const spot = spots[i % spots.length];
  const r = live ?? spot.result;
  const options = useMemo(() => r.actions.map((a) => a.label), [r]);
  const g = answer ? grade(r, answer) : null;

  async function choose(opt: string) {
    if (conn === "connected") {
      try {
        const fresh = await analyze(base, spot.state);
        setLive(fresh);
        setAnswer(opt);
        const gr = grade(fresh, opt);
        setTally((t) => ({ ...t, [gr.cls]: (t[gr.cls] ?? 0) + 1 }));
        return;
      } catch {
        /* fall through to bundled */
      }
    }
    setAnswer(opt);
    const gr = grade(r, opt);
    setTally((t) => ({ ...t, [gr.cls]: (t[gr.cls] ?? 0) + 1 }));
  }

  return (
    <main>
      <h1>Trainer</h1>
      <p className="lead">
        Pick your action first; then see the engine&rsquo;s full distribution.
        Accepted solver mixes grade by frequency; uncertain states are
        review-only and never penalized.
      </p>
      <p>
        <ConnectionBadge state={conn} />
      </p>
      <OfflineNote state={conn} />

      <h2>
        Spot {(i % spots.length) + 1} of {spots.length}
      </h2>
      <p className="muted">{spot.name}</p>
      <div style={{ display: "flex", gap: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
        <Cards text={spot.state.hero_cards ?? ""} />
        <span className="muted">|</span>
        <Cards text={spot.state.board ?? ""} small />
        <span className="muted small mono">pot {spot.state.pot}</span>
      </div>

      {!answer ? (
        <div style={{ display: "flex", gap: "0.6rem", margin: "1rem 0", flexWrap: "wrap" }}>
          {options.map((o) => (
            <button key={o} onClick={() => choose(o)}>
              {o}
            </button>
          ))}
        </div>
      ) : (
        <>
          <p style={{ margin: "1rem 0" }}>
            You chose <b>{answer}</b> —{" "}
            <span className={`badge ${g!.cls}`}>{g!.head}</span>
          </p>
          <DecisionView r={r} />
          <p style={{ marginTop: "1rem" }}>
            <button
              className="primary"
              onClick={() => {
                setI(i + 1);
                setAnswer(null);
                setLive(null);
              }}
            >
              Next spot
            </button>
          </p>
        </>
      )}

      {Object.keys(tally).length > 0 && (
        <p className="muted small" style={{ marginTop: "1.6rem" }}>
          This session:{" "}
          {Object.entries(tally)
            .map(([k, v]) => `${k === "good" ? "common" : k === "warn" ? "mixed/rare" : "review-only"} ${v}`)
            .join(" · ")}{" "}
          — stats stay in this tab.
        </p>
      )}
    </main>
  );
}
