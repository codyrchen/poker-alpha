"use client";

import { useState } from "react";
import { sessionReview } from "@/lib/api";
import { ConnectionBadge, useConnection } from "@/components/Connection";

export default function StudyPage() {
  const { state: conn, base } = useConnection();
  const [review, setReview] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onFile(f: File) {
    setBusy(true);
    setError(null);
    try {
      const doc = JSON.parse(await f.text());
      setReview(await sessionReview(base, doc));
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  const s = review?.summary;
  return (
    <main>
      <h1>Study — session review</h1>
      <p className="lead">
        Upload a <span className="mono">pokeralpha.hand/v1</span> session and
        the engine re-analyzes every one of your decisions: solver gate state,
        strategy alignment, and ESTIMATED decision regret only where a valid
        same-method comparison exists.
      </p>
      <p>
        <ConnectionBadge state={conn} />
      </p>
      {conn === "offline" && (
        <p className="scope-note">
          Session review needs the local engine; there is no bundled output for uploads. Run{" "}
          <span className="mono">python -m poker_alpha.api</span> and open a locally served copy of this
          site (see <a href="/download">Run locally</a>).
        </p>
      )}
      <p className="muted small">
        The file goes to your local engine on 127.0.0.1 and nowhere else. For
        the full experience (drill-down, patterns, trainer integration) use
        the Streamlit Study mode:{" "}
        <span className="mono">streamlit run poker_alpha/ui/app.py</span>.
      </p>

      <input
        type="file"
        accept=".json"
        disabled={conn !== "connected" || busy}
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      {conn !== "connected" && (
        <p className="muted small">
          Start the local engine first: <span className="mono">python -m poker_alpha.api</span>
        </p>
      )}
      {busy && <p className="muted">Reviewing through the engine…</p>}
      {error && <p className="badge err">{error}</p>}

      {s && (
        <>
          <hr className="hair" />
          <div className="status" style={{ borderTop: "none" }}>
            {[
              ["Hands", s.hands],
              ["Decisions", s.decisions],
              ["Solver accepted", s.solver_accepted],
              ["Low confidence", s.solver_low_confidence],
              ["Rejected", s.solver_rejected],
              ["Valid EV regret", s.regret_eligible],
            ].map(([k, v]) => (
              <span key={k as string} className="stat">
                <span className="v mono">{v as number}</span>
                <span className="k">{k}</span>
              </span>
            ))}
          </div>
          {s.avg_estimated_regret_bb != null && (
            <p className="muted small" style={{ marginTop: "0.8rem" }}>
              Average ESTIMATED decision regret{" "}
              {s.avg_estimated_regret_bb >= 0 ? "+" : ""}
              {s.avg_estimated_regret_bb.toFixed(2)} BB over {s.regret_eligible}{" "}
              eligible decisions — a same-method rollout estimate, not money
              lost; decisions without a valid comparison are never averaged in.
            </p>
          )}
          {(s.warnings ?? []).map((w: string) => (
            <p key={w} className="badge warn">
              {w}
            </p>
          ))}

          <h2>Decisions</h2>
          <div style={{ overflowX: "auto" }}>
            <table className="data">
              <thead>
                <tr>
                  <th>hand</th>
                  <th>street</th>
                  <th>pos</th>
                  <th>actual</th>
                  <th>PokerAlpha</th>
                  <th>freq</th>
                  <th>solver</th>
                  <th>est. regret</th>
                  <th>category</th>
                </tr>
              </thead>
              <tbody>
                {review.decisions.map((d: any, i: number) => (
                  <tr key={i}>
                    <td className="mono">{d.hand}</td>
                    <td>{d.street}</td>
                    <td>{d.position ?? "—"}</td>
                    <td>{d.actual_label ?? d.actual}</td>
                    <td>{d.pokeralpha ?? "—"}</td>
                    <td className="mono">
                      {d.actual_frequency == null
                        ? "—"
                        : `${Math.round(d.actual_frequency * 100)}%`}
                    </td>
                    <td>{d.solver_state}</td>
                    <td className="mono">
                      {d.regret.eligible
                        ? `${d.regret.estimated_regret_bb >= 0 ? "+" : ""}${d.regret.estimated_regret_bb.toFixed(2)} BB`
                        : "unavailable"}
                    </td>
                    <td>{d.category}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </main>
  );
}
