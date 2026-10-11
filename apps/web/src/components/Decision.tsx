"use client";

// Shared decision rendering — the same information hierarchy as Play Mode:
// hand facts -> abstention/refusal -> action distribution -> status strip.
// Semantics come from the engine's response; this file only renders.

import { AnalyzeResponse, ActionOut, pct, solverBadgeClass } from "@/lib/api";

const RED = new Set(["h", "d"]);
const SUIT: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };

export function Cards({ text, small }: { text: string; small?: boolean }) {
  const cards = text.split(/[\s,]+/).filter(Boolean);
  if (!cards.length) return <span className="muted small">—</span>;
  return (
    <span className="cardrow">
      {cards.map((c) => (
        <span
          key={c}
          className={`pcard ${RED.has(c[1]) ? "red" : ""} ${small ? "sm" : ""}`}
        >
          {c[0]}
          {SUIT[c[1]] ?? c[1]}
        </span>
      ))}
    </span>
  );
}

function displayName(a: ActionOut): string {
  const k = a.kind || a.label.split("_")[0];
  if (["fold", "check", "call"].includes(k)) return k[0].toUpperCase() + k.slice(1);
  if (k === "all_in") return "All-in";
  const to = a.amount_to != null ? ` ${a.amount_to.toFixed(1)} BB` : "";
  return `${k[0].toUpperCase()}${k.slice(1)}${to}`;
}

export function ActionList({ r }: { r: AnalyzeResponse }) {
  const showEv = r.ev_is_decision_basis;
  return (
    <div className="actions">
      {r.actions.slice(0, 6).map((a) => {
        const f = a.frequency ?? 0;
        const w = Math.round(f * 100) === 0 ? 0 : f * 100;
        return (
          <div key={a.label} className={`action ${a.recommended ? "rec" : ""}`}>
            <span className="name">
              {displayName(a)}
              {a.recommended && <span className="rec-tag">{r.recommended_tag}</span>}
            </span>
            <span className="bar">{w > 0 ? <div style={{ width: `${w}%` }} /> : null}</span>
            <span className="freq">{pct(a.frequency)}</span>
            {showEv && (
              <span className="ev">
                {a.ev_bb == null
                  ? ""
                  : `${a.ev_bb >= 0 ? "+" : ""}${a.ev_bb.toFixed(2)} BB` +
                    (a.ev_se_bb ? ` ±${a.ev_se_bb.toFixed(2)}` : "")}
              </span>
            )}
          </div>
        );
      })}
      <div className="muted small">{r.mix_meaning}</div>
    </div>
  );
}

export function StatusStrip({ r }: { r: AnalyzeResponse }) {
  return (
    <div className="status">
      <span className="stat">
        <span className={solverBadgeClass(r.solver_state)}>{r.solver_state}</span>
        <span className="k">Solver</span>
      </span>
      <span className="stat">
        <span className="v">{r.source}</span>
        <span className="k">Method</span>
      </span>
      <span className="stat">
        <span
          className={`badge ${
            r.decision_confidence === "high"
              ? "good"
              : r.decision_confidence === "medium"
                ? "warn"
                : "mute"
          }`}
        >
          {r.decision_confidence.toUpperCase()}
        </span>
        <span className="k">Decision confidence</span>
      </span>
      {r.equity != null && (
        <span className="stat">
          <span className="v mono">
            {pct(r.equity)}
            {r.equity_se ? ` ±${Math.round(r.equity_se * 100)}%` : ""}
          </span>
          <span className="k">Equity</span>
        </span>
      )}
      {r.pot_odds != null && (
        <span className="stat">
          <span className="v mono">{pct(r.pot_odds)}</span>
          <span className="k">Pot odds</span>
        </span>
      )}
    </div>
  );
}

export function DecisionView({ r, state }: { r: AnalyzeResponse; state?: any }) {
  const hero = (state?.hero_cards as string) ?? "";
  const board = (state?.board as string) ?? "";
  return (
    <div>
      {state && (
        <div style={{ display: "flex", gap: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
          <Cards text={hero} />
          <span className="muted">|</span>
          <Cards text={board} small />
          <span className="muted small mono">
            pot {state.pot} · stacks{" "}
            {state.seats?.map((s: any) => s.stack).join(" / ")}
          </span>
        </div>
      )}
      {r.refusal ? (
        <div className="panel wait">
          <b>{r.refusal.message}</b>
          <div className="muted small">
            No recommendation is shown for a state PokerAlpha cannot trust.
          </div>
        </div>
      ) : (
        <>
          {r.abstention && (
            <div className="panel abstain">
              <h4>{r.abstention.withheld ? "SOLVER WITHHELD" : "LOW CONFIDENCE"}</h4>
              <b>{r.abstention.title}</b>
              <ul className="small" style={{ margin: "0.3rem 0 0 1.1rem" }}>
                {r.abstention.reasons.map((x) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>
              {r.abstention.withheld && r.abstention.fallback && (
                <div className="muted small">Fallback: {r.abstention.fallback}</div>
              )}
            </div>
          )}
          {r.actions.length > 0 && <ActionList r={r} />}
          <StatusStrip r={r} />
          {r.provenance.ev_note && (
            <p className="muted small" style={{ marginTop: "0.6rem" }}>
              {r.provenance.ev_note}
            </p>
          )}
        </>
      )}
    </div>
  );
}
