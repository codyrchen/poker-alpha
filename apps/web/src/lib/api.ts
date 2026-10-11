// pokeralpha.api/v1 client. The web app NEVER computes poker itself — it
// talks to the local Python engine, or falls back to bundled pre-computed
// outputs of that same engine, clearly labeled.

export const API_VERSION = "pokeralpha.api/v1";
export const DEFAULT_API = "http://127.0.0.1:8765";

export interface ActionOut {
  label: string;
  kind: string;
  amount_to: number | null;
  frequency: number | null;
  ev_bb: number | null;
  ev_se_bb: number | null;
  source: string;
  recommended: boolean;
}

export interface AbstentionOut {
  withheld: boolean;
  title: string;
  reasons: string[];
  reason_codes: string[];
  fallback: string | null;
}

export interface AnalyzeResponse {
  api: string;
  state_summary: string;
  actions: ActionOut[];
  recommended: string | null;
  recommended_tag: string;
  mix_meaning: string;
  method: string;
  source: string;
  solver_state: "ACCEPTED" | "LOW CONFIDENCE" | "REJECTED" | "OFF";
  decision_confidence: string;
  ev_is_decision_basis: boolean;
  equity: number | null;
  equity_se: number | null;
  pot_bb: number | null;
  to_call_bb: number | null;
  pot_odds: number | null;
  effective_stack_bb: number | null;
  warnings: string[];
  provenance: {
    input_source: string;
    gate_reasons: string[];
    gate_signals: Record<string, number | null>;
    source_cascade: Record<string, unknown>[];
    ev_note: string | null;
  };
  abstention: AbstentionOut | null;
  refusal: { codes: string[]; message: string } | null;
}

export interface MatrixCell {
  hand: string;
  combo: string;
  status: string;
  dominant: string | null;
  dominant_freq: number | null;
  mix: [string, number][];
  visits: number | null;
  reasons: string[];
  notes: string[];
}

export interface PreflopMatrixResponse {
  api: string;
  node: string;
  nodes: string[];
  cells: MatrixCell[][];
  stats: Record<string, number>;
  note: string;
}

export interface ReleaseInfo {
  strategy_path: string;
  strategy_sha256: string;
  iterations: number;
  config_signature: string;
  encoder: string;
  disclaimer: string;
}

export type ConnState = "checking" | "connected" | "offline";

export function apiBase(): string {
  if (typeof window !== "undefined") {
    const p = new URLSearchParams(window.location.search).get("api");
    if (p && /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(p)) return p;
  }
  return DEFAULT_API;
}

export async function checkHealth(base: string): Promise<boolean> {
  try {
    const r = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) });
    const j = await r.json();
    return j.status === "ok";
  } catch {
    return false;
  }
}

export async function analyze(
  base: string,
  state: Record<string, unknown>,
  settings?: Record<string, unknown>,
): Promise<AnalyzeResponse> {
  const r = await fetch(`${base}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state, settings }),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export async function preflopMatrix(
  base: string,
  node: string,
): Promise<PreflopMatrixResponse> {
  const r = await fetch(`${base}/strategy/preflop?node=${encodeURIComponent(node)}`);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function sessionReview(
  base: string,
  hands: Record<string, unknown>,
): Promise<Record<string, any>> {
  const r = await fetch(`${base}/session/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hands }),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.review;
}

// Formatting helpers shared by every page (semantics live in the engine;
// these only render).
export const pct = (f: number | null | undefined): string =>
  f == null ? "—" : `${Math.round(f * 100)}%`;

export const bb = (x: number | null | undefined): string =>
  x == null ? "—" : `${x.toFixed(1)} BB`;

export const solverBadgeClass = (s: string): string =>
  s === "ACCEPTED" ? "badge good" : s === "LOW CONFIDENCE" ? "badge warn" : "badge mute";
