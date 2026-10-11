// PokerAlpha Arena: heads-up Kuhn poker against the committed
// CFR-trained policy, entirely in the browser. No currency, no stakes —
// chips are the game's utility units. All state is local.
//
// After the session, the X-ray report uses the EXACT evaluation in
// kuhn.ts: the visitor's empirical strategy (with per-infoset sample
// counts) is scored by exact best response, and every hand can be
// replayed against the trained mix.

import {
  ACTIONS,
  ALL_DEALS,
  CARD_NAMES,
  INFOSET_KEYS,
  KuhnAction,
  Strategy,
  bestResponse,
  currentPlayer,
  exploitability,
  infosetKey,
  isTerminal,
  sampleAction,
  utility,
} from "./kuhn";

export interface HandRecord {
  index: number;
  humanSeat: 0 | 1;
  cards: [number, number];
  history: string;
  humanDecisions: { key: string; action: KuhnAction; aiMix: Record<string, number> }[];
  utilityToHuman: number;
}

export interface ArenaSession {
  hands: HandRecord[];
  chips: number; // cumulative utility to the human
}

// Deterministic PRNG (mulberry32) so sessions are replayable from a seed.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class ArenaMatch {
  session: ArenaSession = { hands: [], chips: 0 };
  current: HandRecord | null = null;
  private rng: () => number;

  constructor(
    public aiPolicy: Strategy,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
  }

  /** Start the next hand; human seat alternates (0 first). */
  startHand(): HandRecord {
    const humanSeat = (this.session.hands.length % 2) as 0 | 1;
    const deal = ALL_DEALS[Math.floor(this.rng() * ALL_DEALS.length)];
    this.current = {
      index: this.session.hands.length,
      humanSeat,
      cards: deal,
      history: "",
      humanDecisions: [],
      utilityToHuman: 0,
    };
    this.advanceAI();
    return this.current;
  }

  get toAct(): "human" | "ai" | "done" {
    const h = this.current!;
    if (isTerminal(h.history)) return "done";
    return currentPlayer(h.history) === h.humanSeat ? "human" : "ai";
  }

  /** AI acts until it's the human's turn or the hand ends. */
  private advanceAI(): void {
    const h = this.current!;
    while (!isTerminal(h.history) && currentPlayer(h.history) !== h.humanSeat) {
      const key = infosetKey(h.cards, h.history);
      h.history += sampleAction(this.aiPolicy, key, this.rng());
    }
    if (isTerminal(h.history)) this.settle();
  }

  humanAct(action: KuhnAction): void {
    const h = this.current!;
    if (this.toAct !== "human") throw new Error("not the human's turn");
    const key = infosetKey(h.cards, h.history);
    h.humanDecisions.push({
      key,
      action,
      aiMix: { ...(this.aiPolicy[key] ?? { p: 0.5, b: 0.5 }) },
    });
    h.history += action;
    if (isTerminal(h.history)) this.settle();
    else this.advanceAI();
  }

  private settle(): void {
    const h = this.current!;
    const u0 = utility(h.cards, h.history);
    h.utilityToHuman = h.humanSeat === 0 ? u0 : -u0;
    this.session.hands.push(h);
    this.session.chips += h.utilityToHuman;
  }
}

// ---------------------------------------------------------------------------
// fingerprint: observed tendencies with sample sizes (never personality)
// ---------------------------------------------------------------------------

export interface FingerprintMetric {
  label: string;
  value: number | null; // rate in [0,1], null when insufficient
  n: number; // observations
  se: number | null;
  note: string;
}

const MIN_N = 5;

function rate(hits: number, n: number): { value: number | null; se: number | null } {
  if (n < MIN_N) return { value: null, se: null };
  const p = hits / n;
  return { value: p, se: Math.sqrt((p * (1 - p)) / n) };
}

export function empiricalStrategy(session: ArenaSession): {
  strategy: Strategy;
  counts: Record<string, number>;
} {
  const tally: Record<string, { p: number; b: number }> = {};
  for (const h of session.hands)
    for (const d of h.humanDecisions) {
      tally[d.key] ??= { p: 0, b: 0 };
      tally[d.key][d.action] += 1;
    }
  const strategy: Strategy = {};
  const counts: Record<string, number> = {};
  for (const [k, t] of Object.entries(tally)) {
    const n = t.p + t.b;
    counts[k] = n;
    strategy[k] = { p: t.p / n, b: t.b / n };
  }
  return { strategy, counts };
}

export function fingerprint(session: ArenaSession): FingerprintMetric[] {
  const decs = session.hands.flatMap((h) => h.humanDecisions);
  const firstActs = decs.filter((d) => d.key.length === 1 || d.key.endsWith("p"));
  const facingBet = decs.filter((d) => d.key.endsWith("b"));
  const bluffSpots = decs.filter((d) => d.key[0] === "0" && !d.key.endsWith("b"));
  const valueSpots = decs.filter((d) => d.key[0] === "2" && !d.key.endsWith("b"));
  const surprises = decs
    .map((d) => {
      const p = Math.max(d.aiMix[d.action === "p" ? "p" : "b"] ?? 0.5, 1e-3);
      return -Math.log2(p);
    })
    .filter((x) => isFinite(x));

  const m = (
    label: string,
    hits: number,
    n: number,
    note: string,
  ): FingerprintMetric => ({ label, ...rate(hits, n), n, note });

  const out: FingerprintMetric[] = [
    m(
      "Aggression (bet when not facing a bet)",
      firstActs.filter((d) => d.action === "b").length,
      firstActs.length,
      "share of open/check spots where you bet",
    ),
    m(
      "Fold to a bet",
      facingBet.filter((d) => d.action === "p").length,
      facingBet.length,
      "share of faced bets you folded",
    ),
    m(
      "Bluff rate with the Jack",
      bluffSpots.filter((d) => d.action === "b").length,
      bluffSpots.length,
      "bets holding the worst card (your own card is known, so this IS observable)",
    ),
    m(
      "Value-bet rate with the King",
      valueSpots.filter((d) => d.action === "b").length,
      valueSpots.length,
      "bets holding the best card",
    ),
  ];
  out.push({
    label: "Policy surprise vs trained mix",
    value: surprises.length >= MIN_N
      ? surprises.reduce((a, b) => a + b, 0) / surprises.length
      : null,
    n: surprises.length,
    se: null,
    note: "mean -log2 P(your action) under the trained mix — unusualness, NOT a mistake measure",
  });
  return out;
}

export interface XRay {
  hands: number;
  decisions: number;
  chips: number;
  meanChipsPerHand: number;
  se: number | null;
  verdict: string;
  exploitability: number | null; // of the visitor's empirical strategy
  bestResponsePolicy: Record<string, KuhnAction> | null;
  coverage: { observed: number; total: number };
  empirical: Strategy;
  counts: Record<string, number>;
}

export function xray(session: ArenaSession): XRay {
  const n = session.hands.length;
  const utils = session.hands.map((h) => h.utilityToHuman);
  const mean = n ? utils.reduce((a, b) => a + b, 0) / n : 0;
  const varSum = utils.reduce((a, b) => a + (b - mean) ** 2, 0);
  const se = n > 1 ? Math.sqrt(varSum / (n - 1) / n) : null;
  const { strategy, counts } = empiricalStrategy(session);
  const observed = Object.keys(counts).length;
  // exploitability of the empirical strategy is only meaningful with
  // reasonable coverage; unobserved infosets default to uniform and are
  // reported as such
  const enough = n >= 10 && observed >= 6;
  let expl: number | null = null;
  let brPolicy: Record<string, KuhnAction> | null = null;
  if (enough) {
    const full: Strategy = {};
    for (const k of INFOSET_KEYS) full[k] = strategy[k] ?? { p: 0.5, b: 0.5 };
    expl = exploitability(full);
    brPolicy = { ...bestResponse(full, 0).policy, ...bestResponse(full, 1).policy };
  }
  let verdict: string;
  if (n < 10) verdict = `Only ${n} hands — far too few to say anything.`;
  else if (se !== null && Math.abs(mean) < 1.96 * se)
    verdict = `${mean >= 0 ? "+" : ""}${mean.toFixed(2)} chips/hand over ${n} hands is within noise (±${(1.96 * se).toFixed(2)}).`;
  else
    verdict = `${mean >= 0 ? "+" : ""}${mean.toFixed(2)} chips/hand over ${n} hands (95% CI ±${(1.96 * (se ?? 0)).toFixed(2)}). Still a small sample.`;
  return {
    hands: n,
    decisions: session.hands.reduce((a, h) => a + h.humanDecisions.length, 0),
    chips: session.chips,
    meanChipsPerHand: mean,
    se,
    verdict,
    exploitability: expl,
    bestResponsePolicy: brPolicy,
    coverage: { observed, total: INFOSET_KEYS.length },
    empirical: strategy,
    counts,
  };
}

export { ACTIONS, CARD_NAMES };
