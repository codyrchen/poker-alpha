// Arena game integrity: legal transitions, correct settlement, honest
// statistics, deterministic replay from a seed.

import { describe, expect, it } from "vitest";
import ref from "../data/lab/kuhn_reference.json";
import { ArenaMatch, empiricalStrategy, fingerprint, mulberry32, xray } from "./arena";
import { isTerminal, utility } from "./kuhn";

const policy = (ref as any).payload.final_policy;

function playSession(seed: number, hands: number, chooser: (key: string, u: number) => "p" | "b") {
  const match = new ArenaMatch(policy, seed);
  const rng = mulberry32(seed ^ 0x9e3779b9);
  for (let i = 0; i < hands; i++) {
    match.startHand();
    while (match.toAct === "human") {
      const h = match.current!;
      const key = `${h.cards[h.humanSeat]}${h.history}`;
      match.humanAct(chooser(key, rng()));
    }
  }
  return match;
}

describe("arena integrity", () => {
  it("hands terminate legally and settle with the rules' utilities", () => {
    const m = playSession(42, 50, (_k, u) => (u < 0.5 ? "p" : "b"));
    expect(m.session.hands.length).toBe(50);
    let chips = 0;
    for (const h of m.session.hands) {
      expect(isTerminal(h.history)).toBe(true);
      const u0 = utility(h.cards, h.history);
      expect(h.utilityToHuman).toBe(h.humanSeat === 0 ? u0 : -u0);
      chips += h.utilityToHuman;
      // human decisions recorded with the AI mix at that infoset
      for (const d of h.humanDecisions) {
        expect(["p", "b"]).toContain(d.action);
        expect(d.aiMix.p + d.aiMix.b).toBeCloseTo(1, 6);
      }
    }
    expect(m.session.chips).toBe(chips);
  });

  it("seat alternates and replays deterministically from the seed", () => {
    const a = playSession(7, 20, () => "b");
    const b = playSession(7, 20, () => "b");
    expect(a.session.hands.map((h) => h.history)).toEqual(
      b.session.hands.map((h) => h.history),
    );
    expect(a.session.hands.map((h) => h.humanSeat)).toEqual(
      a.session.hands.map((_, i) => (i % 2) as 0 | 1),
    );
  });

  it("rejects out-of-turn actions", () => {
    const m = new ArenaMatch(policy, 1);
    m.startHand();
    while (m.toAct === "human") m.humanAct("p");
    expect(() => m.humanAct("p")).toThrow();
  });
});

describe("x-ray and fingerprint honesty", () => {
  it("small samples are reported as insufficient, never dramatized", () => {
    const m = playSession(3, 3, () => "b");
    const x = xray(m.session);
    expect(x.verdict).toContain("too few");
    expect(x.exploitability).toBeNull();
    for (const f of fingerprint(m.session)) {
      if (f.n < 5) expect(f.value).toBeNull();
    }
  });

  it("always-bet play is measured as highly exploitable", () => {
    const m = playSession(11, 60, () => "b");
    const x = xray(m.session);
    expect(x.exploitability).not.toBeNull();
    expect(x.exploitability!).toBeGreaterThan(0.2);
    expect(x.bestResponsePolicy).not.toBeNull();
    const fp = fingerprint(m.session);
    const aggression = fp.find((f) => f.label.startsWith("Aggression"))!;
    expect(aggression.value).toBe(1);
    expect(aggression.se).not.toBeNull();
  });

  it("empirical strategy matches recorded decisions exactly", () => {
    const m = playSession(13, 40, (_k, u) => (u < 0.3 ? "b" : "p"));
    const { strategy, counts } = empiricalStrategy(m.session);
    let total = 0;
    for (const [k, n] of Object.entries(counts)) {
      total += n;
      expect(strategy[k].p + strategy[k].b).toBeCloseTo(1, 9);
    }
    expect(total).toBe(
      m.session.hands.reduce((a, h) => a + h.humanDecisions.length, 0),
    );
  });
});
