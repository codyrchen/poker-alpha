// Contract-level checks: the bundled demo data really is pokeralpha.api/v1
// output with the product's semantics intact, and formatting helpers behave.

import { describe, expect, it } from "vitest";
import demo from "../data/demo.json";
import { pct, bb, solverBadgeClass } from "./api";

describe("formatting helpers", () => {
  it("formats percentages and handles nulls", () => {
    expect(pct(0.64)).toBe("64%");
    expect(pct(0)).toBe("0%");
    expect(pct(null)).toBe("—");
    expect(bb(null)).toBe("—");
    expect(bb(6.25)).toBe("6.3 BB");
  });
  it("maps gate states to semantic badge classes only", () => {
    expect(solverBadgeClass("ACCEPTED")).toContain("good");
    expect(solverBadgeClass("LOW CONFIDENCE")).toContain("warn");
    expect(solverBadgeClass("REJECTED")).toContain("mute");
    expect(solverBadgeClass("OFF")).toContain("mute");
  });
});

describe("bundled demo data (real engine output)", () => {
  const d = demo as any;

  it("is labeled as pre-computed, never live", () => {
    expect(d.generated).toMatch(/pre-computed/i);
    expect(d.generated).toMatch(/never presented as live/i);
  });

  it("carries v1 responses for every demo spot", () => {
    expect(d.demo_spots.length).toBeGreaterThanOrEqual(5);
    for (const s of d.demo_spots) {
      expect(s.result.api).toBe("pokeralpha.api/v1");
      expect(["ACCEPTED", "LOW CONFIDENCE", "REJECTED", "OFF"]).toContain(
        s.result.solver_state,
      );
    }
  });

  it("keeps EV-provenance semantics: solver mixes are never EV-basis", () => {
    for (const s of d.demo_spots) {
      if (s.result.source === "solver") {
        expect(s.result.ev_is_decision_basis).toBe(false);
        expect(s.result.recommended_tag).toBe("HIGHEST FREQUENCY");
      }
    }
    // at least one rollout fallback exists and EV is its basis
    const roll = d.demo_spots.filter((s: any) => s.result.source === "rollout");
    expect(roll.length).toBeGreaterThan(0);
    for (const s of roll) expect(s.result.ev_is_decision_basis).toBe(true);
  });

  it("matrix cells show no numbers where the gate rejects", () => {
    for (const m of Object.values<any>(d.preflop)) {
      expect(m.cells.length).toBe(13);
      for (const row of m.cells)
        for (const c of row) {
          if (c.status === "REJECTED" || c.status === "UNSUPPORTED") {
            expect(c.dominant).toBeNull();
            expect(c.mix.length).toBe(0);
          } else {
            const total = c.mix.reduce((a: number, [, p]: any) => a + p, 0);
            expect(Math.abs(total - 1)).toBeLessThan(1e-6);
          }
        }
    }
  });

  it("release info is honest", () => {
    expect(d.release.iterations).toBe(2000000);
    expect(d.release.disclaimer).toMatch(/not GTO/);
  });
});
