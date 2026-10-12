"use client";

// Research Phase II: the strongest new measured results, each with its
// controls, plus the negative results. Every number comes from
// src/data/lab/phase2.json, built from results/phase2/*.json.

import { useState } from "react";
import data from "@/data/lab/phase2.json";
import { ProvenanceFooter, Question } from "@/components/Lab";

const env = data as any;
const P = env.payload;

const ARM_LABEL: Record<string, string> = {
  baseline: "baseline (4 cells/player)",
  random: "random splits",
  uniform_matched: "uniform, same capacity",
  "adaptive:equity_spread": "adaptive: equity spread",
  "adaptive:br_gain": "adaptive: measured error",
  oracle: "oracle (try every split)",
  raw: "no abstraction (floor)",
};

function Bars({
  rows,
  fmt = (v: number) => v.toFixed(3),
  highlight,
  label,
}: {
  rows: { name: string; value: number }[];
  fmt?: (v: number) => string;
  highlight?: string;
  label: string;
}) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <div role="img" aria-label={label} style={{ display: "grid", gap: "0.35rem", margin: "0.8rem 0" }}>
      {rows.map((r) => (
        <div key={r.name} style={{ display: "flex", alignItems: "center", gap: "0.6rem", minWidth: 0 }}>
          <div className="small" style={{ flex: "0 0 11.5rem", maxWidth: "45%", overflow: "hidden", textOverflow: "ellipsis" }}>
            {r.name}
          </div>
          <div style={{ flex: 1, minWidth: 0, background: "rgba(138,143,150,0.12)", borderRadius: 4, height: 14 }}>
            <div
              style={{
                width: `${Math.max(1.5, (100 * r.value) / max)}%`,
                height: "100%",
                borderRadius: 4,
                background: r.name === highlight ? "var(--accent)" : "var(--muted)",
              }}
            />
          </div>
          <div className="mono small" style={{ flex: "0 0 4.2rem", textAlign: "right" }}>
            {fmt(r.value)}
          </div>
        </div>
      ))}
    </div>
  );
}

function Refinement() {
  const R = P.refinement;
  const rows = R.arms.map((a: string) => ({ name: ARM_LABEL[a] ?? a, value: R.confirmation[a] }));
  const pu = R.paired_vs_uniform;
  const pr = R.paired_vs_random;
  return (
    <section>
      <h2>1. Where should abstraction capacity go?</h2>
      <Question>
        Can measured strategic error decide which bucket to split, and does that beat spending the same capacity
        uniformly or at random?
      </Question>
      <Bars rows={rows} highlight={ARM_LABEL["adaptive:br_gain"]} label="Exploitability by refinement arm, confirmation runs" />
      <p className="small">
        Exact exploitability (chips, 10-chip pot) of the abstract strategy in the real river subgame, mean over{" "}
        {R.instances} independent confirmation instances. Splitting the cell with the largest measured error is{" "}
        <b>{Math.round(100 * pu.relative_mean)}% better than a uniform partition with identical capacity</b> (
        {pu.wins}/{pu.wins + pu.losses} wins, 95% CI of the gain {pu.ci95[0].toFixed(3)}–{pu.ci95[1].toFixed(3)}) and{" "}
        {Math.round(100 * pr.relative_mean)}% better than random splits. The solver&apos;s own positive regret as a
        signal is no better than random ({R.negative_signal.toFixed(3)}). These are exact single-street subgames; no new
        Hold&apos;em abstraction or release was trained from this result.
      </p>
    </section>
  );
}

function Native() {
  const N = P.native;
  const last = N.trials[N.trials.length - 1];
  const prof = Object.entries(N.profile_inclusive_before as Record<string, number>);
  return (
    <section>
      <h2>2. Profile-driven native training speedup</h2>
      <Question>Where does training time actually go, and can it be removed without changing a single bit of output?</Question>
      <p className="small muted" style={{ marginTop: "0.6rem" }}>
        Inclusive share of training CPU before optimization (functions call each other, so the shares overlap).
      </p>
      <Bars
        rows={prof.map(([k, v]) => ({ name: k, value: v }))}
        fmt={(v) => `${v.toFixed(1)}%`}
        highlight="FeatureCache::river_values"
        label="Inclusive CPU share before optimization"
      />
      <div style={{ overflowX: "auto" }}>
        <table className="data">
          <thead>
            <tr>
              <th>change</th>
              <th>it/s before → after</th>
              <th>speedup</th>
              <th>output</th>
            </tr>
          </thead>
          <tbody>
            {N.trials.map((t: any) => (
              <tr key={t.id}>
                <td className="small">{t.change}</td>
                <td className="mono">
                  {Math.round(t.baseline)} → {Math.round(t.new)}
                </td>
                <td className="mono">{t.speedup.toFixed(2)}×</td>
                <td>{t.identical ? <span className="badge good">bit-identical</span> : <span className="badge err">differs</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small">
        River percentile buckets rebuilt a 1,081-hand table for almost every new board. Deduplicating canonical
        holdings, counting the board once, and a flat hash index made release-config training{" "}
        <b>{last.speedup.toFixed(2)}× faster</b> with an identical strategy digest; exhaustive 6- and 7-card evaluator
        parity and 10,000-iteration Python/native tape parity still hold.
      </p>
    </section>
  );
}

const SETTING_LABEL: Record<string, string> = {
  "k3:ip_depth1": "coarse blueprint (3 cells), resolve after the first action",
  "k6:ip_depth1": "finer blueprint (6 cells), resolve after the first action",
  "k3:oop_depth2": "coarse blueprint, resolve the second action",
  "k6:oop_depth2": "finer blueprint, resolve the second action",
};

function Resolving() {
  const S = P.resolving;
  const keys = Object.keys(S.settings);
  const [setting, setSetting] = useState(keys[0]);
  const [bi, setBi] = useState(2);
  const s = S.settings[setting];
  const b = S.budgets_ms[bi];
  const rows = [
    { name: "static blueprint", value: s.blueprint },
    { name: `unsafe resolve @${b}ms`, value: s.unsafe[bi] },
    { name: `safe resolve @${b}ms`, value: s.safe[bi] },
    { name: "equilibrium (reference)", value: s.equilibrium },
  ];
  const shift = Math.min(...rows.map((r) => r.value)) - 0.05;
  return (
    <section>
      <h2>3. Decision-time resolving: better, and safe?</h2>
      <Question>Can re-solving a subtree at decision time improve a coarse precomputed strategy without making it exploitable?</Question>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.8rem", alignItems: "center" }}>
        <select value={setting} onChange={(e) => setSetting(e.target.value)} aria-label="Blueprint and resolving point">
          {keys.map((k) => (
            <option key={k} value={k}>
              {SETTING_LABEL[k] ?? k}
            </option>
          ))}
        </select>
        <label className="small">
          time budget <b className="mono">{b} ms</b>
          <input
            type="range"
            min={0}
            max={S.budgets_ms.length - 1}
            value={bi}
            onChange={(e) => setBi(Number(e.target.value))}
            style={{ marginLeft: "0.5rem", verticalAlign: "middle" }}
            aria-label="Resolving time budget"
          />
        </label>
      </div>
      <Bars
        rows={rows.map((r) => ({ ...r, value: r.value - shift }))}
        fmt={(v) => (v + shift).toFixed(3)}
        highlight={`safe resolve @${b}ms`}
        label="Opponent best-response value against each strategy (lower is better)"
      />
      <p className="small">
        Exact value of the opponent&apos;s best response (chips; lower is better), mean over {s.n} instances. Unsafe
        resolving is worse than the blueprint in {s.unsafe_worse_count[bi]}/{s.n} instances at this budget; the safe
        re-solve gadget (Burch et al. 2014) in {s.safe_worse_count[bi]}/{s.n}, and never by more than 1e-4 chips at
        budgets of one second or more. Scope: exact single-street river subgames with a coarse research blueprint;
        &ldquo;safe&rdquo; is a measured property there, not a deployment guarantee for full Hold&apos;em, and the
        release does not resolve at decision time.
      </p>
    </section>
  );
}

function Translation() {
  const T = P.translation;
  const menus = Object.keys(T);
  const M = ["nearest", "nearest_log", "pseudo_harmonic"];
  const NAME: Record<string, string> = {
    nearest: "nearest (pot fraction)",
    nearest_log: "nearest in log space (release)",
    pseudo_harmonic: "pseudo-harmonic (randomized)",
  };
  return (
    <section>
      <h2>4. What does an off-tree bet cost?</h2>
      <Question>
        When the opponent bets a size the strategy never trained on, how much worst-case value does each translation
        rule give away?
      </Question>
      <div style={{ overflowX: "auto" }}>
        <table className="data">
          <thead>
            <tr>
              <th>menu / seat</th>
              {M.map((m) => (
                <th key={m}>{NAME[m]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {menus.flatMap((menu) =>
              ["us=0", "us=1"].map((seat) => (
                <tr key={menu + seat}>
                  <td className="small">
                    {menu.replace(/_/g, " ")} · {seat === "us=1" ? "in position" : "out of position"}
                  </td>
                  {M.map((m) => (
                    <td key={m} className="mono">
                      {T[menu][seat].discovery[m].toFixed(3)} / {T[menu][seat].confirmation[m].toFixed(3)}
                    </td>
                  ))}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p className="small">
        Exact translation cost (chips, discovery / independent confirmation), measured in exact single-street river
        subgames. There, on a release-like 33/75/150% menu, the deterministic nearest-in-log-space rule costs roughly 8×
        more than randomized pseudo-harmonic mapping (Ganzfried &amp; Sandholm 2013). This is a reduced-subgame result
        and a recommendation for follow-up; the release still uses its nearest-in-log-space matcher.
      </p>
    </section>
  );
}

function Negatives() {
  return (
    <section>
      <h2>5. What did not work</h2>
      <Question>Which plausible ideas were tested under controls and failed?</Question>
      <ul>
        {P.negatives.map((n: any) => (
          <li key={n.title} style={{ marginBottom: "0.5rem" }}>
            <b>{n.title}</b> <span className="badge mute">{n.result}</span>
            <div className="small muted">{n.detail}</div>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function Phase2Page() {
  return (
    <main>
      <h1>Research Phase II</h1>
      <Question>
        New evidence from controlled experiments on exact reduced games (single-street river subgames, Leduc):
        abstraction refinement, decision-time resolving, action translation; plus native training performance on the
        release configuration, and the ideas that failed. Every comparison has a capacity- or compute-matched control and
        an independent confirmation run.
      </Question>
      <Refinement />
      <Native />
      <Resolving />
      <Translation />
      <Negatives />
      <ProvenanceFooter
        envelope={env}
        file="phase2.json"
        method="Exact exploitability in reduced exact games (river subgames, Leduc); A/B native benchmarks with identical build flags; discovery and independent confirmation seeds. Full write-ups in docs/phase2/."
        limitation="Exact results come from single-street or small games, not full Hold'em; nothing here changes the release strategy."
        reproduce="python scripts/build_phase2_web_data.py"
      />
    </main>
  );
}
