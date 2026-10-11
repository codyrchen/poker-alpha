"use client";

// Watch a Solver Learn: vanilla CFR on Kuhn poker running LIVE in your
// browser — the same algorithm family as PokerAlpha's Hold'em training,
// validated against the Python reference (see kuhn.test.ts).

import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import ref from "@/data/lab/kuhn_reference.json";
import { LineChart, MixBar, MixLegend, Question } from "@/components/Lab";
import { CARD_NAMES, INFOSET_KEYS, KuhnCFR, exploitability, gameValue } from "@/lib/kuhn";

const refPayload = (ref as any).payload;

const KEY_LABELS: Record<string, string> = {};
for (const k of INFOSET_KEYS) {
  const card = CARD_NAMES[Number(k[0])];
  const hist = k.slice(1);
  const situ =
    hist === ""
      ? "first to act"
      : hist === "p"
        ? "opponent checked"
        : hist === "b"
          ? "facing a bet"
          : "checked, now facing a bet";
  KEY_LABELS[k] = `${card} — ${situ}`;
}

export default function LearnPage() {
  const solverRef = useRef(new KuhnCFR());
  const [, bump] = useState(0);
  const [trace, setTrace] = useState<{ x: number; y: number }[]>([]);
  const [running, setRunning] = useState(false);

  const solver = solverRef.current;
  const avg = useMemo(() => solver.averageStrategy(), [solver.iterations]); // eslint-disable-line react-hooks/exhaustive-deps
  const expl = solver.iterations ? exploitability(avg) : 1 / 3;
  const value = solver.iterations ? gameValue(avg) : 0;

  function run(n: number) {
    // chunked so the UI paints between batches
    setRunning(true);
    const chunk = Math.min(n, 500);
    const step = () => {
      solver.train(chunk);
      setTrace((t) => [
        ...t,
        { x: solver.iterations, y: exploitability(solver.averageStrategy()) },
      ]);
      bump((v) => v + 1);
      n -= chunk;
      if (n > 0) setTimeout(step, 0);
      else setRunning(false);
    };
    step();
  }

  function reset() {
    solver.reset();
    setTrace([]);
    bump((v) => v + 1);
  }

  return (
    <main>
      <h1>Watch a solver learn</h1>
      <Question>
        Can you watch game-theoretic strategy learning actually happen?
        Below, counterfactual regret minimization plays Kuhn poker against
        itself in your browser and converges toward an equilibrium — live,
        not an animation.
      </Question>
      <p className="muted small">
        This is the SAME vanilla-CFR algorithm as PokerAlpha&rsquo;s Python
        reference, ported to TypeScript and test-validated against a committed
        Python-generated trace at 13 checkpoints (strategies agree to 4+
        decimals). Hold&nbsp;em adds abstraction, a 38× C++ backend,
        independent seeds and confidence gating on top of this exact idea.
      </p>

      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: "1.2rem 0" }}>
        <button onClick={() => run(1)} disabled={running}>Step ×1</button>
        <button onClick={() => run(100)} disabled={running}>Run 100</button>
        <button onClick={() => run(1000)} disabled={running}>Run 1,000</button>
        <button onClick={() => run(10000)} disabled={running} className="primary">
          Run 10,000
        </button>
        <button onClick={reset} disabled={running}>Reset</button>
      </div>

      <div className="status" style={{ borderTop: "none" }}>
        <span className="stat">
          <span className="v mono">{solver.iterations.toLocaleString()}</span>
          <span className="k">iterations</span>
        </span>
        <span className="stat">
          <span className="v mono">{expl.toFixed(5)}</span>
          <span className="k">exploitability (chips/hand, exact)</span>
        </span>
        <span className="stat">
          <span className="v mono">{value.toFixed(5)}</span>
          <span className="k">game value (known: −1/18 ≈ −0.05556)</span>
        </span>
      </div>

      {trace.length > 1 && (
        <>
          <h2>Convergence</h2>
          <LineChart
            points={trace}
            logY
            yLabel="exploitability"
            xLabel="iterations"
          />
          <p className="muted small">
            Exploitability = how much an exactly best-responding opponent
            could win per hand. Computed exactly on the full game tree after
            every batch — no sampling.
          </p>
        </>
      )}

      <h2>The strategy, live</h2>
      <MixLegend labels={["p (pass/fold)", "b (bet/call)"]} />
      <table className="data" style={{ marginTop: "0.5rem", maxWidth: "640px" }}>
        <thead>
          <tr>
            <th>situation</th>
            <th style={{ width: "40%" }}>current average strategy</th>
            <th>bet%</th>
            <th>reference @10k</th>
          </tr>
        </thead>
        <tbody>
          {INFOSET_KEYS.map((k) => {
            const s = avg[k] ?? { p: 0.5, b: 0.5 };
            const r = refPayload.final_policy[k];
            return (
              <tr key={k}>
                <td>{KEY_LABELS[k]}</td>
                <td>
                  <MixBar mix={{ p: s.p, b: s.b }} />
                </td>
                <td className="mono">{(s.b * 100).toFixed(1)}%</td>
                <td className="mono muted">{(r.b * 100).toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        Watch the Jack rows develop a genuine BLUFFING frequency and the King
        facing a bet converge to always-call — mixing is the equilibrium, not
        indecision. Kuhn has a one-parameter family of equilibria, so exact
        frequencies can differ between runs while exploitability still goes
        to zero.
      </p>

      <h2>Where this leads</h2>
      <p>
        PokerAlpha&rsquo;s Hold&nbsp;em strategy was trained by the sampling
        variant of this algorithm (external-sampling MCCFR) over an
        abstraction with 144,025 information sets, accelerated 38.1× by a
        bit-exact C++17 backend, across three independent seeds — and then
        gated by a calibrated confidence layer because, unlike here,
        Hold&nbsp;em&rsquo;s exploitability cannot be computed exactly.
      </p>
      <p>
        <Link className="btn" href="/engineering">The 38× engineering story</Link>{" "}
        <Link className="btn" href="/arena">Play against the trained Kuhn policy</Link>{" "}
        <Link className="btn" href="/lab">Research Lab</Link>
      </p>
    </main>
  );
}
