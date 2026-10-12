import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Interactive",
  description: "Interactive PokerAlpha pages: live browser CFR, a Kuhn arena, Phase II results and research tools.",
};

// [href, title, what you do, what it is (scope)]
const START: [string, string, string, string][] = [
  ["/learn", "Watch a solver learn",
   "Run vanilla CFR on Kuhn poker in your browser and watch exploitability fall.",
   "Live computation in your browser, validated against the Python reference."],
  ["/arena", "Arena",
   "Play heads-up Kuhn against the CFR policy, then get the exact exploitability of the strategy you played.",
   "A small-game sandbox. It does not use the Hold'em release strategy."],
  ["/lab/phase2", "Phase II results",
   "Refinement controls, native speedup, a resolving budget slider, translation costs and the negative results.",
   "Precomputed exact results from reduced games."],
];

const DEEPER: [string, string, string][] = [
  ["/surface", "Decision surface", "81 engine analyses over bet size and stack depth; where the recommendation flips"],
  ["/trust", "Trust map", "169 preflop hands × 3 nodes with gate verdicts, visits and cross-seed disagreement"],
  ["/microscope", "Solver microscope", "one information set across checkpoints and seeds"],
  ["/atlas", "Policy atlas", "6,000 stored policies projected to 2-D"],
  ["/break-it", "Red team", "automated search for fragile states, with confirmation"],
  ["/replay", "Hand replay", "one decision with the modelled opponent range street by street"],
  ["/agents", "Agent arena", "seat-swapped tournaments with confidence intervals"],
  ["/games", "Game lab", "Kuhn, Leduc and an auction duel on the same solver stack"],
  ["/tour", "Tour", "the project in ten short steps"],
];

const LOCAL: [string, string, string][] = [
  ["/analyze", "Analyzer", "your own hand states through the full decision pipeline"],
  ["/strategy", "Strategy explorer", "browse the release strategy's preflop matrix"],
  ["/study", "Session review", "upload a hand history for post-session review"],
  ["/trainer", "Trainer", "answer bundled demo spots; graded by frequency against the engine's mix"],
];

export default function InteractivePage() {
  return (
    <main>
      <div className="eyebrow">Interactive</div>
      <h1>Things you can run</h1>
      <p className="lead" style={{ marginTop: "0.6rem" }}>
        Start with the first three. Everything on this site works without installing anything; the
        last group can also talk to a local engine for live analysis.
      </p>

      <section className="block" style={{ marginTop: "2.2rem" }}>
        <div className="demos">
          {START.map(([href, title, what, scope], i) => (
            <Link key={href} className="demo" href={href}>
              <div className="order">{String(i + 1).padStart(2, "0")}</div>
              <h3>{title}</h3>
              <p>{what}</p>
              <p className="scope">{scope}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="block" style={{ marginTop: "3rem" }}>
        <h2>Research surfaces</h2>
        <p className="intro">Precomputed output of the real engine, stamped with the commit that produced it.</p>
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <tbody>
              {DEEPER.map(([href, title, what]) => (
                <tr key={href}>
                  <td style={{ whiteSpace: "nowrap" }}><Link href={href}>{title}</Link></td>
                  <td className="muted">{what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block" style={{ marginTop: "3rem" }}>
        <h2>With a local engine</h2>
        <p className="intro">
          These pages show bundled example output on the public site. Live analysis needs the engine
          running on your machine (<code className="mono">python -m poker_alpha.api</code>), which only
          accepts requests from a locally served copy of this site. <Link href="/download">How to run it</Link>.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <tbody>
              {LOCAL.map(([href, title, what]) => (
                <tr key={href}>
                  <td style={{ whiteSpace: "nowrap" }}><Link href={href}>{title}</Link></td>
                  <td className="muted">{what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
