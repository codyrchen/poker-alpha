import type { Metadata } from "next";
import Link from "next/link";
import { FEATURED } from "@/lib/demos";

export const metadata: Metadata = {
  title: "Demo",
  description:
    "Try PokerAlpha: live CFR in your browser, a Kuhn poker arena, interactive Phase II results, research explorers and local-engine tools.",
};

const EXPLORERS: [string, string, string][] = [
  ["/surface", "Decision surface", "81 engine analyses over bet size and stack depth, showing where the recommendation flips."],
  ["/trust", "Trust map", "169 preflop hands × 3 nodes with gate verdicts, visits and cross-seed disagreement."],
  ["/microscope", "Solver microscope", "One information set followed across training checkpoints and seeds."],
  ["/atlas", "Policy atlas", "6,000 stored policies projected to two dimensions."],
  ["/break-it", "Red team", "Automated search for fragile states, with an independent confirmation run."],
  ["/replay", "Hand replay", "One decision with the modelled opponent range, street by street."],
  ["/agents", "Agent arena", "Seat-swapped agent tournaments with confidence intervals."],
  ["/games", "Game lab", "Kuhn, Leduc and an auction duel on the same solver stack, with exact exploitability."],
];

// [href, title, what it does, what you see on the public site]
const LOCAL: [string, string, string, string][] = [
  ["/analyze", "Analyzer", "Runs your own hand state through the full decision pipeline.", "bundled example spots"],
  ["/strategy", "Strategy explorer", "Browses the release strategy's preflop matrix node by node.", "bundled preflop nodes"],
  ["/study", "Session review", "Reviews an uploaded hand history after the session.", "a notice; uploads need the engine"],
  ["/trainer", "Trainer", "Asks you to play a spot and grades you against the engine's mix.", "bundled demo spots"],
];

const SHOTS: { src: string; title: string; caption: string; small?: boolean }[] = [
  { src: "/screens/learn.webp", title: "Watch a solver learn", caption: "After 10,000 CFR iterations in the browser. The curve is exact exploitability, recomputed as it trains." },
  { src: "/screens/arena.webp", title: "Arena", caption: "Kuhn poker against the CFR policy. Every hand is dealt and played in the browser." },
  { src: "/screens/phase2.webp", title: "Phase II results", caption: "Abstraction refinement against capacity-matched controls, from exact river subgames." },
  { src: "/screens/engineering.webp", title: "Engineering", caption: "How the C++ training backend went from 21.9 to 1,625 iterations per second." },
  { src: "/screens/desktop-overlay.png", title: "Desktop overlay", caption: "The Electron overlay built from source, showing demo data. Desktop builds are not released yet.", small: true },
];

export default function DemoPage() {
  return (
    <main>
      <div className="eyebrow">Demo</div>
      <h1>Try PokerAlpha</h1>
      <p className="lead" style={{ marginTop: "0.6rem" }}>
        Start with the three demos below. They run on this site with nothing to install. The research
        explorers show precomputed engine output, and a few tools need the engine on your own machine.
      </p>

      <section className="block" style={{ marginTop: "2.4rem" }}>
        <div className="eyebrow">Live in-browser demos</div>
        <div className="feature-demos" style={{ marginTop: "0.6rem" }}>
          {FEATURED.map((d) => (
            <Link key={d.href} className="feature" href={d.href}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={d.img} alt={d.alt} loading="lazy" width={1000} height={625} />
              <div className="body">
                <div className="tag">{d.tag}</div>
                <h3>{d.title}</h3>
                <p>{d.text}</p>
                <p className="scope">{d.scope}</p>
                <span className="go">{d.cta} →</span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="block" style={{ marginTop: "3.4rem" }}>
        <div className="eyebrow">Research explorers</div>
        <h2>Look inside the trained strategy</h2>
        <p className="intro">
          Precomputed output of the real engine, stamped with the commit that produced it. These run
          on the public site.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <tbody>
              {EXPLORERS.map(([href, title, what]) => (
                <tr key={href}>
                  <td style={{ whiteSpace: "nowrap" }}><Link href={href}>{title}</Link></td>
                  <td className="muted">{what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block" style={{ marginTop: "3.4rem" }}>
        <div className="eyebrow">Local engine features</div>
        <h2>Tools that need the engine on your machine</h2>
        <p className="intro">
          Live analysis requires the PokerAlpha engine running locally
          (<code className="mono">python -m poker_alpha.api</code>), and it only accepts requests from a
          copy of this site served on your own machine. On the public site these pages show what is
          listed in the last column, not live analysis. <Link href="/download">How to run it locally</Link>.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr><th>tool</th><th>with the local engine</th><th>on this site</th></tr>
            </thead>
            <tbody>
              {LOCAL.map(([href, title, what, publicView]) => (
                <tr key={href}>
                  <td style={{ whiteSpace: "nowrap" }}><Link href={href}>{title}</Link></td>
                  <td className="muted">{what}</td>
                  <td className="muted small">{publicView}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block" style={{ marginTop: "3.4rem" }}>
        <div className="eyebrow">Screenshots</div>
        <h2>What the application looks like</h2>
        <p className="intro">Captures of the current pages and the desktop overlay, not mockups.</p>
        <div className="shots">
          {SHOTS.map((s) => (
            <figure key={s.src} className={s.small ? "small" : undefined}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={s.src} alt={`${s.title}: ${s.caption}`} loading="lazy" />
              <figcaption><b>{s.title}.</b> {s.caption}</figcaption>
            </figure>
          ))}
        </div>
        <p className="small muted" style={{ marginTop: "1.2rem" }}>
          New here? The <Link href="/tour">ten-step tour</Link> walks through the project in order.
        </p>
      </section>
    </main>
  );
}
