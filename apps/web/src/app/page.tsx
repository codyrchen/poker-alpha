import Link from "next/link";
import claims from "@/data/lab/claims.json";
import oneHand from "@/data/lab/one_hand.json";
import phase2 from "@/data/lab/phase2.json";
import { MixBar } from "@/components/Lab";
import { AUTHOR, REPO_URL } from "@/lib/site";

const C = (claims as any).payload.claims;
const H = (oneHand as any).payload;
const P2 = (phase2 as any).payload;

const gh = (path: string) => `${REPO_URL}/blob/main/${path}`;
const pct = (x: number) => `${Math.round(100 * x)}%`;
const fmtInt = (n: number) => n.toLocaleString("en-US");

// ---- derived only from bundled evidence -------------------------------------
const trials = P2.native.trials as { speedup: number }[];
const riverShare = (P2.native.profile_inclusive_before as Record<string, number>)["FeatureCache::river_values"];
const rel = P2.refinement.relative_vs_uniform;
const trC = P2.translation.C_33_75_150["us=1"].confirmation;
const nested = P2.nested_resolving_confirmation.C_33_75_150;
const resolving = P2.resolving;
const b1s = resolving.budgets_ms.indexOf(1000);
const gapClosed = Object.values(resolving.settings).map(
  (s: any) => (s.blueprint - s.safe[b1s]) / (s.blueprint - s.equilibrium),
);

const HEADLINE = [
  {
    id: "native_speedup",
    n: `${C.native_speedup.value}×`,
    what: "warm MCCFR training throughput of the C++ backend vs the Python reference, release config",
    src: "native_mccfr_v2.json",
  },
  {
    id: "evaluator_hands_7card",
    n: fmtInt(C.evaluator_hands_7card.value),
    what: "seven-card hands checked exhaustively, C++ evaluator against Python, zero mismatches",
    src: "evaluator_stress.json",
  },
  {
    id: "release_iterations",
    n: `${C.training_seeds.value} × ${(C.release_iterations.value / 1e6).toFixed(0)}M`,
    what: "independent seeds × MCCFR iterations behind the release strategy",
    src: "solver_config.py",
  },
  {
    id: "coverage_gain",
    n: `${C.coverage_gain.value}×`,
    what: "decisions the confidence gate accepts (v2 vs v1) at equal measured risk, held-out games",
    src: "confidence_signal_quality.json",
  },
];

const BUILD = [
  {
    title: "Python reference",
    rate: C.python_it_per_s.value,
    note: `${C.python_it_per_s.value} it/s`,
    body: "External-sampling MCCFR in NumPy. Correct, and still the reference everything else is checked against, but a 2M-iteration run would take more than a day.",
    milestone: true,
  },
  {
    title: "Move the whole hot path to C++17",
    rate: C.native_it_per_s_phase1.value,
    note: `${Math.round(C.native_it_per_s_phase1.value)} it/s · ${C.native_speedup_phase1.value}×`,
    body: "Rules, dealing, evaluator, infoset encoder, regret matching, sampling and the recursion itself. Moving only helper functions was tried first and lost its gains to per-node boundary crossings.",
    milestone: true,
  },
  {
    title: "Profile the native trainer",
    note: `${riverShare}% of CPU in one function`,
    body: "macOS sample on a symbolized -O3 build. Almost every new river board rebuilt a 1,081-hand percentile table, and external sampling rarely sees the same board twice.",
  },
  {
    title: "Deduplicate river percentile construction",
    note: `${trials[0].speedup.toFixed(2)}× (A/B)`,
    body: "A hand's 7-card value depends only on canonical card ids (exact card for the one flush-capable suit, rank otherwise). Evaluate each distinct pair once and keep cumulative counts instead of sorting.",
  },
  {
    title: "Reuse evaluator board counts",
    note: `${trials[1].speedup.toFixed(2)}× cumulative`,
    body: "Rank and suit histograms for the five board cards are built once per table; each holding adds two cards.",
  },
  {
    title: "Flat open-addressing infoset index",
    note: `${trials[2].speedup.toFixed(2)}× cumulative`,
    body: "Linear probing with a splitmix64 hash replaces std::unordered_map on the per-node lookup.",
  },
  {
    title: "Fresh end-to-end benchmark",
    rate: C.native_it_per_s.value,
    note: `${fmtInt(Math.round(C.native_it_per_s.value))} it/s · ${C.native_speedup.value}×`,
    body: "Same script and protocol as the original benchmark: three trials in fresh processes, warm median.",
    milestone: true,
  },
];
const maxRate = C.native_it_per_s.value;

const ARCH: [string, [string, string, string][]][] = [
  ["Interfaces", [
    ["Web lab", "this site; static, bundled research data", "apps/web"],
    ["Desktop app", "Electron with the engine bundled inside", "apps/desktop"],
    ["Local API", "FastAPI, binds 127.0.0.1 only", "poker_alpha/api"],
    ["Screen observer", "research only, excluded from the product", "poker_alpha/observer"],
  ]],
  ["Decisions", [
    ["Decision engine", "ranges, rollouts with paired SEs, solver / rollout / refuse", "poker_alpha/decision"],
    ["Confidence gate", "visits, seed disagreement, movement, street caps", "poker_alpha/decision/solver_gate.py"],
  ]],
  ["Learning", [
    ["CFR · CFR+ · MCCFR", "one generic extensive-form Game interface", "poker_alpha/solvers"],
    ["Abstraction", "compact imperfect-recall encoder, river percentiles", "poker_alpha/abstraction"],
    ["Training", "multi-seed runs, exact resume, SHA-bound artifacts", "experiments"],
  ]],
  ["Core", [
    ["Game engine", "Hold'em rules: side pots, min-raise reopening, all-ins", "poker_alpha/holdem"],
    ["C++17 backend", "pybind11, GIL released, bit-exact", "cpp/src"],
    ["Evaluator", "histogram 5-7 card evaluator, exhaustively pinned", "poker_alpha/poker"],
  ]],
  ["Research tooling", [
    ["Exact games", "Kuhn, Leduc, reduced Hold'em, river subgames", "poker_alpha/games"],
    ["Red team + bench", "state search, confirmation runs, benchmark suite", "poker_alpha/research"],
  ]],
];

const RESULTS = [
  {
    title: "Error-guided abstraction refinement",
    finding: `Splitting the bucket with the largest measured error lowered exploitability ${pct(rel.confirmation)} below a uniform refinement with identical capacity on independent confirmation (${pct(rel.discovery) === pct(rel.double_capacity) ? `${pct(rel.discovery)} in both the discovery and double-capacity runs` : `${pct(rel.discovery)} in discovery, ${pct(rel.double_capacity)} at double capacity`}).`,
    method: `Exact 52-card river subgames, vectorized CFR+, capacity-matched uniform and random controls, ${P2.refinement.instances} confirmation instances on new seeds and boards.`,
    limit: "Single-street, perfect-recall domain. No Hold'em abstraction was retrained from it.",
    href: gh("docs/phase2/adaptive_abstraction.md"),
  },
  {
    title: "Profile-driven native optimization",
    finding: `A further ${trials[2].speedup.toFixed(2)}× training throughput over the first native port, with an identical strategy digest.`,
    method: "Sampling profile, then A/B builds with identical Release flags run in alternating processes; parity suite and sanitizer self-test re-run.",
    limit: "Training throughput only. End-user analysis latency is a separate measurement.",
    href: gh("docs/native_solver_benchmark.md"),
  },
  {
    title: "Off-tree bet translation",
    finding: `With a release-like 33/75/150% menu, randomized pseudo-harmonic mapping cost ${trC.pseudo_harmonic.toFixed(3)} chips per hand against ${trC.nearest_log.toFixed(3)} for nearest-in-log-space, about ${Math.round(trC.nearest_log / trC.pseudo_harmonic)}× less. In a separate run, safe re-solving after the off-tree bet lowered pseudo-harmonic's cost from ${nested["translate:pseudo_harmonic"].toFixed(3)} to ${nested["resolve:safe"].toFixed(3)}.`,
    method: "Exact best response in river subgames where the opponent may bet off-menu; 27 instances, repeated on independent seeds.",
    limit: "Reduced single-street subgames. The release still uses its nearest-in-log-space matcher; this is a recommendation, not a shipped change.",
    href: gh("docs/phase2/action_translation.md"),
  },
  {
    title: "Decision-time resolving",
    finding: `Re-solving a subtree with a safe gadget closed ${pct(Math.min(...gapClosed))} to ${pct(Math.max(...gapClosed))} of a coarse blueprint's gap to equilibrium at a 1 s budget, and from 1 s up it was never worse than the blueprint by more than 1e-4 chips. Unsafe resolving was worse than the blueprint in most instances.`,
    method: "Re-solve gadget with best-response counterfactual values (Burch et al. 2014), exact evaluation, budgets from 10 ms to 5 s.",
    limit: "Exact single-street subgames with a research blueprint. \"Safe\" is measured there, not guaranteed for full Hold'em; the release does not resolve.",
    href: gh("docs/phase2/local_resolving.md"),
  },
  {
    title: "Calibrated abstention",
    finding: `Confidence v2 accepts ${C.coverage_gain.value}× as many decisions as v1 at a similar accepted wrong-action rate (3.0% vs 3.3%).`,
    method: `${fmtInt(C.calibration_rows.value)} information states with exact EV-regret targets from nine solved games, evaluated leave-one-game-out. An ablation showed which signals carry it.`,
    limit: "Calibrated on small exact games; transfer to full Hold'em is an assumption, which is why flop and turn stay capped.",
    href: gh("docs/phase2/confidence_ablation.md"),
  },
];

const FAILS = [
  ["Full Hold'em is not solved.", "The release strategy is an abstract MCCFR average, not an equilibrium. A restricted local best response still finds at least ~100 bb/100 of exploitability."],
  ["The abstraction forgets.", "The encoder is imperfect-recall, which voids CFR's convergence guarantee. In exact Leduc tests, one forgetful abstraction got worse with more training."],
  ["Flop and turn are lossy.", "Exact flop and turn subgames show the encoder merging strategically different hands, so those streets are permanently capped at low confidence."],
  ["Some decisions stay uncertain.", "Seeds disagree on many preflop spots and on jam / 3-bet defense. Doubling training twice removed only about a tenth of the disagreement each time, so the gate rejects there."],
  ["Screen recognition is unvalidated.", "The observer is tested on synthetic frames only. Real-data validation is blocked, so it is excluded from the product."],
  ["No profitability claim.", "Nothing here is evidence of real-money win rate. The project never automates actions on a poker client."],
];

export default function Home() {
  return (
    <main>
      <section className="home-hero">
        <div className="eyebrow">Independent research project</div>
        <h1>PokerAlpha</h1>
        <p className="tagline">A research system for decision-making under hidden information.</p>
        <p className="detail">
          Custom Hold&rsquo;em engine, CFR/MCCFR solvers, a bit-exact C++ training backend, exact
          reduced-game validation, and a calibrated abstention layer. I built it to study how
          imperfect-information systems behave when their abstractions are wrong, their policies
          disagree, and compute is limited.
        </p>
        <div className="cta-row">
          <Link className="btn solid" href="/learn">Explore the system</Link>
          <a className="textlink" href={REPO_URL}>GitHub</a>
          <Link className="textlink" href="/research">Research</Link>
          <Link className="textlink" href="/engineering">Engineering</Link>
        </div>
        <div className="headline-metrics">
          {HEADLINE.map((m) => (
            <a key={m.id} href={`/evidence#${m.id}`}>
              <div className="hm-n">{m.n}</div>
              <div className="hm-what">{m.what}</div>
              <div className="hm-src">{m.src} →</div>
            </a>
          ))}
        </div>
      </section>

      <section className="block">
        <div className="eyebrow">Why it is hard</div>
        <h2>The answer is a distribution, and nobody tells you whether it is right</h2>
        <p className="intro">
          Heads-up no-limit Hold&rsquo;em has around 10<sup>160</sup> states, so any strategy is
          trained on a lossy abstraction whose error has to be measured separately. Correct play is
          randomized, independent training runs disagree with each other, and none of this is
          testable without a fast training loop. A confident-looking policy table says nothing about
          where it can be trusted.
        </p>
      </section>

      <section className="block" id="build">
        <div className="eyebrow">Build story</div>
        <h2>From {C.python_it_per_s.value} to {fmtInt(Math.round(C.native_it_per_s.value))} iterations per second</h2>
        <p className="intro">
          Warm MCCFR training throughput on the release configuration. The two benchmark endpoints
          and the first port come from the same benchmark script; the three optimization steps are an
          A/B experiment between native builds, reported as cumulative speedups.
        </p>
        <div className="build-grid">
          <ol className="timeline">
            {BUILD.map((s) => (
              <li key={s.title} className={s.milestone ? "milestone" : undefined}>
                <div className="tl-head">
                  <span className="tl-title">{s.title}</span>
                  <span className="tl-rate">{s.note}</span>
                </div>
                <div className="tl-body">{s.body}</div>
                {s.rate !== undefined && (
                  <div className="rate-bar" aria-hidden="true">
                    <div style={{ width: `${Math.max(0.6, (100 * s.rate) / maxRate)}%` }} />
                  </div>
                )}
              </li>
            ))}
          </ol>
          <aside>
            <div className="eyebrow">Kept identical along the way</div>
            <dl>
              <div>
                <dt>one boundary crossing per chunk</dt>
                <dd>pybind11 with the GIL released; no per-node callbacks into Python.</dd>
              </div>
              <div>
                <dt>-ffp-contract=off</dt>
                <dd>Clang&rsquo;s FMA fusion changed betting arithmetic in the last ulp. The parity suite caught it.</dd>
              </div>
              <div>
                <dt>shared random tape</dt>
                <dd>Python and C++ training are bit-identical through {fmtInt(C.parity_iterations.value)} iterations.</dd>
              </div>
              <div>
                <dt>exact checkpoint resume</dt>
                <dd>100,000 continuous iterations equal 10 × 10,000 with save and reload in between.</dd>
              </div>
              <div>
                <dt>exhaustive evaluator parity</dt>
                <dd>every 5-, 6- and 7-card hand ({fmtInt(C.evaluator_hands_7card.value)} seven-card hands).</dd>
              </div>
            </dl>
            <p className="small" style={{ marginTop: "1rem" }}>
              <Link href="/engineering">Engineering details →</Link>
            </p>
          </aside>
        </div>
      </section>

      <section className="block">
        <div className="eyebrow">What I built</div>
        <h2>Scope</h2>
        <p className="intro">
          Everything below is implemented in this repository. Off-the-shelf pieces are the
          foundations: NumPy, pybind11, FastAPI, Next.js, Electron and pytest.
        </p>
        <div className="arch" role="list">
          {ARCH.map(([layer, cells]) => (
            <div className="arch-row" key={layer} role="listitem">
              <div className="arch-layer">{layer}</div>
              <div className="arch-cells">
                {cells.map(([name, text, path]) => (
                  <div className="arch-cell" key={name}>
                    <b>{name}</b>
                    <span>{text}</span>
                    <code>{path}</code>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <h3 style={{ marginTop: "2rem" }}>One real decision through the pipeline</h3>
        <div className="pipeline" style={{ maxWidth: "48rem" }}>
          <div className="pipe-step">
            <b>HAND STATE</b>
            <p className="mono small">
              A♥K♦ on Q♠7♦2♣T♥3♥ · pot {H.state.pot}bb · checked to the hero on the river
            </p>
          </div>
          <div className="pipe-step">
            <b>OPPONENT BELIEF</b>
            <p>
              modelled range after three passive streets: {H.range_stages.at(-1).entropy_bits} bits,
              about {H.range_stages.at(-1).effective_combos} effective combos (a belief, not their cards)
            </p>
          </div>
          <div className="pipe-step">
            <b>TRUST CHECK</b>
            <p>
              gate verdict <span className="badge good">ACCEPTED</span>,{" "}
              {fmtInt(Math.round(H.visits))} training visits, low cross-seed disagreement
            </p>
          </div>
          <div className="pipe-step">
            <b>LEARNED STRATEGY</b>
            <div style={{ maxWidth: "380px" }}>
              <MixBar mix={H.mix} height={14} />
              <p className="mono small" style={{ marginTop: "0.3rem" }}>
                {Object.entries(H.mix as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([a, p]) => `${a} ${(p * 100).toFixed(0)}%`)
                  .join(" · ")}
              </p>
            </div>
          </div>
        </div>
        <p className="small muted">
          Precomputed engine output, stamped with the generating commit.{" "}
          <Link href="/replay">Walk through the whole hand</Link>.
        </p>
      </section>

      <section className="block">
        <div className="eyebrow">Research</div>
        <h2>Five results, with their limits</h2>
        <p className="intro">
          Each comes with a matched control and a confirmation run. Most are exact measurements in
          small or single-street games, which is where exact answers exist; none is a full-Hold&rsquo;em
          theorem.
        </p>
        <div className="results">
          {RESULTS.map((r, i) => (
            <div className="result" key={r.title}>
              <div className="idx">{String(i + 1).padStart(2, "0")}</div>
              <div>
                <h3>{r.title}</h3>
                <p className="finding">{r.finding}</p>
              </div>
              <div>
                <p className="fact"><b>Method</b>{r.method}</p>
                <p className="fact"><b>Limit</b>{r.limit}</p>
                <p className="fact"><a href={r.href}>Write-up and data →</a></p>
              </div>
            </div>
          ))}
        </div>
        <p className="small" style={{ marginTop: "1rem" }}>
          <Link href="/lab/phase2">Charts and controls for these results →</Link>{" "}
          <span className="muted">· the null and negative results are on the same page.</span>
        </p>
      </section>

      <section className="block">
        <div className="eyebrow">Where it fails</div>
        <h2>What the system does not know</h2>
        <p className="pullquote">
          The interesting part is not pretending uncertainty disappears. PokerAlpha measures where
          trust breaks down, and refuses to answer there.
        </p>
        <div className="fails">
          {FAILS.map(([t, d]) => (
            <div className="fail" key={t}>
              <b>{t}</b>
              <p>{d}</p>
            </div>
          ))}
        </div>
        <p className="small" style={{ marginTop: "1rem" }}>
          <Link href="/trust">Trust map</Link> · <Link href="/break-it">Red team</Link> ·{" "}
          <a href={gh("docs/system_card.md")}>System card</a>
        </p>
      </section>

      <section className="block">
        <div className="eyebrow">Try it</div>
        <h2>Three things to open</h2>
        <div className="demos">
          <Link className="demo" href="/learn">
            <div className="order">01</div>
            <h3>Watch a solver learn</h3>
            <p>Vanilla CFR on Kuhn poker, running in your browser. The exploitability curve is computed live from the current strategy.</p>
            <p className="scope">Real computation, not an animation. Checked against the Python reference.</p>
          </Link>
          <Link className="demo" href="/arena">
            <div className="order">02</div>
            <h3>Play against it</h3>
            <p>Heads-up Kuhn against the CFR policy, then see the exact exploitability of the strategy you played.</p>
            <p className="scope">A small-game sandbox, not the Hold&rsquo;em release strategy.</p>
          </Link>
          <Link className="demo" href="/lab/phase2">
            <div className="order">03</div>
            <h3>Phase II results</h3>
            <p>Refinement, native speedup, resolving with a budget slider, translation costs, and five ideas that failed.</p>
            <p className="scope">Exact reduced-game results, with their controls.</p>
          </Link>
        </div>
        <p className="small" style={{ marginTop: "0.9rem" }}>
          <Link href="/interactive">All interactive pages →</Link>
        </p>
      </section>

      <section className="block" aria-labelledby="about">
        <div className="eyebrow">About</div>
        <div className="about">
          <div>
            <div className="name" id="about">{AUTHOR.name}</div>
            <div className="muted small">{AUTHOR.school} · {AUTHOR.program}</div>
            <p className="small" style={{ marginTop: "0.5rem" }}>{AUTHOR.interests}</p>
          </div>
          <div className="links">
            {AUTHOR.links.map((l) =>
              l.href ? (
                <a key={l.label} href={l.href}>{l.label}</a>
              ) : (
                <span key={l.label} className="todo" title="Not added yet">{l.label} (soon)</span>
              ),
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
