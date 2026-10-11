"use client";

// The 90-second tour: plain English first, technical depth expandable.

import Link from "next/link";
import { useEffect, useState } from "react";

const STEPS: {
  title: string;
  plain: React.ReactNode;
  tech: string;
  link?: [string, string];
}[] = [
  {
    title: "The problem",
    plain: (
      <>
        Poker is hard for software for a specific reason: the most important
        information is <b>hidden</b>. Your opponent&rsquo;s cards are unknown,
        good play requires deliberately <b>randomizing</b> (or you become
        predictable), and full no-limit Hold&rsquo;em has around 10
        <sup>160</sup> states — far too many to ever enumerate.
      </>
    ),
    tech: "Imperfect-information extensive-form games; equilibrium strategies are mixed; exact solution of full HU NLHE is intractable, so everything runs through a measured abstraction.",
  },
  {
    title: "The solver",
    plain: (
      <>
        PokerAlpha learns strategy by <b>playing against itself</b> millions
        of times. After each simulated decision it asks &ldquo;how much
        better would another action have been?&rdquo; and shifts toward
        actions it regrets not taking. Over time that regret-driven process
        provably approaches unexploitable play in solvable games.
      </>
    ),
    tech: "Counterfactual Regret Minimization: vanilla CFR and CFR+ for exact small games; external-sampling MCCFR for Hold'em scale. All implemented in this project, not imported.",
    link: ["/learn", "Watch it learn, live in your browser"],
  },
  {
    title: "Why Python became too slow",
    plain: (
      <>
        The first implementation was Python. Profiling showed the training
        loop spending most of its time evaluating hands and encoding states —
        at ~22 iterations/second, a two-million-iteration experiment was a
        day, not a coffee break.
      </>
    ),
    tech: "~60% of runtime in hand evaluation and river percentile tables; moving only helper functions native was measured and found insufficient — crossing the language boundary per node ate the gains.",
  },
  {
    title: "The 38× C++ rewrite",
    plain: (
      <>
        So the <b>entire training hot path</b> moved to C++17: game rules,
        dealing, hand evaluator, state encoding, regret updates, and the
        recursive traversal itself. Python now crosses into C++ once per
        training chunk. Measured result: <b>22 → 837 iterations/second,
        38.1×</b>.
      </>
    ),
    tech: "pybind11, GIL released, packed 64-bit infoset keys over fixed-size native nodes; one boundary crossing per chunk. Benchmark committed: results/benchmarks/native_mccfr_v1.json.",
    link: ["/engineering", "The full engineering story"],
  },
  {
    title: "How correctness was protected",
    plain: (
      <>
        A faster solver is worthless if it quietly becomes a <i>different</i>{" "}
        solver. The C++ and Python implementations run on a shared random
        tape and must produce <b>bit-for-bit identical</b> results — every
        trajectory, every regret value, for 1,000 iterations. The hand
        evaluator is checked on <b>all 2,598,960</b> possible five-card
        hands.
      </>
    ),
    tech: "The parity suite caught the compiler's FMA contraction changing betting arithmetic in the last ulp; -ffp-contract=off is required. Checkpoints resume bit-identically.",
  },
  {
    title: "Why a trained policy can still be wrong",
    plain: (
      <>
        Convergence isn&rsquo;t enough. The abstraction merges states that
        are not truly identical, and three independently-seeded training runs
        genuinely disagree in places. Rather than hide this, PokerAlpha{" "}
        <b>measures</b> it.
      </>
    ),
    tech: "Exact reduced-game subgames quantify abstraction error (flop/turn keys measurably merge strategically different hands); cross-seed disagreement is computed per infoset; restricted LBR gives an honest exploitability floor (≥ ~100 bb/100 in the full game).",
    link: ["/microscope", "See seeds agree and disagree"],
  },
  {
    title: "Confidence: the system refuses",
    plain: (
      <>
        Before trusting any lookup, a gate checks the evidence for that
        specific state — training visits, seed agreement, known abstraction
        risk. If the evidence is weak, <b>the solver refuses to answer</b>{" "}
        and a clearly-labeled fallback takes over. Refusing is a feature.
      </>
    ),
    tech: "The gate was calibrated against TRUE EV-regret on 8,880 infosets from nine exactly solved games. Key negative result: checkpoint movement predicts regret at no horizon; visits + seed disagreement are the real signals (v2: 1.68× coverage at equal accepted risk).",
    link: ["/trust", "The Trust Map: where it trusts itself"],
  },
  {
    title: "The product",
    plain: (
      <>
        The research engine became usable software: a decision analyzer, a
        study suite with hand replay and honest regret semantics, a strategy
        explorer, a trainer — as a local web app and a standalone desktop
        app with the whole engine bundled inside.
      </>
    ),
    tech: "Local FastAPI engine (127.0.0.1 only), Next.js frontend, Electron overlay with the PyInstaller-frozen engine embedded; warm analysis ~0.1s. Every number carries provenance; analyses can emit reproducibility receipts.",
    link: ["/replay", "One real hand through the whole system"],
  },
  {
    title: "A failure case, on purpose",
    plain: (
      <>
        Ask it something it shouldn&rsquo;t answer — an off-tree bet size, a
        distrusted preflop spot, an impossible state — and you see the point
        of the whole design: the gate fires, the reasons are machine-readable,
        and no confident-looking nonsense is produced.
      </>
    ),
    tech: "An automated red team searches legal states for fragility (seed disagreement, translation distance, local sensitivity) and clusters the failures. The failure list is public.",
    link: ["/break-it", "Try to break it"],
  },
  {
    title: "Source and research",
    plain: (
      <>
        Everything above is open: the engine, the solvers, the C++ backend,
        the validation suites, the calibration studies, and the generators
        that produced every number on this site.
      </>
    ),
    tech: "944 tests at the data-generation commit (see /evidence — the count is machine-checked, never quoted); CI across Python 3.11/3.12, native Linux + macOS arm64, web and desktop. Claims live in a machine-checked registry.",
    link: ["/evidence", "Evidence for every claim"],
  },
];

export default function TourPage() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setI((v) => Math.min(v + 1, STEPS.length - 1));
      if (e.key === "ArrowLeft") setI((v) => Math.max(v - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const s = STEPS[i];
  return (
    <main>
      <h1>The 90-second tour</h1>
      <div className="tour-nav">
        <button onClick={() => setI(Math.max(i - 1, 0))} disabled={i === 0}
                aria-label="previous step">
          ← Back
        </button>
        <div className="tour-progress" aria-hidden>
          <div style={{ width: `${((i + 1) / STEPS.length) * 100}%` }} />
        </div>
        <span className="small muted mono">
          {i + 1}/{STEPS.length}
        </span>
        <button
          className="primary"
          onClick={() => setI(Math.min(i + 1, STEPS.length - 1))}
          disabled={i === STEPS.length - 1}
          aria-label="next step"
        >
          Next →
        </button>
      </div>
      <section className="tour-step" aria-live="polite">
        <h2 style={{ marginTop: 0 }}>
          {i + 1}. {s.title}
        </h2>
        <p style={{ fontSize: "1.08rem", maxWidth: "44rem", lineHeight: 1.65 }}>{s.plain}</p>
        <details style={{ marginTop: "1rem", maxWidth: "44rem" }}>
          <summary className="small">technical detail</summary>
          <p className="muted small" style={{ marginTop: "0.4rem" }}>{s.tech}</p>
        </details>
        {s.link && (
          <p style={{ marginTop: "1.2rem" }}>
            <Link className="btn primary" href={s.link[0]}>
              {s.link[1]}
            </Link>
          </p>
        )}
      </section>
      <p className="muted small">
        Use ← → arrow keys. Done?{" "}
        <Link href="/lab">Open the Research Lab</Link> ·{" "}
        <Link href="/overview">One-page overview</Link>
      </p>
    </main>
  );
}
