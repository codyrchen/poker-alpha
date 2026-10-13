import type { Metadata } from "next";
import Link from "next/link";
import claims from "@/data/lab/claims.json";
import phase2 from "@/data/lab/phase2.json";
import { REPO_URL } from "@/lib/site";

export const metadata: Metadata = {
  title: "Research",
  description: "PokerAlpha research: exact validation, calibrated abstention and Phase II controlled experiments.",
};

const C = (claims as any).payload.claims;
const P2 = (phase2 as any).payload;
const doc = (p: string) => `${REPO_URL}/blob/main/${p}`;
const pct = (x: number) => `${Math.round(100 * x)}%`;

const rel = P2.refinement.relative_vs_uniform;
const trC = P2.translation.C_33_75_150["us=1"].confirmation;
const native = P2.native.trials.at(-1);

// [question, result, status, write-up]
const PHASE2: [string, string, string, string][] = [
  ["Can measured error decide where abstraction capacity goes?",
   `Yes in exact river subgames: ${pct(rel.confirmation)}–${pct(rel.double_capacity)} lower exploitability than capacity-matched uniform refinement.`,
   "positive", "docs/phase2/adaptive_abstraction.md"],
  ["Where does native training time go?",
   `69.5% in river percentile tables; three changes gave ${native.speedup.toFixed(2)}× with an identical digest.`,
   "positive", "docs/native_solver_benchmark.md"],
  ["How costly are off-tree bet sizes?",
   `In exact subgames, nearest-in-log-space translation cost ${trC.nearest_log.toFixed(2)} chips vs ${trC.pseudo_harmonic.toFixed(3)} for pseudo-harmonic mapping (release unchanged).`,
   "positive", "docs/phase2/action_translation.md"],
  ["Does decision-time resolving help, and is it safe?",
   "In exact single-street subgames: the safe gadget never did worse than the blueprint beyond 1e-4 chips at ≥ 1 s; unsafe resolving usually did.",
   "positive", "docs/phase2/local_resolving.md"],
  ["Are red-team failures real, and why?",
   "17 of 18 top clusters replicated on an independent sample; the failures are abstraction-boundary jumps, and a pre-registered 2M → 4M training run did not fix them.",
   "positive", "docs/phase2/red_team_confirmation.md"],
  ["Can compute go where the solver is unsure?",
   "No replicated gain from prioritized sampling, even with an exact oracle signal.",
   "null", "docs/phase2/active_compute.md"],
  ["Is there a better confidence score?",
   "A logistic score raised the false-confidence rate on 8 of 9 held-out games. v2 stays.",
   "negative", "docs/phase2/confidence_ablation.md"],
  ["Does more training always help under imperfect recall?",
   "No. In exact Leduc tests one forgetful abstraction got more exploitable as training continued.",
   "negative", "docs/phase2/abstraction_studies.md"],
  ["Is policy distance a usable quality proxy?",
   "No. A distilled Leduc strategy 0.064 TV from equilibrium was 3,600× more exploitable.",
   "negative", "docs/phase2/strategy_compiler.md"],
];

const BADGE: Record<string, string> = { positive: "good", null: "mute", negative: "warn" };

export default function ResearchPage() {
  return (
    <main>
      <div className="eyebrow">Research</div>
      <h1>What the project measures</h1>
      <p className="lead" style={{ marginTop: "0.6rem" }}>
        Poker is used here as an imperfect-information benchmark. The research questions are about
        the methods, not about winning: train fast, validate against games with exact answers, and
        quantify where the result should not be trusted.
      </p>

      <h2>The solver</h2>
      <p style={{ maxWidth: "46rem" }}>
        External-sampling MCCFR over a locked heads-up no-limit Hold&rsquo;em abstraction (legal
        NLHE sizing, river strength percentiles). It is validated two ways. Exact CFR solutions on
        Kuhn and Leduc match published equilibria, and the abstraction&rsquo;s error is measured
        directly by solving exact subgames and comparing, which is also how the per-street
        confidence caps were derived. The encoder is imperfect-recall, so the trained strategy is an
        abstract heuristic strategy, <b>not an equilibrium</b>; a restricted local best response
        shows it remains measurably exploitable.
      </p>

      <h2>The native backend</h2>
      <p style={{ maxWidth: "46rem" }}>
        The training loop is reimplemented in C++17 with bit-exact parity against the Python
        reference: on a shared random tape, regrets and strategies match to the last bit through{" "}
        {C.parity_iterations.value.toLocaleString("en-US")} iterations (FMA contraction disabled to
        keep IEEE semantics). Warm MCCFR training throughput on the release configuration is{" "}
        <b>{C.native_speedup.value}×</b> the Python reference ({C.native_speedup_phase1.value}× from
        the port, {native.speedup.toFixed(2)}× more from Phase II profiling). That speed is what made
        the 2M-iteration release strategy and the three-seed stability analysis practical.{" "}
        <Link href="/engineering">Engineering details</Link>.
      </p>

      <h2>Calibrated abstention</h2>
      <p style={{ maxWidth: "46rem" }}>
        Every information set carries evidence: training visits, movement between checkpoints,
        disagreement across independently seeded runs, and abstraction-collision dispersion. A gate
        turns that evidence into ACCEPT / LOW CONFIDENCE / REJECT per lookup, and the product obeys
        it: a rejected state shows <b>no numbers</b>, only the reasons and the fallback. The gate is
        calibrated on {C.calibration_rows.value.toLocaleString("en-US")} states with exact EV-regret
        targets, and v2 accepts {C.coverage_gain.value}× as many decisions as v1 at a similar
        accepted error rate.
      </p>

      <h2>Phase II: controlled experiments</h2>
      <p className="muted" style={{ maxWidth: "46rem" }}>
        Every comparison has a capacity- or compute-matched control and an independent confirmation
        run. Exact results come from small or single-street games; none of them changes the release
        strategy. Null and negative results are listed with the same weight as positive ones.
      </p>
      <div style={{ overflowX: "auto", marginTop: "0.9rem" }}>
        <table className="data">
          <thead>
            <tr>
              <th>question</th>
              <th>result</th>
              <th>outcome</th>
            </tr>
          </thead>
          <tbody>
            {PHASE2.map(([q, r, status, path]) => (
              <tr key={q}>
                <td style={{ minWidth: "12rem" }}>{q}</td>
                <td className="small" style={{ minWidth: "16rem" }}>
                  {r} <a href={doc(path)}>write-up</a>
                </td>
                <td><span className={`badge ${BADGE[status]}`}>{status.toUpperCase()}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small" style={{ marginTop: "0.8rem" }}>
        <Link href="/lab/phase2">Charts for the strongest results</Link> ·{" "}
        <a href={doc("docs/phase2/research_report.md")}>Phase II report</a> ·{" "}
        <a href={doc("docs/phase2/hypotheses.md")}>30 follow-up hypotheses</a>
      </p>

      <h2>Deep dives</h2>
      <ul style={{ marginLeft: "1.2rem" }}>
        <li><Link href="/evidence">Evidence</Link>: every headline number with its method and source file</li>
        <li><Link href="/lab">Research lab</Link>: interactive surfaces built on precomputed engine output</li>
        <li><a href={doc("docs/solver_validation.md")}>Validation ladder</a> (exact CFR → exact subgames → LBR)</li>
        <li><a href={doc("docs/solver_confidence_v2.md")}>Confidence system v2</a></li>
        <li><a href={doc("docs/native_solver.md")}>Native solver design and parity methodology</a></li>
        <li><a href={doc("docs/research_details.md")}>Full Phase I research details</a></li>
        <li><a href={doc("docs/system_card.md")}>System card</a>: intended use, refused uses, known limitations</li>
      </ul>
    </main>
  );
}
