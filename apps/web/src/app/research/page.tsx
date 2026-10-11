export default function ResearchPage() {
  return (
    <main>
      <h1>Research</h1>
      <p className="lead">
        PokerAlpha treats poker as an imperfect-information benchmark and
        engineering problem: train fast, validate exactly, and quantify when
        the result should not be trusted.
      </p>

      <h2>The solver</h2>
      <p>
        External-sampling MCCFR over a locked heads-up no-limit Hold&nbsp;em
        abstraction (legal NLHE sizing, river strength percentiles). The
        implementation is validated two ways: exact CFR solutions on Kuhn and
        Leduc poker match published equilibria, and the abstraction&rsquo;s
        error is measured directly by solving exact subgames and comparing —
        which is also how the per-street confidence downgrades were derived.
        The encoder is imperfect-recall, so the trained strategy is an
        abstract heuristic strategy, <b>not an equilibrium</b>; restricted
        local best response shows it remains measurably exploitable.
      </p>

      <h2>The native backend</h2>
      <p>
        The training loop is reimplemented in C++17 with bit-exact parity
        against the Python reference — the regret and strategy tapes match to
        the last bit across seeds (FMA contraction disabled to keep IEEE
        semantics). The measured end-to-end speedup is <b>38.1×</b>, which is
        what made the 2M-iteration release strategy and three-seed stability
        analysis practical.
      </p>

      <h2>Calibrated abstention</h2>
      <p>
        Every information set carries evidence: training visits, movement
        between checkpoints, disagreement across independently seeded runs,
        and abstraction-collision dispersion. A gate turns that evidence into
        ACCEPT / LOW CONFIDENCE / REJECT per lookup — and the product surfaces
        obey it: a rejected state shows <b>no numbers</b>, only the reasons
        and the fallback. The same discipline runs through the study tools:
        EV regret is only ever computed where the EVs are the actual decision
        basis and same-method.
      </p>

      <h2>Deep dives</h2>
      <ul style={{ marginLeft: "1.2rem" }}>
        <li>
          <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/native_solver.md">
            Native solver design and parity methodology
          </a>
        </li>
        <li>
          <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/solver_confidence_v2.md">
            Confidence system v2
          </a>
        </li>
        <li>
          <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/solver_validation.md">
            Validation ladder (exact CFR → exact subgames → LBR)
          </a>
        </li>
        <li>
          <a href="https://github.com/codyrchen/poker-alpha/blob/main/docs/research_details.md">
            Full research details
          </a>
        </li>
      </ul>
    </main>
  );
}
