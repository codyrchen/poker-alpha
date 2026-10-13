# A 90-second walkthrough of the site

Run `./scripts/demo_product.sh` (opens the site) or use the static site.

- **0–10s — what it is.** Homepage: an open research and engineering
  project for decision-making under hidden information. The four headline
  numbers each link to their evidence row.
- **10–20s — watch it learn.** Demo → Watch a solver learn → Run 10,000.
  Exploitability falls live on a log chart; the Jack rows learn to bluff.
  This is the real algorithm in the browser, test-matched to the Python
  solver.
- **20–30s — the 74×.** Engineering: 21.9 → 1,625 it/s training throughput,
  the whole hot path in C++17, and the FMA story (the parity suite caught
  the compiler changing the arithmetic in the last bit).
- **30–40s — validation.** Evidence: exhaustive five-, six- and seven-card
  evaluator parity, bitwise cross-language training parity, exact solved
  games.
- **40–50s — it refuses.** Trust map: a mostly rejected preflop map.
  Independent training runs disagree there, so the gate shows no numbers.
- **50–60s — decision surface.** Call turns into fold as the bet grows;
  confidence degrades across the grid. 81 real engine analyses.
- **60–70s — play it.** Arena: a few hands of Kuhn, then the X-ray with the
  exact exploitability of how you played.
- **70–80s — it attacks itself.** Red team: automated search for fragile
  states, failure clusters, confirmation runs.
- **80–90s — limits and source.** Homepage "What the system does not know",
  then Download and GitHub.
