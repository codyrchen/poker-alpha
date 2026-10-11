# The 90-second recruiter demo

Run `./scripts/demo_product.sh` (opens the site) or just the static site.

- **0–10s — what it is.** Homepage hero: "I built an end-to-end poker AI
  research platform." Point at the What-I-Built pipeline: engine, solver,
  C++ trainer, validation, confidence, product.
- **10–20s — watch it learn.** /learn → Run 10,000. Exploitability falls
  live on a log chart; the Jack rows literally learn to bluff. "This is
  the real algorithm in the browser, test-matched to the Python solver."
- **20–30s — the 74×.** /engineering: 22 → 1,625 it/s, whole hot path in
  C++17, and the FMA story ("the parity suite caught the compiler
  changing the algorithm in the last bit").
- **30–40s — validation.** /evidence ladder: all 2,598,960 five-card
  hands; bitwise cross-language parity; exact solved games.
- **40–50s — it refuses.** /trust: a mostly-rejected preflop map. "Three
  independent training runs disagree there, so it says 'I don't know'
  instead of guessing."
- **50–60s — Decision Surface.** /surface: call turns into fold as the
  bet grows; the gate degrades across the grid. 81 real analyses.
- **60–70s — play it.** /arena: three hands of Kuhn; then the X-ray:
  "it computed the exact exploitability of how I just played."
- **70–80s — it attacks itself.** /break-it: automated red team,
  failure clusters, the rule-generated research backlog.
- **80–90s — bigger than poker.** /games: same solver, three games,
  including a sealed-bid auction. Close on /overview + GitHub.
