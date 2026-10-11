# Watch a Solver Learn (/learn)

Vanilla CFR on Kuhn poker, running live in the visitor's browser.

**Backend:** a TypeScript port (`apps/web/src/lib/kuhn.ts`) of the
project's own Python reference — same deterministic full-tree traversal,
simultaneous updates, regret matching with uniform fallback. No WASM was
needed at Kuhn scale: 10,000 iterations run in well under a second.

**Validation (the point):** `kuhn.test.ts` replays the committed
Python-generated trace (`kuhn_reference.json`, 13 checkpoints from 1 to
10,000 iterations) and requires the TS average strategies to match to
4+ decimal places and exact exploitability to 5 — so the browser demo is
provably the same algorithm, not an animation. Exploitability is
computed exactly (information-set-respecting best response on the full
tree) after every batch.

Known-answer checks: game value → −1/18; King-facing-bet → always call;
Jack-facing-bet → never call. Kuhn has a one-parameter equilibrium
family, so individual frequencies may differ across runs while
exploitability still → 0 (stated on the page).

The page ends by connecting the same algorithm family to the Hold'em
system (abstraction + 74× native backend + seeds + gate).
