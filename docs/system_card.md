# PokerAlpha System Card

## Intended use
Research, study, training, simulation and POST-SESSION analysis of
heads-up no-limit Hold'em and small hidden-information games; an
interactive laboratory for solver behavior, uncertainty and failure.
Real-time assistance only in private / play-money / controlled test
environments where assistance is permitted.

## Not intended / refused uses
Automating actions on poker clients (no clicking, betting, or decision
submission — not implemented anywhere); stealth or anti-detection of any
kind; real-money profit claims; use as a "GTO" oracle.

## Algorithm
CFR / CFR+ / external-sampling MCCFR, implemented in this project.
Hold'em runs on a locked imperfect-recall abstraction (compact encoder,
legal NLHE sizing, exact river percentile buckets) — which voids CFR's
equilibrium guarantee; correctness evidence comes from exactly solvable
games instead.

## Training
Release strategy: external-sampling MCCFR, 2,000,000 iterations, native
C++17 backend (bit-exact vs Python), 3 independent seeds, exact
checkpoint resume. Artifacts SHA-bound to their confidence tables.

## Validation
Nine-rung ladder (see /evidence): exact Kuhn/Leduc equilibria → exact
reduced-Hold'em subgames (abstraction error measured directly) →
cross-language bitwise parity → exhaustive evaluator parity →
checkpoint identity → multi-seed disagreement → confidence calibration
on 8,880 exact EV-regret targets → restricted-LBR exploitability floor.

## Uncertainty (kept separate, never one score)
training (cross-seed disagreement) · abstraction (measured subgame
error; permanent flop/turn LOW caps) · Monte Carlo (rollout paired SEs)
· opponent-model (archetype sensitivity) · perception (observer
confidence; real-data validation BLOCKED) · state-mapping (off-tree
translation flags).

## Known limitations
Restricted LBR shows ≥ ~100 bb/100 exploitability remains in the full
game. The preflop gate rejects widely (honest: seeds disagree there).
Rollout fallback uses a simple opponent model. Screen recognition is
unvalidated on real data and excluded from the product.

## Privacy / safety
Local-first: API binds 127.0.0.1 and refuses other hosts; no accounts,
no analytics, no uploads; receipts contain no paths or identifiers;
Arena and fingerprints are browser-local.

# Strategy Card — holdem_v2_native_seed0_2m.npz
config: HoldemSolverConfig v2 (compact_river_pct20, raise-to multiples
2/2.5/3.5x, NLHE min-raise enforced) · iterations: 2,000,000 · seed: 0 ·
infosets: 144,025 · confidence table: SHA-bound, seeds 0-2, recent
movement 1.5M→2M · validation: see ladder above · status: EXPERIMENTAL —
an abstract MCCFR average strategy, measurably exploitable, not an
equilibrium, not GTO · known failure clusters: preflop seed
disagreement, flop/turn abstraction risk, off-tree sizes (see
results/research/redteam_seed7.json).
