# Solver Red Team (poker_alpha/analysis/redteam.py)

Automated discovery of states where PokerAlpha is fragile.

**State generation:** random legal self-play through the real Hold'em
engine at ~100bb HU (the depth the release abstraction covers),
snapshotted at hero decision points. Never arbitrary tuples; validators
are never weakened. Seeded and reproducible (`sample_states(n, seed)`).

**Objectives are separate, never combined into one score:**
gate rejection · cross-seed disagreement signal · training-visit
scarcity · bet-translation distance (relative distance of the faced bet
from the nearest trained pot fraction 33/75/150%) · local sensitivity
(policy TV under a ±10% bet/pot perturbation) · archetype sensitivity
(max TV across the four behavior models).

**Clustering:** interpretable rules only — (street × SPR bucket ×
faced-bet bucket) over flagged states, each cluster labeled
"descriptive grouping, not a causal diagnosis".

**Statistical discipline:** random search over many states WILL surface
extreme values by chance. Everything in a red-team report is a
DISCOVERY result; promotion to a claim requires replication with
independent seeds/samples (discovery/confirmation split). The report
says this in its own `discipline` field.

Run: `python scripts/run_research_audit.py` (writes
`results/research/redteam_seed<k>.json`; the web page consumes a
trimmed copy). A deterministic rule set turns measured clusters into a
research backlog (e.g. HIGH_SEED_DISAGREEMENT with sufficient visits →
abstraction-collision investigation) — no free-form generation.
