# Action translation: exact cost by mapping (Phase II, hypothesis H4)

**Why this experiment.** The red team's confirmed local-sensitivity failures
are overwhelmingly abstraction-boundary effects. A +10 % bet perturbation
changes the abstract key in 80 % of failing states against 4.9 % of controls
(H3, `results/phase2/hypotheses.json`). How off-tree sizes are mapped onto
the tree is therefore a first-order question.

**Setup.** `poker_alpha/research/translation.py`. A strategy is solved in a
river subgame with a coarse menu. The opponent may bet any size in the fine
menu: 25, 50, 75, 100, 150 or 200 % pot. Our strategy translates each
off-tree size and plays from the mapped node. Its exact worst case is the
opponent's best-response value in the fine game. **Cost** is that value
minus the best-response value against the best strategy restricted to the
coarse menu in the same fine game, so the cost is pure translation loss.
Chips of a 10-chip pot. 9 boards x 3 seeds, discovery (seeds 900-902) and
independent confirmation (seeds 4242-4244).

| coarse menu | seat | nearest (pot fraction) | **nearest in log space** (the release's matcher) | pseudo-harmonic (randomized) |
|---|---|---|---|---|
| 50/100 % | OOP | 0.186 / 0.161 | 0.183 / 0.161 | 0.184 / 0.161 |
| 50/100 % | IP | 0.784 / 0.819 | 0.874 / 0.884 | 0.771 / 0.809 |
| 33/75/150 % (release-like) | OOP | 0.081 / 0.065 | 0.112 / 0.096 | **0.033 / 0.024** |
| 33/75/150 % (release-like) | IP | 0.224 / 0.220 | **0.570 / 0.601** | **0.072 / 0.076** |

Cells show discovery / confirmation. Pseudo-harmonic beats nearest-log
in-position on 27/27 (discovery) and 26/27 (confirmation) instances with the
release-like menu.

## Findings

1. **On a release-like menu, nearest-in-log-space mapping costs about 8x
   more than pseudo-harmonic randomized mapping** (Ganzfried & Sandholm,
   IJCAI 2013): 0.60 vs 0.076 chips per hand in position, 6 % vs 0.8 % of the
   pot. Plain nearest-by-pot-fraction sits in between. Randomization removes
   the deterministic threshold that a best responder sizes just past.
2. **Extrapolation dominates the remaining cost.** Adding single off-tree
   sizes one at a time shows that sizes *outside* the menu (25 % below a
   33 % or 50 % minimum, 150 % above a 100 % maximum) cost the same under
   every mapping, because each mapping can only map them to the end of the
   menu. The 200 % size is never worth exploiting here. Inside the menu,
   pseudo-harmonic nearly eliminates the cost (75 % against a 50/100 menu:
   0.005 vs 0.39 for nearest-log).
3. Translation cost is strongly asymmetric. The player who faces the bet
   (IP facing OOP leads, in this tree) pays most of it.

## Relevance to the release (recommendation, not applied)

The release's off-tree matcher (`SolverStrategyProvider._match`) picks the
nearest size in log space of the raise-to amount, deterministically. That
is a close relative of the worst mapping measured here. These are single-
street subgames, not the full game, so the 8x factor is not a release
number. A randomized pseudo-harmonic matcher is a well-founded follow-up
that changes decision support, not training. It is **not** applied, because
changing lookup semantics needs its own validation and the human's sign-off.

Reproduce:

```bash
python experiments/phase2_translation.py --workers 6
python experiments/phase2_translation.py --workers 6 --seed-base 4242 --out results/phase2/translation_confirm.json
```
