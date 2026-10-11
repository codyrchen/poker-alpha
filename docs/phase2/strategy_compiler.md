# Strategy compiler and interpretability (Phase II, Program 17)

APPROXIMATION ONLY. Nothing here replaces the release strategy.

`poker_alpha/research/distill.py` is a dependency-free weighted CART whose
leaves hold action distributions. Splits minimize the visit-weighted squared
error of the probability vectors. Features are ordinal encodings parsed from
the release key: street, position, preflop ranks/suitedness/pair, strength
rung or river percentile, draw, nut, blocker, texture, initiative, raises,
faced size, own prior action and SPR. One tree is fitted per legal-action
set. Artifact: `results/phase2/strategy_compiler.json`. Reproduce:
`python experiments/phase2_strategy_compiler.py` (about 1 min).

## Hold'em release: held-out fidelity

The 8 most-visited legal-action sets are used. The split is by key hash,
20 % test. Metrics are visit-weighted over test keys.

| tree depth | total leaves | TV to release | top-action agreement |
|---|---|---|---|
| 0 (one mean policy per action set) | 8 | 0.473 | 52.6 % |
| 2 | 32 | 0.332 | 72.6 % |
| 4 | 128 | 0.238 | 78.3 % |
| 6 | 512 | 0.195 | 82.2 % |
| 8 | 2,036 | 0.173 | 83.7 % |

For scale, two training seeds of the same configuration differ by a
visit-weighted TV of 0.144 (Program 8). Even 2,036 leaves do not reach the
noise floor. Lookup in pure Python is microseconds per decision.

Example extracted rules, depth 4, facing a bet with fold/call only:

```
[49.4% of weight] IF nut <= 0.5 AND strength <= 2.5 AND pct <= 14.5 AND lo_rank <= 3.5 THEN f 0.87, c 0.13
[ 8.5% of weight] IF nut <= 0.5 AND strength <= 2.5 AND pct > 14.5 AND spr > 0.5 THEN f 0.56, c 0.44
[ 6.3% of weight] IF nut > 0.5 AND pct <= 17.5 AND blocker <= -0.5 AND strength <= 6.5 THEN c 1.00, f 0.00
```

The rules are approximations of the release's average behaviour, not
causal explanations.

## Value consequence (exact, Leduc)

The same tree class distils the exact CFR+ Leduc solution (exploitability
0.0001 chips). Exact exploitability of each distilled strategy:

| depth | leaves (288 infosets) | mean TV to CFR+ | exploitability |
|---|---|---|---|
| 0 | 3 | 0.440 | 2.313 |
| 2 | 12 | 0.224 | 0.903 |
| 4 | 47 | 0.142 | 0.706 |
| 6 | 117 | 0.093 | 0.541 |
| 8 | 171 | **0.064** | **0.454** |

**Small policy distance does not imply small value loss.** A strategy only
0.064 TV from equilibrium, with 171 leaves for 288 infosets, is 3,600x more
exploitable than the strategy it copies. Errors land on infosets where a
best responder can concentrate. Two consequences for the project:

* the Program 8 observation that merging the texture feature moves the release by less TV
  than seed noise is **not** evidence that the merge is safe;
* every quality claim in Phase II is stated in exact exploitability or
  value wherever that can be computed, and TV is labelled as a policy
  distance only.
