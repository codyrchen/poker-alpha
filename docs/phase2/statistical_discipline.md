# Statistical discipline audit (Phase II, Program 19)

| check | how Phase II handled it |
|---|---|
| Multiple seeds | stochastic studies used 5-20 seeds (P1: 5 per board, 9 boards; P2: 10 + 20; P3: 10; P15: 5-30); deterministic solvers run once |
| Error bars | bootstrap 95 % CIs on paired differences (P1, P2, P4, P6), Wilson intervals on rates (P10), seed SDs (P3) |
| Discovery vs confirmation | every positive claim re-run on independent seeds or samples: P1 (new seeds + 3 new boards), P2 (seeds 100-119), P10 (independent 10k states + rescoring), H4 (seeds 4242+) |
| Effect sizes | relative improvements, paired mean differences, lifts and AUCs reported with every test |
| Multiple comparisons | P2's per-budget classifications flipped between runs, so P2 is reported as NULL; P10 clusters are ranked on discovery and measured on confirmation only; P15's 28 pairings are Bonferroni-corrected (one nominal z = 2.05 win dropped) |
| Pre-registration | the W1 loop outcome was predicted and recorded before the 4M artifacts existed |
| Null reporting | P2, the W1 loop, confidence v3, H23 (lazy loading) and every NO-GO are reported with the same prominence as positives |
| Metric hygiene | policy distance (TV) is never presented as value; P17 shows why; exploitability is stated in each game's own chip units |
| Bugs found by the audit | P2 node accounting (outcome walk charged to the budget); resolving gadget used profile cfvs; reduced-game sampling cost confounded P2 equal-time numbers (documented, equal-node numbers unaffected) |
