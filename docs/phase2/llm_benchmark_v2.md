# LLM benchmark v2: preparation without credentials (Phase II, Program 16)

`poker_alpha/llm_bench/v2.py`. **No external model was called.** Real model
runs are blocked on credentials, and the repository ships no responses.

## Dataset (`pokeralpha.llmbench/v2`, 120 items)

| game | infosets | why it is included |
|---|---|---|
| Kuhn | 12 | complete exact game |
| Leduc, round 1 | 18 | two-round poker; round-2 play is completed with the CFR+ reference |
| Auction duel (3 values x 3 bids) | 6 | non-poker, simultaneous-move structure |
| Signaling (2 types x 2 messages, costly) | 4 | one-sided private information, informative messages |

Each infoset is rendered in three representations: structured JSON, natural
prose and compact key=value. **Prompts are generated from the acting
player's information-set key only**, so hidden information cannot leak by
construction. A test also checks that Kuhn deals differing only in the
opponent's card produce identical prompts, and that no prompt names an
unheld card.

## Strict parsing and metrics

The parser never coerces. It reports no JSON, bad schema, illegal action
and unnormalized probabilities as separate categories. Per game:

* parse-failure rate and illegal-action rate;
* inconsistent-action rate: the chosen action has zero stated probability;
* repeat determinism: each item asked twice, same answer;
* **reference-completed exploitability**, exact: unanswered infosets play
  the CFR+ reference, and the count of answered infosets is reported;
* mean TV distance to the reference;
* representation instability: TV between the policies induced by the three
  representations.

## Mock agents (each must be caught; `tests/test_llm_bench_v2.py`)

| mock | Kuhn exploitability (JSON rep.) | determinism | parse failures | detected by |
|---|---|---|---|---|
| perfect (reference mix) | 0.000 | 1.00 | 0 % | baseline |
| always first action | 1.000 | 1.00 | 0 % | exploitability |
| random | 0.458 | 0.39 | 0 % | determinism, exploitability |
| malformed (rotating failures) | 0.157 | n/a | 81 % | all four parse categories plus inconsistency |
| prompt-sensitive (good only on JSON) | 0.000 | 1.00 | 0 % | representation instability (TV > 0.1) |

The prompt-sensitive mock scores a perfect exploitability on the JSON
representation. Only the cross-representation metric exposes it, which is
the reason that metric exists. Artifact: `results/phase2/llm_bench_mocks.json`.

## Provider plugin

`AnthropicProvider` imports the official SDK lazily and resolves
credentials from the environment, so none are embedded. The model comes
from `POKERALPHA_LLM_MODEL`, default `claude-opus-5-5`. It enables
server-side refusal fallback (`fallbacks="default"`). A final refusal is
returned as empty text and is recorded as a parse failure, never as an
invented answer. The existing `provider_from_env` names for other providers
remain credential-gated stubs with no network code.

```python
from poker_alpha.llm_bench import v2
result = v2.run(v2.AnthropicProvider(), v2.build_dataset(), repeats=2)   # needs credentials
```
