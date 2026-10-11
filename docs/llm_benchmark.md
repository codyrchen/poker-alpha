# LLM Hidden-Information Benchmark (poker_alpha/llm_bench/)

A SEPARATE research harness — explicitly not part of the poker
recommendation pipeline, and never will be.

Infrastructure (complete, tested with a deterministic mock provider):
- provider-agnostic adapter (`Provider.complete(prompt) -> text`);
  env-var credentials only (OPENAI_API_KEY / ANTHROPIC_API_KEY /
  LLM_LOCAL_ENDPOINT); no SDK imports unless requested; no secrets;
- versioned seeded dataset (`pokeralpha.llmbench/v1`): all 12 Kuhn
  infosets × 3 semantically equivalent prompt templates (paraphrase-
  stability probe), structured-JSON response contract;
- strict parser: illegal actions are NEVER coerced (failure categories:
  no_json / bad_schema / illegal_action / bad_probabilities; sum
  tolerance 0.01, then exact renormalization — documented);
- exact metrics: exploitability, TV distance to the deterministic CFR
  reference, entropy/determinism, illegal-action rate,
  paraphrase instability (mean TV across templates);
- result format `pokeralpha.llmbench_result/v1` with raw observable
  outputs and latencies; no private chain-of-thought requested.

**Status: cross-provider runs BLOCKED on model credentials.** No scores
exist and none are shown anywhere; the mock provider exists to test the
harness, not to produce results.
