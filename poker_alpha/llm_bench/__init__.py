"""LLM hidden-information benchmark (scaffold).

A SEPARATE research harness for testing language-model agents on small,
exactly solvable hidden-information games. It is NOT part of the poker
recommendation pipeline and never will be.

Status: infrastructure complete and tested with a deterministic mock
provider; actual cross-provider runs are BLOCKED on model credentials
(environment variables only — no secrets in the repository, no results
fabricated).

Research questions this harness can answer once providers run:
mixed-strategy approximation, systematic exploitability, illegal-action
rate, prompt-paraphrase stability, determinism/entropy, calibration of
requested probability vectors.
"""

from .adapter import LLMAgent, MockProvider, Provider, provider_from_env
from .dataset import BENCH_FORMAT, build_dataset
from .harness import run_benchmark
from .metrics import evaluate_policy
from .parser import ParseError, parse_response

__all__ = ["Provider", "MockProvider", "LLMAgent", "provider_from_env",
           "BENCH_FORMAT", "build_dataset", "run_benchmark",
           "evaluate_policy", "ParseError", "parse_response"]
