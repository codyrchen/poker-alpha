"""Provider-agnostic LLM adapter.

A provider is anything with ``complete(prompt) -> text``. Credentials are
environment variables only; nothing here embeds secrets, and no provider
SDK is imported unless that provider is actually requested. Benchmarks
evaluate OBSERVABLE outputs only — no private chain-of-thought is
requested or required.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, prompt: str) -> str:
        """Return the model's raw text response for one prompt."""
        ...


class MockProvider:
    """Deterministic test provider: replays a fixed prompt->response map
    (exercises the full harness without network or credentials)."""

    name = "mock-deterministic"

    def __init__(self, responses: Dict[str, str],
                 default: Optional[str] = None) -> None:
        self.responses = responses
        self.default = default
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if prompt in self.responses:
            return self.responses[prompt]
        if self.default is not None:
            return self.default
        raise KeyError("mock provider has no response for this prompt")


def provider_from_env(name: str) -> Provider:
    """Construct a real provider from environment variables.

    Supported names (each requires its own env credentials; none are
    bundled): ``openai`` (OPENAI_API_KEY), ``anthropic``
    (ANTHROPIC_API_KEY), ``local`` (LLM_LOCAL_ENDPOINT, an
    OpenAI-compatible localhost URL). Raises with a clear message when
    credentials or SDKs are missing — real runs are BLOCKED until then.
    """
    if name == "mock":
        raise ValueError("construct MockProvider directly with its responses")
    envs = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "local": "LLM_LOCAL_ENDPOINT"}
    if name not in envs:
        raise ValueError(f"unknown provider {name!r}; known: {sorted(envs)}")
    if not os.environ.get(envs[name]):
        raise RuntimeError(
            f"provider {name!r} requires the {envs[name]} environment "
            f"variable; cross-provider benchmark runs are BLOCKED on "
            f"credentials")
    raise NotImplementedError(
        f"provider {name!r}: credentials detected but the SDK call is left "
        f"to the run that supplies them (see docs/llm_benchmark.md); this "
        f"scaffold ships no network code")


class LLMAgent:
    """Wraps a provider as a benchmark agent over structured decisions."""

    def __init__(self, provider: Provider, template: str) -> None:
        self.provider = provider
        self.template = template

    def respond(self, rendered_prompt: str) -> str:
        return self.provider.complete(rendered_prompt)


__all__ = ["Provider", "MockProvider", "provider_from_env", "LLMAgent"]
