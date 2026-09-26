"""Provider registry and per-request result types.

Adding a provider: write a subclass in :mod:`.providers` and add one line to
``PROVIDER_REGISTRY``. Nothing else in the codebase needs to change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from .base import LLMProvider, ProviderStatus
from .providers import (
    GroqProvider,
    NvidiaNimProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

log = logging.getLogger("neurosim.llm")

#: name -> provider class.
PROVIDER_REGISTRY: dict[str, Any] = {
    "groq": GroqProvider,
    "nvidia": NvidiaNimProvider,
    "nim": NvidiaNimProvider,
    "nvidia_nim": NvidiaNimProvider,
    "nvidia-nim": NvidiaNimProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}

#: Human label used in "explanation generated using X fallback" messages.
FALLBACK_LABEL = {
    "groq": "Groq",
    "nvidia": "NVIDIA NIM",
    "nim": "NVIDIA NIM",
    "nvidia_nim": "NVIDIA NIM",
    "nvidia-nim": "NVIDIA NIM",
    "openrouter": "OpenRouter",
    "ollama": "local Ollama",
    "openai": "OpenAI",
}


def build_provider(name: str) -> LLMProvider | None:
    """Instantiate a provider by name, or None when it is not registered."""
    factory = PROVIDER_REGISTRY.get(name.strip().lower())
    return factory() if factory else None


def build_chain(order: list[str] | None = None) -> list[LLMProvider]:
    """Instantiate every configured provider, preserving the fallback order."""
    names = order if order is not None else settings.effective_provider_order
    providers: list[LLMProvider] = []
    for name in names:
        p = build_provider(name)
        if p is not None:
            providers.append(p)
        else:
            log.warning("LLM provider=%s is not a registered provider — skipped", name)
    return providers


@dataclass
class Attempt:
    """One provider's outcome during a request."""
    provider: str
    label: str
    state: str
    reason: str = ""
    detail: str = ""
    latency_s: float | None = None
    tries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "label": self.label, "state": self.state,
            "reason": self.reason, "detail": self.detail,
            "latency_s": self.latency_s, "tries": self.tries,
        }


@dataclass
class LLMResult:
    """Outcome of one explanation request.

    Always constructible, even on total failure — callers never need a
    try/except and the deterministic diagnosis is never affected.
    """

    available: bool
    explanation: str | None = None
    provider: str | None = None
    model: str | None = None
    label: str | None = None
    latency_s: float | None = None
    #: "active" | "fallback" | "unavailable"
    outcome: str = "unavailable"
    #: e.g. "Explanation generated using OpenRouter fallback."
    message: str = ""
    error: str | None = None
    chain: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def fallback_used(self) -> bool:
        return self.outcome == "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "explanation": self.explanation,
            "provider": self.provider,
            "model": self.model,
            "label": self.label,
            "latency_s": self.latency_s,
            "outcome": self.outcome,
            "message": self.message,
            "error": self.error,
            "chain": self.chain,
            "attempts": [a.to_dict() for a in self.attempts],
            "usage": self.usage,
        }


def status_to_dict(status: ProviderStatus) -> dict[str, Any]:
    return status.to_dict()
