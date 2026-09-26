"""Optional natural-language explanation layer for the diagnostic engine.

The deterministic engine (``app.diagnostics.engine``) is the source of truth and
always runs. This package only turns its structured findings into prose, via a
fallback chain of OpenAI-compatible providers (Groq → OpenRouter → Ollama by
default).

Public surface::

    from ..llm import LLMService, get_service
    result = get_service().explain(payload, findings)
    result.available, result.explanation, result.provider, result.outcome
"""
from __future__ import annotations

from .base import (
    LLMAuthError,
    LLMError,
    LLMInvalidResponse,
    LLMMessage,
    LLMNotConfigured,
    LLMProvider,
    LLMRateLimited,
    LLMResponse,
    LLMTimeout,
    LLMUnavailable,
    LLMUsage,
    ProviderState,
    ProviderStatus,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .providers import (
    GroqProvider,
    NvidiaNimProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from .registry import PROVIDER_REGISTRY, Attempt, LLMResult, build_chain, build_provider
from .service import LLMService, ProviderManager

__all__ = [
    "Attempt", "LLMAuthError", "LLMError", "LLMInvalidResponse", "LLMMessage",
    "LLMNotConfigured", "LLMProvider", "LLMRateLimited", "LLMResponse", "LLMResult",
    "LLMService", "LLMTimeout", "LLMUnavailable", "LLMUsage", "PROVIDER_REGISTRY",
    "ProviderManager", "ProviderState", "ProviderStatus", "SYSTEM_PROMPT",
    "GroqProvider", "NvidiaNimProvider", "OllamaProvider", "OpenAIProvider", "OpenRouterProvider",
    "build_chain", "build_provider", "build_user_prompt",
]

_service: LLMService | None = None


def get_service() -> LLMService:
    """Process-wide service instance.

    Built lazily so tests (and the app) can swap the provider chain before the
    first diagnosis request.
    """
    global _service
    if _service is None:
        _service = LLMService()
    return _service


def set_service(service: LLMService | None) -> None:
    """Replace (or clear) the process-wide service. Used by tests."""
    global _service
    _service = service
