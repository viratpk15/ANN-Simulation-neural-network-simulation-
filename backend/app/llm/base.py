"""Provider contract and shared value objects.

The diagnostic engine never imports a concrete provider — it only ever sees an
:class:`LLMProvider`. Adding a provider therefore means writing one subclass and
registering it; nothing above this layer changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


class LLMError(RuntimeError):
    """Base class for every provider failure. The service treats all of these
    as "this provider did not work — try the next one"."""

    #: Short machine-readable reason used in logs and in the UI.
    reason = "error"


class LLMNotConfigured(LLMError):
    """The provider is missing credentials/endpoint, so it is not even attempted."""

    reason = "not_configured"


class LLMTimeout(LLMError):
    reason = "timeout"


class LLMAuthError(LLMError):
    """401/403 — usually a wrong or missing key."""

    reason = "auth"


class LLMRateLimited(LLMError):
    reason = "rate_limited"


class LLMUnavailable(LLMError):
    """Connection refused, DNS failure, 5xx, malformed body…"""

    reason = "unavailable"


class LLMInvalidResponse(LLMError):
    reason = "invalid_response"


ProviderState = Literal["available", "not_configured", "active", "failed", "skipped", "disabled"]


@dataclass
class ProviderStatus:
    """Health of one provider, safe to send to the browser.

    Never contains an API key — ``detail`` is a short, already-sanitised
    human-readable reason.
    """
    name: str
    label: str
    state: ProviderState
    model: str
    detail: str = ""
    latency_s: float | None = None
    #: True when this provider produced the current explanation.
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "state": self.state,
            "model": self.model,
            "detail": self.detail,
            "latency_s": self.latency_s,
            "active": self.active,
        }


@dataclass
class LLMMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """A successful completion."""
    text: str
    provider: str
    model: str
    latency_s: float
    usage: LLMUsage = field(default_factory=LLMUsage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_s": self.latency_s,
            "usage": self.usage.to_dict(),
        }


class LLMProvider(ABC):
    """One chat-completions backend.

    Implementations must raise :class:`LLMError` subclasses on *every* failure
    path, and must never raise anything else out of :meth:`generate` — the
    service relies on that to move on to the next provider.
    """

    #: stable id used in config and logs ("groq")
    name: str = "provider"
    #: human-facing name shown in the UI ("Groq")
    label: str = "Provider"
    #: model this provider is configured to use
    model: str = ""
    #: whether credentials/endpoint are present (does not test connectivity)
    requires_api_key: bool = True

    @abstractmethod
    def is_configured(self) -> bool:
        """True when this provider has everything it needs to be attempted."""

    @abstractmethod
    def unavailable_reason(self) -> str:
        """Short reason shown in the UI when :meth:`is_configured` is False."""

    @abstractmethod
    def generate(self, messages: list[LLMMessage], *, timeout_s: float,
                 max_tokens: int, temperature: float) -> LLMResponse:
        """Return a completion or raise an :class:`LLMError` subclass."""

    def status(self, state: ProviderState, detail: str = "",
               latency_s: float | None = None, active: bool = False) -> ProviderStatus:
        return ProviderStatus(
            name=self.name, label=self.label, state=state, model=self.model,
            detail=detail, latency_s=latency_s, active=active,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r} configured={self.is_configured()}>"
