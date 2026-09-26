"""Provider manager: the fallback engine.

    Diagnostic Engine
            ↓  (structured findings)
    LLMService
            ↓
    ProviderManager
            ↓
    ┌──────────┐  ┌────────────┐  ┌─────────┐
    │  Groq    │→ │ OpenRouter │→ │ Ollama  │
    └──────────┘  └────────────┘  └─────────┘

Guarantees:

* **Bounded.** Each provider is attempted at most ``1 + LLM_MAX_RETRIES``
  times and the chain is walked exactly once — no unbounded loops.
* **No secret leakage.** Logs record provider, outcome, reason and latency —
  never keys, headers or prompt content.
* **Never raises.** Any failure is raised as an :class:`LLMError`; the service
  layer converts that into a result object so the app cannot break.
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

from ..config import settings
from .base import (
    LLMError,
    LLMMessage,
    LLMNotConfigured,
    LLMProvider,
    LLMResponse,
    ProviderStatus,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .registry import FALLBACK_LABEL, Attempt, LLMResult, build_chain

log = logging.getLogger("neurosim.llm")


class ProviderManager:
    """Walks the configured provider chain until one answers."""

    def __init__(self, providers: Sequence[LLMProvider] | None = None,
                 *, timeout_s: float | None = None, max_retries: int | None = None,
                 max_tokens: int | None = None, temperature: float | None = None):
        self.timeout_s = settings.llm_timeout_s if timeout_s is None else timeout_s
        self.max_retries = (settings.llm_max_retries if max_retries is None
                            else max_retries)
        self.max_tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
        self.temperature = (settings.llm_temperature if temperature is None
                            else temperature)
        self.providers = list(providers) if providers is not None else build_chain()

    # ---- introspection ------------------------------------------------
    @property
    def chain(self) -> list[str]:
        return [p.name for p in self.providers]

    def chain_labels(self) -> list[str]:
        return [p.label for p in self.providers]

    def status_overview(self) -> list[ProviderStatus]:
        """Configuration-only health. Contacts nothing."""
        out: list[ProviderStatus] = []
        for p in self.providers:
            if p.is_configured():
                out.append(p.status("available"))
            else:
                out.append(p.status("not_configured", p.unavailable_reason()))
        return out

    # ---- the fallback loop ---------------------------------------------
    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        """First successful response, or the last :class:`LLMError`."""
        if not self.providers:
            raise LLMNotConfigured("No LLM providers configured")

        last_error: LLMError | None = None
        for idx, provider in enumerate(self.providers):
            if not provider.is_configured():
                log.info("LLM provider=%s status=skipped reason=%s",
                         provider.name, provider.unavailable_reason())
                last_error = LLMNotConfigured(
                    f"{provider.label}: {provider.unavailable_reason()}")
                continue

            max_tries = self.max_retries + 1
            started = time.perf_counter()
            for try_no in range(1, max_tries + 1):
                try:
                    resp = provider.generate(
                        messages, timeout_s=self.timeout_s,
                        max_tokens=self.max_tokens, temperature=self.temperature)
                except LLMError as exc:
                    last_error = exc
                    log.warning(
                        "LLM provider=%s status=failed reason=%s latency=%.2fs try=%d/%d",
                        provider.name, getattr(exc, "reason", "error"),
                        time.perf_counter() - started, try_no, max_tries)
                    continue
                except Exception as exc:  # a provider must not leak other errors
                    last_error = LLMError(f"{provider.label}: {type(exc).__name__}")
                    log.warning("LLM provider=%s status=failed reason=unexpected(%s)",
                                provider.name, type(exc).__name__)
                    break

                if idx > 0:
                    log.info("LLM fallback provider=%s", provider.name)
                log.info("LLM provider=%s status=success latency=%.2fs model=%s",
                         provider.name, resp.latency_s, resp.model)
                return resp

        raise last_error or LLMError("No LLM provider could answer")


UNAVAILABLE_MESSAGE = ("LLM explanation unavailable. "
                       "Deterministic diagnosis is still available.")


class LLMService:
    """What the diagnostic layer depends on.

    Owns prompt construction and converts the chain outcome into an
    :class:`LLMResult`. Never raises, so an unavailable LLM cannot break
    diagnosis.
    """

    def __init__(self, manager: ProviderManager | None = None):
        self.manager = manager or ProviderManager()

    def explain(self, payload: dict, findings: list[dict]) -> LLMResult:
        """Explain structured findings, falling back through the chain."""
        if not settings.llm_enabled:
            return LLMResult(
                available=False, outcome="unavailable",
                error=("LLM explanation layer is disabled (LLM_ENABLED=false, or no "
                       "providers in LLM_PROVIDER_ORDER). The deterministic findings "
                       "above are unaffected."),
                message=UNAVAILABLE_MESSAGE, chain=self.manager.chain)

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=build_user_prompt(payload, findings)),
        ]

        try:
            resp = self.manager.complete(messages)
        except LLMError as exc:
            log.info("LLM chain exhausted - deterministic diagnosis only")
            return LLMResult(
                available=False, outcome="unavailable",
                error=(f"All configured LLM providers failed ({exc}). "
                       "The deterministic findings below are unaffected."),
                message=UNAVAILABLE_MESSAGE, chain=self.manager.chain,
                attempts=self._config_attempts())
        except Exception as exc:  # pragma: no cover - belt and braces
            log.warning("LLM service unexpected error (%s)", type(exc).__name__)
            return LLMResult(
                available=False, outcome="unavailable",
                error=f"LLM layer error: {type(exc).__name__}.",
                message=UNAVAILABLE_MESSAGE, chain=self.manager.chain)

        label = FALLBACK_LABEL.get(resp.provider, resp.provider)
        is_fallback = (self.manager.chain.index(resp.provider) > 0
                       if resp.provider in self.manager.chain else False)
        return LLMResult(
            available=True, explanation=resp.text, provider=resp.provider,
            model=resp.model, label=label, latency_s=resp.latency_s,
            outcome="fallback" if is_fallback else "active",
            message=(f"Explanation generated using {label} fallback." if is_fallback
                     else f"Explanation generated using {label}."),
            chain=self.manager.chain, usage=resp.usage.to_dict())

    def status(self) -> list[ProviderStatus]:
        return self.manager.status_overview()

    def _config_attempts(self) -> list[Attempt]:
        return [Attempt(provider=p.name, label=p.label,
                        state="available" if p.is_configured() else "not_configured",
                        reason="" if p.is_configured() else p.unavailable_reason())
                for p in self.manager.providers]

