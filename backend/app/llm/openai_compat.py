"""Shared OpenAI-compatible chat-completions transport.

Groq, OpenRouter and Ollama all speak ``POST {base}/chat/completions`` with a
bearer token, but they are *not* identical, so each provider subclasses this and
overrides only what differs: base URL, key handling, model, extra headers, and
how failures are classified.

Every transport error is translated into an :class:`LLMError` subclass so the
service can fall through to the next provider on any failure.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .base import (
    LLMAuthError,
    LLMInvalidResponse,
    LLMMessage,
    LLMNotConfigured,
    LLMProvider,
    LLMRateLimited,
    LLMResponse,
    LLMTimeout,
    LLMUnavailable,
    LLMUsage,
)


class OpenAICompatibleProvider(LLMProvider):
    """Base class for any provider exposing an OpenAI-compatible /v1 API."""

    base_url: str = ""
    api_key: str = ""
    #: extra headers merged into every request (e.g. OpenRouter attribution)
    extra_headers: dict[str, str] = {}

    # ---- configuration ------------------------------------------------
    def is_configured(self) -> bool:
        if not self.base_url:
            return False
        if self.requires_api_key and not self.api_key:
            return False
        return True

    def unavailable_reason(self) -> str:
        if not self.base_url:
            return "No base URL configured"
        if self.requires_api_key and not self.api_key:
            return "API key missing"
        return "Configured"

    # ---- request ------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    def _build_payload(self, messages: list[LLMMessage], *, max_tokens: int,
                       temperature: float) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

    def generate(self, messages: list[LLMMessage], *, timeout_s: float,
                 max_tokens: int, temperature: float) -> LLMResponse:
        if not self.is_configured():
            raise LLMNotConfigured(f"{self.label}: {self.unavailable_reason()}")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = self._build_payload(messages, max_tokens=max_tokens,
                                      temperature=temperature)
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(url, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"{self.label}: timed out after {timeout_s:g}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"{self.label}: {type(exc).__name__}") from exc

        latency = time.perf_counter() - started
        if resp.status_code >= 400:
            raise self._classify_http_error(resp)
        return self._parse(resp, latency)

    # ---- error classification -----------------------------------------
    def _classify_http_error(self, resp: httpx.Response) -> Exception:
        status = resp.status_code
        detail = _short_body(resp)
        if status in (401, 403):
            return LLMAuthError(f"{self.label}: auth failed (HTTP {status}) {detail}")
        if status == 429:
            return LLMRateLimited(f"{self.label}: rate limited (HTTP 429) {detail}")
        if status in (400, 404, 422):
            # 404 here is almost always a wrong model name; 400/422 likewise.
            return LLMInvalidResponse(
                f"{self.label}: request rejected (HTTP {status}) {detail}")
        if status >= 500:
            return LLMUnavailable(f"{self.label}: server error (HTTP {status}) {detail}")
        return LLMUnavailable(f"{self.label}: HTTP {status} {detail}")

    # ---- response parsing ---------------------------------------------
    def _parse(self, resp: httpx.Response, latency: float) -> LLMResponse:
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMInvalidResponse(f"{self.label}: response was not JSON") from exc
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponse(
                f"{self.label}: unexpected response shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMInvalidResponse(f"{self.label}: empty completion")
        u = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=u.get("prompt_tokens"),
            completion_tokens=u.get("completion_tokens"),
            total_tokens=u.get("total_tokens"),
        )
        return LLMResponse(text=text.strip(), provider=self.name, model=self.model,
                           latency_s=round(latency, 4), usage=usage)


def _short_body(resp: httpx.Response, limit: int = 180) -> str:
    """A truncated, safe error string.

    Provider error bodies are short JSON; we take a small slice so a huge or
    HTML error page cannot flood the logs or the UI.
    """
    try:
        body = resp.text.strip().replace("\n", " ")
    except Exception:  # pragma: no cover - defensive
        return ""
    return body[:limit] + ("…" if len(body) > limit else "")
