"""Concrete LLM providers.

Each one is a thin, declarative subclass: the only real differences between
Groq, OpenRouter and Ollama are the endpoint, the credential, the model and a
couple of provider-specific headers. Adding a fourth (Anthropic, Gemini, a
local vLLM…) means adding a class here and registering it — the diagnostic
engine and the service stay untouched.
"""
from __future__ import annotations

from ..config import settings
from .openai_compat import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """Groq — fast LPU inference, OpenAI-compatible. Primary by default."""

    name = "groq"
    label = "Groq"
    requires_api_key = True

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.base_url = (base_url or settings.groq_base_url or "").strip()
        self.model = model or settings.groq_model


class NvidiaNimProvider(OpenAICompatibleProvider):
    """NVIDIA NIM — high-performance accelerated inference. First fallback."""

    name = "nvidia"
    label = "NVIDIA NIM"
    requires_api_key = True

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.base_url = (base_url or settings.nvidia_base_url or "").strip()
        self.model = model or settings.nvidia_model


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter — routes to many hosted models. First cloud fallback."""

    name = "openrouter"
    label = "OpenRouter"
    requires_api_key = True

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.base_url = (base_url or settings.openrouter_base_url or "").strip()
        self.model = model or settings.openrouter_model
        # OpenRouter uses these to attribute traffic to the app. They are not
        # secrets and never include any key material.
        self.extra_headers = {
            "HTTP-Referer": "https://github.com/neurosim-lab",
            "X-Title": "NeuroSim Lab",
        }


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama — local/offline inference. No API key required."""

    name = "ollama"
    label = "Ollama (local)"
    requires_api_key = False

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.api_key = ""
        raw_url = (base_url or settings.ollama_base_url or "").strip().rstrip("/")
        # Ollama's OpenAI-compatible endpoint is at /v1/chat/completions.
        # Auto-normalize so URLs like http://localhost:11434 or http://127.0.0.1:11434 work cleanly.
        if raw_url and not raw_url.endswith("/v1"):
            raw_url = f"{raw_url}/v1"
        self.base_url = raw_url
        self.model = model or settings.ollama_model

    def unavailable_reason(self) -> str:
        if not self.base_url:
            return "No base URL configured"
        return "Configured (start `ollama serve` if it is not reachable)"


class OpenAIProvider(OpenAICompatibleProvider):
    """Direct OpenAI — kept for backwards compatibility with older .env files."""

    name = "openai"
    label = "OpenAI"
    requires_api_key = True

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url or "").strip()
        self.model = model or settings.openai_model
