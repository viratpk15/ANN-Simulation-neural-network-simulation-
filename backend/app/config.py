"""Application configuration.

All settings are read from environment variables (optionally via a local
``.env`` file). Nothing here is required for the core application to run;
the optional LLM settings only enable the natural-language diagnosis
explanation feature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "env")
load_dotenv(BASE_DIR / "backend" / ".env")
load_dotenv(BASE_DIR / "backend" / "env")


def _path(env_name: str, default: str) -> Path:
    raw = os.environ.get(env_name, default)
    p = Path(raw)
    return p if p.is_absolute() else (BASE_DIR / p)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default) or default
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication, so a repeated provider is tried once."""
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _get_app_env() -> str:
    return (
        os.environ.get("APP_ENV")
        or os.environ.get("ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()


@dataclass
class Settings:
    # --- server & environment ---------------------------------------------
    app_env: str = field(default_factory=_get_app_env)
    backend_port: int = int(os.environ.get("BACKEND_PORT", "8000"))
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip()
        for o in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
        ).split(",")
        if o.strip()
    ])

    # --- storage ----------------------------------------------------------
    database_path: Path = field(default_factory=lambda: _path("DATABASE_PATH", "data/neurosim.db"))
    upload_dir: Path = field(default_factory=lambda: _path("UPLOAD_DIR", "data/uploads"))
    runs_dir: Path = field(default_factory=lambda: _path("RUNS_DIR", "data/runs"))
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "10"))

    # --- training guard rails --------------------------------------------
    max_epochs: int = int(os.environ.get("MAX_EPOCHS", "2000"))
    max_neurons_per_layer: int = int(os.environ.get("MAX_NEURONS_PER_LAYER", "512"))
    max_visual_neurons: int = 16  # neurons included in per-neuron traces

    # --- optional LLM (the app is fully functional without these) ---------
    # LLM_ENABLED turns the whole explanation layer on/off.
    # LLM_PROVIDER_ORDER is a comma-separated fallback chain, tried left to
    # right. In production mode, local Ollama is omitted (groq -> nvidia).
    # In development mode, Ollama acts as the final local fallback (groq -> nvidia -> ollama).
    llm_enabled_flag: bool = _env_bool("LLM_ENABLED", True)
    llm_provider_order: list[str] = field(
        default_factory=lambda: _env_list(
            "LLM_PROVIDER_ORDER",
            "groq,nvidia" if _get_app_env() in ("production", "prod")
            else "groq,nvidia,ollama"
        )
    )
    # Legacy single-provider switch (pre-fallback). Still honoured so an older
    # .env keeps working: when set to something other than "none"/empty it wins
    # over LLM_PROVIDER_ORDER.
    llm_provider: str = os.environ.get("LLM_PROVIDER", "").strip().lower()

    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_base_url: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    nvidia_api_key: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVIDIA_NIM_API_KEY")
        or ""
    )
    nvidia_base_url: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_BASE_URL")
        or os.environ.get("NVIDIA_NIM_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    )
    nvidia_model: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_MODEL")
        or os.environ.get("NVIDIA_NIM_MODEL")
        or "meta/llama-3.3-70b-instruct"
    )

    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.environ.get("OPENROUTER_MODEL",
                                           "meta-llama/llama-3.3-70b-instruct")

    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    # Pre-fallback OpenAI-compatible escape hatch, still supported.
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    llm_timeout_s: float = _env_float("LLM_TIMEOUT_S", 30.0)
    # Extra attempts *per provider* after the first. 1 => 2 total tries.
    # Bounded so a failing chain can never loop.
    llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 1)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 500)
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.3)

    def ensure_dirs(self) -> None:
        for p in (self.database_path.parent, self.upload_dir, self.runs_dir):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """True when running in production mode."""
        return self.app_env in ("production", "prod")

    @property
    def effective_provider_order(self) -> list[str]:
        """The fallback chain actually used, de-duplicated and order-preserving.

        In production mode, local Ollama is omitted (groq -> nvidia).
        In development mode, Ollama is allowed as the local fallback (groq -> nvidia -> ollama).
        """
        if self.llm_provider and self.llm_provider != "none":
            order = [self.llm_provider] if self.llm_provider != "chain" \
                else _dedupe(self.llm_provider_order)
        else:
            order = _dedupe(self.llm_provider_order)

        if self.is_production:
            order = [p for p in order if p != "ollama"]

        return order

    @property
    def llm_enabled(self) -> bool:
        """True when the layer is switched on.

        Deliberately *not* "is there a working provider": the UI uses this to
        show the configured chain, and a missing key simply makes that provider
        report itself unavailable at request time.
        """
        return bool(self.llm_enabled_flag) and bool(self.effective_provider_order)


settings = Settings()
settings.ensure_dirs()
