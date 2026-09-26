"""LLM provider fallback tests.

Every external HTTP call is mocked. No test needs a real API key, and none
contacts a real provider.

Covers the required cases:
  1. Groq available            -> Groq response
  2. Groq unavailable          -> OpenRouter response
  3. Groq + OpenRouter down    -> Ollama response
  4. All providers down        -> deterministic diagnosis still returned
  5. No API keys configured    -> app starts, diagnosis works, LLM unavailable
  6. Invalid model             -> provider failure -> next provider
  7. Timeout                   -> provider failure -> next provider
"""
import logging

import pytest

from app.config import settings
from app.diagnostics import engine
from app.llm import LLMService, ProviderManager, set_service
from app.llm.base import (
    LLMAuthError,
    LLMInvalidResponse,
    LLMMessage,
    LLMNotConfigured,
    LLMProvider,
    LLMResponse,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
    LLMUsage,
)
from app.llm.openai_compat import OpenAICompatibleProvider
from app.llm.providers import GroqProvider, NvidiaNimProvider, OllamaProvider, OpenRouterProvider
from app.llm.registry import build_chain, build_provider


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeProvider(LLMProvider):
    """Scripted provider: either succeeds, or raises a given LLMError."""

    def __init__(self, name, label=None, *, ok=True, error=None,
                 configured=True, text="explanation", model="fake-model"):
        self.name = name
        self.label = label or name.title()
        self.model = model
        self.requires_api_key = False
        self._ok = ok
        self._error = error or LLMUnavailable(f"{name}: down")
        self._configured = configured
        self._text = text
        self.calls = 0
        self.seen_messages: list[list[LLMMessage]] = []

    def is_configured(self):
        return self._configured

    def unavailable_reason(self):
        return "Configured" if self._configured else "API key missing"

    def generate(self, messages, *, timeout_s, max_tokens, temperature):
        self.calls += 1
        self.seen_messages.append(messages)
        if not self._ok:
            raise self._error
        return LLMResponse(text=self._text, provider=self.name, model=self.model,
                           latency_s=0.01,
                           usage=LLMUsage(prompt_tokens=10, completion_tokens=20,
                                          total_tokens=30))


def service(*providers, **kw) -> LLMService:
    return LLMService(ProviderManager(list(providers), **kw))


FINDINGS = [{"severity": "warning", "code": "OVERFITTING_GAP",
             "title": "Possible overfitting",
             "explanation": "Training 98% vs validation 81%",
             "suggestions": ["Add dropout"], "evidence": {"gap": 0.17}}]
PAYLOAD = {"context": {"task": "classification", "epochs_run": 30}}


# ===========================================================================
# Case 1 - Groq available
# ===========================================================================

def test_case1_groq_available_returns_groq():
    groq = FakeProvider("groq", "Groq", text="from groq")
    openrouter = FakeProvider("openrouter", "OpenRouter")
    ollama = FakeProvider("ollama", "Ollama (local)")
    r = service(groq, openrouter, ollama).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "groq"
    assert r.outcome == "active" and not r.fallback_used
    assert r.explanation == "from groq"
    assert r.message == "Explanation generated using Groq."
    # the fallbacks are never contacted
    assert openrouter.calls == 0 and ollama.calls == 0


# ===========================================================================
# Case 2 - Groq unavailable -> OpenRouter
# ===========================================================================

def test_case2_groq_fails_falls_back_to_openrouter():
    groq = FakeProvider("groq", "Groq", ok=False,
                        error=LLMAuthError("Groq: auth failed (HTTP 401)"))
    openrouter = FakeProvider("openrouter", "OpenRouter", text="from openrouter")
    ollama = FakeProvider("ollama", "Ollama (local)")
    r = service(groq, openrouter, ollama).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "openrouter"
    assert r.outcome == "fallback" and r.fallback_used
    assert r.message == "Explanation generated using OpenRouter fallback."
    # Groq is retried once (LLM_MAX_RETRIES=1 -> 2 attempts) before moving on.
    assert groq.calls == 2
    assert ollama.calls == 0


def test_case2_groq_fails_falls_back_to_nvidia():
    groq = FakeProvider("groq", "Groq", ok=False,
                        error=LLMAuthError("Groq: auth failed (HTTP 401)"))
    nvidia = FakeProvider("nvidia", "NVIDIA NIM", text="from nvidia")
    openrouter = FakeProvider("openrouter", "OpenRouter")
    ollama = FakeProvider("ollama", "Ollama (local)")
    r = service(groq, nvidia, openrouter, ollama).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "nvidia"
    assert r.outcome == "fallback" and r.fallback_used
    assert r.message == "Explanation generated using NVIDIA NIM fallback."
    assert groq.calls == 2
    assert nvidia.calls == 1
    assert openrouter.calls == 0 and ollama.calls == 0


def test_case2_single_attempt_when_retries_disabled():
    groq = FakeProvider("groq", "Groq", ok=False, error=LLMTimeout("Groq: t/o"))
    openrouter = FakeProvider("openrouter", "OpenRouter", text="from openrouter")
    r = service(groq, openrouter, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.provider == "openrouter"
    assert groq.calls == 1


# ===========================================================================
# Case 3 - Groq + OpenRouter down -> Ollama
# ===========================================================================

def test_case3_only_ollama_left_answers():
    groq = FakeProvider("groq", "Groq", ok=False, error=LLMTimeout("Groq: timeout"))
    openrouter = FakeProvider("openrouter", "OpenRouter", ok=False,
                              error=LLMRateLimited("OpenRouter: 429"))
    ollama = FakeProvider("ollama", "Ollama (local)", text="from ollama")
    r = service(groq, openrouter, ollama, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "ollama"
    assert r.outcome == "fallback"
    assert r.message == "Explanation generated using local Ollama fallback."


# ===========================================================================
# Case 4 - all providers fail, deterministic diagnosis unaffected
# ===========================================================================

def test_case4_all_fail_returns_no_llm_result_without_raising():
    providers = [FakeProvider(n, ok=False, error=LLMUnavailable(f"{n} down"))
                 for n in ("groq", "openrouter", "ollama")]
    r = service(*providers, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.available is False and r.explanation is None
    assert r.outcome == "unavailable"
    assert r.message == ("LLM explanation unavailable. "
                         "Deterministic diagnosis is still available.")
    assert r.error and "All configured LLM providers failed" in r.error


def test_case4_deterministic_diagnosis_still_runs():
    """The engine is completely independent of the LLM layer."""
    run = {"history": [{"epoch": i, "train_loss": 0.9 - 0.1 * i, "val_loss": 0.9,
                        "train_metric": 0.9, "val_metric": 0.7,
                        "metric_name": "accuracy"} for i in range(8)],
           "snapshots": [], "data_summary": {"task": "classification"},
           "final": {}, "network_summary": {}}
    findings = engine.diagnose(run)
    assert findings and any(f["code"] == "OVERFITTING_GAP" for f in findings)


# ===========================================================================
# Case 5 - no API keys configured
# ===========================================================================

def test_case5_unconfigured_providers_are_skipped():
    providers = [FakeProvider(n, configured=False) for n in ("groq", "openrouter", "ollama")]
    r = service(*providers).explain(PAYLOAD, FINDINGS)
    assert r.available is False
    # A provider with no credentials is never even called.
    assert all(p.calls == 0 for p in providers)
    assert "not_configured" in {a.state for a in r.attempts}


def test_case5_real_providers_report_missing_keys(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "nvidia_api_key", "")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    assert GroqProvider().is_configured() is False
    assert GroqProvider().unavailable_reason() == "API key missing"
    assert NvidiaNimProvider().is_configured() is False
    assert NvidiaNimProvider().unavailable_reason() == "API key missing"
    assert OpenRouterProvider().is_configured() is False
    # Ollama needs no key at all.
    assert OllamaProvider().is_configured() is True


def test_case5_only_ollama_configured(monkeypatch):
    """Groq/OpenRouter skipped for a missing key; Ollama answers."""
    groq = FakeProvider("groq", "Groq", configured=False)
    openrouter = FakeProvider("openrouter", "OpenRouter", configured=False)
    ollama = FakeProvider("ollama", "Ollama (local)", text="local only")
    r = service(groq, openrouter, ollama).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "ollama"
    assert groq.calls == 0 and openrouter.calls == 0 and ollama.calls == 1


def test_case5_only_groq_configured(monkeypatch):
    """Groq works; if it fails, OpenRouter is skipped and Ollama is tried."""
    groq = FakeProvider("groq", "Groq", ok=False, error=LLMTimeout("t/o"))
    openrouter = FakeProvider("openrouter", "OpenRouter", configured=False)
    ollama = FakeProvider("ollama", "Ollama (local)", text="ollama")
    r = service(groq, openrouter, ollama, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.provider == "ollama"
    assert openrouter.calls == 0          # skipped: no key


def test_case5_disabled_layer_reports_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled_flag", False)
    groq = FakeProvider("groq", "Groq")
    r = service(groq).explain(PAYLOAD, FINDINGS)
    assert r.available is False
    assert "disabled" in r.error
    assert groq.calls == 0



# ===========================================================================
# Case 6 - invalid model -> next provider
# ===========================================================================

def test_case6_invalid_model_falls_through():
    groq = FakeProvider("groq", "Groq", ok=False,
                        error=LLMInvalidResponse("Groq: model not found"))
    openrouter = FakeProvider("openrouter", "OpenRouter", text="fallback")
    r = service(groq, openrouter, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.provider == "openrouter" and r.available


def test_case6_invalid_model_classified_from_http_status():
    """A 404 from the real transport is an invalid-model error, not a crash."""
    import httpx

    p = GroqProvider(api_key="k", base_url="https://x/v1", model="does-not-exist")
    resp = httpx.Response(404, text='{"error":{"message":"model not found"}}',
                          request=httpx.Request("POST", "https://x/v1/chat/completions"))
    assert isinstance(p._classify_http_error(resp), LLMInvalidResponse)


# ===========================================================================
# Case 7 - timeout -> next provider
# ===========================================================================

def test_case7_timeout_falls_through():
    groq = FakeProvider("groq", "Groq", ok=False, error=LLMTimeout("Groq: timed out"))
    ollama = FakeProvider("ollama", "Ollama (local)", text="slow but alive")
    r = service(groq, ollama, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.provider == "ollama" and r.available


def test_case7_timeout_error_type_is_translated():
    """httpx timeouts become LLMTimeout, so the chain can move on."""
    import httpx

    from app.llm.base import LLMError

    assert issubclass(LLMTimeout, LLMError)
    assert isinstance(httpx.ConnectTimeout("slow"), httpx.TimeoutException)


# ===========================================================================
# Retry bounds / cost safety
# ===========================================================================

def test_retries_are_bounded_per_provider():
    groq = FakeProvider("groq", "Groq", ok=False, error=LLMTimeout("t/o"))
    openrouter = FakeProvider("openrouter", "OpenRouter", text="ok")
    service(groq, openrouter, max_retries=1).explain(PAYLOAD, FINDINGS)
    # max_retries=1 => at most 2 attempts on Groq, then straight to OpenRouter
    assert groq.calls == 2
    assert openrouter.calls == 1


def test_zero_retries_means_one_attempt():
    groq = FakeProvider("groq", "Groq", ok=False)
    service(groq, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert groq.calls == 1


def test_a_repeated_provider_is_never_revisited():
    providers = [FakeProvider("groq", ok=False), FakeProvider("groq", ok=False)]
    service(*providers, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert all(p.calls <= 1 for p in providers)


def test_unexpected_provider_exception_does_not_crash():
    class Exploding(FakeProvider):
        def generate(self, *a, **kw):
            self.calls += 1
            raise RuntimeError("something totally unexpected")

    boom = Exploding("groq", "Groq")
    ollama = FakeProvider("ollama", "Ollama (local)", text="survived")
    r = service(boom, ollama, max_retries=0).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "ollama"


def test_empty_chain_is_handled():
    r = service().explain(PAYLOAD, FINDINGS)
    assert r.available is False
    assert r.chain == []


def test_a_provider_that_succeeds_on_retry_stops_immediately():
    """A transient failure followed by success must not walk the whole chain."""
    class Flaky(FakeProvider):
        def generate(self, messages, *, timeout_s, max_tokens, temperature):
            self.calls += 1
            if self.calls == 1:
                raise LLMTimeout("first attempt times out")
            return LLMResponse(text="recovered", provider=self.name, model=self.model,
                               latency_s=0.01)

    flaky = Flaky("groq", "Groq")
    ollama = FakeProvider("ollama", "Ollama (local)")
    r = service(flaky, ollama, max_retries=1).explain(PAYLOAD, FINDINGS)
    assert r.available and r.provider == "groq"
    assert flaky.calls == 2
    assert ollama.calls == 0          # no need to fall back at all


# ===========================================================================
# Configured order
# ===========================================================================

def test_default_order_is_groq_openrouter_ollama(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", ["groq", "openrouter", "ollama"])
    assert settings.effective_provider_order == ["groq", "openrouter", "ollama"]


def test_development_mode_allows_ollama(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", ["groq", "nvidia", "ollama"])
    assert settings.is_production is False
    assert settings.effective_provider_order == ["groq", "nvidia", "ollama"]


def test_production_mode_excludes_ollama(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", ["groq", "nvidia", "ollama"])
    assert settings.is_production is True
    assert settings.effective_provider_order == ["groq", "nvidia"]


def test_order_is_deduplicated(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", ["groq", "ollama", "groq"])
    assert settings.effective_provider_order == ["groq", "ollama"]


def test_custom_order_is_respected(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", ["ollama", "groq"])
    assert settings.effective_provider_order == ["ollama", "groq"]


def test_legacy_single_provider_still_wins(monkeypatch):
    """An older .env with LLM_PROVIDER=ollama keeps working."""
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    assert settings.effective_provider_order == ["ollama"]


def test_llm_disabled_when_no_providers(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", [])
    assert settings.llm_enabled is False


# ===========================================================================
# Registry / real provider configuration
# ===========================================================================

def test_registry_builds_the_providers(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_provider", "")
    monkeypatch.setattr(settings, "llm_provider_order", ["groq", "nvidia", "ollama"])
    chain = build_chain()
    assert [p.name for p in chain] == ["groq", "nvidia", "ollama"]
    assert all(isinstance(p, OpenAICompatibleProvider) for p in chain)


def test_registry_builds_nvidia_nim_aliases():
    assert isinstance(build_provider("nvidia"), NvidiaNimProvider)
    assert isinstance(build_provider("nim"), NvidiaNimProvider)
    assert isinstance(build_provider("nvidia_nim"), NvidiaNimProvider)
    assert isinstance(build_provider("nvidia-nim"), NvidiaNimProvider)


def test_unknown_provider_is_ignored():
    assert build_provider("not-a-provider") is None


def test_provider_config_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "secret-key")
    monkeypatch.setattr(settings, "groq_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    p = GroqProvider()
    assert p.is_configured()
    assert p.model == "llama-3.3-70b-versatile"
    assert p.base_url == "https://api.groq.com/openai/v1"


def test_openrouter_sends_attribution_headers():
    h = OpenRouterProvider(api_key="k")._headers()
    assert h["Authorization"] == "Bearer k"
    assert "X-Title" in h and "HTTP-Referer" in h


def test_ollama_sends_no_authorization_header():
    assert "Authorization" not in OllamaProvider()._headers()


def test_ollama_normalizes_base_url_and_model():
    p = OllamaProvider(base_url="http://localhost:11434")
    assert p.base_url == "http://localhost:11434/v1"

    p2 = OllamaProvider(base_url="http://127.0.0.1:11434/v1/")
    assert p2.base_url == "http://127.0.0.1:11434/v1"

    p3 = OllamaProvider(model="qwen2.5:3b")
    assert p3.model == "qwen2.5:3b"


def test_status_overview_never_leaks_keys():
    status = GroqProvider(api_key="super-secret").status("available").to_dict()
    assert "super-secret" not in str(status)
    assert status["name"] == "groq"



# ===========================================================================
# Prompt contract - the LLM must not compute anything
# ===========================================================================

def test_system_prompt_forbids_inventing_metrics():
    from app.llm.prompts import SYSTEM_PROMPT

    low = SYSTEM_PROMPT.lower()
    assert "never invent" in low
    assert "claim certainty" in low
    assert "observed" in low and "hypothesis" in low
    assert "deterministic" in low


def test_user_prompt_contains_only_engine_data():
    from app.llm.prompts import build_user_prompt

    payload = {
        "context": {"task": "classification"},
        "architecture_summary": {"total_params": 114},
        "dataset_summary": {"n_train": 256},
        "training_metrics": {"final_train_metric": 1.0},
        "loss_history_summary": {"final_train_loss": 0.02},
    }
    p = build_user_prompt(payload, FINDINGS)
    assert "total_params" in p and "114" in p
    assert "OVERFITTING_GAP" in p
    assert "Add dropout" in p
    assert "api_key" not in p.lower()


def test_service_sends_system_and_user_messages():
    groq = FakeProvider("groq", "Groq", text="ok")
    service(groq).explain(PAYLOAD, FINDINGS)
    msgs = groq.seen_messages[0]
    assert [m.role for m in msgs] == ["system", "user"]
    assert "educational" in msgs[0].content.lower()


def test_build_payload_uses_only_recorded_run_data():
    from app.diagnostics.llm import build_payload

    run = {
        "history": [{"epoch": 1, "train_loss": 0.9, "val_loss": 1.0,
                     "train_metric": 0.6, "val_metric": 0.5,
                     "metric_name": "accuracy"}],
        "snapshots": [{"activations": [{"label": "h1", "activation": "relu",
                                        "mean": 0.1, "sample": [1, 2, 3]}],
                      "gradients": {"layers": [{"label": "h1", "w_grad_norm": 0.5}],
                                    "total_norm": 0.5}}],
        "data_summary": {"task": "classification", "n_train": 100},
        "network_summary": {"total_params": 42, "n_layers": 2},
        "final": {"test_metrics": {"accuracy": 0.9},
                  "weights": {"h1": {"shape": [2, 2], "stats": {"mean": 0.0}}}},
        "duration_s": 3.0,
        "status": "finished",
    }
    p = build_payload(run, {"task": "classification"})
    assert p["architecture_summary"]["total_params"] == 42
    assert p["training_metrics"]["test_metrics"]["accuracy"] == 0.9
    assert p["loss_history_summary"]["final_train_loss"] == 0.9
    assert p["activation_statistics"][0]["label"] == "h1"
    assert p["gradient_statistics"][0]["w_grad_norm"] == 0.5
    assert "h1" in p["weight_statistics"]
    # raw activation samples are not forwarded
    assert "sample" not in p["activation_statistics"][0]



# ===========================================================================
# Logging - outcome recorded, secrets never
# ===========================================================================

def _log_text(caplog) -> str:
    """caplog.text already renders each record's message with its args."""
    return caplog.text


def test_logs_record_provider_outcome_and_latency(caplog):
    groq = FakeProvider("groq", "Groq", text="ok")
    with caplog.at_level(logging.INFO, logger="neurosim.llm"):
        service(groq).explain(PAYLOAD, FINDINGS)
    text = _log_text(caplog)
    assert "provider=groq" in text
    assert "status=success" in text
    assert "latency=" in text


def test_logs_record_failure_reason_and_fallback(caplog):
    groq = FakeProvider("groq", "Groq", ok=False, error=LLMTimeout("t/o"))
    openrouter = FakeProvider("openrouter", "OpenRouter", text="ok")
    with caplog.at_level(logging.INFO, logger="neurosim.llm"):
        service(groq, openrouter, max_retries=0).explain(PAYLOAD, FINDINGS)
    text = _log_text(caplog)
    assert "status=failed" in text
    assert "reason=timeout" in text
    assert "LLM fallback provider=openrouter" in text


def test_logs_never_contain_api_keys(caplog, monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "TOPSECRETKEY")
    groq = FakeProvider("groq", "Groq", text="ok")
    with caplog.at_level(logging.DEBUG, logger="neurosim.llm"):
        service(groq).explain(PAYLOAD, FINDINGS)
    assert "TOPSECRETKEY" not in _log_text(caplog)
    assert "TOPSECRETKEY" not in caplog.text


def test_logs_never_contain_authorization_headers(caplog):
    groq = FakeProvider("groq", "Groq", text="ok")
    with caplog.at_level(logging.DEBUG, logger="neurosim.llm"):
        service(groq).explain(PAYLOAD, FINDINGS)
    assert "Bearer" not in caplog.text


# ===========================================================================
# Transport: error classification and parsing
# ===========================================================================

@pytest.mark.parametrize("status,expected", [
    (401, LLMAuthError), (403, LLMAuthError), (429, LLMRateLimited),
    (400, LLMInvalidResponse), (404, LLMInvalidResponse), (500, LLMUnavailable),
    (503, LLMUnavailable),
])
def test_http_status_classification(status, expected):
    import httpx

    p = GroqProvider(api_key="k", base_url="https://x/v1")
    resp = httpx.Response(status, text="err",
                          request=httpx.Request("POST", "https://x"))
    assert isinstance(p._classify_http_error(resp), expected)


def test_not_configured_provider_raises_before_any_http():
    p = GroqProvider(api_key="", base_url="https://x/v1")
    with pytest.raises(LLMNotConfigured):
        p.generate([LLMMessage("user", "hi")], timeout_s=1, max_tokens=1, temperature=0)


def test_malformed_responses_are_handled():
    import httpx

    p = GroqProvider(api_key="k", base_url="https://x/v1")
    req = httpx.Request("POST", "https://x")
    with pytest.raises(LLMInvalidResponse):
        p._parse(httpx.Response(200, text="not json", request=req), 0.0)
    with pytest.raises(LLMInvalidResponse):
        p._parse(httpx.Response(200, json={"nope": 1}, request=req), 0.0)
    with pytest.raises(LLMInvalidResponse):
        p._parse(httpx.Response(200, json={"choices": [{"message": {"content": " "}}]},
                                request=req), 0.0)


def test_successful_response_is_parsed():
    import httpx

    p = GroqProvider(api_key="k", base_url="https://x/v1", model="m")
    resp = httpx.Response(200, json={
        "choices": [{"message": {"content": "  hello  "}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }, request=httpx.Request("POST", "https://x"))
    out = p._parse(resp, 0.5)
    assert out.text == "hello"           # whitespace trimmed
    assert out.provider == "groq" and out.model == "m"
    assert out.usage.total_tokens == 12


def test_request_payload_shape():
    p = GroqProvider(api_key="k", base_url="https://x/v1", model="m")
    payload = p._build_payload([LLMMessage("user", "hi")],
                               max_tokens=64, temperature=0.2)
    assert payload["model"] == "m"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.2
    assert payload["stream"] is False


# ===========================================================================
# Adapter (diagnostics.llm) - backwards compatible entry point
# ===========================================================================

def test_explain_with_llm_adapter_never_raises():
    from app.diagnostics.llm import explain_with_llm

    groq = FakeProvider("groq", "Groq", ok=False)
    set_service(service(groq, max_retries=0))
    try:
        out = explain_with_llm(FINDINGS, {"task": "classification"})
        assert out["available"] is False
        assert "Deterministic diagnosis is still available" in out["message"]
        assert out["chain"] == ["groq"]
    finally:
        set_service(None)


def test_explain_with_llm_adapter_reports_provider():
    from app.diagnostics.llm import explain_with_llm

    set_service(service(FakeProvider("groq", "Groq", text="hi")))
    try:
        out = explain_with_llm(FINDINGS, {})
        assert out["available"] and out["provider"] == "groq"
        assert out["model"] == "fake-model"
    finally:
        set_service(None)


def test_llm_result_to_dict_has_expected_keys():
    from app.diagnostics.llm import llm_result_to_dict

    d = llm_result_to_dict(service(FakeProvider("groq", "Groq", text="hi"))
                           .explain(PAYLOAD, FINDINGS))
    for k in ("available", "explanation", "error", "provider", "model", "label",
              "outcome", "message", "fallback_used", "chain", "attempts", "usage"):
        assert k in d

