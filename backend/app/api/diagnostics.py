"""AI Diagnosis endpoints (deterministic engine + optional LLM explanation).

The deterministic engine always runs. The LLM layer is a strict add-on: if it
is disabled, misconfigured or every provider fails, the response still contains
the complete structured findings.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..llm import get_service
from ..diagnostics import engine
from ..diagnostics.llm import explain_run
from ..models.schemas import DiagnoseRequest
from .deps import get_run_or_404

router = APIRouter()


@router.get("/llm-status")
def llm_status():
    """Provider health for the AI Diagnosis panel.

    Reports configuration only — it never contacts a provider, so it is always
    fast and never blocks. No API key is ever included in the response.
    """
    service = get_service()
    manager = service.manager
    return {
        "enabled": settings.llm_enabled,
        "chain": manager.chain,
        "chain_labels": manager.chain_labels(),
        "providers": [p.to_dict() for p in manager.status_overview()],
        "models": {p.name: p.model for p in manager.providers},
        "timeout_s": manager.timeout_s,
        "max_retries": manager.max_retries,
        "note": ("The app and its deterministic diagnosis work fully without any LLM. "
                 "API keys are never sent to the browser."),
    }


@router.post("/analyze")
def analyze(req: DiagnoseRequest):
    """Deterministic findings, plus an LLM explanation when one is available."""
    run = get_run_or_404(req.run_id)
    findings = engine.diagnose(run)
    history = run.get("history") or []
    final = (run.get("final") or {}).get("test_metrics") or {}
    context = {
        "task": (run.get("data_summary") or {}).get("task"),
        "dataset": (run.get("dataset_meta") or {}).get("dataset_name"),
        "epochs_run": len(history),
        "total_params": (run.get("network_summary") or {}).get("total_params"),
        "n_train": (run.get("data_summary") or {}).get("n_train"),
        "test_metrics": final,
        "duration_s": run.get("duration_s"),
        "status": run.get("status"),
    }
    service = get_service()
    result = {
        "run_id": req.run_id,
        "context": context,
        "findings": findings,
        "llm": {
            "requested": req.use_llm,
            "enabled": settings.llm_enabled,
            "available": False,
            "explanation": None,
            "error": None,
            "provider": None,
            "model": None,
            "label": None,
            "outcome": "unavailable",
            "message": "",
            "fallback_used": False,
            "chain": service.manager.chain,
            "attempts": [],
        },
    }
    if req.use_llm:
        # explain_run never raises: a broken provider chain degrades to
        # available=False and the deterministic findings are returned unchanged.
        result["llm"].update(explain_run(run, findings, context))
    return result


@router.post("/analyze/preview")
def analyze_preview(req: DiagnoseRequest):
    """Same as /analyze but returns the exact structured payload sent to the LLM.

    Useful for debugging prompt contents and for verifying that the model only
    ever receives engine-computed statistics. Contains no credentials.
    """
    from ..diagnostics.llm import build_payload

    run = get_run_or_404(req.run_id)
    findings = engine.diagnose(run)
    context = {
        "task": (run.get("data_summary") or {}).get("task"),
        "epochs_run": len(run.get("history") or []),
    }
    return {"run_id": req.run_id, "findings": findings,
            "payload": build_payload(run, context)}

