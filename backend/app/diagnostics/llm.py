"""Explanation layer for the diagnostic engine.

The deterministic engine runs first and is the source of truth. This module is
a thin adapter over :mod:`app.llm`, which owns the provider abstraction and the
Groq → OpenRouter → Ollama fallback chain. Nothing here talks to a provider
directly.

The application is fully functional when no LLM is configured.
"""
from __future__ import annotations

from typing import Any

from ..llm import LLMResult, get_service


def explain_with_llm(findings: list[dict], context: dict) -> dict:
    """Explain structured findings. Returns a dict; never raises.

    Backwards-compatible signature. The returned dict carries the original
    keys (``available``/``explanation``/``error``) plus the richer
    provider/fallback metadata the UI needs.
    """
    return llm_result_to_dict(get_service().explain({"context": context}, findings))


def explain_run(run: dict[str, Any], findings: list[dict], context: dict) -> dict:
    """Explain a full run, forwarding the structured sections the engine owns.

    The LLM only ever sees statistics the deterministic engine computed; it is
    never asked to calculate anything itself.
    """
    return llm_result_to_dict(get_service().explain(build_payload(run, context), findings))


def build_payload(run: dict[str, Any], context: dict) -> dict[str, Any]:
    """Assemble the structured, LLM-facing view of a run.

    Everything here is *observed* data recorded by the training engine. There is
    deliberately no raw user data, no prompt content and no credentials.
    """
    history = run.get("history") or []
    snaps = run.get("snapshots") or []
    net = run.get("network_summary") or {}
    data = run.get("data_summary") or {}
    final = (run.get("final") or {}).get("test_metrics") or {}
    last = history[-1] if history else {}

    payload: dict[str, Any] = {
        "context": context,
        "architecture_summary": {
            "total_params": net.get("total_params"),
            "n_layers": net.get("n_layers"),
            "layer_shapes": [
                {k: s.get(k) for k in ("id", "kind", "in_shape", "out_shape", "params")}
                for s in (net.get("layer_shapes") or [])[:32]
            ],
        },
        "dataset_summary": {
            k: data.get(k) for k in
            ("task", "n_train", "n_val", "n_test", "n_features", "n_classes",
             "class_names", "class_counts_train", "scaled")
        },
        "training_metrics": {
            "metric_name": last.get("metric_name"),
            "final_train_metric": last.get("train_metric"),
            "final_val_metric": last.get("val_metric"),
            "test_metrics": final,
            "epochs_run": len(history),
            "duration_s": run.get("duration_s"),
            "status": run.get("status"),
        },
        "loss_history_summary": {
            "first_train_loss": history[0].get("train_loss") if history else None,
            "final_train_loss": last.get("train_loss"),
            "first_val_loss": history[0].get("val_loss") if history else None,
            "final_val_loss": last.get("val_loss"),
            "loss_decreased": (history[0].get("train_loss") > last.get("train_loss")
                               if history and last else None),
        },
    }

    if snaps:
        snap = snaps[-1]
        acts = snap.get("activations") or []
        grads = (snap.get("gradients") or {}).get("layers") or []
        payload["activation_statistics"] = [
            {k: a.get(k) for k in ("label", "activation", "mean", "std", "min", "max",
                                   "frac_near_zero", "frac_saturated", "dead_neuron_frac")}
            for a in acts[:24]
        ]
        payload["gradient_statistics"] = [
            {k: g.get(k) for k in ("label", "w_grad_norm", "w_grad_mean_abs",
                                   "w_grad_max_abs", "b_grad_norm")}
            for g in grads[:24]
        ]
        payload["gradient_statistics"].append(
            {"total_norm": (snap.get("gradients") or {}).get("total_norm")})

    weights = (run.get("final") or {}).get("weights") or {}
    if weights:
        payload["weight_statistics"] = {
            lid: {"shape": w.get("shape"), "stats": w.get("stats"), "kind": w.get("kind")}
            for lid, w in list(weights.items())[:24]
        }
    return payload


def llm_result_to_dict(result: LLMResult) -> dict[str, Any]:
    """Flatten an :class:`LLMResult` into the API response shape."""
    return {
        "available": result.available,
        "explanation": result.explanation,
        "error": result.error,
        "provider": result.provider,
        "model": result.model,
        "label": result.label,
        "outcome": result.outcome,
        "message": result.message,
        "fallback_used": result.fallback_used,
        "latency_s": result.latency_s,
        "chain": result.chain,
        "attempts": [a.to_dict() for a in result.attempts],
        "usage": result.usage,
    }
