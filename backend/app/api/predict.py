"""Prediction + forward-trace + backprop-inspection endpoints."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from ..ml import datasets as ds
from ..simulation import forward_trace as ft
from .deps import err, get_run_or_404, load_run_model

router = APIRouter()


class TraceRequest(BaseModel):
    run_id: str
    features: list[float | str] | dict[str, float | str]


class BackpropRequest(TraceRequest):
    target: float | int | str


def _trace(run_id: str, features):
    run = get_run_or_404(run_id)
    meta = run.get("dataset_meta")
    if not meta:
        raise err(409, "This run has no stored preprocessing metadata.")
    model, spec = load_run_model(run)
    try:
        x = ds.transform_single(meta, features)
    except ValueError as exc:
        raise err(400, str(exc)) from exc
    trace = ft.forward_trace(model, spec, x.reshape(-1))
    out = np.array(trace["output"], dtype=np.float32)
    details = ft.classification_details(
        spec, model, out, meta.get("class_names"), (run.get("config") or {}).get("loss", ""))
    # top-activated neurons per computed layer (educational "what fired most")
    top_neurons = []
    for layer in trace["layers"]:
        if "a" in layer:
            a = np.array(layer["a"], dtype=float)
            idx = np.argsort(-np.abs(a))[:3]
            top_neurons.append({
                "layer": layer["label"], "layer_id": layer["id"],
                "top": [{"neuron": int(i), "value": round(float(a[i]), 4)} for i in idx],
            })
    return run, trace, details, top_neurons


@router.post("/predict")
def predict(req: TraceRequest):
    """Run one feature row through the trained network (real forward pass)."""
    _run, trace, details, top_neurons = _trace(req.run_id, req.features)
    last = [l for l in trace["layers"] if l["kind"] in ("dense", "output")]
    final_calc = None
    if last:
        out_layer = last[-1]
        winner = int(np.argmax(details["probabilities"])) if details.get("probabilities") else 0
        if out_layer["neurons"] and winner < len(out_layer["neurons"]):
            final_calc = {"layer": out_layer["label"], "neuron": winner,
                          **out_layer["neurons"][winner]}
    return {
        "prediction": details,
        "probabilities": details.get("probabilities"),
        "output_raw": trace["output"],
        "top_neurons": top_neurons,
        "final_layer_calculation": final_calc,
        "note": "Neuron-importance view is an educational visualisation (activation magnitude), "
                "not a rigorous attribution method.",
    }


@router.post("/inspect/forward-trace")
def forward_trace_endpoint(req: TraceRequest):
    """Full per-layer/per-neuron forward computation for the Math view."""
    _run, trace, details, top_neurons = _trace(req.run_id, req.features)
    return {"trace": trace["layers"], "prediction": details, "top_neurons": top_neurons,
            "output": trace["output"]}


@router.post("/inspect/backprop")
def backprop_endpoint(req: BackpropRequest):
    """Show one real weight update: w_new = w_old − η·∂L/∂w, computed for real."""
    run = get_run_or_404(req.run_id)
    meta = run.get("dataset_meta")
    if not meta:
        raise err(409, "This run has no stored preprocessing metadata.")
    model, _spec = load_run_model(run)
    cfg = run.get("config") or {}
    try:
        x = ds.transform_single(meta, req.features)
    except ValueError as exc:
        raise err(400, str(exc)) from exc
    # map target through the stored class encoding when needed
    target = req.target
    if meta.get("task") == "classification" and not isinstance(target, (int, float)):
        mapping = (meta.get("target") or {}).get("class_to_idx") or {}
        if str(target) not in mapping:
            raise err(400, f"Unknown class '{target}'. Valid: {list(mapping)}")
        target = int(mapping[str(target)])
    result = ft.backprop_example(model, x, target, cfg.get("loss", "cross_entropy"),
                                 meta.get("task", "classification"),
                                 cfg.get("learning_rate", 0.01))
    return result


@router.get("/inspect/weights/{run_id}")
def weights(run_id: str):
    run = get_run_or_404(run_id)
    final = run.get("final") or {}
    w = final.get("weights")
    if not w:
        # fall back to computing from the saved state
        model, _spec = load_run_model(run)
        from ..ml.builder import weight_matrices
        w = weight_matrices(model)
    return {"run_id": run_id, "layers": w}
