"""Research Mode: experiments (named, repeatable training runs) + comparison."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..models.schemas import TrainRequest
from ..services import storage, training_manager
from .deps import err
from .training import preflight

router = APIRouter()


@router.post("/run")
def run_experiment(req: TrainRequest):
    """Start a training run recorded under an experiment name (research mode)."""
    if not req.experiment_name:
        raise err(400, "experiment_name is required for research-mode runs.")
    preflight(req)
    exp_id = training_manager._experiment_id_for(req.experiment_name)
    storage.save_experiment(req.experiment_name, {"request": req.model_dump()}, exp_id)
    try:
        job = training_manager.start_job(req)
    except RuntimeError as exc:
        raise err(429, str(exc)) from exc
    return {"job_id": job.id, "experiment_id": exp_id, "state": job.state,
            "note": "When training finishes, the run (config, metrics, curves, activation & "
                    "gradient stats) is stored under this experiment with a timestamp."}


@router.get("")
def list_experiments():
    return {"experiments": storage.list_experiments()}


@router.get("/compare")
def compare(run_ids: str = Query(..., description="comma-separated run ids")):
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if not (2 <= len(ids) <= 6):
        raise err(400, "Provide between 2 and 6 run ids to compare.")
    runs = []
    for rid in ids:
        run = storage.get_run(rid)
        if not run:
            raise err(404, f"Run '{rid}' not found.")
        runs.append(run)
    return {"runs": runs}


@router.get("/runs")
def all_runs():
    return {"runs": storage.list_runs()}


@router.get("/{exp_id}")
def get_experiment(exp_id: str):
    exp = storage.get_experiment(exp_id)
    if not exp:
        raise err(404, "Experiment not found.")
    return exp


@router.delete("/{exp_id}")
def delete_experiment(exp_id: str):
    return {"deleted": storage.delete_experiment(exp_id)}
