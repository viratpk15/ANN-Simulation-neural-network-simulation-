"""Registry of live training jobs + persistence of finished runs."""
from __future__ import annotations

import threading

import torch

from ..config import settings
from ..ml.trainer import TrainingJob
from ..models.schemas import TrainRequest
from . import storage

_lock = threading.Lock()
_jobs: dict[str, TrainingJob] = {}
_stored: set[str] = set()


def ensure_stored(job: TrainingJob) -> None:
    """Idempotently persist a finished run (weights + record)."""
    with _lock:
        if job.id in _stored:
            return
        _stored.add(job.id)
    try:
        finalize_and_store(job, job.request.experiment_name)
    except Exception:
        with _lock:
            _stored.discard(job.id)


def start_job(request: TrainRequest) -> TrainingJob:
    existing_running = [j for j in _jobs.values() if j.state in ("queued", "running", "paused")]
    if len(existing_running) >= 3:
        raise RuntimeError("Too many concurrent training jobs — stop one first.")
    job = TrainingJob(request, settings_max_epochs=settings.max_epochs)
    with _lock:
        _jobs[job.id] = job
    job.start()
    return job


def get_job(job_id: str) -> TrainingJob | None:
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    return [j.status() for j in _jobs.values()]


def finalize_and_store(job: TrainingJob, experiment_name: str | None = None) -> dict:
    """Persist a finished job: weights file + DB record (+ experiment link)."""
    from ..ml.builder import validate_network

    rid = job.id
    exp_id = None
    if experiment_name:
        exp_id = _experiment_id_for(experiment_name)
    weights_path = None
    if job.model is not None:
        weights_path = str(settings.runs_dir / f"{rid}.pt")
        torch.save(job.model.state_dict(), weights_path)

    val = validate_network(job.request.network)
    record = {
        "id": rid,
        "created_at": job.created_at,
        "status": job.state,
        "network": job.request.network.model_dump(),
        "network_summary": {
            "total_params": val.total_params,
            "n_layers": len(val.order),
            "layer_shapes": [s.model_dump() for s in val.layers],
        },
        "dataset_meta": job.data.meta if job.data else None,
        "config": job.request.config.model_dump(),
        "history": job.history,
        "snapshots": job.snapshots[-10:],      # keep the most recent snapshots
        "data_summary": job.data_summary,
        "final": job.final,
        "weights_path": weights_path,
        "experiment_id": exp_id,
        "duration_s": round(job.duration_s, 2),
        "error": job.error,
    }
    storage.save_run(record)
    return record


def _experiment_id_for(name: str) -> str:
    """Stable id per experiment name (create on first use)."""
    for exp in storage.list_experiments():
        if exp["name"] == name:
            return exp["id"]
    config_snapshot: dict = {"name": name}
    return storage.save_experiment(name, config_snapshot)["id"]


def update_experiment_config(exp_id: str, config: dict) -> None:
    exp = storage.get_experiment(exp_id)
    if exp:
        storage.save_experiment(exp["name"], config, exp_id)


def job_to_status(job: TrainingJob) -> dict:
    return job.status()
