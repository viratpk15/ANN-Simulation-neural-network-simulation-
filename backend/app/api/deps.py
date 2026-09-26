"""Shared dependencies/helpers for routers."""
from __future__ import annotations

import os

import torch
from fastapi import HTTPException

from ..ml.builder import ChainModel, build_model, validate_network
from ..models.schemas import NetworkSpec
from ..services import storage, training_manager
from ..ml.trainer import TrainingJob


def err(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def get_run_or_404(run_id: str) -> dict:
    # live job first (in case it was never stored), then DB
    job = training_manager.get_job(run_id)
    if job is not None and job.state in ("finished", "stopped", "failed"):
        training_manager.ensure_stored(job)
    run = storage.get_run(run_id)
    if not run:
        raise err(404, f"Run '{run_id}' not found.")
    return run


def load_run_model(run: dict) -> tuple[ChainModel, NetworkSpec]:
    if not run.get("weights_path") or not os.path.exists(run["weights_path"]):
        raise err(409, "This run has no saved weights (was training completed?).")
    spec = NetworkSpec(**run["network"])
    val = validate_network(spec)
    if not val.ok:
        raise err(409, "The network stored with this run is invalid.")
    model = build_model(spec, val)
    state = torch.load(run["weights_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, spec


def ensure_stored_if_done(job: TrainingJob) -> None:
    if job.state in ("finished", "stopped", "failed"):
        training_manager.ensure_stored(job)
