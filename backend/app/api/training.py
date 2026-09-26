"""Training endpoints + live WebSocket stream."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..ml import builder, datasets
from ..ml.trainer import TrainingError, check_compatibility
from ..models.schemas import TrainRequest
from ..services import training_manager
from .deps import err

router = APIRouter()
ws_router = APIRouter()


def preflight(req: TrainRequest) -> None:
    """Fail fast (HTTP 400) before a thread is even started."""
    val = builder.validate_network(req.network)
    if not val.ok:
        first = val.errors[0]
        raise err(400, f"Network validation failed: {first['message']}"
                       + (f" Suggestion: {first['suggestion']}" if first.get("suggestion") else ""))
    try:
        data = datasets.prepare(req.dataset)
    except KeyError as exc:
        # A KeyError here can come from a genuine bug in dataset resolution, not
        # only from a missing dataset. Chain it so the log keeps the real cause
        # instead of the client seeing a bare 404.
        raise err(404, str(exc)) from exc
    except ValueError as exc:
        raise err(400, f"Dataset error: {exc}") from exc
    if val.input_dim != data.n_features:
        raise err(400, f"Input layer expects {val.input_dim} features but the prepared dataset has "
                       f"{data.n_features}. Adjust the input layer or the selected feature columns.")
    last_layer = {l.id: l for l in req.network.layers}[val.order[-1]]
    try:
        check_compatibility(val.output_dim or 0, last_layer.params.activation, data, req.config.loss)
    except TrainingError as exc:
        raise err(400, str(exc)) from exc


@router.post("/start")
def start_training(req: TrainRequest):
    preflight(req)
    if req.experiment_name:
        from ..services import storage, training_manager as tm
        exp_id = tm._experiment_id_for(req.experiment_name)
        storage.save_experiment(req.experiment_name, {"request": req.model_dump()}, exp_id)
    try:
        job = training_manager.start_job(req)
    except RuntimeError as exc:
        # Usually "too many concurrent jobs" — expected control flow, but the
        # chain costs nothing and aids debugging if it ever isn't.
        raise err(429, str(exc)) from exc
    return {"job_id": job.id, "state": job.state}


@router.post("/{job_id}/pause")
def pause(job_id: str):
    job = training_manager.get_job(job_id) or (_ for _ in ()).throw(err(404, "Job not found."))
    job.pause()
    return {"state": job.state}


@router.post("/{job_id}/resume")
def resume(job_id: str):
    job = training_manager.get_job(job_id) or (_ for _ in ()).throw(err(404, "Job not found."))
    job.resume()
    return {"state": job.state}


@router.post("/{job_id}/stop")
def stop(job_id: str):
    job = training_manager.get_job(job_id) or (_ for _ in ()).throw(err(404, "Job not found."))
    job.stop()
    return {"state": "stopping"}


@router.get("/{job_id}/status")
def status(job_id: str):
    job = training_manager.get_job(job_id)
    if job is None:
        # maybe the server restarted — the run may still exist on disk
        from ..services import storage
        run = storage.get_run(job_id)
        if run:
            return {"job_id": job_id, "state": run["status"], "final": run.get("final"),
                    "epoch": len(run.get("history") or []), "total_epochs": (run.get("config") or {}).get("epochs"),
                    "persisted": True, "error": run.get("error")}
        raise err(404, "Job not found.")
    if job.state in ("finished", "stopped", "failed"):
        training_manager.ensure_stored(job)
    return job.status()


@router.get("/{job_id}/metrics")
def metrics(job_id: str):
    job = training_manager.get_job(job_id)
    if job is not None:
        if job.state in ("finished", "stopped", "failed"):
            training_manager.ensure_stored(job)
        return {
            "job_id": job_id, "state": job.state, "history": job.history,
            "snapshots": job.snapshots, "final": job.final,
            "data_summary": job.data_summary, "duration_s": job.duration_s,
        }
    from ..services import storage
    run = storage.get_run(job_id)
    if not run:
        raise err(404, "Job/run not found.")
    return {
        "job_id": job_id, "state": run["status"], "history": run.get("history") or [],
        "snapshots": run.get("snapshots") or [], "final": run.get("final"),
        "data_summary": run.get("data_summary"), "duration_s": run.get("duration_s"),
    }


@router.get("")
def jobs():
    return {"jobs": training_manager.list_jobs()}


TERMINAL = ("finished", "stopped", "failed")


@ws_router.websocket("/ws/training/{job_id}")
async def training_ws(ws: WebSocket, job_id: str):
    """Replay-then-stream job events until a terminal state is reached."""
    await ws.accept()
    job = training_manager.get_job(job_id)
    if job is None:
        await ws.send_text(json.dumps({"type": "error", "message": "Job not found."}))
        await ws.close(code=4404)
        return
    idx = 0
    try:
        while True:
            n, events = job.events_after(idx)
            for ev in events:
                await ws.send_text(json.dumps(ev))
            idx = n
            if job.state in TERMINAL:
                training_manager.ensure_stored(job)
                await ws.send_text(json.dumps({
                    "type": "done", "state": job.state, "error": job.error,
                    "final": job.final, "data_summary": job.data_summary,
                    "run_id": job.id, "duration_s": round(job.duration_s, 2),
                }))
                break
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
