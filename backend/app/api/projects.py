"""Project save/load/export/import (.nnsim files)."""
from __future__ import annotations

import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models.schemas import ProjectPayload
from ..services import storage
from .deps import err

router = APIRouter()

FORMAT = "neurosim-lab/project"
VERSION = 1


@router.post("")
def save_project(payload: ProjectPayload):
    rec = storage.save_project(payload.model_dump())
    return {"project_id": rec["id"], "updated_at": rec["updated_at"]}


@router.get("")
def list_projects():
    return {"projects": storage.list_projects()}


@router.get("/{project_id}")
def get_project(project_id: str):
    rec = storage.get_project(project_id)
    if not rec:
        raise err(404, "Project not found.")
    return rec


@router.delete("/{project_id}")
def delete_project(project_id: str):
    return {"deleted": storage.delete_project(project_id)}


@router.post("/export")
def export_project(payload: ProjectPayload):
    """Download a portable .nnsim project file (JSON format)."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", payload.name.strip() or "project").strip("-")
    doc = {"format": FORMAT, "version": VERSION, "payload": payload.model_dump()}
    return JSONResponse(
        content=doc,
        headers={"Content-Disposition": f'attachment; filename="{slug}.nnsim"'},
    )


@router.post("/import")
def import_project(doc: dict):
    if doc.get("format") != FORMAT:
        raise err(400, "Not a NeuroSim Lab project file (.nnsim).")
    try:
        payload = ProjectPayload(**doc["payload"])
    except Exception as exc:
        # Validation error caused by the client's file, but keep the chain so the
        # underlying pydantic error stays in the server log.
        raise err(400, f"Project file is malformed: {exc}") from exc
    rec = storage.save_project(payload.model_dump())
    return {"project_id": rec["id"], "name": payload.name}
