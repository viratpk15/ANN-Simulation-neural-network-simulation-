"""NeuroSim Lab — FastAPI application entry point."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import api_router
from .api.training import ws_router
from .config import settings
from .services.storage import init_db

app = FastAPI(
    title="NeuroSim Lab API",
    version=__version__,
    description="Interactive Neural Network Simulation, Visualization & AI Diagnosis Platform",
)

cors = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in cors else cors,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(_req: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "detail": "Invalid request payload.",
        "errors": [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg")} for e in exc.errors()],
    })


@app.exception_handler(ValueError)
async def value_error_handler(_req: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "name": "NeuroSim Lab API",
        "version": __version__,
        "llm_enabled": settings.llm_enabled,
        "llm_provider_order": settings.effective_provider_order,
    }


app.include_router(api_router)
app.include_router(ws_router)

# Serve the built frontend (single-port production mode) if it exists.
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
