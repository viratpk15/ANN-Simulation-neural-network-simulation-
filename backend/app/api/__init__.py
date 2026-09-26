from fastapi import APIRouter

from . import activations, datasets, diagnostics, experiments, networks, predict, projects, training

api_router = APIRouter(prefix="/api")
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(networks.router, prefix="/networks", tags=["networks"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(predict.router, tags=["predict", "inspect"])
api_router.include_router(activations.router, prefix="/activations", tags=["activations"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(diagnostics.router, prefix="/diagnostics", tags=["diagnostics"])
