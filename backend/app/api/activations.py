"""Custom activation functions: safe validation, plotting, persistence."""
from __future__ import annotations

from fastapi import APIRouter

from ..ml.activations import list_builtin_activations
from ..ml.expressions import ExpressionError, analyze_expression, compile_expression
from ..models.schemas import ActivationValidateRequest, SavedActivation
from ..services import storage
from .deps import err

router = APIRouter()


@router.get("/library")
def library():
    """Builtin activation catalogue + the user's saved custom functions."""
    return {
        "builtin": list_builtin_activations(),
        "custom": storage.list_activations(),
    }


@router.post("/validate")
def validate(req: ActivationValidateRequest):
    """Compile the formula *safely*, evaluate it and its derivative on a grid,
    and report numerical-health issues. Never executes arbitrary code."""
    try:
        result = analyze_expression(req.formula, req.lo, req.hi)
    except ExpressionError as exc:
        return {"ok": False, "error": str(exc), "xs": [], "ys": [], "dys": [],
                "stats": None, "issues": [str(exc)]}
    return result


@router.post("")
def save_custom(act: SavedActivation):
    if not act.name.replace("_", "").isalnum():
        raise err(400, "Activation names may only contain letters, digits and underscores.")
    try:
        compile_expression(act.formula)
    except ExpressionError as exc:
        raise err(400, f"Cannot save invalid formula: {exc}") from exc
    storage.save_activation(act.name, act.formula, act.notes)
    return {"saved": act.name, "usage": f"Use as activation id 'custom::{act.name}'"}


@router.delete("/{name}")
def delete_custom(name: str):
    return {"deleted": storage.delete_activation(name)}
