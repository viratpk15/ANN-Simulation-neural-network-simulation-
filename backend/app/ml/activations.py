"""Activation-function registry: built-ins plus user-defined expressions."""
from __future__ import annotations

import torch
from torch import nn

from .expressions import CompiledExpression, ExpressionError, compile_expression


class ExpressionActivation(nn.Module):
    """nn.Module wrapper around a compiled safe expression (autograd-able)."""

    def __init__(self, compiled: CompiledExpression, label: str = "custom"):
        super().__init__()
        self.compiled = compiled
        self.label = label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.compiled.eval_torch(x)
        # keep training numerically contained; activations exploding to inf
        # poison gradients downstream
        return torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


_BUILTIN: dict[str, type[nn.Module]] = {
    "linear": Identity,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "softplus": nn.Softplus,
    "softmax": nn.Softmax,
}

# activations for which saturation statistics are meaningful
SATURATING = {"sigmoid", "tanh"}
RELU_FAMILY = {"relu", "leaky_relu", "gelu", "elu", "selu", "swish"}


def list_builtin_activations() -> list[dict]:
    return [
        {"id": "linear", "name": "Linear (identity)", "hint": "f(x) = x — typical for regression outputs"},
        {"id": "relu", "name": "ReLU", "hint": "f(x) = max(0, x) — default for hidden layers"},
        {"id": "leaky_relu", "name": "Leaky ReLU", "hint": "small slope for x<0 — avoids dead neurons"},
        {"id": "sigmoid", "name": "Sigmoid", "hint": "squashes to (0,1) — binary outputs"},
        {"id": "tanh", "name": "Tanh", "hint": "squashes to (-1,1), zero-centred"},
        {"id": "gelu", "name": "GELU", "hint": "smooth ReLU alternative used in transformers"},
        {"id": "elu", "name": "ELU", "hint": "exponential for negatives, helps push mean to zero"},
        {"id": "swish", "name": "Swish / SiLU", "hint": "x * sigmoid(x)"},
        {"id": "softplus", "name": "Softplus", "hint": "smooth approximation of ReLU"},
        {"id": "softmax", "name": "Softmax", "hint": "probabilities over classes — output layer only"},
    ]


def is_valid_activation(name: str, custom: dict[str, str] | None = None) -> bool:
    if name in _BUILTIN:
        return True
    if name.startswith("custom::"):
        key = name.split("::", 1)[1]
        return bool(custom) and key in custom
    return False


def make_activation(name: str, custom: dict[str, str] | None = None, dim: int | None = None) -> nn.Module:
    """Create the activation module.

    ``name`` is a builtin id or ``custom::<key>`` where ``custom`` maps keys
    to formula strings. Raises ValueError/ExpressionError on problems.
    """
    if name == "softmax":
        return nn.Softmax(dim=dim or -1)
    if name in _BUILTIN:
        return _BUILTIN[name]()
    if name.startswith("custom::"):
        key = name.split("::", 1)[1]
        if not custom or key not in custom:
            raise ExpressionError(f"Custom activation '{key}' was not provided with the request.")
        return ExpressionActivation(compile_expression(custom[key]), label=key)
    raise ValueError(
        f"Unknown activation '{name}'. Builtins: {', '.join(_BUILTIN)} or 'custom::<name>'."
    )
