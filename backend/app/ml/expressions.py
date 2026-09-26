"""Safe mathematical-expression compiler for custom activation functions.

User formulas are parsed with :mod:`ast` and compiled into a restricted
tree of PyTorch operations. Arbitrary Python is *never* executed:

* only a whitelist of AST node types is accepted,
* only a whitelist of math functions may be called,
* the only allowed variable name is ``x`` (plus constants ``pi``/``e``),
* attribute access, subscripts, lambdas, names like ``__import__`` etc.
  are all rejected before evaluation.

Because the compiled form is built from differentiable torch ops, custom
activations get correct gradients *for free* through autograd — no
hand-written derivative is required (though the caller may inspect the
autograd derivative numerically).
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import torch


class ExpressionError(ValueError):
    """Raised when a user-supplied formula is unsafe or invalid."""


# -> torch implementations -------------------------------------------------
def _torch_pow(base, exp):
    return torch.pow(base, exp)


def _torch_sigmoid(x):
    return torch.sigmoid(x)


def _torch_relu(x):
    return torch.clamp_min(x, 0.0)


def _torch_clip(x, lo, hi):
    return torch.clamp(x, float(lo), float(hi))


_FUNC_IMPLS = {
    "exp": (torch.exp, 1),
    "log": (torch.log, 1),
    "log10": (torch.log10, 1),
    "sqrt": (torch.sqrt, 1),
    "sin": (torch.sin, 1),
    "cos": (torch.cos, 1),
    "tan": (torch.tan, 1),
    "asin": (torch.asin, 1),
    "acos": (torch.acos, 1),
    "atan": (torch.atan, 1),
    "abs": (torch.abs, 1),
    "tanh": (torch.tanh, 1),
    "sinh": (torch.sinh, 1),
    "cosh": (torch.cosh, 1),
    "sigmoid": (_torch_sigmoid, 1),
    "relu": (_torch_relu, 1),
    "softplus": (torch.nn.functional.softplus, 1),
    "sign": (torch.sign, 1),
    "floor": (torch.floor, 1),
    "ceil": (torch.ceil, 1),
    "pow": (_torch_pow, 2),
    "min": (torch.minimum, 2),
    "max": (torch.maximum, 2),
    "clip": (_torch_clip, 3),
}

_CONSTS = {"pi": math.pi, "e": math.e}
_VAR = "x"

_BIN_OPS = {
    ast.Add: torch.add,
    ast.Sub: torch.sub,
    ast.Mult: torch.mul,
    ast.Div: torch.div,
    ast.Pow: _torch_pow,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

MAX_FORMULA_LEN = 400


@dataclass
class CompiledExpression:
    source: str
    tree: ast.Expression

    # -- evaluation -------------------------------------------------------
    def eval_torch(self, x: torch.Tensor) -> torch.Tensor:
        return _eval_node(self.tree.body, x)

    def eval_scalar(self, v: float) -> float:
        t = torch.tensor(float(v), dtype=torch.float64)
        return float(self.eval_torch(t))


def compile_expression(formula: str) -> CompiledExpression:
    """Parse + validate a formula string into a :class:`CompiledExpression`.

    Raises :class:`ExpressionError` with a human-readable message on any
    syntax, safety or semantic problem.
    """
    if not formula or not formula.strip():
        raise ExpressionError("The formula is empty.")
    formula = formula.strip()
    if len(formula) > MAX_FORMULA_LEN:
        raise ExpressionError(f"Formula is too long (max {MAX_FORMULA_LEN} characters).")
    # Friendly syntax: allow '^' as power like many CAS tools.
    normalised = formula.replace("^", "**")
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - message varies
        raise ExpressionError(f"Syntax error in formula: {exc.msg}") from exc
    _validate(tree)
    return CompiledExpression(source=formula, tree=tree)


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"Disallowed syntax '{type(node).__name__}'. Only numbers, the variable "
                "'x', operators (+ - * / ^) and whitelisted functions are allowed."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ExpressionError("Only numeric constants are allowed.")
        if isinstance(node, ast.Name) and node.id not in _CONSTS and node.id != _VAR \
                and node.id not in _FUNC_IMPLS:
            raise ExpressionError(
                f"Unknown name '{node.id}'. Use 'x' as the input variable "
                f"(constants pi and e are available)."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNC_IMPLS:
                fn = ast.dump(node.func) if not isinstance(node.func, ast.Name) else node.func.id
                raise ExpressionError(
                    f"Function '{fn}' is not allowed. Available: {', '.join(sorted(_FUNC_IMPLS))}."
                )
            _, arity = _FUNC_IMPLS[node.func.id]
            if len(node.args) != arity or node.keywords:
                raise ExpressionError(
                    f"Function '{node.func.id}' expects exactly {arity} argument(s)."
                )


def _eval_node(node: ast.AST, x: torch.Tensor) -> torch.Tensor:
    if isinstance(node, ast.Constant):
        return torch.as_tensor(float(node.value), dtype=x.dtype)
    if isinstance(node, ast.Name):
        if node.id == _VAR:
            return x
        if node.id in _CONSTS:
            return torch.as_tensor(_CONSTS[node.id], dtype=x.dtype)
        raise ExpressionError(f"Function '{node.id}' must be called, e.g. {node.id}(x).")
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, x)
        if isinstance(node.op, ast.USub):
            return -val
        return val
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, x)
        right = _eval_node(node.right, x)
        op = _BIN_OPS[type(node.op)]
        if isinstance(node.op, ast.Pow):
            # guard numerical blowups while staying differentiable
            base = torch.clamp(left, -1e6, 1e6)
            return op(base, right)
        return op(left, right)
    if isinstance(node, ast.Call):
        fn, _ = _FUNC_IMPLS[node.func.id]
        args = [_eval_node(a, x) for a in node.args]
        return fn(*args)
    raise ExpressionError(f"Unsupported expression element: {type(node).__name__}")


# --------------------------------------------------------------------------
def analyze_expression(formula: str, lo: float = -5.0, hi: float = 5.0, points: int = 121) -> dict:
    """Evaluate a formula on a grid; return curves + numerical health info.

    Used by the "validate custom activation" API. The derivative is computed
    with autograd (per scalar point) so it is exact, not a finite difference.
    """
    compiled = compile_expression(formula)
    xs = torch.linspace(lo, hi, points, dtype=torch.float64)
    with torch.no_grad():
        ys = compiled.eval_torch(xs)
    # elementwise autograd derivative (exact, point by point)
    xs_g = xs.clone().requires_grad_(True)
    dys: list[float] = []
    try:
        for i in range(points):
            yi = compiled.eval_torch(xs_g[i])
            g = torch.autograd.grad(yi, xs_g, allow_unused=True)[0]
            dys.append(float(g[i]) if g is not None else float("nan"))
    except RuntimeError:
        dys = [float("nan")] * points

    ys_list = [float(v) for v in ys]
    finite = [math.isfinite(v) for v in ys_list]
    has_nan = any(math.isnan(v) for v in ys_list)
    has_inf = any(math.isinf(v) for v in ys_list)
    d_finite = [math.isfinite(d) for d in dys]

    issues: list[str] = []
    if has_nan:
        issues.append(
            "Formula produced NaN for some inputs in the tested range — check domains "
            "(log/sqrt of negatives, division by zero).")
    if has_inf:
        issues.append("Formula produced +/- infinity in the tested range — training will likely diverge.")
    if all(d == 0 or not math.isfinite(d) for d in dys):
        issues.append(
            "Derivative is zero/undefined everywhere in the tested range — gradients "
            "cannot flow (e.g. floor/sign only).")
    overflow_risk = any(abs(v) > 1e4 for v in ys_list if math.isfinite(v))
    if overflow_risk:
        issues.append(
            "Output magnitude is very large in parts of the range — consider clipping "
            "or rescaling for training stability.")

    finite_vals = [v for v in ys_list if math.isfinite(v)]
    finite_ders = [d for d in dys if math.isfinite(d)]
    return {
        "ok": not (has_nan or has_inf),
        "formula": formula,
        "range": [lo, hi],
        "xs": [float(v) for v in xs],
        "ys": [v if math.isfinite(v) else None for v in ys_list],
        "dys": [d if math.isfinite(d) else None for d in dys],
        "stats": {
            "min": min(finite_vals) if finite_vals else None,
            "max": max(finite_vals) if finite_vals else None,
            "mean": (sum(finite_vals) / len(finite_vals)) if finite_vals else None,
            "max_abs_derivative": max((abs(d) for d in finite_ders), default=None),
            "fraction_finite": sum(finite) / len(finite),
            "derivative_defined_fraction": sum(d_finite) / len(d_finite),
        },
        "issues": issues,
    }
