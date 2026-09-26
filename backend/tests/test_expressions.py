"""Safe expression compiler tests — correctness AND security."""

import pytest
import torch

from app.ml.expressions import ExpressionError, analyze_expression, compile_expression


def test_sigmoid_formula_numerics():
    expr = compile_expression("1 / (1 + exp(-x))")
    xs = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    ys = expr.eval_torch(xs)
    expected = torch.sigmoid(xs)
    assert torch.allclose(ys, expected, atol=1e-12)


def test_caret_power_sugar():
    expr = compile_expression("x^2 + 2*x + 1")
    assert expr.eval_scalar(3.0) == pytest.approx(16.0)


def test_function_whitelist():
    for f in ["tanh(x)", "sin(x) + cos(x)", "sqrt(abs(x))", "min(x, 0)", "clip(x, -1, 1)", "softplus(x)"]:
        compile_expression(f)  # should not raise


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')",
    "x.__class__",
    "open('/etc/passwd')",
    "(lambda y: y)(x)",
    "y + 1",
    "eval('1')",
    "import os",
    "x if x > 0 else 0",       # ternary not whitelisted
    "print(x)",
    "[x, x]",
    "{'a': 1}",
])
def test_injection_rejected(bad):
    with pytest.raises(ExpressionError):
        compile_expression(bad)


def test_unknown_function_rejected_with_names():
    with pytest.raises(ExpressionError) as exc:
        compile_expression("weird(x)")
    assert "weird" in str(exc.value)


def test_syntax_error_reported():
    with pytest.raises(ExpressionError):
        compile_expression("x +* 2")


def test_analyze_reports_nan_domain():
    res = analyze_expression("log(x)", -5, 5)
    assert res["ok"] is False
    assert any("NaN" in i for i in res["issues"])
    assert res["stats"]["fraction_finite"] < 1.0


def test_analyze_derivative_is_exact():
    res = analyze_expression("x^2", -5, 5, points=11)
    # f'(x) = 2x
    for x, d in zip(res["xs"], res["dys"]):
        assert d == pytest.approx(2 * x, abs=1e-8)


def test_stable_swish_like():
    res = analyze_expression("x / (1 + exp(-x))", -5, 5)
    assert res["ok"] is True
    assert res["stats"]["mean"] is not None


def test_gradient_death_detected():
    res = analyze_expression("floor(x)", -5, 5)
    assert any("zero/undefined" in i for i in res["issues"])
