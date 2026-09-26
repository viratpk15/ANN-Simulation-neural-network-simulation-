"""Network spec validation + model building tests."""
import torch

from app.ml.builder import build_model, validate_network
from app.models.schemas import NetworkSpec

from .conftest import make_xor_spec


def spec(d):
    return NetworkSpec.model_validate(d)


def test_valid_chain_shapes_and_param_count():
    val = validate_network(make_xor_spec())
    assert val.ok, val.errors
    assert val.input_dim == 2 and val.output_dim == 2
    assert val.total_params == (2 * 8 + 8) + (8 * 8 + 8) + (8 * 2 + 2)  # 114
    assert val.order == ["input", "h1", "h2", "output"]


def test_no_bias_param_count():
    s = make_xor_spec()
    s.layers[1].params.use_bias = False
    val = validate_network(s)
    h1 = next(l for l in val.layers if l.id == "h1")
    assert h1.params == 2 * 8


def test_missing_input():
    s = make_xor_spec()
    s.layers = [l for l in s.layers if l.kind != "input"]
    val = validate_network(s)
    assert not val.ok and any(e["code"] == "NO_INPUT" for e in val.errors)


def test_missing_output():
    s = make_xor_spec()
    s.layers = [l for l in s.layers if l.kind != "output"]
    val = validate_network(s)
    assert not val.ok and any(e["code"] == "NO_OUTPUT" for e in val.errors)


def test_disconnected_layer():
    s = make_xor_spec()
    s.edges = [e for e in s.edges if e.id != "e2"]
    val = validate_network(s)
    assert not val.ok and any(e["code"] == "DISCONNECTED" for e in val.errors)


def test_branching_rejected():
    s = make_xor_spec()
    s.edges.append(s.edges[0].model_copy(update={"id": "extra", "target": "h2"}))
    val = validate_network(s)
    assert not val.ok and any(e["code"] == "BRANCH_IN" for e in val.errors)


def test_cycle_detected():
    d = {
        "name": "cyclic",
        "layers": [
            {"id": "input", "kind": "input", "params": {"features": 2}},
            {"id": "a", "kind": "dense", "params": {"neurons": 4}},
            {"id": "b", "kind": "dense", "params": {"neurons": 4}},
            {"id": "output", "kind": "output", "params": {"neurons": 2}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "a"},
            {"id": "e2", "source": "a", "target": "output"},
            # disconnected self-consistent cycle a->b? b->b is a cycle; b->a forms loop a->b->a
            {"id": "e3", "source": "b", "target": "b"},
        ],
    }
    val = validate_network(spec(d))
    assert not val.ok
    assert any(e["code"] in ("CYCLE", "UNREACHABLE_OR_CYCLE") for e in val.errors)


def test_bad_input_features():
    s = make_xor_spec()
    s.layers[0].params.features = 0
    val = validate_network(s)
    assert not val.ok and any(e["code"] == "BAD_FEATURES" for e in val.errors)


def test_invalid_activation_rejected():
    s = make_xor_spec()
    s.layers[1].params.activation = "does-not-exist"
    val = validate_network(s)
    assert not val.ok and any(e["code"] == "BAD_ACTIVATION" for e in val.errors)


def test_forward_pass_shapes():
    s = make_xor_spec()
    val = validate_network(s)
    model = build_model(s, val, seed=0)
    x = torch.rand(5, 2)
    out = model(x)
    assert out.shape == (5, 2)
    # softmax output sums to 1
    assert torch.allclose(out.sum(dim=1), torch.ones(5), atol=1e-5)


def test_deterministic_init_with_seed():
    s = make_xor_spec()
    m1 = build_model(s, seed=123)
    m2 = build_model(s, seed=123)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_custom_activation_in_model():
    s = make_xor_spec()
    s.layers[1].params.activation = "custom::swishy"
    s.custom_activations = {"swishy": "x / (1 + exp(-x))"}
    val = validate_network(s)
    assert val.ok, val.errors
    model = build_model(s, val, seed=0)
    out = model(torch.rand(4, 2))
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_custom_activation_gradient_flows():
    s = make_xor_spec()
    s.layers[1].params.activation = "custom::swishy"
    s.custom_activations = {"swishy": "x / (1 + exp(-x))"}
    model = build_model(s, seed=0)
    out = model(torch.rand(4, 2))
    # (out**2).sum() is non-constant w.r.t. softmax outputs, unlike out.sum()
    loss = (out ** 2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
