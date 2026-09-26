"""Tests for the layer catalogue, shape inference, model building and validation
of every newly supported layer kind, plus backward compatibility with v1
projects (Input / Dense / Dropout / Output only)."""
import pytest
import torch
from torch import nn

from app.ml import shapes as shp
from app.ml.builder import build_model, validate_network, weight_matrices
from app.ml.shapes import (
    ShapeError,
    TensorShape,
    batchnorm_params,
    conv_output_size,
    conv_params,
    dense_params,
    flatten_output_shape,
    global_pool_output_shape,
    pool_output_size,
)
from app.models.schemas import (
    CURRENT_SCHEMA_VERSION,
    IMAGE_ONLY_KINDS,
    LayerParams,
    NetworkSpec,
)


def spec(layers, edges=None):
    ids = [lay["id"] for lay in layers]
    return NetworkSpec.model_validate({
        "name": "t",
        "layers": layers,
        "edges": edges if edges is not None else
        [{"id": f"e{i}", "source": a, "target": b} for i, (a, b) in enumerate(zip(ids, ids[1:]))],
    })


def codes(val):
    return [e["code"] for e in val.errors]


CNN = [
    {"id": "in", "kind": "input", "params": {"features": 784, "input_shape": [28, 28, 1]}},
    {"id": "norm", "kind": "batchnorm", "params": {}},
    {"id": "conv", "kind": "conv2d", "params": {"filters": 8, "kernel_size": 3, "stride": 1}},
    {"id": "act", "kind": "activation", "params": {"activation": "relu"}},
    {"id": "mp", "kind": "maxpool", "params": {"pool_size": 2, "pool_stride": 2}},
    {"id": "ap", "kind": "avgpool", "params": {"pool_size": 2, "pool_stride": 2}},
    {"id": "flat", "kind": "flatten", "params": {}},
    {"id": "hid", "kind": "dense", "params": {"neurons": 16, "activation": "relu"}},
    {"id": "out", "kind": "output", "params": {"neurons": 3, "activation": "softmax"}},
]


# ===========================================================================
# catalogue
# ===========================================================================

def test_catalog_covers_every_kind():
    assert {i.kind for i in shp.LAYER_CATALOG} == set(shp.SUPPORTED_KINDS) | set(shp.UNSUPPORTED_KINDS)
    assert shp.UNSUPPORTED_KINDS == ("embedding", "lstm", "gru")


def test_coming_soon_layers_are_flagged_and_explained():
    payload = {e["kind"]: e for c in shp.catalog_payload() for e in c["layers"]}
    for kind in shp.UNSUPPORTED_KINDS:
        assert payload[kind]["supported"] is False
        # a user must be told *why* it is unavailable
        assert len(payload[kind]["coming_soon_note"]) > 20
    for kind in shp.SUPPORTED_KINDS:
        assert payload[kind]["supported"] is True
        assert payload[kind]["description"]
        assert payload[kind]["formula"]


def test_every_supported_kind_has_a_default_param_set():
    """Each supported kind must be constructible from its defaults alone, given an
    input shape appropriate to that kind (image layers need a feature map)."""
    for kind in shp.SUPPORTED_KINDS:
        p = LayerParams(**shp.default_params(kind))
        if kind == "input":
            inp = None
        elif kind in IMAGE_ONLY_KINDS or kind in ("batchnorm", "flatten"):
            inp = TensorShape.image(3, 8, 8)
        else:
            inp = TensorShape.vector(4)
        out = shp.output_shape(kind, inp, p)
        assert out is not None and out.numel > 0


# ===========================================================================
# shape arithmetic
# ===========================================================================

def test_tensor_shape_describe_and_order():
    assert TensorShape.vector(784).describe() == "784"
    # display is H x W x C even though the internal layout is C, H, W
    assert TensorShape.image(1, 28, 28).describe() == "28 × 28 × 1"
    assert TensorShape.image(3, 8, 8).as_list() == [8, 8, 3]
    assert TensorShape.image(3, 8, 8).torch_shape() == (3, 8, 8)
    assert TensorShape.image(3, 8, 8).numel == 192


def test_conv_and_pool_output_sizes():
    assert conv_output_size(28, 3, 1, 0) == 26      # 28 -> 26 with a 3x3 kernel
    assert conv_output_size(28, 3, 1, 1) == 28      # 'same' padding preserves the size
    assert conv_output_size(28, 3, 2, 0) == 13      # stride 2 roughly halves it
    assert pool_output_size(26, 2, 2, 0) == 13
    assert pool_output_size(13, 2, 2, 0) == 6


def test_parameter_count_formulas():
    # Dense: inputs x neurons (+ neurons for the bias)
    assert dense_params(4, 16, True) == 4 * 16 + 16
    assert dense_params(4, 16, False) == 4 * 16
    # Conv2D: (k x k x C_in + bias) x filters
    assert conv_params(1, 3, 32, True) == (3 * 3 * 1 + 1) * 32
    assert conv_params(3, 3, 4, True) == (3 * 3 * 3 + 1) * 4
    assert conv_params(3, 3, 4, False) == 3 * 3 * 3 * 4
    # BatchNorm: scale + shift per channel, only when affine
    assert batchnorm_params(8, True) == 16
    assert batchnorm_params(8, False) == 0


def test_shape_transformations():
    img = TensorShape.image(1, 28, 28)
    assert flatten_output_shape(img) == TensorShape.vector(784)
    assert global_pool_output_shape(TensorShape.image(8, 6, 6)) == TensorShape.vector(8)
    with pytest.raises(ShapeError):
        flatten_output_shape(TensorShape.vector(10))


# ===========================================================================
# shape inference through a full chain
# ===========================================================================

def test_cnn_shape_propagation():
    val = validate_network(spec(CNN))
    assert val.ok, val.errors
    assert val.is_cnn is True
    shapes = {s.id: s for s in val.layers}
    # channels-last display, exactly as the spec describes
    assert shapes["conv"].in_shape == [28, 28, 1]
    assert shapes["conv"].out_shape == [26, 26, 8]
    assert shapes["mp"].out_shape == [13, 13, 8]
    assert shapes["ap"].out_shape == [6, 6, 8]
    assert shapes["flat"].out_shape == [288]
    assert shapes["hid"].out_shape == [16]
    assert shapes["out"].out_shape == [3]
    # shape-preserving layers really preserve it
    for lid in ("norm", "act"):
        assert shapes[lid].in_shape == shapes[lid].out_shape


def test_cnn_parameter_counts_add_up():
    val = validate_network(spec(CNN))
    p = {s.id: s.params for s in val.layers}
    assert p["conv"] == (3 * 3 * 1 + 1) * 8                 # 80
    assert p["norm"] == 2 * 1                                # 1 channel
    assert p["act"] == 0 and p["mp"] == 0 and p["ap"] == 0 and p["flat"] == 0
    assert p["hid"] == 288 * 16 + 16                        # 4624
    assert p["out"] == 16 * 3 + 3                           # 51
    assert val.total_params == sum(p.values())


def test_batchnorm_affine_false_has_no_parameters():
    layers = [dict(lay) for lay in CNN]
    layers[1] = {"id": "norm", "kind": "batchnorm", "params": {"affine": False}}
    val = validate_network(spec(layers))
    assert val.ok, val.errors
    assert next(s for s in val.layers if s.id == "norm").params == 0
    assert val.total_params == (3 * 3 * 1 + 1) * 8 + 288 * 16 + 16 + 16 * 3 + 3


def test_globalavgpool_collapses_to_channels():
    s = spec([
        {"id": "in", "kind": "input", "params": {"features": 192, "input_shape": [8, 8, 3]}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 4, "kernel_size": 3, "padding_mode": "same"}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
    ])
    val = validate_network(s)
    assert val.ok, val.errors
    g = next(x for x in val.layers if x.id == "g")
    assert g.in_shape == [8, 8, 4] and g.out_shape == [4]



# ===========================================================================
# model building — real PyTorch modules, real forward/backward
# ===========================================================================

def test_cnn_model_builds_and_runs():
    s = spec(CNN)
    val = validate_network(s)
    model = build_model(s, val, seed=0)
    out = model(torch.rand(5, 784))
    assert out.shape == (5, 3)
    assert torch.allclose(out.sum(dim=1), torch.ones(5), atol=1e-5)  # softmax
    assert torch.isfinite(out).all()
    # the real modules are there, not placeholders
    assert isinstance(model.layers["conv"].conv, nn.Conv2d)
    assert isinstance(model.layers["norm"], nn.BatchNorm2d)
    assert isinstance(model.layers["mp"], nn.MaxPool2d)
    assert isinstance(model.layers["ap"], nn.AvgPool2d)
    assert isinstance(model.layers["flat"], nn.Flatten)


def test_conv_output_matches_manual_convolution():
    """The built Conv2d must agree with an independent hand computation."""
    s = spec([
        {"id": "in", "kind": "input", "params": {"features": 64, "input_shape": [8, 8, 1]}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 2, "kernel_size": 3, "activation": "linear"}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 1, "activation": "linear"}},
    ])
    model = build_model(s, seed=0)
    x = torch.rand(1, 1, 8, 8)
    got = model.layers["c"].conv(x)[0].detach().numpy()          # (2, 6, 6)
    W = model.layers["c"].conv.weight.detach().numpy()
    b = model.layers["c"].conv.bias.detach().numpy()
    for f in range(2):
        for i in range(6):
            for j in range(6):
                want = float((W[f, 0] * x[0, 0, i:i + 3, j:j + 3].numpy()).sum() + b[f])
                assert got[f, i, j] == pytest.approx(want, abs=1e-5)


def test_cnn_trains_end_to_end():
    """Gradients must flow through every trainable layer of a CNN."""
    s = spec(CNN)
    model = build_model(s, seed=0)
    out = model(torch.rand(8, 784))
    (out ** 2).sum().backward()
    for lid in ("conv", "norm", "hid", "out"):
        w = model.primary_weight(lid)
        assert w is not None, f"{lid} has no primary weight"
        assert w.grad is not None and torch.isfinite(w.grad).all()
        assert w.grad.abs().sum() > 0, f"{lid} received no gradient"


def test_standalone_activation_layer_is_a_real_module():
    s = spec([
        {"id": "in", "kind": "input", "params": {"features": 4}},
        {"id": "a", "kind": "activation", "params": {"activation": "gelu"}},
        {"id": "out", "kind": "output", "params": {"neurons": 4, "activation": "softmax"}},
    ])
    val = validate_network(s)
    assert val.ok, val.errors
    model = build_model(s, val, seed=0)
    x = torch.rand(3, 4)
    # the Activation layer must actually transform its input
    a = model.layers["a"](x)
    assert not torch.allclose(a, x)
    # the whole chain is then Output(Activation(x))
    assert torch.allclose(model(x), model.layers["out"](a), atol=1e-6)


@pytest.mark.parametrize("fn", ["relu", "sigmoid", "tanh", "gelu", "linear", "softmax"])
def test_every_builtin_activation_works_in_the_activation_layer(fn):
    s = spec([
        {"id": "in", "kind": "input", "params": {"features": 4}},
        {"id": "a", "kind": "activation", "params": {"activation": fn}},
        {"id": "out", "kind": "output", "params": {"neurons": 4, "activation": "softmax"}},
    ])
    val = validate_network(s)
    assert val.ok, val.errors
    out = build_model(s, val, seed=0)(torch.rand(3, 4))
    assert torch.isfinite(out).all()


def test_custom_activation_layer_is_validated_by_the_safe_parser():
    s = spec([
        {"id": "in", "kind": "input", "params": {"features": 4}},
        {"id": "a", "kind": "activation", "params": {"activation": "custom::swishy"}},
        {"id": "out", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
    ])
    s.custom_activations = {"swishy": "x / (1 + exp(-x))"}
    val = validate_network(s)
    assert val.ok, val.errors
    model = build_model(s, val, seed=0)
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    want = x / (1 + torch.exp(-x))
    assert torch.allclose(model.layers["a"](x), want, atol=1e-6)
    # gradients must reach the trained weights through the custom activation
    (model(x) ** 2).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_custom_activation_layer_never_yields_nan():
    """A custom formula must be numerically contained so it cannot poison training."""
    s = spec([
        {"id": "in", "kind": "input", "params": {"features": 4}},
        {"id": "a", "kind": "activation", "params": {"activation": "custom::risky"}},
        {"id": "out", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
    ])
    s.custom_activations = {"risky": "log(x)"}    # undefined for x <= 0
    val = validate_network(s)
    assert val.ok, val.errors
    out = build_model(s, val, seed=0)(torch.rand(4, 4))
    assert torch.isfinite(out).all()


def test_weight_matrices_expose_conv_and_batchnorm():
    s = spec(CNN)
    val = validate_network(s)
    w = weight_matrices(build_model(s, val, seed=0))
    assert w["conv"]["kind"] == "conv"
    assert w["conv"]["full_shape"] == [8, 1, 3, 3]
    assert w["conv"]["shape"] == [8, 9]           # one flattened filter per row
    assert w["norm"]["kind"] == "batchnorm"
    assert w["norm"]["shape"] == [2, 1]           # [gamma; beta] for 1 channel
    assert "running_mean" in w["norm"]
    assert w["hid"]["kind"] == "linear"
    assert w["hid"]["shape"] == [16, 288]



# ===========================================================================
# validation — every new failure mode
# ===========================================================================

def test_conv_on_flat_vector_is_rejected_with_a_suggestion():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 784}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 4, "kernel_size": 3}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "CONV_NEEDS_IMAGE" in codes(val)
    err = next(e for e in val.errors if e["code"] == "CONV_NEEDS_IMAGE")
    assert "784" in err["message"]
    assert err["suggestion"]


def test_dense_on_an_image_tensor_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 784, "input_shape": [28, 28, 1]}},
        {"id": "d", "kind": "dense", "params": {"neurons": 8}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "DENSE_NEEDS_VECTOR" in codes(val)
    msg = next(e for e in val.errors if e["code"] == "DENSE_NEEDS_VECTOR")["message"]
    assert "28 × 28 × 1" in msg


def test_flatten_on_a_vector_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 10}},
        {"id": "f", "kind": "flatten", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "NEEDS_IMAGE" in codes(val)


def test_pooling_on_a_vector_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 16}},
        {"id": "p", "kind": "maxpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "CONV_NEEDS_IMAGE" in codes(val)


def test_globalavgpool_on_a_vector_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 16}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "NEEDS_IMAGE" in codes(val)


def test_kernel_larger_than_the_feature_map_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 4, "input_shape": [2, 2, 1]}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 2, "kernel_size": 3}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "SHAPE_MISMATCH" in codes(val)


def test_same_padding_with_stride_two_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 64, "input_shape": [8, 8, 1]}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 2, "kernel_size": 3,
                                                 "padding_mode": "same", "stride": 2}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "BAD_PADDING" in codes(val)


def test_padding_not_smaller_than_the_kernel_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 256, "input_shape": [16, 16, 1]}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 2, "kernel_size": 3, "padding": 3}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "BAD_PADDING" in codes(val)



@pytest.mark.parametrize("patch,code", [
    ({"eps": 0.0}, "BAD_EPS"),
    ({"eps": -1.0}, "BAD_EPS"),
    ({"momentum": 1.5}, "BAD_MOMENTUM"),
    ({"momentum": -0.1}, "BAD_MOMENTUM"),
])
def test_invalid_batchnorm_parameters_are_rejected(patch, code):
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 8}},
        {"id": "bn", "kind": "batchnorm", "params": patch},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and code in codes(val)


@pytest.mark.parametrize("patch,code", [
    ({"pool_size": 0}, "BAD_POOL"),
    ({"pool_stride": 0}, "BAD_POOL"),
    ({"pool_padding": -1}, "BAD_POOL"),
])
def test_invalid_pooling_parameters_are_rejected(patch, code):
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 64, "input_shape": [8, 8, 1]}},
        {"id": "p", "kind": "maxpool", "params": patch},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and code in codes(val)


def test_overlapping_pooling_windows_warn():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 256, "input_shape": [16, 16, 1]}},
        {"id": "p", "kind": "avgpool", "params": {"pool_size": 3, "pool_stride": 2}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert val.ok
    assert "POOL_OVERLAP" in [w["code"] for w in val.warnings]


@pytest.mark.parametrize("patch", [{"kernel_size": 0}, {"filters": 0}, {"stride": 0}])
def test_invalid_conv_parameters_are_rejected(patch):
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 64, "input_shape": [8, 8, 1]}},
        {"id": "c", "kind": "conv2d", "params": patch},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok, f"{patch} should be rejected"


def test_invalid_activation_on_the_activation_layer_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 4}},
        {"id": "a", "kind": "activation", "params": {"activation": "nope"}},
        {"id": "out", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
    ]))
    assert not val.ok and "BAD_ACTIVATION" in codes(val)


def test_invalid_custom_activation_on_a_conv_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 64, "input_shape": [8, 8, 1]}},
        {"id": "c", "kind": "conv2d", "params": {"filters": 2, "kernel_size": 3,
                                                 "activation": "custom::missing"}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "BAD_ACTIVATION" in codes(val)


def test_input_shape_must_agree_with_features():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 10, "input_shape": [4, 4, 1]}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "BAD_FEATURES" in codes(val)


def test_batchnorm_over_a_single_spatial_position_is_rejected():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 1, "input_shape": [1, 1, 1]}},
        {"id": "bn", "kind": "batchnorm", "params": {}},
        {"id": "g", "kind": "globalavgpool", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2}},
    ]))
    assert not val.ok and "BATCHNORM_TOO_SMALL" in codes(val)


def test_stack_without_any_hidden_layer_warns():
    val = validate_network(spec([
        {"id": "in", "kind": "input", "params": {"features": 4}},
        {"id": "a", "kind": "activation", "params": {}},
        {"id": "out", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
    ]))
    assert val.ok
    assert "SHALLOW" in [w["code"] for w in val.warnings]



# ===========================================================================
# backward compatibility with v1 projects (Input/Dense/Dropout/Output only)
# ===========================================================================

V1 = {
    "name": "legacy v1 project",
    "layers": [
        {"id": "i", "kind": "input", "params": {"features": 2}},
        {"id": "d1", "kind": "dense", "params": {"neurons": 8, "activation": "tanh",
                                                "init": "xavier", "use_bias": True}},
        {"id": "dr", "kind": "dropout", "params": {"dropout_rate": 0.5}},
        {"id": "d2", "kind": "dense", "params": {"neurons": 8, "activation": "tanh"}},
        {"id": "o", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
    ],
    "edges": [{"id": "e1", "source": "i", "target": "d1"},
              {"id": "e2", "source": "d1", "target": "dr"},
              {"id": "e3", "source": "dr", "target": "d2"},
              {"id": "e4", "source": "d2", "target": "o"}],
}


def test_v1_project_loads_and_is_migrated():
    s = NetworkSpec.model_validate(V1)          # no `version` field, as v1 wrote it
    assert s.version == CURRENT_SCHEMA_VERSION
    val = validate_network(s)
    assert val.ok, val.errors
    assert val.order == ["i", "d1", "dr", "d2", "o"]
    assert val.total_params == (2 * 8 + 8) + (8 * 8 + 8) + (8 * 2 + 2)
    assert val.is_cnn is False


def test_v1_project_still_builds_and_runs():
    s = NetworkSpec.model_validate(V1)
    model = build_model(s, seed=0)
    out = model(torch.rand(4, 2))
    assert out.shape == (4, 2)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)
    # Dropout is still a real nn.Dropout and is inactive at inference
    assert isinstance(model.layers["dr"], nn.Dropout)
    model.eval()
    assert torch.allclose(model.layers["dr"](torch.ones(3, 8)), torch.ones(3, 8))


def test_v1_project_shapes_keep_their_original_meaning():
    val = validate_network(NetworkSpec.model_validate(V1))
    shapes = {s.id: s for s in val.layers}
    assert shapes["d1"].in_features == 2 and shapes["d1"].out_features == 8
    assert shapes["dr"].in_features == 8 and shapes["dr"].out_features == 8
    assert shapes["i"].in_features is None and shapes["i"].out_features == 2
    # ...and the new rank-aware fields agree with them
    assert shapes["d1"].in_shape == [2] and shapes["d1"].out_shape == [8]


def test_v1_project_weight_export_unchanged():
    w = weight_matrices(build_model(NetworkSpec.model_validate(V1), seed=0))
    assert set(w) == {"d1", "d2", "o"}
    assert w["d1"]["kind"] == "linear"
    assert w["d1"]["shape"] == [8, 2]
    assert w["o"]["shape"] == [2, 8]


def test_explicit_version_2_document_is_accepted():
    s = NetworkSpec.model_validate({**V1, "version": 2})
    assert s.version == 2
    assert validate_network(s).ok


def test_new_params_default_sensibly_for_v1_layers():
    p = LayerParams()                       # what a v1 layer's params deserialize to
    assert p.dropout_mode == "train_only"
    assert p.momentum == 0.1 and p.eps == 1e-5 and p.affine is True
    assert p.input_shape is None
    assert p.task == "auto"
    assert p.padding_mode == "valid"


def test_layer_params_rejects_an_unknown_kind():
    """An unsupported layer kind must not silently validate."""
    with pytest.raises(Exception):
        NetworkSpec.model_validate({
            "name": "bad", "layers": [{"id": "l", "kind": "lstm", "params": {}}], "edges": []})

