"""Network validation/building + ready-made educational presets."""
from __future__ import annotations

from fastapi import APIRouter

from ..ml import builder
from ..ml.shapes import catalog_payload
from ..models.schemas import NetworkSpec

router = APIRouter()


@router.get("/layers")
def layers():
    """The layer palette: every kind, grouped by category.

    The frontend renders its palette from this, so a layer can never appear in
    the UI without a backend definition. Unsupported kinds come back with
    ``supported: false`` and are rendered disabled ("Coming Soon").
    """
    return {"categories": catalog_payload()}


@router.post("/validate")
def validate(spec: NetworkSpec):
    """Validate the architecture and return per-layer shapes/params + errors."""
    return builder.validate_network(spec)


@router.post("/build")
def build(spec: NetworkSpec):
    """Validate **and** actually instantiate the PyTorch model (catches runtime issues)."""
    val = builder.validate_network(spec)
    if not val.ok:
        return val
    try:
        model = builder.build_model(spec, val, seed=0)
    except Exception as exc:
        val.ok = False
        val.errors.append({"code": "BUILD_FAILED", "message": str(exc), "layer_id": None,
                           "suggestion": "Check activations and layer parameters."})
        return val
    n_tensors = sum(1 for _ in model.parameters())
    out = val.model_dump()
    out["parameter_tensors"] = n_tensors
    return out


def _layer(lid, kind, x, y, **params):
    return {"id": lid, "kind": kind, "position": {"x": x, "y": y}, "params": params}


def _edges(ids):
    # `ids[1:]` is deliberately one element shorter, so this is the one case where
    # `strict=True` would be wrong — it would raise on every call. The offset is
    # intentional: consecutive pairs form the chain. Not switching to
    # itertools.pairwise here keeps the intent obvious at the call site.
    return [{"id": f"e-{a}-{b}", "source": a, "target": b} for a, b in zip(ids, ids[1:])]


@router.get("/presets")
def presets():
    """One-click demo configurations used by the frontend 'Try Example' button."""
    xor_net = NetworkSpec(
        name="XOR Demo Network",
        layers=[
            _layer("input", "input", 0, 140, features=2),
            _layer("hidden1", "dense", 220, 60, neurons=8, activation="tanh", init="xavier"),
            _layer("hidden2", "dense", 440, 220, neurons=8, activation="tanh", init="xavier"),
            _layer("output", "output", 660, 140, neurons=2, activation="softmax"),
        ],
        edges=_edges(["input", "hidden1", "hidden2", "output"]),
    )
    iris_net = NetworkSpec(
        name="Iris Classifier 4-16-8-3",
        layers=[
            _layer("input", "input", 0, 140, features=4),
            _layer("hidden1", "dense", 220, 60, neurons=16, activation="relu", init="he"),
            _layer("hidden2", "dense", 440, 220, neurons=8, activation="relu", init="he"),
            _layer("output", "output", 660, 140, neurons=3, activation="softmax"),
        ],
        edges=_edges(["input", "hidden1", "hidden2", "output"]),
    )
    cnn_net = NetworkSpec(
        name="CNN Classifier 8x8-16-3",
        layers=[
            _layer("input", "input", 0, 140, features=64, input_shape=[8, 8, 1]),
            _layer("norm1", "batchnorm", 170, 60, momentum=0.1, eps=1e-5, affine=True),
            _layer("conv1", "conv2d", 350, 40, filters=8, kernel_size=3, stride=1,
                   padding=0, padding_mode="valid", activation="relu", init="he"),
            _layer("norm2", "batchnorm", 540, 40, momentum=0.1, eps=1e-5, affine=True),
            _layer("act1", "activation", 720, 40, activation="relu"),
            _layer("pool1", "maxpool", 890, 40, pool_size=2, pool_stride=2),
            _layer("pool2", "avgpool", 1060, 40, pool_size=2, pool_stride=2),
            _layer("flat", "flatten", 1230, 140),
            _layer("hidden", "dense", 1400, 60, neurons=16, activation="relu", init="he"),
            _layer("drop", "dropout", 1560, 220, dropout_rate=0.3),
            _layer("output", "output", 1720, 140, neurons=3, activation="softmax"),
        ],
        edges=_edges(["input", "norm1", "conv1", "norm2", "act1", "pool1",
                      "pool2", "flat", "hidden", "drop", "output"]),
    )
    return {
        "presets": [
            {
                "id": "xor-demo", "name": "XOR Demo (Try Example)",
                "description": "2 → 8 → 8 → 2 network on the XOR dataset. The classic demo: "
                               "a hidden layer is required because XOR is not linearly separable.",
                "network": xor_net.model_dump(),
                "dataset": {"kind": "builtin", "name": "xor",
                            "preprocessing": {"scale": "none", "test_split": 0.2, "val_split": 0.2, "seed": 42}},
                "config": {"epochs": 300, "batch_size": 16, "learning_rate": 0.05,
                           "optimizer": "adam", "loss": "cross_entropy", "seed": 42,
                           "snapshot_every": 15},
            },
            {
                "id": "iris-research", "name": "Iris Research Experiment",
                "description": "The sample research experiment from the project spec: 4 → 16 → 8 → 3, "
                               "ReLU + softmax, Adam lr=0.001.",
                "network": iris_net.model_dump(),
                "dataset": {"kind": "builtin", "name": "iris",
                            "preprocessing": {"scale": "standard", "test_split": 0.2, "val_split": 0.2, "seed": 7}},
                "config": {"epochs": 200, "batch_size": 16, "learning_rate": 0.001,
                           "optimizer": "adam", "loss": "cross_entropy", "seed": 7,
                           "snapshot_every": 10},
                "experiment_name": "iris-baseline",
            },
            {
                "id": "cnn-shapes", "name": "CNN Demo (Conv + Pool + BatchNorm)",
                "description": "A real convolutional network on 8×8 synthetic images: "
                               "8×8×1 → Conv2D(8 filters, 3×3) → 6×6×8 → MaxPool → 3×3×8 → "
                               "AvgPool → 1×1×8 → Flatten → 8 → Dense 16 → 3. Exercises every "
                               "newly supported layer kind end to end.",
                "network": cnn_net.model_dump(),
                "dataset": {"kind": "builtin", "name": "shapes8",
                            "preprocessing": {"scale": "none", "test_split": 0.2, "val_split": 0.2, "seed": 42}},
                "config": {"epochs": 60, "batch_size": 32, "learning_rate": 0.01,
                           "optimizer": "adam", "loss": "cross_entropy", "seed": 42,
                           "snapshot_every": 10},
                "experiment_name": "cnn-baseline",
            },
        ]
    }
