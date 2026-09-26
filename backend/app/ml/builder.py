"""Network specification validation, shape inference and model building.

The engine supports *feed-forward chains* (one input, one output, every
intermediate node in-degree/out-degree 1). This keeps shape inference exact and
the mathematics explainable. Branching/merging (residuals, concatenation) is a
documented extension point — see docs/ARCHITECTURE.md.

Layer kinds and their shape rules are declared once in :mod:`.shapes`; this
module turns those rules into actionable validation errors, reports the
per-layer shape/parameter table, and instantiates the real PyTorch model.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from . import shapes as shp
from .activations import make_activation
from .shapes import ShapeError, TensorShape, param_count
from ..models.schemas import LayerShape, NetworkSpec, ValidationResult


def _err(code: str, message: str, layer_id: str | None = None, suggestion: str | None = None) -> dict:
    return {"code": code, "message": message, "layer_id": layer_id, "suggestion": suggestion}


def _warn(code: str, message: str, layer_id: str | None = None, suggestion: str | None = None) -> dict:
    return _err(code, message, layer_id, suggestion)


def validate_network(spec: NetworkSpec) -> ValidationResult:
    errors: list[dict] = []
    warnings: list[dict] = []
    layers = {l.id: l for l in spec.layers}

    if not spec.layers:
        errors.append(_err("EMPTY", "The network has no layers.",
                           suggestion="Drag an Input, at least one Dense and an Output layer onto the canvas."))
        return ValidationResult(ok=False, errors=errors)

    ids = [l.id for l in spec.layers]
    if len(ids) != len(set(ids)):
        errors.append(_err("DUP_ID", "Duplicate layer ids found."))

    inputs = [l for l in spec.layers if l.kind == "input"]
    outputs = [l for l in spec.layers if l.kind == "output"]
    if len(inputs) == 0:
        errors.append(_err("NO_INPUT", "Missing input layer.",
                           suggestion="Add an Input layer and set its feature count to match the dataset."))
    if len(outputs) == 0:
        errors.append(_err("NO_OUTPUT", "Missing output layer.",
                           suggestion="Add an Output layer whose neuron count matches the task "
                                      "(1 for regression/binary, n_classes for multi-class)."))
    if len(inputs) > 1:
        errors.append(_err("MULTI_INPUT", "Multiple input layers are not supported in v1.",
                           suggestion="Use a single Input layer with the total number of features."))
    if len(outputs) > 1:
        errors.append(_err("MULTI_OUTPUT", "Multiple output layers are not supported in v1."))

    # referencing edges
    srcs: dict[str, list[str]] = defaultdict(list)
    tgts: dict[str, list[str]] = defaultdict(list)
    for e in spec.edges:
        if e.source not in layers or e.target not in layers:
            errors.append(_err("BAD_EDGE", f"Edge {e.id} references a layer that does not exist."))
            continue
        srcs[e.source].append(e.target)
        tgts[e.target].append(e.source)

    # structural chain checks
    for l in spec.layers:
        if l.kind == "input" and tgts.get(l.id):
            errors.append(_err("EDGE_INTO_INPUT", "The input layer cannot have incoming connections.", l.id))
        if l.kind == "output" and srcs.get(l.id):
            errors.append(_err("EDGE_FROM_OUTPUT", "The output layer cannot have outgoing connections.", l.id))
        if l.kind not in ("input",) and len(tgts.get(l.id, [])) > 1:
            errors.append(_err("BRANCH_IN", f"'{l.label or l.id}' receives {len(tgts[l.id])} inputs. "
                                             "Branching/merging is not supported in v1.", l.id,
                               suggestion="Keep the network as a single chain: Input → Dense → … → Output."))
        if l.kind not in ("output",) and len(srcs.get(l.id, [])) > 1:
            errors.append(_err("BRANCH_OUT", f"'{l.label or l.id}' feeds {len(srcs[l.id])} layers. "
                                             "Branching is not supported in v1.", l.id))
    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # connectivity
    connected_targets = {e.target for e in spec.edges}
    for l in spec.layers:
        if l.kind == "input":
            continue
        if l.id not in connected_targets:
            errors.append(_err("DISCONNECTED", f"'{l.label or l.id}' ({l.kind}) is not connected from any layer.",
                               l.id, "Connect the layers in order so data can flow Input → Output."))
    for l in inputs + [x for x in spec.layers if x.kind not in ("input", "output")]:
        if not srcs.get(l.id) and l.kind != "output":
            errors.append(_err("DEAD_END", f"'{l.label or l.id}' ({l.kind}) does not connect to the next layer.",
                               l.id))
    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # topological walk — detect cycles
    order: list[str] = []
    cur = inputs[0].id if inputs else None
    visited: set[str] = set()
    while cur:
        if cur in visited:
            errors.append(_err("CYCLE", "A cycle was detected — feed-forward networks must be acyclic.", cur))
            break
        visited.add(cur)
        order.append(cur)
        nxt = srcs.get(cur, [])
        cur = nxt[0] if nxt else None

    if not errors and outputs and outputs[0].id not in visited:
        errors.append(_err("NO_PATH", "No path from the input layer to the output layer."))
    if not errors and len(visited) != len(spec.layers):
        unreachable = [lid for lid in layers if lid not in visited]
        errors.append(_err("UNREACHABLE_OR_CYCLE",
                           f"Layers {unreachable} are not on the path from input to output "
                           "(unreachable or inside a cycle).",
                           unreachable[0], "Remove or reconnect these layers into the main chain."))
    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # ---- per-layer parameter sanity (before shape inference) -------------
    for l in spec.layers:
        p = l.params
        name = l.label or l.id
        if l.kind == "dropout" and not (0.0 < p.dropout_rate < 1.0):
            errors.append(_err("BAD_DROPOUT", f"'{name}': dropout rate must be in (0, 1).", l.id))
        if l.kind == "batchnorm":
            if not (0.0 <= p.momentum < 1.0):
                errors.append(_err("BAD_MOMENTUM",
                                   f"'{name}': BatchNorm momentum must be in [0, 1).", l.id,
                                   "0.1 is the PyTorch default; values close to 1 make the running "
                                   "statistics adapt very slowly."))
            if p.eps <= 0:
                errors.append(_err("BAD_EPS", f"'{name}': BatchNorm epsilon must be > 0.", l.id,
                                   "1e-5 is the standard value; ε stops division by zero when the "
                                   "variance is tiny."))
        if l.kind == "conv2d":
            if p.kernel_size < 1 or p.kernel_size > 15:
                errors.append(_err("BAD_KERNEL", f"'{name}': kernel size must be between 1 and 15.", l.id))
            if p.filters < 1 or p.filters > 512:
                errors.append(_err("BAD_FILTERS", f"'{name}': filters must be between 1 and 512.", l.id))
            if p.stride < 1:
                errors.append(_err("BAD_STRIDE", f"'{name}': stride must be >= 1.", l.id))
            if p.padding_mode == "same" and p.stride != 1:
                errors.append(_err("BAD_PADDING",
                                   f"'{name}': 'same' padding requires stride 1, but stride is {p.stride}.",
                                   l.id, "Either set the stride to 1 or switch padding to 'valid'."))
            if p.padding_mode == "valid" and p.padding < 0:
                errors.append(_err("BAD_PADDING", f"'{name}': padding cannot be negative.", l.id))
        if l.kind in ("maxpool", "avgpool"):
            if p.pool_size < 1:
                errors.append(_err("BAD_POOL", f"'{name}': pool size must be >= 1.", l.id))
            if p.pool_stride < 1:
                errors.append(_err("BAD_POOL", f"'{name}': pool stride must be >= 1.", l.id))
            if p.pool_padding < 0:
                errors.append(_err("BAD_POOL", f"'{name}': pool padding cannot be negative.", l.id))
            if p.pool_size > p.pool_stride and p.pool_padding == 0:
                warnings.append(_warn("POOL_OVERLAP",
                                      f"'{name}': pool size ({p.pool_size}) exceeds the stride "
                                      f"({p.pool_stride}), so windows overlap and some values are "
                                      "counted more than once.", l.id,
                                      f"Set the stride to {p.pool_size} for non-overlapping windows."))
    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # ---- parameter & shape inference -------------------------------------
    shapes: list[LayerShape] = []
    total_params = 0
    shape: TensorShape | None = None
    is_cnn = False
    for lid in order:
        layer = layers[lid]
        p = layer.params
        name = layer.label or lid
        in_shape = shape

        # -- rank compatibility --------------------------------------------
        if in_shape is not None:
            if in_shape.is_image:
                is_cnn = True
            if layer.kind in ("dense", "output") and in_shape.is_image:
                errors.append(_err(
                    "DENSE_NEEDS_VECTOR",
                    f"'{name}' ({layer.kind}) expects a flat feature vector but receives a "
                    f"{in_shape.describe()} image tensor.", lid,
                    "Insert a Flatten layer (or GlobalAveragePooling2D) before this Dense layer."))
                break
            if layer.kind in ("flatten", "globalavgpool") and not in_shape.is_image:
                errors.append(_err(
                    "NEEDS_IMAGE",
                    f"'{name}' ({layer.kind}) expects a 3-D image tensor but receives a "
                    f"{in_shape.numel}-element vector.", lid,
                    "Remove this layer, or give the Input layer an image shape "
                    "(height × width × channels) so the chain starts from a feature map."))
                break
            if layer.kind in ("conv2d", "maxpool", "avgpool") and not in_shape.is_image:
                errors.append(_err(
                    "CONV_NEEDS_IMAGE",
                    f"'{name}' ({layer.kind}) expects a 3D image tensor.\n\n"
                    f"Current input shape: {in_shape.numel}", lid,
                    "Set an image input shape (height × width × channels) on the Input layer — "
                    "e.g. 28 × 28 × 1 for MNIST-style data."))
                break
            if layer.kind == "batchnorm" and in_shape.is_image \
                    and in_shape.height * in_shape.width < 2:
                errors.append(_err(
                    "BATCHNORM_TOO_SMALL",
                    f"'{name}': BatchNorm over a {in_shape.describe()} tensor has a single spatial "
                    "position, so the variance cannot be estimated.", lid,
                    "Increase the spatial size or add a layer that expands it."))
                break

        # -- per-kind parameter validation ----------------------------------
        if layer.kind in ("dense", "output") and (not p.neurons or p.neurons < 1):
            errors.append(_err("BAD_NEURONS", f"'{name}' needs 'neurons' >= 1.", lid))
            break
        if layer.kind == "conv2d" and p.padding_mode == "valid" and p.padding >= p.kernel_size:
            errors.append(_err("BAD_PADDING",
                               f"'{name}': padding ({p.padding}) must be smaller than the kernel "
                               f"size ({p.kernel_size}).", lid))
            break

        # -- activation validity -------------------------------------------
        if layer.kind in ("dense", "output", "activation", "conv2d"):
            try:
                make_activation(p.activation, spec.custom_activations)
            except Exception as exc:
                errors.append(_err("BAD_ACTIVATION", f"Invalid activation on '{name}': {exc}", lid,
                                   "Choose a builtin activation, or fix the custom formula in the "
                                   "Activation Lab (it is checked by a safe expression parser — "
                                   "no user code is ever executed)."))
                break
            if p.activation == "softmax" and layer.kind != "output":
                warnings.append(_warn("SOFTMAX_HIDDEN",
                                      f"Softmax on layer '{name}' is unusual; it is normally used "
                                      "on the output.", lid))

        # -- shape resolution ----------------------------------------------
        try:
            out = shp.output_shape(layer.kind, in_shape, p)
        except ShapeError as exc:
            msg = str(exc)
            code = "BAD_FEATURES" if layer.kind == "input" else (
                "CONV_NEEDS_IMAGE" if "expects a 3D image tensor" in msg else "SHAPE_MISMATCH")
            errors.append(_err(code, f"'{name}': {msg}", lid,
                               "Set it to the number of dataset feature columns."
                               if layer.kind == "input" else
                               "Adjust the previous layer's output size or this layer's parameters."))
            break

        try:
            cnt = 0 if layer.kind == "input" else param_count(layer.kind, in_shape, p)
        except ShapeError as exc:
            errors.append(_err("BAD_PARAMS", f"'{name}': {exc}", lid))
            break
        total_params += cnt

        shapes.append(LayerShape(
            id=lid, kind=layer.kind,
            in_features=in_shape.numel if in_shape else None,
            out_features=out.numel,
            params=cnt,
            in_shape=in_shape.as_list() if in_shape else None,
            out_shape=out.as_list(),
            note=shp.layer_note(layer.kind, in_shape, p, out),
        ))
        shape = out

    if not errors:
        hidden = sum(1 for s in shapes if s.kind in ("dense", "conv2d"))
        if hidden == 0:
            warnings.append(_warn("SHALLOW",
                                  "The network has no hidden Dense or Conv2D layers — it can only "
                                  "learn a linear function of the input.",
                                  suggestion="Add at least one hidden layer for non-linear problems "
                                             "like XOR."))
        if total_params > 2_000_000:
            warnings.append(_warn("HUGE", f"Network has {total_params:,} parameters — training "
                                          "may be slow in this browser-based lab."))

    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        layers=shapes,
        input_dim=shapes[0].out_features if shapes else None,
        output_dim=shapes[-1].out_features if shapes else None,
        total_params=total_params,
        order=order,
        is_cnn=is_cnn,
    )


class ChainModel(nn.Module):
    """A named feed-forward chain that keeps layer ids for inspection."""

    def __init__(self, modules: list[tuple[str, nn.Module]], linear_key: dict[str, str],
                 param_key: dict[str, str] | None = None,
                 shapes: dict[str, TensorShape] | None = None):
        super().__init__()
        self.layers = nn.ModuleDict(dict(modules))
        self.order = [k for k, _ in modules]
        self.linear_key = linear_key  # layer id -> module key of the Linear inside its block
        # layer id -> "weight-bearing" submodule path, for the Weights tab and
        # gradient statistics. Covers Linear, Conv2d and BatchNorm.
        self.param_key: dict[str, str] = dict(param_key or {})
        # layer id -> the shape this layer consumes / produces (per sample).
        self.shapes: dict[str, TensorShape] = dict(shapes or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for key in self.order:
            x = self.layers[key](x)
        return x

    def _resolve(self, dotted: str) -> nn.Module:
        """Look up a submodule by a layer-relative path such as ``h1.linear``.

        Paths in :attr:`param_key` start at the layer, so resolution begins in
        the ``layers`` ModuleDict.
        """
        obj: nn.Module = self.layers
        for part in dotted.split("."):
            obj = getattr(obj, part)  # type: ignore[assignment]
        return obj

    def primary_weight(self, layer_id: str) -> nn.Parameter | None:
        """The main trainable tensor of a layer (used for gradient examples)."""
        key = self.param_key.get(layer_id)
        if not key:
            return None
        mod = self._resolve(key)
        for attr in ("weight", "gamma"):
            t = getattr(mod, attr, None)
            if isinstance(t, nn.Parameter):
                return t
        return None


class Reshape(nn.Module):
    """Reshape a flat feature row into an image tensor (channels-first).

    Lets a single ``Input`` layer feed convolutional layers: the dataset still
    delivers ``[N, H*W*C]`` rows, and this module turns each row into
    ``[C, H, W]`` so PyTorch's Conv2d can consume it.
    """

    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        self.shape = shape  # (C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 2:
            return x.reshape(-1, *self.shape)
        return x.reshape(*self.shape)  # pragma: no cover - single-sample path


class GlobalAvgPool(nn.Module):
    """Average every spatial position of each channel independently.

    ``nn.AdaptiveAvgPool2d(1)`` alone would leave a trailing singleton axis
    (``[N, C, 1, 1]``), which the next Dense layer cannot consume; the flatten
    here makes the output exactly ``[N, C]``, matching the shape the validator
    reports (``H × W × C → C``).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.mean(dim=(2, 3))
        return x  # pragma: no cover - already rank-2


class AlwaysDropout(nn.Module):
    """Dropout that also fires in eval mode (the "training mode" config option).

    ``nn.Dropout`` is a no-op during inference, which is the default and almost
    always what you want. This variant exists so the effect of dropout can be
    *observed* during simulation; it is opt-in per layer.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.dropout(x, p=self.p, training=True)


class ActivationLayer(nn.Module):
    """Standalone activation layer — the real, trainable-in-the-graph counterpart
    of a Dense layer's ``activation`` field.

    Semantics:

    * rank-1 input (a feature vector) — the function is applied over the
      features, i.e. ``dim=-1``.
    * rank-3 input (a feature map, ``[N, C, H, W]``) — every function is applied
      *element-wise*, except softmax, which is applied per channel over the
      spatial positions (``dim=1``). Applying softmax over the last axis of a
      feature map would normalise across neighbouring pixels, which is almost
      never what you want.
    """

    def __init__(self, act: nn.Module, name: str = "", is_image: bool = False):
        super().__init__()
        self.act = act
        self.activation_name = name
        self.is_image = is_image
        if is_image and isinstance(act, nn.Softmax):
            act.dim = 1  # per-channel distribution over H×W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)



def _init_linear(linear: nn.Linear, method: str, activation: str) -> None:
    act = activation.lower()
    if method == "xavier":
        gain = nn.init.calculate_gain("relu" if act in ("relu", "leaky_relu")
                                      else "tanh" if act == "tanh" else "linear")
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    elif method == "he":
        nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
    elif method == "normal":
        nn.init.normal_(linear.weight, mean=0.0, std=0.05)
    elif method == "uniform":
        nn.init.uniform_(linear.weight, -0.1, 0.1)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_conv(conv: nn.Conv2d, method: str, activation: str) -> None:
    """Initialise convolution filters.

    Uses the same four schemes as :func:`_init_linear`, but scaled to a filter's
    fan-in (``k × k × C_in``) rather than a matrix row.
    """
    act = activation.lower()
    if method == "xavier":
        gain = nn.init.calculate_gain("relu" if act in ("relu", "leaky_relu", "gelu")
                                      else "tanh" if act == "tanh" else "linear")
        nn.init.xavier_uniform_(conv.weight, gain=gain)
    elif method == "he":
        nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
    elif method == "normal":
        nn.init.normal_(conv.weight, mean=0.0, std=0.05)
    elif method == "uniform":
        nn.init.uniform_(conv.weight, -0.1, 0.1)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def build_model(spec: NetworkSpec, validation: ValidationResult | None = None, seed: int | None = None) -> ChainModel:
    """Build a deterministic (seeded) PyTorch model from a validated spec.

    Every supported palette layer maps to a real ``nn.Module``; there is no
    placeholder or no-op path. The per-layer shapes resolved during validation
    are carried on the model so the trainer and the simulator never have to
    re-derive them.
    """
    val = validation or validate_network(spec)
    if not val.ok:
        messages = "; ".join(e["message"] for e in val.errors)
        raise ValueError(f"Cannot build an invalid network: {messages}")
    if seed is not None:
        torch.manual_seed(seed)

    layers = {l.id: l for l in spec.layers}
    custom = spec.custom_activations or {}
    modules: list[tuple[str, nn.Module]] = []
    linear_key: dict[str, str] = {}
    param_key: dict[str, str] = {}
    shape_map: dict[str, TensorShape] = {}
    cur: TensorShape | None = None

    for lid in val.order:
        layer = layers[lid]
        p = layer.params
        kind = layer.kind

        if kind == "input":
            cur = shp.input_shape(p)
            shape_map[lid] = cur
            # Identity for the original flat-vector case; Reshape only when the
            # layer declares an image shape that the dataset must be folded into.
            modules.append((lid, Reshape(cur.torch_shape()) if cur.is_image else nn.Identity()))
            continue

        assert cur is not None, f"Layer '{lid}' ({kind}) has no input shape."
        in_shape = cur

        if kind in ("dense", "output"):
            n = int(p.neurons or 0)
            linear = nn.Linear(in_shape.numel, n, bias=p.use_bias)
            _init_linear(linear, p.init, p.activation)
            block = nn.Sequential()
            block.add_module("linear", linear)
            block.add_module("activation", make_activation(p.activation, custom))
            modules.append((lid, block))
            linear_key[lid] = f"{lid}.linear"
            param_key[lid] = f"{lid}.linear"

        elif kind == "activation":
            modules.append((lid, ActivationLayer(
                make_activation(p.activation, custom), p.activation, in_shape.is_image)))

        elif kind == "dropout":
            mod = (AlwaysDropout(p.dropout_rate) if p.dropout_mode == "always"
                   else nn.Dropout(p.dropout_rate))
            modules.append((lid, mod))

        elif kind == "batchnorm":
            bn_cls = nn.BatchNorm2d if in_shape.is_image else nn.BatchNorm1d
            channels = in_shape.channels if in_shape.is_image else in_shape.numel
            modules.append((lid, bn_cls(channels, momentum=p.momentum, eps=p.eps, affine=p.affine)))
            if p.affine:
                param_key[lid] = f"{lid}"

        elif kind == "flatten":
            modules.append((lid, nn.Flatten(start_dim=1)))

        elif kind == "conv2d":
            k = int(p.kernel_size)
            pad = shp.effective_conv_padding(p, k)
            conv = nn.Conv2d(in_shape.channels, int(p.filters), kernel_size=k,
                             stride=int(p.stride), padding=pad, bias=p.use_bias)
            _init_conv(conv, p.init, p.activation)
            block = nn.Sequential()
            block.add_module("conv", conv)
            block.add_module("activation", make_activation(p.activation, custom))
            modules.append((lid, block))
            param_key[lid] = f"{lid}.conv"

        elif kind == "maxpool":
            modules.append((lid, nn.MaxPool2d(kernel_size=int(p.pool_size),
                                               stride=int(p.pool_stride),
                                               padding=int(p.pool_padding))))
        elif kind == "avgpool":
            modules.append((lid, nn.AvgPool2d(kernel_size=int(p.pool_size),
                                               stride=int(p.pool_stride),
                                               padding=int(p.pool_padding))))
        elif kind == "globalavgpool":
            modules.append((lid, GlobalAvgPool()))
        else:  # pragma: no cover - validation rejects unknown kinds first
            raise ValueError(f"Layer '{lid}' has unsupported kind '{kind}'.")

        cur = shp.output_shape(kind, in_shape, p)
        shape_map[lid] = cur

    return ChainModel(modules, linear_key, param_key, shape_map)


def weight_matrices(model: ChainModel, max_elems: int = 4096) -> dict[str, dict]:
    """Extract per-layer parameters for the Weights tab.

    Three payload flavours, distinguished by ``kind``:

    * ``linear``    — ``shape: [out, in]``, a 2-D matrix (the original format).
    * ``conv``      — ``shape: [filters, k*k*C_in]``; each row is one filter
      flattened to a vector, so it renders in the same heat-map component. The
      original ``[filters, C_in, k, k]`` is kept in ``full_shape``.
    * ``batchnorm`` — ``shape: [2, channels]``: row 0 is the scale γ, row 1 the
      shift β, alongside the running statistics.

    Everything is downsampled by a stride when it would exceed ``max_elems`` so
    a large convolution never ships megabytes to the browser.
    """
    import numpy as np

    def _round(a):
        return np.round(a, 5).tolist()

    def _stats(a):
        return {
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
        }

    out: dict[str, dict] = {}
    for lid in model.order:
        mod = model.layers[lid]

        # ---- Conv2d ---------------------------------------------------
        conv = getattr(mod, "conv", None)
        if isinstance(conv, nn.Conv2d):
            W = conv.weight.detach().cpu().numpy()            # (F, C, k, k)
            F, C, kh, kw = W.shape
            flat = W.reshape(F, C * kh * kw)
            b = conv.bias.detach().cpu().numpy() if conv.bias is not None else None
            note = (f"Each row is one filter of shape {C}×{kh}×{kw} "
                    f"(flattened for display). Full tensor: {F}×{C}×{kh}×{kw}.")
            if flat.size > max_elems:
                step = int(np.ceil(np.sqrt(flat.size / max_elems)))
                flat = flat[::step, ::step]
                note += f" Showing every {step}th filter."
            out[lid] = {
                "kind": "conv",
                "shape": [int(flat.shape[0]), int(flat.shape[1])],
                "full_shape": [F, C, kh, kw],
                "weights": _round(flat),
                "bias": _round(b) if b is not None else None,
                "stats": _stats(W),
                "display_note": note,
            }
            continue

        # ---- BatchNorm ------------------------------------------------
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if mod.weight is None:      # affine=False → nothing learnable here
                continue
            g = mod.weight.detach().cpu().numpy()
            b = mod.bias.detach().cpu().numpy()
            rm = mod.running_mean.detach().cpu().numpy()
            rv = mod.running_var.detach().cpu().numpy()
            out[lid] = {
                "kind": "batchnorm",
                "shape": [2, int(g.size)],
                "weights": _round(np.stack([g, b])),
                "scale": _round(g),
                "shift": _round(b),
                "running_mean": _round(rm),
                "running_var": _round(rv),
                "stats": _stats(g),
                "display_note": ("Row 0 = learnable scale γ, row 1 = learnable shift β "
                                 f"(ε = {mod.eps:g}, momentum = {mod.momentum:g})."),
            }
            continue

        # ---- Linear ---------------------------------------------------
        if lid not in model.linear_key:
            continue
        linear = mod.linear  # type: ignore[attr-defined]
        W = linear.weight.detach().cpu().numpy()
        b = linear.bias.detach().cpu().numpy() if linear.bias is not None else None
        note = None
        Ws, bs = W, b
        if W.size > max_elems:
            # stride-based downsample for display only
            step = int(np.ceil(np.sqrt(W.size / max_elems)))
            Ws = W[::step, ::step]
            note = f"Downsampled by stride {step} for display (full matrix {W.shape[0]}×{W.shape[1]})."
            if bs is not None:
                bs = b[::step]
        out[lid] = {
            "kind": "linear",
            "shape": [int(W.shape[0]), int(W.shape[1])],
            "weights": _round(Ws),
            "bias": _round(bs) if bs is not None else None,
            "stats": _stats(W),
            "display_note": note,
        }
    return out
