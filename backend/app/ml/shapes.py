"""Shared tensor-shape arithmetic and the layer catalogue.

Shape inference lives here (not inside the validator) so validation, model
building, the forward-trace simulator and the parameter-count display all use
*one* implementation and can never disagree.

Two tensor ranks are supported, matching what the engine can actually run:

* **vector** — rank 1, ``[N]``. What the original v1 engine used.
* **image**  — rank 3, ``[C, H, W]`` (PyTorch/NCHW order internally).

Internally everything is ``[C, H, W]``; the UI and docs display ``H × W × C``
because that is the convention people read in papers. :meth:`TensorShape.describe`
does the reordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ..models.schemas import LayerParams

TensorRank = Literal["vector", "image"]


class ShapeError(ValueError):
    """Raised when a layer's parameters cannot produce a valid output shape."""


@dataclass(frozen=True)
class TensorShape:
    """An immutable, hashable tensor shape of rank 1 or 3."""

    rank: TensorRank
    dims: tuple[int, int, int]  # vector: (N, 1, 1); image: (C, H, W)

    # ---- constructors ---------------------------------------------------
    @staticmethod
    def vector(n: int) -> "TensorShape":
        if n < 1:
            raise ShapeError(f"A vector shape needs at least 1 element, got {n}.")
        return TensorShape("vector", (int(n), 1, 1))

    @staticmethod
    def image(c: int, h: int, w: int) -> "TensorShape":
        if min(c, h, w) < 1:
            raise ShapeError(f"An image shape needs positive C, H and W, got C={c}, H={h}, W={w}.")
        return TensorShape("image", (int(c), int(h), int(w)))

    @staticmethod
    def from_hwc(h: int, w: int, c: int) -> "TensorShape":
        """Build an image shape from the channels-*last* [H, W, C] UI order."""
        return TensorShape.image(c, h, w)

    # ---- queries --------------------------------------------------------
    @property
    def is_image(self) -> bool:
        return self.rank == "image"

    @property
    def channels(self) -> int:
        return self.dims[0]

    @property
    def height(self) -> int:
        return self.dims[1]

    @property
    def width(self) -> int:
        return self.dims[2]

    @property
    def numel(self) -> int:
        """Total number of scalars — this is what v1 called ``features``."""
        c, h, w = self.dims
        return c * h * w

    def describe(self) -> str:
        """Human/UI string: ``784`` for a vector, ``28 × 28 × 1`` (H×W×C) for images."""
        if not self.is_image:
            return str(self.numel)
        c, h, w = self.dims
        return f"{h} × {w} × {c}"

    def as_list(self) -> list[int]:
        """Shape list for the API/UI: ``[N]`` or ``[H, W, C]`` (channels last)."""
        if not self.is_image:
            return [self.numel]
        c, h, w = self.dims
        return [h, w, c]

    def torch_shape(self) -> tuple[int, int, int]:
        """Shape of a single sample as PyTorch sees it (NCHW, no batch dim)."""
        return self.dims

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.describe()


# ---------------------------------------------------------------------------
# Convolution / pooling output-size arithmetic
# ---------------------------------------------------------------------------

def conv_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    """Output extent for one spatial dimension of a convolution.

    ``out = floor((in + 2*pad - kernel) / stride) + 1``
    """
    return (size + 2 * padding - kernel) // stride + 1


def pool_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    """Output extent for one spatial dimension of a pooling window.

    Uses the same floor formula as convolution, floored at 1 so a legal but
    aggressive configuration still yields a usable (if tiny) feature map.
    """
    return max(1, (size + 2 * padding - kernel) // stride + 1)


def effective_conv_padding(p: LayerParams, kernel: int) -> int:
    """Resolve ``padding_mode`` into a concrete per-side pad.

    ``same`` requires stride 1 and pads by ``kernel // 2``; that constraint is
    enforced by the validator, so here it is a pure lookup.
    """
    if p.padding_mode == "same":
        return kernel // 2
    return int(p.padding)


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

def dense_params(in_features: int, neurons: int, use_bias: bool) -> int:
    """``inputs × neurons (+ neurons)``"""
    return in_features * neurons + (neurons if use_bias else 0)


# ---------------------------------------------------------------------------
# Per-kind shape resolution
# ---------------------------------------------------------------------------

def input_shape(p: LayerParams) -> TensorShape:
    """Resolve the input layer's declared shape.

    ``input_shape`` (channels-last ``[H, W, C]``) wins when present; otherwise
    the flat ``features`` count is used, giving a rank-1 vector as in v1.
    """
    if p.input_shape:
        if len(p.input_shape) != 3:
            raise ShapeError("Input shape must be [height, width, channels].")
        h, w, c = (int(v) for v in p.input_shape)
        shape = TensorShape.from_hwc(h, w, c)
        if p.features and p.features != shape.numel:
            raise ShapeError(
                f"Input shape {shape.describe()} contains {shape.numel} values but "
                f"'features' is set to {p.features}. Keep the two consistent."
            )
        return shape
    if not p.features or p.features < 1:
        raise ShapeError("Input layer needs 'features' >= 1.")
    return TensorShape.vector(int(p.features))


def conv_output_shape(inp: TensorShape, p: LayerParams) -> TensorShape:
    if not inp.is_image:
        raise ShapeError(
            f"Conv2D expects a 3D image tensor, but the incoming shape is a "
            f"{inp.numel}-element vector."
        )
    k = int(p.kernel_size)
    pad = effective_conv_padding(p, k)
    h = conv_output_size(inp.height, k, int(p.stride), pad)
    w = conv_output_size(inp.width, k, int(p.stride), pad)
    if h < 1 or w < 1:
        raise ShapeError(
            f"A {k}×{k} kernel with padding {pad} and stride {p.stride} does not fit "
            f"a {inp.height}×{inp.width} feature map (result would be {h}×{w})."
        )
    return TensorShape.image(int(p.filters), h, w)


def pool_output_shape(inp: TensorShape, p: LayerParams) -> TensorShape:
    if not inp.is_image:
        raise ShapeError(
            f"Pooling expects a 3D image tensor, but the incoming shape is a "
            f"{inp.numel}-element vector."
        )
    k = int(p.pool_size)
    h = pool_output_size(inp.height, k, int(p.pool_stride), int(p.pool_padding))
    w = pool_output_size(inp.width, k, int(p.pool_stride), int(p.pool_padding))
    return TensorShape.image(inp.channels, h, w)


def global_pool_output_shape(inp: TensorShape) -> TensorShape:
    if not inp.is_image:
        raise ShapeError(
            f"GlobalAveragePooling2D expects a 3D image tensor, but the incoming "
            f"shape is a {inp.numel}-element vector."
        )
    return TensorShape.vector(inp.channels)


def flatten_output_shape(inp: TensorShape) -> TensorShape:
    if not inp.is_image:
        raise ShapeError(
            f"Flatten expects a 3D image tensor, but the incoming shape is already "
            f"a {inp.numel}-element vector."
        )
    return TensorShape.vector(inp.numel)


def param_count(kind: str, inp: TensorShape, p: LayerParams) -> int:
    """Trainable parameter count for one layer, given its resolved input shape."""
    if kind in ("dense", "output"):
        if not p.neurons or p.neurons < 1:
            raise ShapeError("needs 'neurons' >= 1")
        return dense_params(inp.numel, int(p.neurons), p.use_bias)
    if kind == "conv2d":
        return conv_params(inp.channels, int(p.kernel_size), int(p.filters), p.use_bias)
    if kind == "batchnorm":
        return batchnorm_params(inp.channels, p.affine)
    return 0


def output_shape(kind: str, inp: Optional[TensorShape], p: LayerParams) -> TensorShape:
    """Resolve a layer's output shape.

    ``activation``, ``dropout`` and ``batchnorm`` preserve the incoming shape.
    """
    if kind == "input":
        return input_shape(p)
    if inp is None:  # pragma: no cover - guarded by the validator
        raise ShapeError("no input shape resolved for this layer")
    if kind in ("dense", "output"):
        if not p.neurons or p.neurons < 1:
            raise ShapeError("needs 'neurons' >= 1")
        return TensorShape.vector(int(p.neurons))
    if kind == "flatten":
        return flatten_output_shape(inp)
    if kind == "globalavgpool":
        return global_pool_output_shape(inp)
    if kind == "conv2d":
        return conv_output_shape(inp, p)
    if kind in ("maxpool", "avgpool"):
        return pool_output_shape(inp, p)
    # activation / dropout / batchnorm preserve the incoming shape
    return inp


def layer_note(kind: str, inp: Optional[TensorShape], p: LayerParams, out: TensorShape) -> Optional[str]:
    """Short human annotation shown on the node and the configuration panel —
    the maths used to derive this layer's output shape."""
    if kind == "conv2d" and inp is not None:
        k = int(p.kernel_size)
        pad = effective_conv_padding(p, k)
        bias = " + 1" if p.use_bias else ""
        return (f"out = ⌊({inp.height} + 2·{pad} − {k}) / {p.stride}⌋ + 1 = {out.height}  ·  "
                f"params = ({k}×{k}×{inp.channels}{bias}) × {p.filters} = "
                f"{param_count(kind, inp, p):,}")
    if kind == "flatten" and inp is not None:
        return f"{inp.describe()} → {out.numel:,} values"
    if kind == "globalavgpool" and inp is not None:
        return (f"mean over the {inp.height}×{inp.width} positions of each of "
                f"{inp.channels} channel(s) → {out.numel} values")
    if kind in ("maxpool", "avgpool") and inp is not None:
        k = int(p.pool_size)
        op = "max" if kind == "maxpool" else "mean"
        return (f"out = ⌊({inp.height} + 2·{p.pool_padding} − {k}) / {p.pool_stride}⌋ + 1 = {out.height}"
                f"  ·  {op} over each {k}×{k} window, stride {p.pool_stride}")
    if kind == "batchnorm" and inp is not None:
        return (f"per channel: x̂ = (x − μ)/√(σ² + {p.eps:g}),  y = γx̂ + β  ·  "
                f"momentum {p.momentum:g}  ·  {param_count(kind, inp, p):,} params")
    if kind == "activation":
        return f"a = {p.activation.replace('custom::', 'custom ')}(z)  ·  shape is preserved"
    if kind in ("dense", "output") and inp is not None:
        bias = f" + {p.neurons}" if p.use_bias else ""
        return (f"z = Wx + b,  a = {p.activation.replace('custom::', 'custom ')}(z)  ·  "
                f"({inp.numel:,} × {p.neurons}{bias}) = {param_count(kind, inp, p):,} params")
    if kind == "dropout":
        return (f"p = {p.dropout_rate}  ·  " +
                ("always applied" if p.dropout_mode == "always" else "training mode only"))
    return None


def conv_params(in_channels: int, kernel: int, filters: int, use_bias: bool) -> int:
    """``(k_h × k_w × C_in + bias) × filters`` — the spec's formula."""
    per_filter = kernel * kernel * in_channels + (1 if use_bias else 0)
    return per_filter * filters


def batchnorm_params(channels: int, affine: bool) -> int:
    """Learnable scale γ and shift β — 2 per channel when affine."""
    return 2 * channels if affine else 0


# ---------------------------------------------------------------------------
# The layer catalogue — the single source of truth for the palette
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerInfo:
    """Palette metadata for one layer kind."""
    kind: str
    name: str
    category: str            # BASIC | CNN | ADVANCED
    icon: str
    description: str         # one line, shown on the palette card
    formula: str             # the maths, shown in the tooltip
    supported: bool
    coming_soon_note: str = ""
    summary: str = ""        # parameter count, or a short note


LAYER_CATALOG: tuple[LayerInfo, ...] = (
    LayerInfo(
        "input", "Input", "BASIC", "⬇",
        "Entry point — declares the number of dataset features, or an image shape.",
        "x ∈ ℝᴺ   (or x ∈ ℝ^(C×H×W) for image data)",
        True, summary="no parameters",
    ),
    LayerInfo(
        "dense", "Dense", "BASIC", "⬡",
        "Fully connected layer — every input connects to every neuron.",
        "z = Wx + b,   a = f(z)", True,
        summary="(inputs × neurons) + neurons",
    ),
    LayerInfo(
        "activation", "Activation", "BASIC", "ƒ",
        "Applies a non-linear mathematical function element-wise.",
        "a = f(z)", True, summary="no parameters",
    ),
    LayerInfo(
        "dropout", "Dropout", "BASIC", "✂",
        "Randomly disables activations during training (regularization).",
        "aᵢ = 0 with probability p, else aᵢ / (1 − p)", True,
        summary="no parameters",
    ),
    LayerInfo(
        "batchnorm", "BatchNorm", "BASIC", "⊞",
        "Normalizes intermediate activations, then rescales and shifts them.",
        "x̂ = (x − μ) / √(σ² + ε),   y = γx̂ + β", True,
        summary="2 parameters per channel",
    ),
    LayerInfo(
        "flatten", "Flatten", "BASIC", "▤",
        "Converts a multi-dimensional feature map into a flat vector.",
        "28 × 28 × 1  →  784", True, summary="no parameters",
    ),
    LayerInfo(
        "output", "Output", "BASIC", "⬆",
        "Final layer — neurons must match the task (classes / 1 real value).",
        "ŷ = softmax(Wx + b)", True, summary="(inputs × neurons) + neurons",
    ),
    LayerInfo(
        "conv2d", "Conv2D", "CNN", "▦",
        "Learns spatial features by sliding learned filters over the input.",
        "y = f( Σ_f w_f ⊛ x + b )", True,
        summary="(k × k × C_in + bias) × filters",
    ),
    LayerInfo(
        "maxpool", "MaxPooling2D", "CNN", "▩",
        "Downsamples feature maps by keeping the maximum of each window.",
        "out = max(window)", True, summary="no parameters",
    ),
    LayerInfo(
        "avgpool", "AveragePooling2D", "CNN", "▨",
        "Downsamples feature maps by averaging each window.",
        "out = mean(window)", True, summary="no parameters",
    ),
    LayerInfo(
        "globalavgpool", "GlobalAveragePooling2D", "CNN", "▧",
        "Reduces each feature map to one value by averaging all its positions.",
        "out_c = (1 / H·W) Σ x_c", True, summary="no parameters",
    ),
    # ---- ADVANCED: declared, but not trainable end-to-end yet ------------
    LayerInfo(
        "embedding", "Embedding", "ADVANCED", "❑",
        "Maps integer indices to dense vectors (text / categorical input).",
        "e_i = W[i]", False,
        coming_soon_note=(
            "Needs an integer-token input pipeline and sequence-shaped data. "
            "It stays disabled so you cannot build a network that cannot train."
        ),
    ),
    LayerInfo(
        "lstm", "LSTM", "ADVANCED", "↻",
        "Recurrent layer with gated memory, for sequence and time-series data.",
        "h_t, c_t = LSTM(x_t, h_prev, c_prev)", False,
        coming_soon_note=(
            "Needs a recurrent data pipeline (variable-length sequences) and a "
            "sequence-aware trainer. Not implemented yet."
        ),
    ),
    LayerInfo(
        "gru", "GRU", "ADVANCED", "⇄",
        "Simplified recurrent layer with fewer gates than LSTM.",
        "h_t = GRU(x_t, h_prev)", False,
        coming_soon_note=(
            "Needs a recurrent data pipeline and a sequence-aware trainer. "
            "Not implemented yet."
        ),
    ),
)

CATALOG_BY_KIND: dict[str, LayerInfo] = {i.kind: i for i in LAYER_CATALOG}
SUPPORTED_KINDS: tuple[str, ...] = tuple(i.kind for i in LAYER_CATALOG if i.supported)
UNSUPPORTED_KINDS: tuple[str, ...] = tuple(i.kind for i in LAYER_CATALOG if not i.supported)
CATEGORY_ORDER: tuple[str, ...] = ("BASIC", "CNN", "ADVANCED")


def catalog_payload() -> list[dict[str, Any]]:
    """Palette payload for ``GET /api/networks/layers`` (grouped by category)."""
    return [
        {
            "category": cat,
            "layers": [
                {
                    "kind": i.kind, "name": i.name, "icon": i.icon,
                    "description": i.description, "formula": i.formula,
                    "supported": i.supported,
                    "coming_soon_note": i.coming_soon_note,
                    "summary": i.summary,
                }
                for i in LAYER_CATALOG if i.category == cat
            ],
        }
        for cat in CATEGORY_ORDER
    ]


def default_params(kind: str) -> dict[str, Any]:
    """Sensible starting parameters per kind - mirrors the v1 frontend defaults."""
    if kind == "input":
        return {"features": 2}
    if kind == "dense":
        return {"neurons": 8, "activation": "relu", "init": "he"}
    if kind == "activation":
        return {"activation": "relu"}
    if kind == "output":
        return {"neurons": 2, "activation": "softmax"}
    if kind == "dropout":
        return {"dropout_rate": 0.5}
    if kind == "batchnorm":
        return {"momentum": 0.1, "eps": 1e-5, "affine": True}
    if kind == "conv2d":
        return {"filters": 8, "kernel_size": 3, "stride": 1,
                "padding": 0, "padding_mode": "valid", "activation": "relu"}
    if kind in ("maxpool", "avgpool"):
        return {"pool_size": 2, "pool_stride": 2, "pool_padding": 0}
    return {}
