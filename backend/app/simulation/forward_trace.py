"""Transparent forward/backward inspection.

These functions recompute the mathematics *manually* (numpy, layer by
layer) from the real learned parameters, so the UI can display every
weighted sum, bias addition and activation — no black box, no fake numbers.
"""
from __future__ import annotations

import numpy as np

from ..config import settings
from ..ml.activations import make_activation
from ..ml.builder import ChainModel
from ..models.schemas import NetworkSpec

CAP = settings.max_visual_neurons   # per-layer neurons shipped to the UI
CAP_IN = 12                          # incoming connections expanded per neuron


def _apply_activation(name: str, custom: dict[str, str], z: np.ndarray) -> np.ndarray:
    import torch

    mod = make_activation(name, custom, dim=-1)
    with torch.no_grad():
        t = torch.tensor(z, dtype=torch.float32)
        return mod(t).numpy()


# ---------------------------------------------------------------------------
# Helpers for the image-shaped layers
# ---------------------------------------------------------------------------

def _run(module, a: np.ndarray) -> np.ndarray:
    """Apply a real ``nn.Module`` to one numpy sample and return numpy.

    The traced values therefore come from the *same* PyTorch modules the trainer
    optimises, not from a re-implementation, so the Math tab can never drift
    away from what the network actually computes.
    """
    import torch

    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
        return module(t).numpy()[0]


def _grid(mat: np.ndarray, rows: int, cols: int, cap: int = 8) -> list[list[float]]:
    """A 2-D slice rendered as a small grid of rounded floats for the UI.

    Layout-agnostic: accepts an already-2-D array *or* a flat array of
    ``rows × cols`` values, so it works for patches, filters and feature maps no
    matter how the upstream array was shaped.
    """
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(rows, cols) if arr.size == rows * cols else arr.reshape(1, -1)
    return [[round(float(v), 4) for v in row[:cap]] for row in arr[:cap]]


def _conv_example(x: np.ndarray, W: np.ndarray, b, pad: int, stride: int,
                  max_side: int = 4) -> dict:
    """One worked convolution, computed by hand from the real learned filters.

    Picks the top-left output position of the first filter and reports the input
    patch, the corresponding filter, their element-wise product and the summed
    pre-activation — exactly the arithmetic PyTorch performs, so the Math tab
    shows the genuine operation rather than an illustration.
    """
    C, H, Wd = x.shape
    F, C2, kh, kw = W.shape
    # pad the two *spatial* axes only, then take the top-left kh×kw window
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    win = padded[:, 0:kh, 0:kw, :] if padded.ndim == 4 else padded[:, 0:kh, 0:kw]
    win = np.transpose(win, (1, 2, 0))                   # [kh, kw, C]
    filt = np.transpose(W[0], (1, 2, 0))                # [kh, kw, C]
    prod = win * filt
    z = float(prod.sum()) + (float(b[0]) if b is not None else 0.0)
    def _patch(arr: np.ndarray) -> list[list[float]]:
        """Render a [kh, kw, C] patch. With one channel the natural view is the
        kh × kw kernel window itself; with several, one row per kernel position."""
        if C == 1:
            return _grid(arr[:, :, 0], kh, kw, max_side)
        return _grid(arr.reshape(kh * kw, C), kh * kw, C, max_side)

    return {
        "output_position": [0, 0],
        "input_patch": _patch(win),
        "filter": _patch(filt),
        "products": _patch(prod),
        "channels": C,
        "sum_of_products": round(float(prod.sum()), 5),
        "bias": round(float(b[0]), 5) if b is not None else 0.0,
        "z": round(z, 5),
        "formula": f"z[0,0] = Σ(w ⊛ x) + b = {prod.sum():.4f}"
                   f"{' + ' + f'{float(b[0]):.4f}' if b is not None else ''} = {z:.4f}",
    }


def _pool_window(x: np.ndarray, k: int, stride: int, pad: int, mode: str,
                 cap: int = 4) -> dict:
    """The first pooling window, its value, and where that value came from."""
    C, H, Wd = x.shape
    # x is an unbatched sample: (C, H, W). Padded values are marked NaN so a
    # window touching the border is not silently averaged with fake zeros.
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)),
                    mode="constant", constant_values=float("nan") if pad else 0.0)
    win = padded[0, 0:k, 0:k]        # first channel's top-left k×k window
    finite = win[np.isfinite(win)]
    if mode == "max":
        value = float(finite.max()) if finite.size else 0.0
    else:
        value = float(finite.mean()) if finite.size else 0.0
    # where the winning value came from (max only)
    src = None
    if mode == "max" and finite.size:
        r, c = np.unravel_index(int(np.nanargmax(np.where(np.isnan(win), -np.inf, win))), win.shape)
        src = [int(r), int(c)]
    return {
        "window": [[("—" if not np.isfinite(v) else round(float(v), 4)) for v in row]
                   for row in win[:cap, :cap]],
        "value": round(value, 5),
        "argmax_position": src,
        "window_size": [k, k],
        "stride": stride,
        "formula": (f"out[0,0] = max(window) = {value:.4f}" if mode == "max"
                    else f"out[0,0] = mean(window) = {value:.4f}"),
    }


def _batchnorm_stats(x: np.ndarray, bn) -> dict:
    """Recompute BatchNorm by hand for the current (evaluation-mode) input.

    Reports the statistics PyTorch is actually using, the normalized values, the
    learnable scale γ and shift β, and the resulting output — so the Math tab
    can show the real ``x̂ = (x − μ)/√(σ² + ε)`` arithmetic.

    ``x`` is a single unbatched sample: (C, H, W) for a feature map, (N,) for a
    vector, so the per-channel means are computed over the spatial axes.
    """
    # x is a single unbatched sample: (C, H, W) for a feature map, (N,) for a
    # vector, so per-channel statistics come from the spatial axes.
    axes = (1, 2) if x.ndim == 3 else None
    mean = x.mean(axis=axes)
    var = x.var(axis=axes)
    # γ and β hold one value per channel, so give them a trailing singleton pair
    # to broadcast across the two spatial axes of a (C, H, W) sample.
    shape = (-1, 1, 1) if x.ndim == 3 else ()
    gamma = (bn.weight.detach().cpu().numpy() if bn.weight is not None
             else np.ones_like(mean)).reshape(shape)
    beta = (bn.bias.detach().cpu().numpy() if bn.bias is not None
            else np.zeros_like(mean)).reshape(shape)
    inv = 1.0 / np.sqrt(var + bn.eps)
    xhat = (x - mean.reshape(shape)) / np.sqrt(var + bn.eps).reshape(shape)
    y = gamma * xhat + beta

    def _flat(v) -> list[float]:
        return [round(float(t), 5) for t in np.atleast_1d(v).reshape(-1)[:8]]

    first = xhat.reshape(-1)[:8]
    return {
        "mode": "eval (running statistics)",
        "momentum": float(bn.momentum),
        "eps": float(bn.eps),
        "affine": bn.weight is not None,
        "mean": _flat(mean),
        "variance": _flat(var),
        "running_mean": _flat(bn.running_mean.detach().cpu().numpy()),
        "running_var": _flat(bn.running_var.detach().cpu().numpy()),
        "scale_gamma": _flat(gamma),
        "shift_beta": _flat(beta),
        "inv_std": _flat(inv),
        "x_hat_sample": _flat(first),
        "y_sample": _flat(y.reshape(-1)[:8]),
        "formula": "x̂ = (x − μ) / √(σ² + ε),  y = γx̂ + β",
    }


def forward_trace(model: ChainModel, spec: NetworkSpec, x: np.ndarray) -> dict:
    """Trace one input row through the network, computing the maths by hand.

    Every supported layer kind contributes a real, explainable entry:

    * ``input``        — the raw probe values, reshaped if the network is a CNN.
    * ``dense``/``output`` — per-neuron ``z = w₁x₁ + … + b`` and ``a = f(z)``.
    * ``activation``   — ``z`` (the incoming values) and ``a = f(z)``.
    * ``batchnorm``    — μ, σ², x̂, γ, β and y.
    * ``flatten``      — the shape transformation itself.
    * ``conv2d``       — a worked ``Σ (w ⊛ x) + b`` for the first filter.
    * ``maxpool``      — the first window, its max and where that max came from.
    * ``avgpool``      — the first window and its mean.
    * ``globalavgpool``— the per-channel spatial means.
    * ``dropout``      — a pass-through note (inactive at inference).

    Nothing here is illustrative: every number comes from the trained parameters
    of the current run.
    """
    from ..ml.shapes import effective_conv_padding

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    layers = {l.id: l for l in spec.layers}
    order = model.order
    custom = spec.custom_activations or {}
    trace: list[dict] = []

    # The Input layer decides whether this is a vector or an image chain; mirror
    # exactly what the model's Reshape module does.
    first_shape = model.shapes.get(order[0]) if order else None
    is_cnn = bool(first_shape and first_shape.is_image)
    a: np.ndarray = (x.reshape(first_shape.torch_shape())
                    if (first_shape and first_shape.is_image) else x)

    for lid in order:
        layer = layers[lid]
        p = layer.params
        kind = layer.kind
        label = layer.label or lid
        block = model.layers[lid]
        a_np = np.asarray(a, dtype=np.float32)
        flat = a_np.reshape(-1)
        C, H, Wd = a_np.shape if a_np.ndim == 3 else (0, 0, 0)
        if kind == "input":
            trace.append({
                "id": lid, "kind": "input", "label": label,
                "n": int(flat.size), "shape": list(a_np.shape),
                "is_image": bool(a_np.ndim == 3),
                "values": np.round(flat[:256], 6).tolist(),
                "matrix": _grid(a_np[0], H, Wd) if a_np.ndim == 3 else None,
                "note": (f"Reshaped to {H} × {Wd} × {C} so the convolutional layers receive "
                         "a proper image tensor.") if a_np.ndim == 3 else None,
            })
            continue

        if kind == "dropout":
            trace.append({
                "id": lid, "kind": "dropout", "label": label,
                "n": int(flat.size), "values": np.round(flat[:256], 6).tolist(),
                "note": ("Dropout is inactive during inference; values pass through unchanged."
                         if p.dropout_mode == "train_only" else
                         "This layer is set to always apply dropout, so values are randomly "
                         "rescaled and zeroed even at inference."),
            })
            continue

        # ---------------- standalone activation ----------------
        if kind == "activation":
            z = a_np
            a_next = _run(block, z)
            trace.append({
                "id": lid, "kind": "activation", "label": label,
                "activation": p.activation,
                "n": int(flat.size), "n_shown": min(int(flat.size), CAP),
                "z": np.round(flat[:CAP], 6).tolist(),
                "a": np.round(a_next.reshape(-1)[:CAP], 6).tolist(),
                "in_shape": list(a_np.shape),
                "formula": f"a = {p.activation}(z)   (element-wise, shape preserved)",
                "note": "The Activation layer is a real module in the PyTorch model — it sits "
                        "between two layers and changes the forward pass.",
            })
            a = a_next
            continue

        # ---------------- batch normalization ----------------
        if kind == "batchnorm":
            y = _run(block, a_np)
            detail = _batchnorm_stats(a_np, block)
            trace.append({
                "id": lid, "kind": "batchnorm", "label": label,
                "activation": "linear",
                "n": int(flat.size), "n_shown": min(int(flat.size), CAP),
                "in_shape": list(a_np.shape), "out_shape": list(y.shape),
                "inputs": np.round(flat[:CAP], 6).tolist(),
                "a": np.round(y.reshape(-1)[:CAP], 6).tolist(),
                "batchnorm": detail,
                "formula": "x̂ = (x − μ) / √(σ² + ε),  y = γx̂ + β",
                "note": (f"Shown in inference mode using the running statistics (updated each "
                         f"training step with momentum {detail['momentum']:g}). Learnable scale γ "
                         f"and shift β: {'yes' if detail['affine'] else 'no (affine is off)'}."),
            })
            a = y
            continue

        # ---------------- flatten ----------------
        if kind == "flatten":
            y = _run(block, a_np)
            trace.append({
                "id": lid, "kind": "flatten", "label": label,
                "activation": "linear",
                "n": int(flat.size), "n_shown": min(int(flat.size), CAP),
                "in_shape": list(a_np.shape), "out_shape": list(y.shape),
                "a": np.round(y.reshape(-1)[:CAP], 6).tolist(),
                "formula": f"{H} × {Wd} × {C}  →  {int(flat.size)}",
                "note": "Flatten only re-orders memory: every feature-map value becomes one "
                        "element of the vector handed to the next layer. No values change.",
            })
            a = y.reshape(-1)
            continue

        # ---------------- convolution ----------------
        if kind == "conv2d":
            conv = block.conv
            W = conv.weight.detach().cpu().numpy()
            b = conv.bias.detach().cpu().numpy() if conv.bias is not None else None
            pad = effective_conv_padding(p, p.kernel_size)
            z_map = _run(conv, a_np)
            a_next = _run(block, a_np)
            out_c, out_h, out_w = z_map.shape
            n_params = int(W.size + (b.size if b is not None else 0))
            trace.append({
                "id": lid, "kind": "conv2d", "label": label,
                "activation": p.activation,
                "n": int(z_map.size), "n_shown": min(int(z_map.size), CAP),
                "in_shape": list(a_np.shape), "out_shape": list(z_map.shape),
                "params": n_params,
                "convolution": _conv_example(a_np, W, b, pad, p.stride),
                "z": np.round(z_map.reshape(-1)[:CAP], 6).tolist(),
                "a": np.round(a_next.reshape(-1)[:CAP], 6).tolist(),
                "feature_map": _grid(a_next[0], out_h, out_w),
                "input_grid": _grid(a_np[0], H, Wd) if a_np.ndim == 3 else None,
                "formula": (f"out = ⌊({H} + 2·{pad} − {p.kernel_size}) / {p.stride}⌋ + 1 = {out_h}"
                            f"  ·  params = ({p.kernel_size}×{p.kernel_size}×{C}"
                            f"{' + 1' if p.use_bias else ''}) × {p.filters} = {n_params:,}"),
                "note": (f"{p.filters} learned filters of {p.kernel_size}×{p.kernel_size} slide "
                         f"over the input with stride {p.stride} and padding {pad}, producing "
                         f"{out_h} × {out_w} × {out_c} values, then {p.activation} is applied."),
            })
            a = a_next
            continue

        # ---------------- pooling ----------------
        if kind in ("maxpool", "avgpool"):
            y = _run(block, a_np)
            mode = "max" if kind == "maxpool" else "mean"
            out_c, out_h, out_w = y.shape
            trace.append({
                "id": lid, "kind": kind, "label": label,
                "activation": "linear",
                "n": int(y.size), "n_shown": min(int(y.size), CAP),
                "in_shape": list(a_np.shape), "out_shape": list(y.shape),
                "pooling": _pool_window(a_np, p.pool_size, p.pool_stride, p.pool_padding, mode),
                "a": np.round(y.reshape(-1)[:CAP], 6).tolist(),
                "feature_map": _grid(y[0], out_h, out_w),
                "input_grid": _grid(a_np[0], H, Wd) if a_np.ndim == 3 else None,
                "formula": (f"out = ⌊({H} + 2·{p.pool_padding} − {p.pool_size}) / {p.pool_stride}⌋ + 1"
                            f" = {out_h}  ·  {mode} over each {p.pool_size}×{p.pool_size} window"),
                "note": (f"Each {p.pool_size}×{p.pool_size} window collapses to one number (the "
                         f"{mode}), so the feature map shrinks to {out_h} × {out_w} while keeping "
                         f"{out_c} channels."),
            })
            a = y
            continue

        # ---------------- global average pooling ----------------
        if kind == "globalavgpool":
            y = _run(block, a_np)
            per_ch = a_np.mean(axis=(1, 2)) if a_np.ndim == 3 else a_np
            trace.append({
                "id": lid, "kind": "globalavgpool", "label": label,
                "activation": "linear",
                "n": int(y.size), "n_shown": min(int(y.size), CAP),
                "in_shape": list(a_np.shape), "out_shape": list(y.shape),
                "channel_means": [round(float(v), 5) for v in np.atleast_1d(per_ch)[:16]],
                "a": np.round(y.reshape(-1)[:CAP], 6).tolist(),
                "formula": f"out_c = (1 / {H}·{Wd}) Σ x_c   →   {C} values",
                "note": (f"Each of the {C} feature maps is averaged over all {H}×{Wd} spatial "
                         f"positions, collapsing {H} × {Wd} × {C} into just {C} numbers — which "
                         "is why a CNN can reach a Dense layer without a Flatten."),
            })
            a = y.reshape(-1)
            continue
        # ---------------- dense / output ----------------
        W = block.linear.weight.detach().cpu().numpy()          # (out, in)
        b = block.linear.bias.detach().cpu().numpy() if block.linear.bias is not None else None
        z = flat @ W.T + (b if b is not None else 0.0)
        a_next = _apply_activation(p.activation, custom, z)
        neurons = []
        for j in range(min(len(z), CAP)):
            wj = W[j]
            contrib = flat * wj
            neurons.append({
                "index": j,
                "bias": round(float(b[j]), 6) if b is not None else 0.0,
                "z": round(float(z[j]), 6),
                "a": round(float(a_next[j]), 6),
                "weights": np.round(wj[:CAP_IN], 5).tolist(),
                "weighted_inputs": np.round(contrib[:CAP_IN], 6).tolist(),
                "n_inputs": int(flat.size),
                "truncated": int(flat.size) > CAP_IN,
            })
        trace.append({
            "id": lid, "kind": kind, "label": label,
            "activation": p.activation,
            "n": int(len(z)), "n_shown": min(len(z), CAP),
            "inputs": np.round(flat[:CAP_IN], 6).tolist(),
            "in_features": int(flat.size),
            "z": np.round(z, 6).tolist(),
            "a": np.round(a_next, 6).tolist(),
            "neurons": neurons,
            "formula": (f"z = Wx + b with {int(flat.size)} inputs × {len(z)} neurons  ·  "
                        f"params = {int(W.size + (b.size if b is not None else 0)):,}"),
        })
        a = a_next

    out = np.asarray(a, dtype=np.float32).reshape(-1)
    return {
        "output": np.round(out, 6).tolist(),
        "prediction": None,
        "probabilities": None,
        "is_cnn": is_cnn,
        "layers": trace,
    }


def classification_details(spec: NetworkSpec, model: ChainModel, out: np.ndarray, class_names, loss: str) -> dict:
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    last_id = model.order[-1]
    last_layer = {l.id: l for l in spec.layers}[last_id]
    out_act = last_layer.params.activation
    if class_names is None:
        return {"type": "regression", "value": round(float(out[0]), 6)}
    if len(out) == 1:
        p = float(1.0 / (1.0 + np.exp(-out[0]))) if out_act != "sigmoid" else float(out[0])
        p = min(max(p, 0.0), 1.0)
        pred = int(p >= 0.5)
        return {
            "type": "classification", "prediction": pred,
            "label": class_names[pred],
            "probabilities": [1.0 - p, p],
        }
    if out_act == "softmax":
        probs = out
    else:
        e = np.exp(out - out.max())
        probs = e / e.sum()
    pred = int(np.argmax(probs))
    return {
        "type": "classification", "prediction": pred,
        "label": class_names[pred], "probabilities": [round(float(p), 5) for p in probs],
    }


def backprop_example(model: ChainModel, x: np.ndarray, y, loss_name: str, task: str,
                     lr: float, layer_id: str | None = None) -> dict:
    """Compute one real optimisation step on a throwaway copy of the model and
    report before/after for an example weight:  w_new = w_old − η · ∂L/∂w."""
    import copy

    import torch

    from ..ml.trainer import _loss
    from ..ml.datasets import PreparedData

    model_copy = copy.deepcopy(model)
    model_copy.train()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xb = torch.tensor(arr)
    if task == "classification":
        yb = torch.tensor([int(y)], dtype=torch.long) if loss_name == "cross_entropy" \
            else torch.tensor([[float(y)]], dtype=torch.float32)
    else:
        yb = torch.tensor(np.asarray(y, dtype=np.float32).reshape(1, -1))
    data_stub = PreparedData(task=task, X_train=np.zeros((1, 1)), y_train=np.zeros(1),
                             X_val=np.zeros((0, 1)), y_val=np.zeros(0),
                             X_test=np.zeros((0, 1)), y_test=np.zeros(0),
                             n_features=1, n_classes=None, class_names=None,
                             feature_names_in=[], feature_names_model=[])
    out = model_copy(xb)
    lv = _loss(out, yb, loss_name, data_stub)
    lv.backward()

    # choose the example weight: the first layer that owns a primary tensor.
    # Its rank varies (Linear/Conv2d are 2-D, BatchNorm's γ is 1-D), so index the
    # flattened tensor rather than assuming two dimensions.
    lid = layer_id or next((k for k in model_copy.order
                            if model_copy.primary_weight(k) is not None), None)
    if lid is None:
        raise ValueError("This network has no trainable layers.")
    w = model_copy.primary_weight(lid)
    assert w is not None and w.grad is not None
    w_old = float(w.detach().reshape(-1)[0].item())
    g = float(w.grad.reshape(-1)[0].item())
    with torch.no_grad():
        for p in model_copy.parameters():
            if p.grad is not None:
                p -= lr * p.grad
    w_new = float(w.detach().reshape(-1)[0].item())
    return {
        "layer_id": lid,
        "index": [0, 0] if w.dim() >= 2 else [0],
        "loss": round(float(lv.detach()), 6),
        "w_old": round(w_old, 6),
        "gradient": round(g, 6),
        "learning_rate": lr,
        "w_new_computed": round(w_old - lr * g, 6),
        "w_new_actual": round(w_new, 6),
        "formula": "w_new = w_old − η · ∂L/∂w",
    }
