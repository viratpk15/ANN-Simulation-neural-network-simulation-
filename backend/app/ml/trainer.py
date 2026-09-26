"""Background training engine with pause/resume/stop and live event stream.

Each :class:`TrainingJob` runs in its own thread. Metrics/snapshots are
appended to an in-memory event list which the WebSocket layer polls — this
avoids any cross-thread asyncio complexity and makes status cheap to read.
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .activations import RELU_FAMILY, SATURATING
from .builder import ChainModel, build_model, validate_network, weight_matrices
from .datasets import PreparedData, prepare
from ..models.schemas import TrainRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingError(ValueError):
    pass


def check_compatibility(output_dim: int, out_activation: str, data: PreparedData, loss: str) -> None:
    """Raise TrainingError if dataset/loss/output-layer shape don't line up."""
    if data.task == "classification":
        if loss == "cross_entropy":
            if output_dim != (data.n_classes or 0):
                raise TrainingError(
                    f"Output layer has {output_dim} neurons but the dataset has {data.n_classes} classes. "
                    f"Set the output layer to {data.n_classes} neurons (softmax) for cross-entropy, "
                    "or use BCE with 1 output neuron for binary problems."
                )
            if out_activation not in ("softmax", "linear"):
                raise TrainingError(
                    f"Output activation '{out_activation}' is unusual for cross-entropy — "
                    "use 'softmax' (probabilities) or 'linear' (raw logits)."
                )
        elif loss == "bce":
            if (data.n_classes or 0) != 2:
                raise TrainingError("Binary cross-entropy requires a 2-class dataset (this one has "
                                    f"{data.n_classes} classes). Use cross-entropy instead.")
            if output_dim != 1:
                raise TrainingError("BCE expects exactly 1 output neuron with a sigmoid activation.")
            if out_activation != "sigmoid":
                raise TrainingError("BCE expects the output activation to be 'sigmoid' (probabilities in (0,1)).")
        else:
            raise TrainingError(f"Loss '{loss}' is a regression loss but the dataset is a classification task.")
    else:  # regression
        if loss not in ("mse", "mae"):
            raise TrainingError(f"Loss '{loss}' is a classification loss but the dataset is "
                                "a regression task. Use MSE or MAE.")
        if output_dim != 1:
            raise TrainingError("Regression expects exactly 1 output neuron (use 'linear' output activation).")


class TrainingJob:
    STATES = ("queued", "running", "paused", "stopped", "finished", "failed")

    def __init__(self, request: TrainRequest, settings_max_epochs: int = 2000):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.max_epochs = settings_max_epochs
        self.state = "queued"
        self.error: str | None = None
        self.created_at = _now_iso()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.duration_s: float = 0.0

        self.events: list[dict] = []            # epoch/snapshot/status events for the WS stream
        self.history: list[dict] = []           # per-epoch scalar metrics
        self.snapshots: list[dict] = []         # activation/gradient snapshots
        self.final: dict | None = None
        self.data_summary: dict | None = None

        self._pause = threading.Event()         # set => running, clear => paused
        self._pause.set()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.model: ChainModel | None = None
        self.data: PreparedData | None = None

    # -- controls ----------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"train-{self.id}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.state == "running":
            self._pause.clear()
            self._set_state("paused")

    def resume(self) -> None:
        if self.state == "paused":
            self._pause.set()
            self._set_state("running")

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()  # release a paused loop so it can observe the stop

    # -- introspection -----------------------------------------------------
    def _emit(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)

    def events_after(self, idx: int) -> tuple[int, list[dict]]:
        with self._lock:
            return len(self.events), self.events[idx:]

    def _set_state(self, s: str) -> None:
        self.state = s
        self._emit({"type": "status", "state": s, "at": _now_iso()})

    def status(self) -> dict:
        return {
            "job_id": self.id,
            "state": self.state,
            "error": self.error,
            "epoch": self.history[-1]["epoch"] if self.history else 0,
            "total_epochs": self.request.config.epochs,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "n_events": len(self.events),
            "final": self.final,
            "data_summary": self.data_summary,
        }

    # -- the actual training -----------------------------------------------
    def _run(self) -> None:  # noqa: C901 - training loop is intentionally explicit
        t0 = time.time()
        self.started_at = _now_iso()
        req, cfg = self.request, self.request.config
        try:
            if cfg.epochs < 1 or cfg.epochs > self.max_epochs:
                raise TrainingError(f"epochs must be between 1 and {self.max_epochs}.")
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

            # ---- data ---------------------------------------------------
            data = prepare(req.dataset)
            self.data = data
            class_counts = None
            if data.task == "classification":
                class_counts = np.bincount(data.y_train, minlength=data.n_classes or 2).tolist()
            self.data_summary = {
                "task": data.task,
                "n_train": int(len(data.y_train)),
                "n_val": int(len(data.y_val)),
                "n_test": int(len(data.y_test)),
                "n_features": data.n_features,
                "n_classes": data.n_classes,
                "class_names": data.class_names,
                "class_counts_train": class_counts,
                "feature_names_in": data.feature_names_in,
                "scaled": req.dataset.preprocessing.scale,
                "raw_feature_stds": data.meta.get("raw_feature_stds"),
            }

            # ---- network -------------------------------------------------
            val = validate_network(req.network)
            if not val.ok:
                raise TrainingError("Network validation failed: " +
                                    "; ".join(e["message"] for e in val.errors))
            assert val.input_dim is not None and val.output_dim is not None
            if val.input_dim != data.n_features:
                raise TrainingError(
                    f"Input layer expects {val.input_dim} features but the prepared dataset has "
                    f"{data.n_features} features (after encoding). Change the input layer or the "
                    "selected feature columns."
                )
            last = req.network.layers[[l.id for l in req.network.layers].index(val.order[-1])]
            check_compatibility(val.output_dim, last.params.activation, data, cfg.loss)
            # let the loss function know whether softmax is already applied
            data._softmax_output = (last.params.activation == "softmax")  # type: ignore[attr-defined]

            model = build_model(req.network, val, seed=cfg.seed)
            self.model = model

            # ---- tensors --------------------------------------------------
            Xtr = torch.tensor(data.X_train, dtype=torch.float32)
            Xval = torch.tensor(data.X_val, dtype=torch.float32) if len(data.X_val) else None
            if data.task == "classification":
                if cfg.loss == "bce":
                    ytr = torch.tensor(data.y_train.astype(np.float32)).view(-1, 1)
                    yval = torch.tensor(data.y_val.astype(np.float32)).view(-1, 1) if Xval is not None else None
                else:
                    ytr = torch.tensor(data.y_train, dtype=torch.long)
                    yval = torch.tensor(data.y_val, dtype=torch.long) if Xval is not None else None
            else:
                ytr = torch.tensor(data.y_train, dtype=torch.float32)
                yval = torch.tensor(data.y_val, dtype=torch.float32) if Xval is not None else None

            opt = _make_optimizer(model, cfg)
            n = len(Xtr)
            bs = max(1, min(cfg.batch_size, n))
            fixed_x = Xtr[: min(128, n)]
            fixed_y = ytr[: min(128, n)]
            metric_name = "accuracy" if data.task == "classification" else "mae"
            rng = torch.Generator().manual_seed(cfg.seed)  # job-local RNG: no cross-job interference

            self._set_state("running")

            for epoch in range(1, cfg.epochs + 1):
                self._pause.wait()
                if self._stop.is_set():
                    break

                model.train()
                order = torch.randperm(n, generator=rng) if cfg.shuffle else torch.arange(n)
                tot_loss, nb = 0.0, 0
                for i in range(0, n, bs):
                    if self._stop.is_set():
                        break
                    self._pause.wait()
                    xb, yb = Xtr[order[i:i + bs]], ytr[order[i:i + bs]]
                    opt.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = _loss(out, yb, cfg.loss, data)
                    if cfg.l2 > 0:
                        loss = loss + cfg.l2 * sum((p ** 2).sum() for p in model.parameters())
                    loss.backward()
                    opt.step()
                    tot_loss += float(loss.detach())
                    nb += 1
                if self._stop.is_set():
                    break

                # ---- epoch evaluation ------------------------------------
                tr_loss, tr_metric = _evaluate(model, Xtr, ytr, cfg.loss, data, metric_name)
                if Xval is not None and len(Xval):
                    vl_loss, vl_metric = _evaluate(model, Xval, yval, cfg.loss, data, metric_name)
                else:
                    vl_loss = vl_metric = None
                rec = {
                    "epoch": epoch,
                    "train_loss": round(tot_loss / max(nb, 1), 6),
                    "train_loss_full": round(tr_loss, 6),
                    "val_loss": round(vl_loss, 6) if vl_loss is not None else None,
                    "train_metric": round(tr_metric, 5),
                    "val_metric": round(vl_metric, 5) if vl_metric is not None else None,
                    "metric_name": metric_name,
                    "lr": cfg.learning_rate,
                }
                self.history.append(rec)
                self._emit({"type": "epoch", **rec, "total_epochs": cfg.epochs, "batches": nb})

                if epoch % max(1, cfg.snapshot_every) == 0 or epoch == cfg.epochs:
                    snap = _capture_snapshot(model, req, fixed_x, fixed_y, cfg, data, epoch, val.order)
                    self.snapshots.append(snap)
                    self._emit({"type": "snapshot", "epoch": epoch, "snapshot": snap})

            # ---- final evaluation on the held-out TEST set ------------------
            self.final = self._final_eval(model, data, cfg, val.order)
            self.duration_s = time.time() - t0
            self.finished_at = _now_iso()
            self._set_state("stopped" if self._stop.is_set() else "finished")
        except TrainingError as exc:
            self.error = str(exc)
            self.duration_s = time.time() - t0
            self._set_state("failed")
        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"Unexpected training error: {exc}"
            self.duration_s = time.time() - t0
            self._set_state("failed")

    def _final_eval(self, model: ChainModel, data: PreparedData, cfg, order) -> dict:
        Xte = torch.tensor(data.X_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            out = model(Xte)
        if data.task == "classification":
            if data.n_classes == 2 and cfg.loss == "bce":
                pred = (out.view(-1).numpy() >= 0.5).astype(int)
            else:
                pred = out.argmax(dim=1).numpy()
            y_true = data.y_test
            fm = M.final_metrics("classification", y_true, pred, data.n_classes)
            cm = M.confusion_matrix(y_true, pred, int(data.n_classes or 2))
        else:
            pred = out.numpy().reshape(-1)
            y_true = data.y_test.reshape(-1)
            fm = M.final_metrics("regression", y_true, pred, None)
            cm = None
        return {
            "test_metrics": fm,
            "confusion_matrix": cm,
            "n_test": int(len(data.y_test)),
            "weights": weight_matrices(model),
            "total_epochs_run": self.history[-1]["epoch"] if self.history else 0,
        }


# --------------------------------------------------------------------------
def _make_optimizer(model, cfg) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate)
    if cfg.optimizer == "momentum":
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate)
    return torch.optim.RMSprop(params, lr=cfg.learning_rate)


def _loss(out, y, loss_name: str, data: PreparedData) -> torch.Tensor:
    if data.task == "classification":
        if loss_name == "bce":
            return F.binary_cross_entropy(torch.clamp(out, 1e-6, 1 - 1e-6), y)
        # cross entropy — supports raw logits (linear) or softmax probabilities.
        # Softmax outputs already sum to 1, so NLL on log(probs) is exactly the
        # mathematically equivalent Softmax+CE composition. For linear outputs
        # we apply softmax first. Same gradients either way.
        if out.shape[-1] == 1:
            raise TrainingError("Cross-entropy requires more than one output neuron; use "
                                "BCE for 1-neuron binary output.")
        if getattr(data, "_softmax_output", False):
            probs = torch.clamp(out, 1e-8, 1.0)
        else:
            probs = torch.softmax(out, dim=-1)
        return F.nll_loss(torch.log(probs), y)
    if loss_name == "mse":
        return F.mse_loss(out, y)
    return F.l1_loss(out, y)


def _evaluate(model, X, y, loss_name, data, metric_name) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        out = model(X)
        lv = float(_loss(out, y, loss_name, data))
        if data.task == "classification":
            if data.n_classes == 2 and y.ndim == 2:
                pred = (out.view(-1).numpy() >= 0.5).astype(int)
            else:
                pred = out.argmax(dim=1).numpy()
            met = M.accuracy(y.numpy().reshape(-1).astype(int), pred)
        else:
            met = M.mae(y.numpy().reshape(-1), out.numpy().reshape(-1))
    return lv, met


def _effective_activation(layer) -> str:
    """The activation that produced this layer's output.

    For Dense/Conv2D/Output it is the layer's own ``activation`` field; for the
    standalone Activation layer it is the selected function; other kinds
    (BatchNorm, pooling, dropout, flatten) apply no non-linearity, reported as
    ``"linear"`` so the saturation heuristics simply do not fire.
    """
    kind = layer.kind
    if kind in ("dense", "output", "conv2d", "activation"):
        return layer.params.activation
    return "linear"


def _bias_grad_norm(model: ChainModel, layer_id: str) -> float | None:
    """Gradient norm of a layer's bias vector, when it has one."""
    mod = model.layers[layer_id]
    for holder, attr in ((mod, "linear"), (mod, "conv"), (mod, None)):
        target = getattr(holder, attr, None) if attr else mod
        if target is None:
            continue
        bias = getattr(target, "bias", None)
        if isinstance(bias, torch.Tensor) and bias.grad is not None:
            return float(bias.grad.norm())
    return None


def _capture_snapshot(model: ChainModel, req: TrainRequest, xb, yb, cfg, data, epoch, order) -> dict:
    """Activation stats + per-neuron samples + gradient statistics, all real."""
    acts: dict[str, torch.Tensor] = {}
    handles = []
    # Hook *every* layer rather than only Linear blocks, so Conv2d, BatchNorm,
    # the standalone Activation layer and the pooling layers all contribute
    # real statistics to the Activations tab. A layer's *output* is what the
    # next layer receives, so hooking the layer module is the right level.
    def _grab(key):
        def hook(_m, _i, o):
            acts[key] = o.detach()
        return hook

    for lid in order:
        handles.append(model.layers[lid].register_forward_hook(_grab(lid)))
    try:
        model.zero_grad(set_to_none=True)
        model.eval()
        out = model(xb)
        loss = _loss(out, yb, cfg.loss, data)
        loss.backward()

        layers = {l.id: l for l in req.network.layers}
        act_stats, grad_stats = [], []
        for lid in order:
            layer = layers[lid]
            if lid in acts:
                a = acts[lid]
                # collapse (N, C, H, W) and (N, C) to (N, features) so the same
                # statistics apply to every layer kind
                a_np = a.cpu().numpy().reshape(a.shape[0], -1)
                act_name = _effective_activation(layer)
                per_neuron_all_zero = (np.abs(a_np).max(axis=0) < 1e-6)
                sat_frac = None
                if act_name in SATURATING:
                    if act_name == "sigmoid":
                        sat_frac = float(((a_np <= 0.02) | (a_np >= 0.98)).mean())
                    else:
                        sat_frac = float((np.abs(a_np) >= 0.98).mean())
                act_stats.append({
                    "layer_id": lid,
                    "label": layer.label or lid,
                    "activation": act_name,
                    "mean": round(float(a_np.mean()), 5),
                    "std": round(float(a_np.std()), 5),
                    "min": round(float(a_np.min()), 5),
                    "max": round(float(a_np.max()), 5),
                    "frac_near_zero": round(float((np.abs(a_np) < 1e-3).mean()), 4),
                    "frac_saturated": round(sat_frac, 4) if sat_frac is not None else None,
                    "dead_neuron_frac": (round(float(per_neuron_all_zero.mean()), 4)
                                        if act_name in RELU_FAMILY else None),
                    "sample": [round(float(v), 4) for v in a_np.flatten()[:256]],
                })
            # gradients: every layer that owns a primary trainable tensor
            w = model.primary_weight(lid)
            if w is not None and w.grad is not None:
                wg = w.grad
                # w may be rank-2 (Linear/Conv2d) or rank-1 (BatchNorm's γ), so
                # index the flattened tensor rather than w[0, 0].
                wf = w.detach().reshape(-1)
                gf = wg.reshape(-1)
                grad_stats.append({
                    "layer_id": lid,
                    "label": layers[lid].label or lid,
                    "w_grad_norm": float(wg.norm()),
                    "w_grad_mean_abs": float(wg.abs().mean()),
                    "w_grad_max_abs": float(wg.abs().max()),
                    "b_grad_norm": _bias_grad_norm(model, lid),
                    "example_weight": {
                        "index": [0, 0] if w.dim() >= 2 else [0],
                        "w": float(wf[0].item()),
                        "grad": float(gf[0].item()),
                    },
                })
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.norm()) ** 2
        return {
            "epoch": epoch,
            "activations": act_stats,
            "gradients": {
                "layers": grad_stats,
                "total_norm": round(total_norm ** 0.5, 6),
                "loss_on_batch": round(float(loss.detach()), 6),
                "input_sample": [round(float(v), 4) for v in xb[0].tolist()],
            },
        }
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
