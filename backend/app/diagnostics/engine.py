"""Deterministic, rule-based network/training diagnosis.

Every finding is derived from *actual* recorded numbers (loss history,
metrics, activation statistics, gradient statistics, weight statistics,
dataset statistics). Language is deliberately hedged ("possible",
"potential", "consider") — these are indications, not certainties.
"""
from __future__ import annotations

import math
from typing import Any


def _f(sev: str, code: str, title: str, explanation: str, evidence: dict, suggestions: list[str]) -> dict:
    return {"severity": sev, "code": code, "title": title,
            "explanation": explanation, "evidence": evidence, "suggestions": suggestions}


def diagnose(run: dict[str, Any]) -> list[dict]:
    history: list[dict] = run.get("history") or []
    snaps: list[dict] = run.get("snapshots") or []
    data_summary: dict = run.get("data_summary") or {}
    final: dict = run.get("final") or {}
    net_summary: dict = (run.get("network_summary") or {})
    findings: list[dict] = []

    if not history:
        return [_f("info", "NO_DATA", "No training data available",
                   "This run has no recorded epochs — diagnosis needs at least a completed training run.",
                   {}, ["Train the network for a few epochs, then run diagnosis again."])]

    task = data_summary.get("task", "classification")
    metric = history[-1].get("metric_name", "accuracy")
    last = history[-1]
    first = history[0]
    train_losses = [h["train_loss"] for h in history]
    tr_m = [h["train_metric"] for h in history]
    vl_m = [h["val_metric"] for h in history if h.get("val_metric") is not None]

    # ---- divergence ---------------------------------------------------
    if any(not math.isfinite(v) for v in train_losses):
        findings.append(_f("critical", "NAN_LOSS", "Training diverged (NaN/Inf loss)",
            "The loss became NaN or infinite during training — the network's numbers overflowed. "
            "This is usually caused by a learning rate that is far too high, unscaled inputs, or an "
            "unstable activation (e.g. an exploding custom function).",
            {"first_bad_epoch": next((h["epoch"] for h in history if not math.isfinite(h["train_loss"])), None)},
            ["Reduce the learning rate by 10-100×.",
             "Standardize/normalize input features.",
             "Check custom activation functions for exponential blow-ups.",
             "Use Xavier/He initialization."]))
        return findings  # everything else would be noise

    # ---- overfitting ----------------------------------------------------
    if len(history) >= 6 and vl_m:
        gap = tr_m[-1] - vl_m[-1]
        if task == "classification" and gap >= 0.12:
            findings.append(_f("warning", "OVERFITTING_GAP", "Possible overfitting",
                f"Training {metric}: {tr_m[-1]:.1%} vs validation {metric}: {vl_m[-1]:.1%} — "
                f"a gap of {gap:.1%}. The model may be memorising the training set rather than generalising.",
                {"train_metric": tr_m[-1], "val_metric": vl_m[-1], "gap": gap},
                ["Reduce network size (fewer neurons/layers).",
                 "Add a Dropout layer between hidden layers.",
                 "Enable L2 weight regularization.",
                 "Gather more training data if possible."]))
        best_val = max(vl_m)
        best_ep = next(h["epoch"] for h in history if h.get("val_metric") == best_val)
        if best_ep <= len(history) * 0.5 and vl_m[-1] < best_val - 2e-3:
            findings.append(_f("info", "VAL_PLATEAU", "Validation performance peaked early",
                f"Best validation {metric} ({best_val:.1%}) occurred at epoch {best_ep} and has "
                "not improved since — later epochs are likely not helping.",
                {"best_epoch": best_ep, "total_epochs": len(history)},
                ["Stop training earlier (fewer epochs).",
                 "Consider a lower learning rate for fine convergence."]))

    # ---- underfitting ----------------------------------------------------
    if task == "classification" and vl_m:
        majority = 1.0 / max(1, data_summary.get("n_classes", 2))
        counts = data_summary.get("class_counts_train") or []
        if counts:
            majority = max(counts) / sum(counts)
        if vl_m[-1] <= majority + 0.05 and len(history) >= 10:
            findings.append(_f("warning", "UNDERFITTING", "Possible underfitting",
                f"Validation {metric} is {vl_m[-1]:.1%}, close to the majority-class baseline "
                f"({majority:.1%}). The model may lack capacity or training to capture the pattern.",
                {"val_metric": vl_m[-1], "majority_baseline": majority},
                ["Add neurons or another hidden layer.",
                 "Train longer / increase the learning rate moderately.",
                 "Check that relevant features are selected and well-scaled."]))
    if task == "regression":
        r2 = (final.get("test_metrics") or {}).get("r2")
        if r2 is not None and r2 < 0.5:
            findings.append(_f("warning", "LOW_R2", "Low explanatory power (possible underfitting)",
                f"Test R² is {r2:.3f} — the model explains less than half of the target variance.",
                {"r2": r2},
                ["Increase network capacity.",
                 "Train longer.",
                 "Verify the target actually depends on the selected features.",
                 "Try scaling the target or using MSE loss if outliers dominate."]))

    # ---- learning-rate heuristics -----------------------------------------
    if len(history) >= 8:
        last_k = train_losses[-min(10, len(train_losses)):]
        mean_k = sum(last_k) / len(last_k)
        if mean_k > 1e-12:
            std_k = (sum((v - mean_k) ** 2 for v in last_k) / len(last_k)) ** 0.5
            if std_k / mean_k > 0.5 and train_losses[0] > 0:
                findings.append(_f("warning", "LR_HIGH_OSC", "Unstable loss — learning rate may be too high",
                    "The training loss oscillates strongly (relative std > 50% over the last epochs) — "
                    "a classic indication of a learning rate that overshoots minima.",
                    {"mean_loss": mean_k, "std_loss": std_k},
                    ["Reduce the learning rate by 5-10×.",
                     "Use Adam/RMSprop which adapt per-parameter step sizes."]))
        if last["train_loss"] > first["train_loss"] * 1.2:
            findings.append(_f("warning", "LOSS_INCREASED", "Loss increased during training",
                f"Final training loss ({last['train_loss']:.4f}) is higher than the first epoch "
                f"({first['train_loss']:.4f}).",
                {"first": first["train_loss"], "last": last["train_loss"]},
                ["Lower the learning rate.",
                 "Check that the chosen loss matches the task and output activation."]))
        # slow learning
        if len(history) >= 12:
            span = train_losses[-1] - train_losses[-10]
            rel = abs(span) / max(1e-12, abs(train_losses[-10]))
            if rel < 0.01 and ((task == "classification" and tr_m[-1] < 0.9)
                               or (task == "regression" and (final.get("test_metrics") or {}).get("r2", 1) < 0.9)):
                findings.append(_f("info", "SLOW_LEARNING", "Learning is very slow",
                    "The loss hardly changed over the last 10 epochs while the model is still far from "
                    "converged — the learning rate may be too small, or the network hit a plateau.",
                    {"relative_change_last_10": rel},
                    ["Increase the learning rate moderately.",
                     "Try Adam, which adapts step sizes automatically.",
                     "Check for vanishing gradients below."]))

    # ---- gradient statistics ---------------------------------------------
    if snaps:
        snap = snaps[-1]
        grads = (snap.get("gradients") or {}).get("layers") or []
        if grads:
            norms = [g["w_grad_norm"] for g in grads]
            first_norm, last_norm = norms[0], norms[-1]
            if any(n > 1e3 for n in norms):
                findings.append(_f("warning", "GRAD_EXPLODE", "Possible exploding gradients",
                    f"A layer's weight-gradient norm reaches {max(norms):.1f} — very large updates "
                    "destabilise training and can cause the loss to blow up.",
                    {"layer_norms": {g["label"]: g["w_grad_norm"] for g in grads}},
                    ["Reduce the learning rate.",
                     "Ensure inputs/activations are well scaled.",
                     "Consider fewer stacked layers or a gentler activation."]))
            elif first_norm < 1e-7 or (last_norm > 0 and first_norm / last_norm < 1e-2 and first_norm < 1e-5):
                findings.append(_f("warning", "GRAD_VANISH", "Possible vanishing gradients",
                    f"The earliest layer's gradient norm ({first_norm:.2e}) is tiny relative to the "
                    f"last layer's ({last_norm:.2e}) — gradients shrink as they propagate backwards, "
                    "so early layers barely learn.",
                    {"first_layer": first_norm, "last_layer": last_norm,
                     "ratio": (first_norm / last_norm) if last_norm else None},
                    ["Use ReLU/GELU activations instead of sigmoid/tanh in hidden layers.",
                     "Use He (for ReLU) or Xavier initialization.",
                     "Reduce network depth or add normalization."]))
        for a in snap.get("activations") or []:
            if a.get("dead_neuron_frac") is not None and a["dead_neuron_frac"] >= 0.5:
                findings.append(_f("warning", "DEAD_NEURONS", f"Many dead neurons in '{a['label']}'",
                    f"{a['dead_neuron_frac']:.0%} of the neurons in layer '{a['label']}' output ≈0 for "
                    "the entire inspected batch (dead ReLU-type units). They contribute nothing.",
                    {"layer": a["label"], "dead_fraction": a["dead_neuron_frac"]},
                    ["Increase the learning rate slightly, or decrease it if weights collapsed at init.",
                     "Try Leaky ReLU/GELU which keep a small gradient for negative inputs.",
                     "Re-initialise with He initialization."]))
            if a.get("frac_saturated") is not None and a["frac_saturated"] >= 0.3 and \
               a["activation"] in ("sigmoid", "tanh"):
                findings.append(_f("info", "SATURATION", f"Activation saturation in '{a['label']}'",
                    f"{a['frac_saturated']:.0%} of activations in '{a['label']}' sit in the flat tails "
                    f"of {a['activation']} — gradients there vanish, slowing learning in earlier layers.",
                    {"layer": a["label"], "saturated_fraction": a["frac_saturated"]},
                    ["Scale inputs so pre-activations stay near 0.",
                     "Prefer ReLU-family activations in hidden layers.",
                     "Use Xavier initialization for tanh/sigmoid layers."]))

    # ---- dataset-level ------------------------------------------------------
    counts = data_summary.get("class_counts_train") or []
    if counts and len(counts) >= 2 and max(counts) > 0:
        ratio = min(counts) / max(counts)
        if ratio < 0.2:
            names = data_summary.get("class_names") or [str(i) for i in range(len(counts))]
            # `counts` comes from np.bincount(..., minlength=n_classes) and
            # `names` from the dataset's class_names, so the two are *not*
            # guaranteed to be the same length. zip() would silently truncate to
            # the shorter one and report a class distribution that quietly
            # disagrees with the recorded numbers. Validate explicitly instead:
            # on a mismatch fall back to positional labels, which is at worst
            # less readable but never wrong.
            if len(names) != len(counts):
                names = [str(i) for i in range(len(counts))]
            class_counts = dict(zip(names, counts, strict=True))
            findings.append(_f("info", "CLASS_IMBALANCE", "Class imbalance detected",
                f"The smallest class has only {ratio:.0%} of the largest class's samples "
                f"({class_counts}). Accuracy alone can be misleading here.",
                {"counts": class_counts},
                ["Check precision/recall/F1 in the metrics, not only accuracy.",
                 "Consider oversampling the minority class or collecting more data.",
                 "A stratified split (default) keeps proportions stable across splits."]))
    raw_stds = data_summary.get("raw_feature_stds") or []
    if data_summary.get("scaled") == "none" and raw_stds:
        s = sorted(v for v in raw_stds if v > 1e-12)
        if s and s[-1] / s[0] > 10:
            findings.append(_f("info", "FEATURE_SCALING", "Features are on very different scales",
                f"Raw feature standard-deviations span a {s[-1]/s[0]:.0f}× range and no scaling was "
                "applied — large-scale features can dominate gradients and slow training.",
                {"min_std": s[0], "max_std": s[-1]},
                ["Enable standardization (zero mean, unit variance) in the dataset settings.",
                 "Or use min-max normalization for bounded features."]))

    # ---- architecture --------------------------------------------------------
    params = net_summary.get("total_params") or 0
    n_train = data_summary.get("n_train") or 0
    if params and n_train and params > 10 * n_train:
        findings.append(_f("info", "PARAMS_VS_DATA", "Model capacity is large relative to the data",
            f"The network has {params:,} parameters but only {n_train} training samples "
            f"({params/max(n_train,1):.0f} params per sample). Such models can memorise data.",
            {"params": params, "n_train": n_train},
            ["Shrink the network unless the task is genuinely complex.",
             "Add dropout/L2 regularization if overfitting appears."]))

    if not findings:
        findings.append(_f("ok", "HEALTHY", "No obvious problems detected",
            "Loss curves, metrics, gradients and activations all look reasonable for this run. "
            "This does not guarantee an optimal model — only that none of the checked failure "
            "patterns were found.",
            {"epochs": len(history), "final_train_loss": last["train_loss"],
             f"final_train_{metric}": tr_m[-1]},
            ["Try a research-mode experiment varying activation or learning rate to see if "
             "you can improve further."]))
    sev_rank = {"critical": 0, "warning": 1, "info": 2, "ok": 3}
    findings.sort(key=lambda f: sev_rank.get(f["severity"], 4))
    return findings
