"""Rule-based diagnostic engine tests (findings from crafted signals)."""
import math

from app.diagnostics.engine import diagnose


def base_run(**kw):
    run = {
        "history": [
            {"epoch": i + 1, "train_loss": 0.6 - 0.1 * i, "val_loss": 0.62 - 0.09 * i,
             "train_metric": m_t, "val_metric": m_v, "metric_name": "accuracy"}
            for i, (m_t, m_v) in enumerate(kw.pop("metrics", [(0.7, 0.68), (0.8, 0.78), (0.85, 0.83),
                                                              (0.9, 0.88), (0.93, 0.9), (0.95, 0.92)]))
        ],
        "snapshots": kw.pop("snapshots", []),
        "data_summary": kw.pop("data_summary", {"task": "classification", "n_train": 500,
                                                "n_classes": 3, "class_counts_train": [170, 165, 165],
                                                "class_names": ["a", "b", "c"], "scaled": "standard"}),
        "final": kw.pop("final", {"test_metrics": {"accuracy": 0.9}}),
        "network_summary": {"total_params": 500},
    }
    run.update(kw)
    return run


def codes(findings):
    return {f["code"] for f in findings}


def test_overfitting_detected():
    m = [(0.80, 0.78), (0.90, 0.85), (0.95, 0.83), (0.98, 0.81), (0.995, 0.80), (0.999, 0.80)]
    f = diagnose(base_run(metrics=m))
    assert "OVERFITTING_GAP" in codes(f)
    gap = next(x for x in f if x["code"] == "OVERFITTING_GAP")
    assert any("Dropout" in s or "dropout" in s for s in gap["suggestions"])


def test_underfitting_detected():
    m = [(0.36, 0.35)] * 12
    ds = {"task": "classification", "n_train": 900, "n_classes": 3,
          "class_counts_train": [300, 300, 300], "class_names": ["a", "b", "c"], "scaled": "standard"}
    f = diagnose(base_run(metrics=m, data_summary=ds,
                          history=None) if False else base_run(metrics=m, data_summary=ds))
    assert "UNDERFITTING" in codes(f)


def test_nan_loss_dominates():
    hist = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.5, "train_metric": 0.5,
         "val_metric": 0.5, "metric_name": "accuracy"},
        {"epoch": 2, "train_loss": math.nan, "val_loss": math.nan, "train_metric": 0.1,
         "val_metric": 0.1, "metric_name": "accuracy"},
    ]
    f = diagnose(base_run(history=hist))
    assert codes(f) == {"NAN_LOSS"}


def test_vanishing_gradients():
    snaps = [{"epoch": 10, "activations": [], "gradients": {"layers": [
        {"layer_id": "h1", "label": "h1", "w_grad_norm": 1e-9, "w_grad_mean_abs": 1e-10,
         "w_grad_max_abs": 1e-9, "b_grad_norm": 1e-9, "example_weight": {"index": [0, 0], "w": 0.1, "grad": 1e-9}},
        {"layer_id": "out", "label": "out", "w_grad_norm": 0.5, "w_grad_mean_abs": 0.01,
         "w_grad_max_abs": 0.1, "b_grad_norm": 0.1, "example_weight": {"index": [0, 0], "w": 0.1, "grad": 0.01}},
    ], "total_norm": 0.5, "loss_on_batch": 0.5, "input_sample": []}}]
    f = diagnose(base_run(snapshots=snaps))
    assert "GRAD_VANISH" in codes(f)


def test_exploding_gradients():
    snaps = [{"epoch": 10, "activations": [], "gradients": {"layers": [
        {"layer_id": "h1", "label": "h1", "w_grad_norm": 5000.0, "w_grad_mean_abs": 100,
         "w_grad_max_abs": 900, "b_grad_norm": 10, "example_weight": {"index": [0, 0], "w": 0.1, "grad": 900}},
    ], "total_norm": 5000, "loss_on_batch": 10, "input_sample": []}}]
    f = diagnose(base_run(snapshots=snaps))
    assert "GRAD_EXPLODE" in codes(f)


def test_dead_neurons_and_saturation():
    snaps = [{"epoch": 5, "gradients": {"layers": [], "total_norm": 0.1, "loss_on_batch": 0.4,
                                        "input_sample": []},
              "activations": [
                  {"layer_id": "h1", "label": "Hidden 1", "activation": "relu", "mean": 0.01,
                   "std": 0.01, "min": 0, "max": 0.2, "frac_near_zero": 0.9,
                   "frac_saturated": None, "dead_neuron_frac": 0.75, "sample": []},
                  {"layer_id": "h2", "label": "Hidden 2", "activation": "sigmoid", "mean": 0.99,
                   "std": 0.005, "min": 0.95, "max": 0.999, "frac_near_zero": 0.0,
                   "frac_saturated": 0.6, "dead_neuron_frac": None, "sample": []}]}]
    cs = codes(diagnose(base_run(snapshots=snaps)))
    assert "DEAD_NEURONS" in cs and "SATURATION" in cs


def test_class_imbalance():
    ds = {"task": "classification", "n_train": 1000, "n_classes": 2,
          "class_counts_train": [950, 50], "class_names": ["neg", "pos"], "scaled": "standard"}
    assert "CLASS_IMBALANCE" in codes(diagnose(base_run(data_summary=ds)))


def test_unscaled_features_warning():
    ds = {"task": "classification", "n_train": 500, "n_classes": 2,
          "class_counts_train": [250, 250], "scaled": "none",
          "raw_feature_stds": [0.01, 100.0]}
    assert "FEATURE_SCALING" in codes(diagnose(base_run(data_summary=ds)))


def test_healthy_run_gives_ok_finding():
    m = [(0.7, 0.7), (0.8, 0.8), (0.85, 0.85), (0.88, 0.87), (0.9, 0.89), (0.91, 0.9)]
    hist = [
        {"epoch": i + 1, "train_loss": 0.7 - 0.08 * i, "val_loss": 0.72 - 0.075 * i,
         "train_metric": a, "val_metric": b, "metric_name": "accuracy"}
        for i, (a, b) in enumerate(m)
    ]
    f = diagnose(base_run(history=hist))
    assert "HEALTHY" in codes(f)
    healthy = next(x for x in f if x["code"] == "HEALTHY")
    assert healthy["severity"] == "ok"


def test_no_history():
    f = diagnose({"history": [], "snapshots": [], "data_summary": {}, "final": {}, "network_summary": {}})
    assert codes(f) == {"NO_DATA"}


# ---------------------------------------------------------------------------
# CLASS_IMBALANCE: names/counts pairing
#
# `class_counts_train` is produced by np.bincount(y_train, minlength=n_classes)
# and `class_names` by the dataset loader, so the two can legitimately differ in
# length. A bare zip() would silently truncate and report a distribution that
# disagrees with the recorded numbers.
# ---------------------------------------------------------------------------

def _imbalance_finding(ds):
    f = next(x for x in diagnose(base_run(data_summary=ds)) if x["code"] == "CLASS_IMBALANCE")
    return f


def test_class_imbalance_pairs_names_with_counts():
    ds = {"task": "classification", "n_train": 500, "n_classes": 3,
          "class_counts_train": [470, 20, 10], "class_names": ["a", "b", "c"],
          "scaled": "standard"}
    f = _imbalance_finding(ds)
    # Equal lengths: real names are used, and every count is reported.
    assert f["evidence"]["counts"] == {"a": 470, "b": 20, "c": 10}
    assert len(f["evidence"]["counts"]) == 3
    assert "'a': 470" in f["explanation"]


def test_class_imbalance_mismatched_lengths_do_not_truncate():
    """More names than counts (and vice versa) must not silently mislabel data.

    The finding is still emitted, but the payload must describe exactly the
    recorded counts — falling back to positional labels rather than pairing a
    name with the wrong count.
    """
    # 2 names, 4 counts: a bare zip() would drop counts[2:] entirely.
    ds = {"task": "classification", "n_train": 500, "n_classes": 4,
          "class_counts_train": [970, 15, 10, 5], "class_names": ["a", "b"],
          "scaled": "standard"}
    f = _imbalance_finding(ds)
    counts = f["evidence"]["counts"]
    # All four recorded counts survive; none is dropped.
    assert sorted(counts.values()) == [5, 10, 15, 970]
    assert len(counts) == 4
    # The truncated names are discarded in favour of positional labels, so no
    # name is left paired with a count it may not belong to.
    assert "a" not in counts and "b" not in counts


def test_class_imbalance_more_names_than_counts():
    """4 names, 2 counts: must not invent counts or reuse names wrongly."""
    ds = {"task": "classification", "n_train": 500, "n_classes": 2,
          "class_counts_train": [480, 20], "class_names": ["a", "b", "c", "d"],
          "scaled": "standard"}
    f = _imbalance_finding(ds)
    counts = f["evidence"]["counts"]
    assert sorted(counts.values()) == [20, 480]
    assert len(counts) == 2


def test_class_imbalance_missing_names_falls_back_to_positions():
    """No class_names recorded: positional labels, still complete."""
    ds = {"task": "classification", "n_train": 500, "n_classes": 3,
          "class_counts_train": [300, 100, 5], "class_names": None,
          "scaled": "standard"}
    f = _imbalance_finding(ds)
    counts = f["evidence"]["counts"]
    assert len(counts) == 3
    assert sorted(counts.values()) == [5, 100, 300]
