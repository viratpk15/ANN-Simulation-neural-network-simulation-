"""End-to-end training tests: real optimization on real data.

The convolutional-network cases (Conv/Pool/BatchNorm/Flatten/Activation) are at
the end of this file.
"""
import time

import pytest

from app.ml.trainer import TrainingJob
from app.models.schemas import TrainRequest

from .conftest import make_train_request, make_xor_spec


def wait_for(job, states=("finished", "stopped", "failed"), timeout=120.0):
    t0 = time.time()
    while job.state not in states:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job stuck in state {job.state}")
        time.sleep(0.05)
    return job.state


def test_xor_training_reaches_high_accuracy():
    job = TrainingJob(make_train_request())
    job.start()
    state = wait_for(job)
    assert state == "finished", job.error
    assert len(job.history) == 250
    acc = job.final["test_metrics"]["accuracy"]
    assert acc >= 0.9, f"XOR test accuracy too low: {acc}"
    # learning actually happened
    assert job.history[-1]["train_loss"] < job.history[0]["train_loss"]
    assert job.history[-1]["train_metric"] > 0.95
    # snapshots were captured
    assert job.snapshots
    snap = job.snapshots[-1]
    assert snap["activations"] and snap["gradients"]["layers"]
    assert snap["gradients"]["total_norm"] > 0


def test_determinism_same_seed():
    r1, r2 = make_train_request(), make_train_request()
    j1, j2 = TrainingJob(r1), TrainingJob(r2)
    j1.start()
    wait_for(j1)
    j2.start()
    wait_for(j2)  # sequential: same seed must reproduce identical runs
    assert j1.history[-1]["train_loss"] == pytest.approx(j2.history[-1]["train_loss"], rel=1e-6)


def test_pause_resume_stop():
    req = make_train_request()
    req.config.epochs = 2000
    job = TrainingJob(req)
    job.start()
    # wait until it's clearly running
    t0 = time.time()
    while job.state != "running" and time.time() - t0 < 10:
        time.sleep(0.02)
    while not job.history and time.time() - t0 < 10:
        time.sleep(0.02)
    job.pause()
    time.sleep(0.1)
    assert job.state == "paused"
    epoch_at_pause = job.history[-1]["epoch"]
    time.sleep(0.4)
    assert job.history[-1]["epoch"] == epoch_at_pause  # frozen while paused
    job.resume()
    time.sleep(0.1)
    assert job.state == "running"
    job.stop()
    assert wait_for(job, timeout=30) in ("stopped", "finished")
    assert job.history[-1]["epoch"] < 2000


def test_input_dim_mismatch_raises():
    req = make_train_request()
    req.network = make_xor_spec(features=5)  # xor data has 2 features
    job = TrainingJob(req)
    job.start()
    wait_for(job)
    assert job.state == "failed"
    assert "expects 5 features" in job.error


def test_wrong_loss_for_task():
    req = make_train_request()
    req.config.loss = "mse"
    job = TrainingJob(req)
    job.start()
    wait_for(job)
    assert job.state == "failed"
    assert "regression loss" in job.error


def test_regression_training():
    payload = {
        "network": {
            "name": "reg-test",
            "layers": [
                {"id": "input", "kind": "input", "params": {"features": 2}},
                {"id": "h1", "kind": "dense", "params": {"neurons": 16, "activation": "relu", "init": "he"}},
                {"id": "output", "kind": "output", "params": {"neurons": 1, "activation": "linear"}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "h1"},
                {"id": "e2", "source": "h1", "target": "output"},
            ],
        },
        "dataset": {"kind": "builtin", "name": "linear_regression",
                    "preprocessing": {"scale": "standard", "test_split": 0.2, "val_split": 0.2, "seed": 3}},
        "config": {"epochs": 200, "batch_size": 32, "learning_rate": 0.01,
                   "optimizer": "adam", "loss": "mse", "seed": 3, "snapshot_every": 50},
    }
    job = TrainingJob(TrainRequest.model_validate(payload))
    job.start()
    assert wait_for(job) == "finished", job.error
    r2 = job.final["test_metrics"]["r2"]
    assert r2 > 0.8, f"linear regression R2 too low: {r2}"


def test_bce_binary_training():
    payload = make_train_request().model_dump()
    payload["network"]["layers"][-1] = {"id": "output", "kind": "output",
                                        "params": {"neurons": 1, "activation": "sigmoid"}}
    payload["config"]["loss"] = "bce"
    job = TrainingJob(TrainRequest.model_validate(payload))
    job.start()
    assert wait_for(job) == "finished", job.error
    assert job.final["test_metrics"]["accuracy"] >= 0.9


# ---------------------------------------------------------------------------
# Convolutional networks train end to end
# (Conv2D / MaxPool / AvgPool / BatchNorm / Activation / Flatten)
# ---------------------------------------------------------------------------

CNN_IDS = ["input", "norm1", "conv", "norm2", "act", "mp", "ap", "flat",
           "hidden", "drop", "output"]
CNN_KINDS = ["input", "batchnorm", "conv2d", "batchnorm", "activation",
             "maxpool", "avgpool", "flatten", "dense", "dropout", "output"]


def make_cnn_request(epochs=30):
    from app.models.schemas import NetworkSpec
    network = NetworkSpec.model_validate({
        "name": "cnn-test",
        "layers": [
            {"id": i, "kind": k, "params": p}
            for i, k, p in zip(CNN_IDS, CNN_KINDS, [
                {"features": 64, "input_shape": [8, 8, 1]},
                {"momentum": 0.1, "eps": 1e-5},
                {"filters": 8, "kernel_size": 3, "stride": 1, "padding": 0,
                 "activation": "relu", "init": "he"},
                {},
                {"activation": "relu"},
                {"pool_size": 2, "pool_stride": 2},
                {"pool_size": 2, "pool_stride": 2},
                {},
                {"neurons": 16, "activation": "relu"},
                {"dropout_rate": 0.3},
                {"neurons": 3, "activation": "softmax"},
            ])
        ],
        "edges": [{"id": f"e{i}", "source": a, "target": b}
                  for i, (a, b) in enumerate(zip(CNN_IDS, CNN_IDS[1:]))],
    })
    return TrainRequest.model_validate({
        "network": network.model_dump(),
        "dataset": {"kind": "builtin", "name": "shapes8",
                    "preprocessing": {"scale": "none", "test_split": 0.2,
                                      "val_split": 0.2, "seed": 42}},
        "config": {"epochs": epochs, "batch_size": 32, "learning_rate": 0.01,
                   "optimizer": "adam", "loss": "cross_entropy", "seed": 42,
                   "snapshot_every": 10},
    })


def test_cnn_trains_on_image_data_and_generalises():
    job = TrainingJob(make_cnn_request())
    job.start()
    assert wait_for(job) == "finished", job.error
    assert job.final["test_metrics"]["accuracy"] >= 0.9, job.final["test_metrics"]
    # the loss really went down
    assert job.history[-1]["train_loss"] < job.history[0]["train_loss"]


def test_cnn_snapshots_cover_every_layer_and_trainable_layer():
    job = TrainingJob(make_cnn_request(epochs=10))
    job.start()
    assert wait_for(job) == "finished", job.error
    snap = job.snapshots[-1]
    # activation statistics exist for every layer in the chain
    assert {a["layer_id"] for a in snap["activations"]} == set(CNN_IDS)
    # gradient statistics exist for every trainable layer
    assert {g["layer_id"] for g in snap["gradients"]["layers"]} == {
        "norm1", "conv", "norm2", "hidden", "output"}
    assert snap["gradients"]["total_norm"] > 0
    # weights of every trainable layer are exported for the Weights tab
    assert set(job.final["weights"]) == {"norm1", "conv", "norm2", "hidden", "output"}
    assert job.final["weights"]["conv"]["kind"] == "conv"
    assert job.final["weights"]["norm1"]["kind"] == "batchnorm"


def test_cnn_architecture_summary_reports_shapes_and_params():
    from app.services import storage, training_manager
    storage.init_db()
    job = TrainingJob(make_cnn_request(epochs=5))
    job.start()
    assert wait_for(job) == "finished", job.error
    # the persisted record keeps the full per-layer shape table
    rec = training_manager.finalize_and_store(job, "cnn-shapes-test")
    table = {s["id"]: s for s in rec["network_summary"]["layer_shapes"]}
    assert table["conv"]["in_shape"] == [8, 8, 1]
    assert table["conv"]["out_shape"] == [6, 6, 8]
    assert table["flat"]["out_shape"] == [8]
    assert rec["network_summary"]["total_params"] > 0


def test_cnn_rejects_a_dataset_whose_width_does_not_match_the_input():
    """A CNN whose Input layer expects 64 pixels must refuse a 4-feature dataset."""
    payload = make_cnn_request(epochs=1).model_dump()
    payload["dataset"] = {"kind": "builtin", "name": "iris",
                          "preprocessing": {"scale": "standard", "test_split": 0.2,
                                             "val_split": 0.2, "seed": 1}}
    job = TrainingJob(TrainRequest.model_validate(payload))
    job.start()
    assert wait_for(job) == "failed"
    assert "4 features" in (job.error or "")


def test_cnn_refuses_to_train_when_validation_fails():
    """Training must not start on an invalid architecture."""
    payload = make_cnn_request(epochs=1).model_dump()
    # drop the Flatten so a Dense layer receives a 3-D tensor
    payload["network"]["layers"] = [l for l in payload["network"]["layers"]
                                    if l["id"] != "flat"]
    payload["network"]["edges"] = [e for e in payload["network"]["edges"]
                                   if e["target"] != "flat"]
    job = TrainingJob(TrainRequest.model_validate(payload))
    job.start()
    assert wait_for(job) == "failed"
    assert "validation failed" in (job.error or "").lower()

