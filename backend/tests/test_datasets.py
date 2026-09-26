"""Dataset loading/preprocessing tests."""

import numpy as np
import pytest

from app.ml import datasets as ds
from app.models.schemas import DatasetRef

CSV = """height,weight,city,label
180,75,London,A
165,60,,B
,80,Paris,A
172,,London,B
190,95,Paris,A
166,58,London,B
177,72,Paris,A
169,63,London,B
182,81,Paris,A
161,55,London,B
"""


def ref(name, **pre):
    p = {"scale": "standard", "test_split": 0.2, "val_split": 0.2, "seed": 1}
    p.update(pre)
    return DatasetRef(kind="builtin", name=name, preprocessing=p)


def test_builtin_summaries_complete():
    sums = ds.builtin_summaries()
    names = {s.id for s in sums}
    assert {"xor", "and", "or", "iris", "wine", "breast_cancer", "moons", "circles",
            "synthetic_classification", "synthetic_regression", "linear_regression"} <= names
    iris = next(s for s in sums if s.id == "iris")
    assert iris.samples == 150 and iris.n_features == 4 and iris.task == "classification"
    assert "Classes: 3" in iris.description


def test_prepare_xor():
    data = ds.prepare(ref("xor", scale="none"))
    assert data.task == "classification"
    assert data.n_classes == 2
    assert data.n_features == 2
    assert len(data.X_train) + len(data.X_test) + len(data.X_val) > 200


def test_prepare_iris_standardized():
    data = ds.prepare(ref("iris"))
    assert data.n_classes == 3
    # standardized on train => column means near 0
    assert np.allclose(data.X_train.mean(axis=0), 0, atol=0.2)


def test_prepare_regression():
    data = ds.prepare(ref("synthetic_regression"))
    assert data.task == "regression"
    assert data.y_train.ndim == 2 and data.y_train.shape[1] == 1


def test_unknown_builtin_raises():
    with pytest.raises(KeyError):
        ds.prepare(ref("nope"))


def test_csv_upload_categorical_and_missing(tmp_path):
    meta = ds.save_upload(CSV.encode(), "toys.csv")
    assert meta["rows"] == 10
    assert meta["total_missing"] == 3  # 1 city + 1 height + 1 weight
    assert meta["suggested_target"] == "label"
    prepared = ds.prepare(DatasetRef(
        kind="upload", upload_id=meta["upload_id"],
        feature_columns=["height", "weight", "city"], target_column="label",
        preprocessing={"scale": "none", "test_split": 0.2, "val_split": 0.2, "seed": 1},
    ))
    # 2 numeric + one-hot city (London, Paris + explicit "__missing__" indicator)
    assert prepared.n_features == 2 + 3
    assert not np.isnan(prepared.X_train).any()
    # prediction-time transform handles unknown categories without crashing
    x = ds.transform_single(prepared.meta, {"height": 175, "weight": 70, "city": "Berlin"})
    assert x.shape == (1, prepared.n_features)


def test_reject_non_csv():
    with pytest.raises(ValueError):
        ds.save_upload(b"hello world not a csv at all", "evil.txt")


def test_transform_single_bad_length():
    data = ds.prepare(ref("xor", scale="none"))
    with pytest.raises(ValueError):
        ds.transform_single(data.meta, [1.0, 2.0, 3.0])
