"""Built-in educational datasets + user CSV loading and preprocessing."""
from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import settings
from ..models.schemas import DatasetRef, DatasetSummary

MAX_PREVIEW_ROWS = 8
MAX_ROWS = 200_000
MAX_COLS = 500


@dataclass
class PreparedData:
    """Fully-preprocessed tensors + everything needed to reproduce the
    transform at prediction time."""
    task: str                                    # classification | regression
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int | None
    class_names: list[str] | None
    feature_names_in: list[str]                  # raw columns the user picked
    feature_names_model: list[str]               # after one-hot encoding
    meta: dict = field(default_factory=dict)     # serialisable transform metadata


# --------------------------------------------------------------------------
# Built-in datasets
# --------------------------------------------------------------------------

def _logic_gate(op: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    if op == "xor":
        y = (X[:, 0] != X[:, 1]).astype(int)
    elif op == "and":
        y = ((X[:, 0] == 1) & (X[:, 1] == 1)).astype(int)
    else:
        y = ((X[:, 0] == 1) | (X[:, 1] == 1)).astype(int)
    return X, y


def _synthetic_shapes(n_per_class: int = 200, size: int = 8, seed: int = 42):
    """Tiny 8×8 synthetic *image* dataset: vertical line / horizontal line / box.

    Real image data at a size a browser lab can train on in seconds. The rows
    are stored flat (64 columns) exactly like the tabular datasets, and the
    network's Input layer declares the 8×8×1 shape that folds them back into an
    image tensor — so the whole existing preprocessing pipeline is reused
    unchanged.
    """
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    m = size // 2
    for cls in range(3):
        for _ in range(n_per_class):
            img = np.zeros((size, size), dtype=np.float32)
            if cls == 0:            # vertical bar
                img[:, m - 1:m + 1] = 1.0
            elif cls == 1:          # horizontal bar
                img[m - 1:m + 1, :] = 1.0
            else:                   # hollow square
                img[1:-1, 1:-1] = 1.0
                img[2:-2, 2:-2] = 0.0
            # random shift (roll) + noise so the classes are not pixel-identical
            img = np.roll(img, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), axis=(0, 1))
            img = np.clip(img + rng.normal(0, 0.08, img.shape).astype(np.float32), 0, 1)
            imgs.append(img.reshape(-1))
            labels.append(cls)
    X = np.stack(imgs).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    # shuffle so classes are interleaved
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _load_builtin_raw(name: str, seed: int = 42) -> tuple[pd.DataFrame, str, str, list[str] | None]:
    """Return (df, target_column, task, class_names)."""
    rng = np.random.default_rng(seed)
    if name in ("xor", "and", "or"):
        X, y = _logic_gate(name)
        # replicate with jitter so there are enough samples to batch
        reps = 64
        X = np.tile(X, (reps, 1)) + rng.normal(0, 0.03, (reps * 4, 2)).astype(np.float32)
        X = np.clip(X, -0.2, 1.2)
        y = np.tile(y, reps)
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y
        desc_names = ["0", "1"]
        return df, "target", "classification", desc_names
    if name == "iris":
        d = skds.load_iris()
        df = pd.DataFrame(d.data, columns=[c.replace(" (cm)", "").replace(" ", "_") for c in d.feature_names])
        df["target"] = d.target
        return df, "target", "classification", list(d.target_names)
    if name == "wine":
        d = skds.load_wine()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
        return df, "target", "classification", list(d.target_names)
    if name == "breast_cancer":
        d = skds.load_breast_cancer()
        df = pd.DataFrame(d.data, columns=[c.replace(" ", "_") for c in d.feature_names])
        df["target"] = d.target
        return df, "target", "classification", list(d.target_names)
    if name == "moons":
        X, y = skds.make_moons(n_samples=600, noise=0.25, random_state=seed)
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y
        return df, "target", "classification", ["0", "1"]
    if name == "circles":
        X, y = skds.make_circles(n_samples=600, noise=0.12, factor=0.5, random_state=seed)
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y
        return df, "target", "classification", ["0", "1"]
    if name == "synthetic_classification":
        X, y = skds.make_classification(
            n_samples=800, n_features=6, n_informative=4, n_redundant=1,
            n_classes=3, random_state=seed)
        df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
        df["target"] = y
        return df, "target", "classification", ["0", "1", "2"]
    if name == "synthetic_regression":
        X, y = skds.make_regression(n_samples=500, n_features=4, noise=8.0, random_state=seed)
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
        df["target"] = y
        return df, "target", "regression", None
    if name == "linear_regression":
        X = rng.uniform(-3, 3, (400, 2))
        y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + 0.7 + rng.normal(0, 0.3, 400)
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y
        return df, "target", "regression", None
    if name == "shapes8":
        X, y = _synthetic_shapes(seed=seed)
        df = pd.DataFrame(X, columns=[f"p{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df, "target", "classification", ["vertical", "horizontal", "box"]
    raise KeyError(f"Unknown builtin dataset '{name}'.")


def _bi(label: str, description: str) -> dict:
    """Helper for the BUILTIN_INFO table below."""
    return {"label": label, "description": description}


BUILTIN_INFO: dict[str, dict] = {
    "xor": _bi(
        "XOR Gate",
        "The classic non-linearly-separable problem: 2 inputs, output 1 when inputs "
        "differ. The historical reason hidden layers exist.",
    ),
    "and": _bi(
        "AND Gate",
        "Linearly separable logic gate (2 inputs → 1 output). A single neuron can "
        "learn this.",
    ),
    "or": _bi(
        "OR Gate",
        "Linearly separable logic gate (2 inputs → 1 output).",
    ),
    "iris": _bi(
        "Iris",
        "Fisher's 1936 flower dataset: 4 measurements classify 3 iris species. "
        "The 'hello world' of classification.",
    ),
    "wine": _bi(
        "Wine",
        "13 chemical measurements classify 3 wine cultivars grown in the same "
        "region of Italy.",
    ),
    "breast_cancer": _bi(
        "Breast Cancer (Wisconsin)",
        "30 features from cell nuclei images; binary malignant/benign diagnosis.",
    ),
    "moons": _bi(
        "Two Moons",
        "Two interleaving half-circles — a non-linear 2-D binary problem you can "
        "visualise.",
    ),
    "circles": _bi(
        "Concentric Circles",
        "Inner circle vs outer ring — needs a genuinely non-linear decision boundary.",
    ),
    "synthetic_classification": _bi(
        "Synthetic Classification (3-class)",
        "Randomly generated 6-feature, 3-class problem with informative + redundant "
        "features.",
    ),
    "synthetic_regression": _bi(
        "Synthetic Regression",
        "Random linear combination of 4 features plus noise — a generic regression task.",
    ),
    "linear_regression": _bi(
        "Linear Regression (manual)",
        "y = 2.5·x1 − 1.2·x2 + 0.7 + noise. Perfect for watching a network learn a "
        "line/hyperplane.",
    ),
    "shapes8": _bi(
        "Synthetic Shapes 8×8 (images)",
        "600 real 8×8 grayscale images in 3 classes — a vertical bar, a horizontal bar "
        "and a hollow square. Set the Input layer to 8 × 8 × 1 to train a CNN on it. "
        "Stored as 64 flat pixels, so the standard preprocessing pipeline applies "
        "unchanged.",
    ),
}


def builtin_summaries() -> list[DatasetSummary]:
    out: list[DatasetSummary] = []
    for name, info in BUILTIN_INFO.items():
        try:
            df, target, task, classes = _load_builtin_raw(name)
            feats = [c for c in df.columns if c != target]
            out.append(DatasetSummary(
                id=name, name=info["label"], kind="builtin", samples=len(df),
                n_features=len(feats), task=task,
                n_classes=(len(classes) if classes else None),
                description=f"{info['description']}  Samples: {len(df)} · Features: {len(feats)}" +
                            (f" · Classes: {len(classes)}" if classes else " · Regression"),
                target_name=target, feature_names=feats))
        except Exception as exc:  # pragma: no cover
            out.append(DatasetSummary(id=name, name=info["label"], kind="builtin", samples=0,
                                      n_features=0, task="classification", error=str(exc)))
    return out


# --------------------------------------------------------------------------
# CSV uploads
# --------------------------------------------------------------------------

def save_upload(content: bytes, filename: str) -> dict:
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"File exceeds the {settings.max_upload_mb} MB limit.")
    name_safe = Path(filename).name
    if not name_safe.lower().endswith(".csv"):
        raise ValueError("Only .csv files are supported.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV must contain at least one feature column and one target column.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLS:
        raise ValueError(f"CSV too large (max {MAX_ROWS} rows × {MAX_COLS} columns).")
    upload_id = uuid.uuid4().hex[:12]
    path = settings.upload_dir / f"{upload_id}.csv"
    df.to_csv(path, index=False)
    meta = _analyse_df(df, upload_id, name_safe)
    meta["original_name"] = name_safe
    (settings.upload_dir / f"{upload_id}.json").write_text(json.dumps(meta))
    return meta


def load_upload(upload_id: str) -> pd.DataFrame:
    path = settings.upload_dir / f"{upload_id}.csv"
    if not path.exists():
        raise KeyError("Upload not found.")
    return pd.read_csv(path)


def upload_meta(upload_id: str) -> dict:
    path = settings.upload_dir / f"{upload_id}.json"
    if not path.exists():
        # regenerate analysis if csv exists but meta lost
        df = load_upload(upload_id)
        return _analyse_df(df, upload_id, f"{upload_id}.csv")
    return json.loads(path.read_text())


def upload_summaries() -> list[dict]:
    out = []
    for f in sorted(settings.upload_dir.glob("*.json")):
        try:
            m = json.loads(f.read_text())
            out.append({
                "id": m["upload_id"], "name": m.get("original_name", m["upload_id"]),
                "kind": "upload", "samples": m["rows"], "n_features": max(m["columns"] - 1, 0),
                "task": m["suggested_task"], "n_classes": m.get("suggested_classes"),
                "target_name": m["suggested_target"],
                "description": f"Uploaded CSV · {m['rows']} rows × {m['columns']} cols · "
                               f"{m['numeric_columns']} numeric, {m['categorical_columns']} categorical, "
                               f"{m['total_missing']} missing values",
                "feature_names": m["feature_candidates"],
            })
        except Exception:
            continue
    return out


def _sample_value(v, is_num: bool):
    """Render one column value for the upload-preview table."""
    if pd.isna(v):
        return None
    if is_num and isinstance(v, (int, float, np.floating)):
        return float(v)
    return str(v)


def _analyse_df(df: pd.DataFrame, upload_id: str, filename: str) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        is_num = pd.api.types.is_numeric_dtype(s)
        cols.append({
            "name": str(c),
            "dtype": str(s.dtype),
            "numeric": bool(is_num),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "sample_values": [_sample_value(v, is_num) for v in s.head(4).tolist()],
        })
    target = str(df.columns[-1])
    suggested_task = _detect_task(df[target], len(df))
    return {
        "upload_id": upload_id,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": sum(1 for c in cols if c["numeric"]),
        "categorical_columns": sum(1 for c in cols if not c["numeric"]),
        "total_missing": int(df.isna().sum().sum()),
        "columns_detail": cols,
        "suggested_target": target,
        "suggested_task": suggested_task,
        "suggested_classes": int(df[target].nunique(dropna=True)) if suggested_task == "classification" else None,
        "feature_candidates": [str(c) for c in df.columns if c != target],
        "preview": df.head(MAX_PREVIEW_ROWS).where(pd.notna(df.head(MAX_PREVIEW_ROWS)), None).to_dict(orient="records"),
    }


def _detect_task(y: pd.Series, n_rows: int) -> str:
    """classification ⟺ non-numeric target, or few distinct values relative
    to the row count (a handful of distinct numbers in thousands of rows is a
    class label; 15 distinct numbers in 16 rows is regression)."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    uniques = int(y.nunique(dropna=True))
    threshold = min(20, max(2, int(0.2 * n_rows)))
    return "classification" if uniques <= threshold else "regression"


# --------------------------------------------------------------------------
# Preparation pipeline
# --------------------------------------------------------------------------

def _load_raw(ref: DatasetRef) -> tuple[pd.DataFrame, str | None, list[str] | None]:
    """Return (df, forced_target, builtin_class_names)."""
    if ref.kind == "builtin":
        df, target, _task, classes = _load_builtin_raw(ref.name or "xor")
        return df, target, classes
    df = load_upload(ref.upload_id or "")
    return df, None, None


def prepare(ref: DatasetRef) -> PreparedData:
    df, forced_target, builtin_classes = _load_raw(ref)
    target_col = forced_target or ref.target_column or str(df.columns[-1])
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    df = df.loc[df[target_col].notna()].reset_index(drop=True)
    feature_cols = ref.feature_columns or [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns selected.")
    X_df = df[feature_cols].copy()
    y_ser = df[target_col]

    # --- task detection ----------------------------------------------------
    pre = ref.preprocessing
    task = _detect_task(y_ser, len(df))

    # --- target ------------------------------------------------------------
    class_names: list[str] | None = None
    if task == "classification":
        classes = sorted(y_ser.dropna().unique().tolist(), key=lambda v: str(v))
        class_to_idx = {str(c): i for i, c in enumerate(classes)}
        class_names = [str(c) for c in classes]
        if builtin_classes and len(builtin_classes) == len(classes):
            class_names = builtin_classes
        y = y_ser.map(lambda v: class_to_idx[str(v)]).to_numpy(dtype=np.int64)
        n_classes = len(classes)
        target_meta = {"class_to_idx": class_to_idx}
    else:
        y_ser = pd.to_numeric(y_ser, errors="coerce")
        med = float(y_ser.median()) if y_ser.isna().any() else 0.0
        y = y_ser.fillna(med).to_numpy(dtype=np.float32).reshape(-1, 1)
        n_classes = None
        target_meta = {"imputed_median": med}

    # --- features ----------------------------------------------------------
    cat_maps: dict[str, list] = {}
    parts: list[pd.DataFrame] = []
    model_feature_names: list[str] = []
    for c in feature_cols:
        s = X_df[c]
        if pd.api.types.is_numeric_dtype(s):
            mean = float(s.mean()) if s.isna().any() else 0.0
            parts.append(s.fillna(mean).astype(np.float32).to_frame())
            model_feature_names.append(str(c))
            cat_maps[str(c)] = {"type": "numeric", "impute": mean}  # type: ignore[dict-item]
        else:
            vals = sorted(s.dropna().astype(str).unique().tolist())
            use_cats = vals + (["__missing__"] if "__missing__" not in vals else [])
            onehot = pd.get_dummies(
                pd.Categorical(s.astype(str).where(s.notna(), "__missing__"), categories=use_cats),
                prefix=str(c),
            ).astype(np.float32)
            parts.append(onehot)
            model_feature_names.extend(onehot.columns.tolist())
            # store the exact category list backing the one-hot width
            cat_maps[str(c)] = {"type": "categorical", "categories": use_cats}  # type: ignore[dict-item]
    X = pd.concat(parts, axis=1).to_numpy(dtype=np.float32)
    raw_stds = np.nanstd(X, axis=0)
    raw_feature_stds = [float(v) for v in np.nan_to_num(raw_stds, nan=1.0)]

    # --- splits --------------------------------------------------------------
    idx = np.arange(len(X))
    stratify = y if task == "classification" and min(np.bincount(y)) >= 2 else None
    i_train, i_tmp = train_test_split(idx, test_size=pre.test_split, random_state=pre.seed, stratify=stratify)
    val_rel = pre.val_split / max(1e-9, (1 - pre.test_split))
    if val_rel > 0 and len(i_tmp) > 1:
        strat2 = y[i_tmp] if stratify is not None and min(np.bincount(y[i_tmp])) >= 2 else None
        i_test, i_val = train_test_split(i_tmp, test_size=val_rel, random_state=pre.seed, stratify=strat2)
    else:
        i_test, i_val = i_tmp, np.array([], dtype=int)

    # --- scaling (fit on train only) ---------------------------------------
    scaler = None
    if pre.scale == "standard":
        scaler = StandardScaler().fit(X[i_train])
        scale_meta = {"kind": "standard", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    elif pre.scale == "minmax":
        scaler = MinMaxScaler().fit(X[i_train])
        scale_meta = {"kind": "minmax", "min": scaler.min_.tolist(), "scale": scaler.scale_.tolist()}
    else:
        scale_meta = {"kind": "none"}
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)

    meta = {
        "task": task,
        "feature_names_in": [str(c) for c in feature_cols],
        "feature_names_model": model_feature_names,
        "feature_transform": cat_maps,
        "scaler": scale_meta,
        "target": target_meta,
        "class_names": class_names,
        "target_column": target_col,
        "dataset_name": ref.name if ref.kind == "builtin" else (ref.upload_id or "upload"),
        "raw_feature_stds": raw_feature_stds,
    }
    return PreparedData(
        task=task,
        X_train=X[i_train], y_train=y[i_train],
        X_val=X[i_val] if len(i_val) else X[i_test][:0], y_val=y[i_val] if len(i_val) else y[i_test][:0],
        X_test=X[i_test], y_test=y[i_test],
        n_features=X.shape[1], n_classes=n_classes, class_names=class_names,
        feature_names_in=[str(c) for c in feature_cols],
        feature_names_model=model_feature_names, meta=meta,
    )


def transform_single(meta: dict, values) -> np.ndarray:
    """Apply a stored preprocessing pipeline to one raw input row.

    ``values`` may be a dict {raw_feature: value} or an ordered list matching
    meta['feature_names_in']."""
    names = meta["feature_names_in"]
    if isinstance(values, dict):
        row = [values.get(n) for n in names]
    else:
        row = list(values)
        if len(row) != len(names):
            raise ValueError(f"Expected {len(names)} feature values ({', '.join(names)}), got {len(row)}.")
    out: list[float] = []
    for name, v in zip(names, row):
        spec = meta["feature_transform"].get(name, {"type": "numeric", "impute": 0.0})
        if spec["type"] == "numeric":
            try:
                out.append(float(v) if v is not None and str(v) != "" else float(spec.get("impute", 0.0)))
            except (TypeError, ValueError):
                out.append(float(spec.get("impute", 0.0)))
        else:
            cats: list = spec["categories"]
            s = "__missing__" if v is None else str(v)
            for c in cats:
                out.append(1.0 if s == str(c) else 0.0)
            if s == "__missing__" and "__missing__" in cats:
                pass  # handled above via category match; unknown -> all zeros
    X = np.array(out, dtype=np.float32).reshape(1, -1)
    sc = meta["scaler"]
    if sc["kind"] == "standard":
        X = (X - np.array(sc["mean"], dtype=np.float32)) / np.array(sc["scale"], dtype=np.float32)
    elif sc["kind"] == "minmax":
        X = X * np.array(sc["scale"], dtype=np.float32) + np.array(sc["min"], dtype=np.float32)
    return X.astype(np.float32)
