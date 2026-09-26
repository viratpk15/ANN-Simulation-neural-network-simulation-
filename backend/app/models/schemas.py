"""Pydantic request/response schemas shared across the API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Network definition
# --------------------------------------------------------------------------

# Layer kinds. The four original kinds (input/dense/dropout/output) keep their
# exact meaning and default parameters so v1 projects load unchanged; every
# later kind is additive.
LayerKind = Literal[
    "input",
    "dense",
    "activation",
    "dropout",
    "batchnorm",
    "flatten",
    "conv2d",
    "maxpool",
    "avgpool",
    "globalavgpool",
    "output",
]
ActivationName = str  # builtin id or custom::<name>

CURRENT_SCHEMA_VERSION = 2

# Kinds that operate on a feature *vector* (rank-1) only.
VECTOR_ONLY_KINDS = {"dense", "output"}
# Kinds that require a 3-D image tensor (N, C, H, W).
IMAGE_ONLY_KINDS = {"conv2d", "maxpool", "avgpool", "globalavgpool"}
# Kinds with no trainable parameters.
PARAMETERLESS_KINDS = {
    "input", "activation", "dropout", "flatten",
    "maxpool", "avgpool", "globalavgpool",
}
TRAINABLE_KINDS = {"dense", "output", "conv2d", "batchnorm"}


class LayerParams(BaseModel):
    # --- original v1 parameters (unchanged defaults) ----------------------
    features: Optional[int] = None            # input layer
    neurons: Optional[int] = None             # dense/output
    activation: ActivationName = "relu"       # dense; output often softmax/sigmoid/linear
    use_bias: bool = True
    init: Literal["xavier", "he", "normal", "uniform"] = "xavier"
    dropout_rate: float = 0.5
    dropout_mode: Literal["train_only", "always"] = "train_only"

    # --- input layer: optional 3-D image shape -----------------------------
    # [H, W, C] — channels last, the convention used in the UI and the docs.
    # When set, the flat feature row is reshaped into a (C, H, W) tensor so
    # convolutional layers can consume it. features must equal H*W*C.
    input_shape: Optional[list[int]] = None

    # --- batchnorm ---------------------------------------------------------
    momentum: float = 0.1
    eps: float = 1e-5
    affine: bool = True

    # --- conv2d ------------------------------------------------------------
    filters: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    padding_mode: Literal["valid", "same"] = "valid"

    # --- pooling -----------------------------------------------------------
    pool_size: int = 2
    pool_stride: int = 2
    pool_padding: int = 0

    # --- output ------------------------------------------------------------
    task: Literal["auto", "classification", "regression", "binary"] = "auto"


class LayerSpec(BaseModel):
    id: str
    kind: LayerKind
    params: LayerParams = Field(default_factory=LayerParams)
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    label: Optional[str] = None


class EdgeSpec(BaseModel):
    id: str
    source: str
    target: str


class NetworkSpec(BaseModel):
    name: str = "Untitled Network"
    layers: list[LayerSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    custom_activations: dict[str, str] = Field(default_factory=dict)  # name -> formula
    # Schema version. Documents written by v1 omit it and are migrated to v2
    # on load (the migration is additive: new layer params have defaults).
    version: int = 1

    @model_validator(mode="after")
    def _migrate(self) -> "NetworkSpec":
        if self.version < CURRENT_SCHEMA_VERSION:
            # v1 -> v2: purely additive. Every new LayerParams field has a
            # default, so existing input/dense/dropout/output projects are
            # valid v2 networks with no data loss.
            self.version = CURRENT_SCHEMA_VERSION
        return self


class LayerShape(BaseModel):
    """Per-layer shape/parameter report.

    ``in_features``/``out_features`` are kept (vector width, or the total
    element count for image tensors) because the original UI and saved run
    records depend on them. ``in_shape``/``out_shape`` carry the full rank-1
    or rank-3 shape for the educational shape display.
    """
    id: str
    kind: LayerKind
    in_features: Optional[int]
    out_features: Optional[int]
    params: int
    in_shape: Optional[list[int]] = None
    out_shape: Optional[list[int]] = None
    note: Optional[str] = None


class ValidationResult(BaseModel):
    ok: bool
    errors: list[dict] = Field(default_factory=list)
    warnings: list[dict] = Field(default_factory=list)
    layers: list[LayerShape] = Field(default_factory=list)
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    total_params: int = 0
    order: list[str] = Field(default_factory=list)  # layer ids in data-flow order
    # True when the chain contains at least one image (3-D) tensor, which the
    # trainer uses to decide whether the dataset must be image-shaped.
    is_cnn: bool = False


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

class DatasetSummary(BaseModel):
    id: str
    name: str
    kind: Literal["builtin", "upload"]
    samples: int
    n_features: int
    task: str
    n_classes: Optional[int] = None
    description: str = ""
    target_name: str = "target"
    feature_names: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class Preprocessing(BaseModel):
    scale: Literal["none", "standard", "minmax"] = "standard"
    test_split: float = 0.2
    val_split: float = 0.1
    seed: int = 42


class DatasetRef(BaseModel):
    """Which dataset to train on + exactly how to prepare it."""
    kind: Literal["builtin", "upload"] = "builtin"
    name: Optional[str] = None               # builtin name
    upload_id: Optional[str] = None          # uploaded csv id
    feature_columns: Optional[list[str]] = None   # raw columns (pre one-hot); None = auto numeric
    target_column: Optional[str] = None      # None = auto (last col)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: Literal["sgd", "momentum", "adam", "rmsprop"] = "adam"
    loss: Literal["cross_entropy", "bce", "mse", "mae"] = "cross_entropy"
    l2: float = 0.0
    seed: int = 42
    shuffle: bool = True
    snapshot_every: int = 5    # epochs between activation/gradient snapshots


class TrainRequest(BaseModel):
    network: NetworkSpec
    dataset: DatasetRef
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    experiment_name: Optional[str] = None
    project_id: Optional[str] = None


class PredictRequest(BaseModel):
    run_id: str
    features: list[float | str] | dict[str, float | str]


class ActivationValidateRequest(BaseModel):
    formula: str
    lo: float = -5.0
    hi: float = 5.0


class SavedActivation(BaseModel):
    name: str
    formula: str
    notes: str = ""


class ProjectPayload(BaseModel):
    name: str
    network: NetworkSpec
    dataset: Optional[DatasetRef] = None
    training_config: Optional[TrainingConfig] = None
    run_id: Optional[str] = None          # last known weights to restore
    custom_activations: dict[str, str] = Field(default_factory=dict)


class DiagnoseRequest(BaseModel):
    run_id: str
    use_llm: bool = True
