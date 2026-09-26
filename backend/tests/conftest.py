"""Test environment: isolate all persistence into a temp dir *before* the app
package reads its configuration."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="neurosim-test-")
os.environ["DATABASE_PATH"] = os.path.join(_TMP, "test.db")
os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")
os.environ["RUNS_DIR"] = os.path.join(_TMP, "runs")
os.environ["LLM_PROVIDER"] = "none"
os.environ["GROQ_API_KEY"] = ""
os.environ["NVIDIA_API_KEY"] = ""
os.environ["NVIDIA_NIM_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_xor_spec(features=2):
    from app.models.schemas import NetworkSpec
    return NetworkSpec.model_validate({
        "name": "xor-test",
        "layers": [
            {"id": "input", "kind": "input", "params": {"features": features}},
            {"id": "h1", "kind": "dense", "params": {"neurons": 8, "activation": "tanh", "init": "xavier"}},
            {"id": "h2", "kind": "dense", "params": {"neurons": 8, "activation": "tanh", "init": "xavier"}},
            {"id": "output", "kind": "output", "params": {"neurons": 2, "activation": "softmax"}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "h1"},
            {"id": "e2", "source": "h1", "target": "h2"},
            {"id": "e3", "source": "h2", "target": "output"},
        ],
    })


def make_train_request(**over):
    from app.models.schemas import TrainRequest
    payload = {
        "network": make_xor_spec().model_dump(),
        "dataset": {"kind": "builtin", "name": "xor",
                    "preprocessing": {"scale": "none", "test_split": 0.2, "val_split": 0.2, "seed": 42}},
        "config": {"epochs": 250, "batch_size": 16, "learning_rate": 0.05,
                   "optimizer": "adam", "loss": "cross_entropy", "seed": 42, "snapshot_every": 50},
    }
    for k, v in over.items():
        payload[k] = v
    return TrainRequest.model_validate(payload)
