# API Reference — NeuroSim Lab

Base URL: `http://localhost:8000`. Interactive Swagger UI: `/docs`.
Errors: `{"detail": "human-readable message"}` with proper HTTP status codes.

## Health
### `GET /api/health`
→ `{ status, name, version, llm_enabled }`

## Networks
### `POST /api/networks/validate`
Body: `NetworkSpec`. → `ValidationResult { ok, errors[], warnings[], layers[], input_dim, output_dim, total_params, order[] }`.
### `POST /api/networks/build`
Same + actually instantiates the PyTorch model (catches runtime issues) → adds `parameter_tensors`.
### `GET /api/networks/presets`
→ `{ presets[] }` — `xor-demo` (the "Try Example" preset) and `iris-research` (spec §37: 4→16→8→3, ReLU+softmax, Adam lr=0.001).

**NetworkSpec**
```jsonc
{
  "name": "My net",
  "layers": [
    {"id": "input", "kind": "input",  "params": {"features": 4},
     "position": {"x": 0, "y": 100}, "label": "Input"},
    {"id": "h1", "kind": "dense", "params": {"neurons": 16, "activation": "relu", "use_bias": true, "init": "he"}},
    {"id": "d1", "kind": "dropout", "params": {"dropout_rate": 0.3}},
    {"id": "out", "kind": "output", "params": {"neurons": 3, "activation": "softmax"}}
  ],
  "edges": [{"id": "e1", "source": "input", "target": "h1"}],
  "custom_activations": {"mishy": "x * tanh(log(1 + exp(x)))"}
}
```
`activation`: builtin id (`linear relu leaky_relu sigmoid tanh gelu elu selu swish softplus softmax`) or `custom::<name>`.

## Datasets
| method | path | notes |
|---|---|---|
| GET | `/api/datasets` | builtins + uploads with samples/features/classes/task/description |
| GET | `/api/datasets/builtin/{name}/preview` | column analysis + first rows |
| POST | `/api/datasets/upload` | multipart `file` (.csv, ≤ MAX_UPLOAD_MB) → analysis + `upload_id` |
| GET | `/api/datasets/upload/{upload_id}` | analysis again |
| DELETE | `/api/datasets/upload/{upload_id}` | |

**DatasetRef** (inside training requests)
```jsonc
{
  "kind": "builtin", "name": "iris",                 // or {"kind": "upload", "upload_id": "..."}
  "feature_columns": ["sepal_length", "petal_width"], // optional — default: all non-target
  "target_column": "target",                          // optional for uploads
  "preprocessing": {"scale": "standard", "test_split": 0.2, "val_split": 0.1, "seed": 42}
}
```

## Training
| method | path | notes |
|---|---|---|
| POST | `/api/training/start` | `TrainRequest` → `{job_id}`; 400 with clear message on incompatibility |
| POST | `/api/training/{job_id}/pause` / `resume` / `stop` | |
| GET | `/api/training/{job_id}/status` | state, epoch counter, final eval when done |
| GET | `/api/training/{job_id}/metrics` | full history + snapshots + final metrics (works for stored runs) |
| WS | `/ws/training/{job_id}` | replays then streams: `epoch`, `snapshot`, `status`, `done`, `error` |

**TrainingConfig**
```jsonc
{"epochs": 300, "batch_size": 16, "learning_rate": 0.05,
 "optimizer": "adam",            // sgd | momentum | adam | rmsprop
 "loss": "cross_entropy",        // cross_entropy | bce | mse | mae
 "l2": 0.0, "seed": 42, "shuffle": true, "snapshot_every": 15}
```
Add `"experiment_name"` to the request to archive the run under Research Mode.

## Prediction & inspection
| method | path | notes |
|---|---|---|
| POST | `/api/predict` | `{run_id, features}` → prediction, probabilities, top neurons, winning-neuron calc |
| POST | `/api/inspect/forward-trace` | per-layer per-neuron inputs/weights/z/a (real numbers) |
| POST | `/api/inspect/backprop` | `{run_id, features, target}` → one real update `w_new = w_old − η·∂L/∂w` |
| GET | `/api/inspect/weights/{run_id}` | per-layer matrices, bias, stats |

`features`: ordered list matching the run's raw input features, **or** an object `{feature_name: value}`. Raw (pre-scaling) values — the stored preprocessing pipeline is applied server-side.

## Custom activations
| method | path | notes |
|---|---|---|
| POST | `/api/activations/validate` | `{formula, lo?, hi?}` → `{ok, xs, ys, dys, stats, issues}` |
| GET | `/api/activations/library` | builtin catalogue + saved customs |
| POST | `/api/activations` | `{name, formula, notes}` — compiled before saving |
| DELETE | `/api/activations/{name}` | |

## AI Diagnosis
| method | path | notes |
|---|---|---|
| POST | `/api/diagnostics/analyze` | `{run_id, use_llm}` → findings (+ optional LLM explanation or graceful error) |
| GET | `/api/diagnostics/llm-status` | whether an LLM provider is configured |

## Experiments (Research Mode)
| method | path | notes |
|---|---|---|
| POST | `/api/experiments/run` | like training start but requires `experiment_name` |
| GET | `/api/experiments` | experiments with their archived runs |
| GET | `/api/experiments/{id}` | one experiment incl. config snapshot |
| GET | `/api/experiments/compare?run_ids=a,b,…` | 2–6 full run records with histories |
| GET | `/api/experiments/runs` | all runs (most recent first) |
| DELETE | `/api/experiments/{id}` | |

## Projects
| method | path | notes |
|---|---|---|
| POST | `/api/projects` | save `ProjectPayload` → `{project_id}` |
| GET | `/api/projects` / `/{id}` | list / fetch |
| DELETE | `/api/projects/{id}` | |
| POST | `/api/projects/export` | returns a downloadable `.nnsim` document |
| POST | `/api/projects/import` | body = `.nnsim` JSON → new project id |

`.nnsim` format: `{"format": "neurosim-lab/project", "version": 1, "payload": ProjectPayload}` where
`ProjectPayload = {name, network, dataset?, training_config?, run_id?, custom_activations{}}`.
