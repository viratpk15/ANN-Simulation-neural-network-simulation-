# Architecture — NeuroSim Lab

## High level

```
┌────────────────────────────┐        REST + WebSocket         ┌──────────────────────────────┐
│  Frontend (React 18 + TS)  │  ───────── /api/* ───────────▶  │  Backend (FastAPI)           │
│  Vite dev proxy :5173      │  ◀──────── JSON ────────────    │  /ws/training/<job> :8000    │
│                            │  ◀──────── WS events ─────────  │                              │
│  store: zustand            │                                 │  app/ml        PyTorch engine │
│  canvas: React Flow        │                                 │  app/services  job manager    │
│  charts: Recharts          │                                 │  app/diagnostics rule engine  │
│  math:   KaTeX             │                                 │  SQLite   runs/projects/exps  │
└────────────────────────────┘                                 └──────────────────────────────┘
```

The backend can also serve the built SPA from the same port (production mode).

## Backend modules

| module | responsibility |
|---|---|
| `app/config.py` | env-driven settings; every variable optional |
| `app/models/schemas.py` | Pydantic contracts shared by API, ML and storage |
| `app/ml/builder.py` | NetworkSpec validation (structure, cycles, chain, shapes, params), model construction, weight extraction |
| `app/ml/expressions.py` | **safe** expression compiler: AST whitelist → torch ops; analysis (grid eval + autograd derivative + NaN/Inf/health) |
| `app/ml/activations.py` | builtin registry + `custom::<name>` expression-backed modules |
| `app/ml/datasets.py` | builtin generators, CSV validation/analysis, preprocessing (impute, one-hot, scale, stratified splits), single-row transform for prediction |
| `app/ml/trainer.py` | `TrainingJob` thread: real training loop, pause/resume/stop events, epoch metrics, activation+gradient snapshots, final test evaluation |
| `app/services/training_manager.py` | job registry, idempotent persistence of finished runs (weights → `data/runs/<id>.pt`, record → SQLite) |
| `app/services/storage.py` | thin sqlite3 layer: projects, runs, experiments, activations |
| `app/simulation/forward_trace.py` | manual per-neuron forward pass (numpy) from real parameters; one-step real backprop example |
| `app/ml/shapes.py` | layer catalogue (single source of truth for the palette), tensor-shape arithmetic, conv/pool output sizes, parameter-count formulas |
| `app/diagnostics/engine.py` | deterministic findings from recorded numbers only |
| `app/llm/` | OPTIONAL explanation layer: `LLMProvider` abstraction, Groq/NVIDIA NIM/OpenRouter/Ollama implementations, and the bounded fallback manager. Never blocks or fabricates. |
| `app/diagnostics/llm.py` | thin adapter: builds the structured payload and delegates to `app/llm` |
| `app/api/*` | routers; `training.py` also hosts the WebSocket streamer (replay + live poll) |

## Request flows

**Training.** `POST /api/training/start` → synchronous *preflight* (network
validation + dataset preparation + input/output/loss compatibility) →
`TrainingJob` thread. Every epoch emits an `epoch` event; every *N* epochs a
`snapshot` (real activation stats + a real backward pass for gradient stats).
The WS endpoint replays in-memory events to late joiners, then polls every
250 ms. Terminal states trigger idempotent storage of the run + weights.

**Inspection.** Endpoints load the persisted run, rebuild the `ChainModel`
from the stored spec, load `state_dict`, and recompute — so traces, weight
matrices and backprop examples always reflect exactly what was learned.

**Diagnosis.** `engine.diagnose(run)` consumes only stored numbers and
returns structured findings (`severity`, `evidence`, `suggestions`). If
enabled, the LLM receives *those findings + a small context summary* and
merely rephrases them.

## Feed-forward chain constraint (v1) and extension points

v1 deliberately supports one input → one output chains with in/out-degree ≤ 1
(plus Dropout). Guarantees this buys: exact static shape inference, unambiguous
per-neuron math rendering, and simple gradient attribution to layers.

To add Conv/RNN/branching later:
1. extend `LayerKind` + params in `schemas.py`,
2. teach `builder.validate_network` a shape rule & relax the degree checks with
   explicit merge semantics (`concat`/`add`),
3. add module construction in `build_model`,
4. extend `forward_trace.forward_trace` with per-layer rendering metadata
   (the trace format already supports heterogeneous `kind`s).

## Security notes

- Custom formulas: restricted AST whitelist; no `eval`, no builtins access,
  no attributes/subscripts/imports. Verified by `tests/test_expressions.py`
  (generic injection payloads rejected) and an e2e check that nothing executes.
- Uploads: `.csv` only, size-limited (`MAX_UPLOAD_MB`, default 10), row/column
  capped, parsed by pandas inside a guard.
- LLM keys live only in backend env; the frontend never sees them.
- CORS allow-list is env-configured; the frontend uses same-origin requests.

## Performance choices

- Threads (not processes) for training → shared memory, cheap events; fine for
  educational-scale models. Guard rails: `MAX_EPOCHS`, neuron caps, ≤3 concurrent jobs.
- Per-neuron traces are capped (`max_visual_neurons = 16`), weight matrices
  downsampled for display beyond 4096 elements.
- Frontend sends full spec JSON (small); charts use `isAnimationActive={false}`
  for smooth 300+ point updates.
