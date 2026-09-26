# 🧠 NeuroSim Lab

**Interactive Neural Network Simulation, Visualization & AI Diagnosis Platform**

A web-based neural-network *laboratory* for students, teachers and researchers.
Build a network visually (drag → drop → connect), train it on real datasets,
watch the forward pass neuron-by-neuron, inspect the actual mathematics
(`z = Σ wᵢxᵢ + b`, `a = f(z)`), examine weights / activations / gradients,
run predictions, get rule-based **AI diagnosis** of training problems, define
your own **custom activation functions** safely, and run comparable
**research experiments** — all real computation, no mock numbers.

Built with **React 18 + TypeScript + Vite + Tailwind + React Flow + Recharts + KaTeX**
(frontend) and **FastAPI + PyTorch + NumPy + pandas + scikit-learn** (backend).

---

## Quick start

### Requirements
- **Python ≥ 3.10** (3.11/3.12/3.13 tested)
- **Node.js ≥ 18** (Node 20 LTS tested) + npm
- ~500 MB disk for Python deps (PyTorch CPU), ~250 MB for npm deps

### 1 — Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs (auto-generated): http://localhost:8000/docs

### 2 — Frontend (dev mode)
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api + /ws to :8000)
```

### Single-port production mode
```bash
cd frontend && npm run build   # outputs frontend/dist
# the backend automatically serves dist/ → open http://localhost:8000
```

### 3 — Try the demo
Click **⚡ Try Example** in the top bar → loads the XOR demo network + dataset +
training config → press **▶ Train** → explore the tabs:
Training → Simulation → Math → Weights → Activations → Gradients → Predict → AI Diagnosis.

---

## What’s inside

| Area | Highlights |
|---|---|
| Visual builder | React Flow canvas: drag/drop Input·Dense·Dropout·Output, connect, multi-select, delete, live architecture validation with exact shapes & parameter counts |
| Datasets | 11 built-ins (XOR/AND/OR/Iris/Wine/BreastCancer/moons/circles/synthetic/linear…), CSV upload with column-type & missing-value analysis, one-hot encoding, imputation, standard/min-max scaling, stratified train/val/test splits |
| Training | Background threads + WebSocket live stream; epochs/batches/loss/accuracy/val metrics; pause/resume/stop/reset; SGD/Momentum/Adam/RMSprop; CE/BCE/MSE/MAE; L2; seeds for reproducibility |
| Simulation | Animated signal flow coloured by **real** activations; auto / slow / fast / step-by-layer modes |
| Math view | Per-neuron `z = w₁x₁ + … + b` and `a = f(z)` with the model’s **actual** numbers (KaTeX); winning-neuron output calc |
| Backprop | One **real** fwd+bwd pass on a model copy: `w_new = w_old − η·∂L/∂w` verified against the optimiser step |
| Weights | Matrix heatmaps, bias strips, exact tables, min/max/mean/std |
| Activations | Real per-layer histograms + dead-neuron % (ReLU family) + saturation % (sigmoid/tanh), sampled every N epochs |
| Gradients | Layer-wise ‖∇W‖ (log scale), global norm, vanishing/exploding flags |
| AI Diagnosis | Deterministic rule engine over recorded metrics (overfitting, underfitting, vanishing/exploding gradients, dead neurons, saturation, LR issues, imbalance, scaling, capacity). An optional LLM layer (**Groq → OpenRouter → Ollama** fallback chain) rephrases those findings — **the app is fully functional without any LLM**. See [LLM Architecture](#llm-architecture) |
| Activation Lab | Safe custom formulas (`x / (1 + exp(-x))` etc.) parsed to a restricted AST → whitelisted tensor ops (no code execution); autograd derivatives; f & f′ plots; numerical-health report; save to library and train with them |
| Research Mode | Named experiments = archived runs (config, curves, metrics, activation/gradient stats, timestamp); compare 2–6 runs side by side |
| Projects | Save / load / delete in SQLite; export / import portable `.nnsim` files |
| Persistence | SQLite via a thin data-access layer (easy to move to PostgreSQL) |

**Scope (documented extension points):** v1 implements **feed-forward chains**
(`input → dense/dropout … → output`) with exact shape inference. Conv/RNN/
branching layers are intentionally not in v1 — see `docs/ARCHITECTURE.md` for
the extension design. Dropout is included.

---

## Repository layout

```
neural-network-simulator/
├── frontend/            # React + TS + Vite SPA
│   ├── src/components/  # canvas, sidebar, inspector, panels…
│   ├── src/store/       # zustand app store
│   └── ...
├── backend/
│   ├── app/
│   │   ├── main.py      # FastAPI app (+ SPA serving, CORS, error model)
│   │   ├── api/         # routers: datasets/networks/training/predict/...
│   │   ├── ml/          # builder, trainer, datasets, expressions, metrics, activations
│   │   ├── simulation/  # manual forward trace + backprop example
│   │   ├── llm/         # provider abstraction + Groq→OpenRouter→Ollama fallback
│   │   ├── diagnostics/ # rule engine + optional LLM explainer
│   │   └── services/    # SQLite storage + training job manager
│   └── tests/           # 185 pytest tests (core ML, layers, LLM fallback, API e2e)
├── examples/            # sample CSVs + .nnsim networks/experiments
├── docs/                # ARCHITECTURE · API · USER_GUIDE · MATHEMATICS · RESEARCH_MODE · TROUBLESHOOTING
├── scripts/             # dev/run/test/e2e helpers
├── .env.example
└── docker-compose.yml   # optional convenience
```

## Environment

Copy `.env.example` → `.env`. **All variables are optional** — the platform
runs with zero configuration. The LLM variables only enable the
natural-language explanation of diagnostics. Never put real keys in version
control; `.env` is git-ignored.

---

## LLM Architecture

The AI Diagnosis panel has two independent layers:

```
Real ML system
     ↓  training → metrics → activations → gradients
Deterministic Diagnostic Engine        (rule-based, source of truth)
     ↓  structured findings
LLM Service                            (prompting, never computes)
     ↓
Provider Manager                       (the fallback engine)
     ↓
Groq → NVIDIA NIM → OpenRouter → Ollama (local, dev mode only)
```

The **LLM never calculates or invents** accuracy, loss, gradients, weights,
parameter counts, dataset statistics or architecture facts. It only receives
statistics the deterministic engine already computed, and rephrases them. If
the LLM layer is disabled, misconfigured, or every provider fails, the
deterministic findings are still returned in full.

### Fallback order

| Order | Provider | Needs a key | Notes |
|-------|----------|-------------|-------|
| 1 | **Groq** | yes (`GROQ_API_KEY`) | fast LPU inference; primary by default |
| 2 | **NVIDIA NIM** | yes (`NVIDIA_API_KEY`) | high-performance accelerated inference; first fallback |
| 3 | **OpenRouter** | yes (`OPENROUTER_API_KEY`) | cloud model router; second fallback |
| 4 | **Ollama** | no | local & offline; development mode only (`ENV=development`) |

In **production mode** (`ENV=production`), Ollama is automatically omitted, maintaining `Groq → NVIDIA NIM → OpenRouter`.

Change the order with `LLM_PROVIDER_ORDER=groq,nvidia,openrouter,ollama`. Providers not
listed are never contacted, and a provider with no key is skipped instantly
rather than being retried.

### What happens on failure

| Situation | Result |
|-----------|--------|
| Groq answers | "Explanation generated using Groq." |
| Groq fails, NVIDIA NIM answers | "Explanation generated using NVIDIA NIM fallback." |
| Groq & NVIDIA fail, OpenRouter answers | "Explanation generated using OpenRouter fallback." |
| Only Ollama answers (dev mode) | "Explanation generated using local Ollama fallback." |
| All providers fail | "LLM explanation unavailable. Deterministic diagnosis is still available." |
| No keys configured | app starts normally; diagnosis works; explanation marked unavailable |

The AI Diagnosis panel shows a small provider badge (and a collapsible
`▸ providers` list with per-provider health), so you can always see which
provider produced the text. **API keys are never sent to the browser.**

### Environment variables

```bash
ENV=development
LLM_ENABLED=true
LLM_PROVIDER_ORDER=groq,nvidia,openrouter,ollama

GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile

NVIDIA_API_KEY=
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct

OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1

LLM_TIMEOUT_S=30
LLM_MAX_RETRIES=1
```

`LLM_MAX_RETRIES` is *extra* attempts per provider (so `1` means at most 2
tries). The chain is walked exactly once, so a failing chain can never loop.

### Setup

**Cloud providers (Groq / OpenRouter).** Create `.env` and add the key(s):

```bash
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
```

**Local Ollama (no key).**

```bash
# install from https://ollama.com, then:
ollama serve            # start the local server
ollama pull llama3.1    # fetch the model
```

Make sure `OLLAMA_BASE_URL` includes the `/v1` suffix (Ollama exposes an
OpenAI-compatible API under it).

**You only need one.** With only Groq configured, a Groq failure skips
OpenRouter (no key) and tries Ollama. With nothing configured, the app starts
and diagnosis still works — the explanation is simply marked unavailable.

### Adding a new provider

Write a subclass in `backend/app/llm/providers.py` (or a plain
`LLMProvider` subclass), then add one line to `PROVIDER_REGISTRY` in
`backend/app/llm/registry.py` and name it in `LLM_PROVIDER_ORDER`. The
diagnostic engine, the API layer and the frontend need no changes.

### Inspecting prompts

`POST /api/diagnostics/analyze/preview` returns the exact structured payload
that would be sent to the LLM, so you can verify that only engine-computed
statistics are included.

## Tests

```bash
cd backend && python -m pytest tests -q      # 185 tests
cd frontend && npm test                       # vitest — 18 tests
```

All LLM tests mock the network; none needs a real API key.

## License — MIT (see `LICENSE`)
