# User Guide — NeuroSim Lab

## The workspace

```
┌──────────────────────────────────────────────────────────────────────────┐
│ TOP BAR:  project name · New · Projects · ⚡Try Example · Validate · ▶Train │
│           (pause/resume/stop while training) · API status · theme         │
├──────────┬───────────────────────────────────────────────┬───────────────┤
│ SIDEBAR  │              CANVAS                           │  INSPECTOR    │
│ Layers   │   [Input] → [Dense] → [Dense] → [Output]     │  selected     │
│ (drag!)  │                                               │  layer config │
│ Datasets │                                               │  validation   │
├──────────┴───────────────────────────────────────────────┴───────────────┤
│ BOTTOM TABS: Training · Simulation · Math · Weights · Activations ·       │
│ Gradients · Predict · AI Diagnosis · Activation Lab · Research · Dataset  │
│ · Console                                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

## 30-second demo (recommended first run)

1. **⚡ Try Example** (top bar) — loads: XOR dataset, a 2→8→8→2 tanh/softmax network, Adam lr=0.05, 300 epochs.
2. Press **▶ Train** — watch live loss/accuracy curves in the **Training** tab. XOR should reach ~100% test accuracy in seconds.
3. **Simulation** tab → **⟳ Run forward pass**, then **▶ Animate flow** or **Step forward ▶** — neuron colours/numbers are the real activations.
4. **Math** tab — pick a layer, read the actual `z = w₁x₁ + … + b` computation for every neuron; scroll down for the real backprop weight update.
5. **Weights / Activations / Gradients** tabs — heatmaps, distributions, gradient norms with dead-neuron/saturation/vanishing warnings.
6. **Predict** tab — enter `1, 0` → predicts class `1` with probabilities; see which neurons fired.
7. **AI Diagnosis** tab → **Analyze this run** — findings with evidence and suggestions.
8. **Activation Lab** — try `x * tanh(log(1 + exp(x)))` (Mish), save it, set a hidden layer's activation to `custom::…`, retrain, then compare the two runs in **Research Mode**.

## Building your own network

- **Drag** layers from the sidebar (or double-click them), connect left→right by dragging from the right handle of one node to the left handle of the next.
- **Click** a node to configure it in the Inspector (neurons, activation + hint, init + hint, bias, dropout rate, display name).
- Shift-drag multi-selects; `Delete`/`Backspace` removes; zoom via wheel; pan by dragging the background; the minimap is live.
- The Inspector shows **live validation**: chain order, per-layer `in → out` shapes and parameter counts, and human-readable errors with suggestions (missing input/output, disconnected layer, cycle/branch, invalid activation…).

### Rules of thumb enforced by the trainer
| Task | Output layer | Loss |
|---|---|---|
| Multi-class (k classes) | k neurons, `softmax` (recommended) or `linear` | `cross_entropy` |
| Binary | 1 neuron, `sigmoid` | `bce` |
| Regression | 1 neuron, `linear` | `mse` or `mae` |

Input features must equal the dataset's prepared feature count (after one-hot encoding of categorical columns).

## Datasets

- **Built-ins** include descriptions with samples/features/classes; open **Dataset → Analyse & preview** for column stats, class info and the first 8 rows.
- **CSV upload**: auto-detected numeric/categorical columns, missing values, suggested target/task. Pick feature columns (checkboxes) and the target, choose scaling (standardization recommended), test/val splits and a seed, then **Apply**. Missing numerics are imputed with the mean, categoricals get an explicit `__missing__` bucket, everything is fitted on the train split only.

## Training tab

Epochs, batch size, learning rate, optimizer, loss, L2, seed, shuffle, snapshot cadence, optional **experiment name** (records the run into Research Mode). Live stats row + loss/accuracy charts; pause/resume/stop; final **test-set** metrics (accuracy, precision/recall/F1 macro — or MSE/MAE/R² — and a colour-coded confusion matrix for classification).

## Understanding the visualisation tabs

| Tab | What you see | Source |
|---|---|---|
| Simulation | Signals flowing through the validated chain; neuron colour/labels | Real forward trace of your probe input |
| Math | Exact z/a formulas with the model's actual parameters; real backprop update | Stored weights + numpy recompute |
| Weights | Matrix heatmaps/tables + stats per layer | `state_dict` of the run |
| Activations | Distribution histograms, dead-neuron %, saturation % | Real batches every N epochs |
| Gradients | Layer-wise ‖∇W‖ on log scale + a single weight's real update | Real backward passes |
| Predict | Probabilities + strongest-firing neurons (educational, not formal XAI) | Real forward pass |
| AI Diagnosis | Rule-based findings with hedged language + evidence + suggestions | Your recorded metrics |
| Activation Lab | f(x) and autograd f′(x) plots, numerical health, save to library | Safe AST compiler |
| Research Mode | Experiments and run comparison overlays | Archived runs |

## Tips

- XOR needs **at least one hidden layer** — that's the point of the demo.
- If the loss oscillates: lower the learning rate (Diagnosis will tell you the same thing).
- Deep tanh/sigmoid networks on purpose → watch gradients vanish in the Gradients tab.
- Set the same seed to reproduce a run exactly; change one variable at a time in Research Mode.
- Everything in the UI is computed from the actual model — screenshot graphs freely for reports.
