# Research Mode — Experiments & Comparison

Research Mode turns one-off trainings into a small, reproducible experiment
journal stored in SQLite.

## What an experiment run archives

For every run started with an **experiment name** (Training tab field, or
`POST /api/experiments/run`):

- full `TrainRequest` configuration (network spec, dataset ref, training config, experiment name),
- random seed → reproducibility,
- per-epoch history (train/val loss + metric),
- activation statistics & gradient statistics snapshots,
- final held-out **test** metrics (accuracy / P / R / F1 macro, or MSE / MAE / R²) + confusion matrix,
- learned weights (as a file) and Unix-timestamped records.

## Recommended protocol (and the built-in example)

The preset **Iris Research Experiment** (top-bar `Projects` area -> presets API,
spec §37) is: dataset `iris`, architecture **4 → 16 → 8 → 3**, ReLU + softmax,
Adam, lr = 0.001, seed 7.

Then create variations, changing **one factor at a time**:

| experiment_name | change |
|---|---|
| `iris-baseline` | everything default |
| `iris-tanh` | hidden activation → tanh |
| `iris-custom-mish` | hidden activation → `custom::<your formula>` from the Activation Lab |
| `iris-lr-0.01` | learning rate 0.001 → 0.01 |
| `iris-bigger` | 4 → 32 → 16 → 3 |

Nothing is hard-coded — every run is computed live and archived.

## Comparing (UI)

**Research Mode tab** → tick 2–6 runs → **Compare**: overlay of training-loss
curves + a table of test metrics, hyper-parameters and wall-clock time.
"load" re-opens any run in every inspection tab (Math, Weights, Gradients, …),
so you can correlate *why* the winner won (e.g. fewer dead neurons, healthier
gradient norms).

## Comparing custom vs standard activations

1. Activation Lab: validate + save e.g. `mishy = x * tanh(log(1 + exp(x)))`.
2. Set each hidden layer's activation to `custom::mishy`.
3. Train with experiment name `…-mish` using the **same seed & splits**.
4. Compare with the ReLU/tanh runs — accuracy, F1, training time, gradient norms.

## API

See `docs/API.md` → *Experiments*: `POST /api/experiments/run`,
`GET /api/experiments`, `GET /api/experiments/compare?run_ids=…` （returns the
complete run records, e.g. for plotting in a notebook).
