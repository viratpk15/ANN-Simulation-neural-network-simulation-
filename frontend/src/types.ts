// Shared API/domain types mirroring the backend schemas.

export type LayerKind =
  // BASIC
  | 'input' | 'dense' | 'activation' | 'dropout' | 'batchnorm' | 'flatten' | 'output'
  // CNN
  | 'conv2d' | 'maxpool' | 'avgpool' | 'globalavgpool'
  // ADVANCED (declared but not yet trainable end-to-end)
  | 'embedding' | 'lstm' | 'gru'

export interface LayerParams {
  // --- original v1 fields (unchanged) ---
  features?: number | null
  neurons?: number | null
  activation: string
  use_bias: boolean
  init: 'xavier' | 'he' | 'normal' | 'uniform'
  dropout_rate: number
  dropout_mode: 'train_only' | 'always'
  // --- input: optional 3-D image shape, channels-last [H, W, C] ---
  input_shape?: number[] | null
  // --- batchnorm ---
  momentum: number
  eps: number
  affine: boolean
  // --- conv2d ---
  filters: number
  kernel_size: number
  stride: number
  padding: number
  padding_mode: 'valid' | 'same'
  // --- pooling ---
  pool_size: number
  pool_stride: number
  pool_padding: number
  // --- output ---
  task: 'auto' | 'classification' | 'regression' | 'binary'
}

// ---- layer palette catalogue (mirrors backend app/ml/shapes.py) ---------

export interface LayerCatalogEntry {
  kind: LayerKind
  name: string
  icon: string
  description: string
  formula: string
  supported: boolean
  coming_soon_note: string
  summary: string
}

export interface LayerCatalogCategory {
  category: 'BASIC' | 'CNN' | 'ADVANCED'
  layers: LayerCatalogEntry[]
}

export interface LayerSpec {
  id: string
  kind: LayerKind
  params: LayerParams
  position: { x: number; y: number }
  label?: string | null
}

export interface EdgeSpec { id: string; source: string; target: string }

export interface NetworkSpec {
  name: string
  layers: LayerSpec[]
  edges: EdgeSpec[]
  custom_activations: Record<string, string>
  /** Schema version; absent in v1 documents, migrated on load by the backend. */
  version?: number
}

export interface ValidationIssue {
  code: string
  message: string
  layer_id?: string | null
  suggestion?: string | null
}

export interface LayerShape {
  id: string; kind: LayerKind
  in_features?: number | null; out_features?: number | null; params: number
  /** Full shape: [N] for a vector, [H, W, C] (channels last) for a feature map. */
  in_shape?: number[] | null
  out_shape?: number[] | null
  /** Short human annotation, e.g. the formula used to derive the output shape. */
  note?: string | null
}

export interface ValidationResult {
  ok: boolean
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
  layers: LayerShape[]
  input_dim?: number | null
  output_dim?: number | null
  total_params: number
  order: string[]
  /** True when the chain contains image (3-D) tensors. */
  is_cnn: boolean
}

export interface DatasetSummary {
  id: string; name: string; kind: 'builtin' | 'upload'
  samples: number; n_features: number; task: string
  n_classes?: number | null; description: string
  target_name: string; feature_names: string[]
}

export interface Preprocessing {
  scale: 'none' | 'standard' | 'minmax'
  test_split: number
  val_split: number
  seed: number
}

export interface DatasetSelection {
  kind: 'builtin' | 'upload'
  name?: string
  upload_id?: string
  feature_columns?: string[]
  target_column?: string
  preprocessing: Preprocessing
}

export interface TrainingConfig {
  epochs: number
  batch_size: number
  learning_rate: number
  optimizer: 'sgd' | 'momentum' | 'adam' | 'rmsprop'
  loss: 'cross_entropy' | 'bce' | 'mse' | 'mae'
  l2: number
  seed: number
  shuffle: boolean
  snapshot_every: number
}

export interface TrainRequest {
  network: NetworkSpec
  dataset: DatasetSelection
  config: TrainingConfig
  experiment_name?: string | null
  project_id?: string | null
}

export interface EpochMetric {
  epoch: number
  train_loss: number
  train_loss_full: number
  val_loss?: number | null
  train_metric: number
  val_metric?: number | null
  metric_name: string
  lr: number
  total_epochs?: number
  batches?: number
}

export interface ActivationStat {
  layer_id: string; label: string; activation: string
  mean: number; std: number; min: number; max: number
  frac_near_zero: number
  frac_saturated?: number | null
  dead_neuron_frac?: number | null
  sample: number[]
}

export interface GradStat {
  layer_id: string; label: string
  w_grad_norm: number; w_grad_mean_abs: number; w_grad_max_abs: number
  b_grad_norm?: number | null
  example_weight: { index: number[]; w: number; grad: number }
}

export interface Snapshot {
  epoch: number
  activations: ActivationStat[]
  gradients: {
    layers: GradStat[]
    total_norm: number
    loss_on_batch: number
    input_sample: number[]
  }
}

export interface DataSummary {
  task: string
  n_train: number; n_val: number; n_test: number
  n_features: number
  n_classes?: number | null
  class_names?: string[] | null
  class_counts_train?: number[] | null
  feature_names_in: string[]
  scaled: string
  raw_feature_stds?: number[] | null
}

export interface FinalEval {
  test_metrics: Record<string, number>
  confusion_matrix?: number[][] | null
  n_test: number
  weights: Record<string, WeightLayer>
  total_epochs_run: number
}

export interface WeightLayer {
  shape: number[]
  weights: number[][]
  bias?: number[] | null
  stats: { min: number; max: number; mean: number; std: number }
  display_note?: string | null
  /**
   * `linear`    — shape [out, in] (the original v1 format).
   * `conv`      — shape [filters, k*k*C_in]; each row is one flattened filter,
   *               with the unflattened tensor in `full_shape`.
   * `batchnorm` — shape [2, channels]: row 0 = scale γ, row 1 = shift β.
   */
  kind?: 'linear' | 'conv' | 'batchnorm'
  full_shape?: number[]
  scale?: number[]
  shift?: number[]
  running_mean?: number[]
  running_var?: number[]
}

export type JobStateName = 'idle' | 'queued' | 'running' | 'paused' | 'stopped' | 'finished' | 'failed'

export interface JobState {
  id: string | null
  state: JobStateName
  history: EpochMetric[]
  snapshots: Snapshot[]
  final: FinalEval | null
  dataSummary: DataSummary | null
  duration: number | null
  error: string | null
}

// ---- forward trace --------------------------------------------------------
export interface TraceNeuron {
  index: number; bias: number; z: number; a: number
  weights: number[]; weighted_inputs: number[]
  n_inputs: number; truncated: boolean
}

export interface TraceLayer {
  id: string; kind: LayerKind
  label: string
  activation?: string
  n: number
  n_shown?: number
  values?: number[]
  inputs?: number[]
  z?: number[]
  a?: number[]
  neurons?: TraceNeuron[]
  note?: string
  /** Plain-language formula for this layer's operation. */
  formula?: string
  /** [C, H, W] shape of the tensor entering / leaving this layer. */
  in_shape?: number[]
  out_shape?: number[]
  in_features?: number
  params?: number
  /** Small rendered 2-D grid (patch, filter, or feature map). */
  input_grid?: (number | null)[][]
  feature_map?: number[][]
  /** A worked Conv2D step: patch ⊛ filter → products → sum + bias. */
  convolution?: {
    output_position: number[]
    input_patch: number[][]
    filter: number[][]
    products: number[][]
    channels: number
    sum_of_products: number
    bias: number
    z: number
    formula: string
  }
  /** A worked pooling step: the first window, its value and its argmax. */
  pooling?: {
    window: (number | null)[][]
    value: number
    argmax_position?: number[] | null
    window_size: number[]
    stride: number
    formula: string
  }
  /** BatchNorm statistics actually used, recomputed for this probe input. */
  batchnorm?: {
    mode: string
    momentum: number
    eps: number
    affine: boolean
    mean: number[]
    variance: number[]
    running_mean: number[]
    running_var: number[]
    scale_gamma: number[]
    shift_beta: number[]
    inv_std: number[]
    x_hat_sample: number[]
    y_sample: number[]
    formula: string
  }
  /** Per-channel spatial means produced by GlobalAveragePooling2D. */
  channel_means?: number[]
  /** `true` when the input layer declared an image shape. */
  is_image?: boolean
  matrix?: number[][]
  shape?: number[]
}

export interface PredictionDetails {
  type: 'classification' | 'regression'
  prediction?: number
  label?: string
  value?: number
  probabilities?: number[]
}

// ---- diagnosis -------------------------------------------------------------

export type ProviderState =
  | 'available' | 'not_configured' | 'active' | 'failed' | 'skipped' | 'disabled'

/** Health of one LLM provider. Never contains an API key. */
export interface LlmProviderStatus {
  name: string
  label: string
  state: ProviderState
  model: string
  detail?: string
  latency_s?: number | null
  active?: boolean
}

export interface LlmAttempt {
  provider: string
  label: string
  state: string
  reason?: string
  detail?: string
  latency_s?: number | null
  tries?: number
}

export interface LlmInfo {
  requested?: boolean
  enabled: boolean
  available: boolean
  explanation: string | null
  error?: string | null
  /** Which provider produced the explanation, e.g. 'groq'. */
  provider?: string | null
  model?: string | null
  /** Human label, e.g. 'Groq' or 'local Ollama'. */
  label?: string | null
  /** 'active' | 'fallback' | 'unavailable' */
  outcome?: string
  /** e.g. 'Explanation generated using OpenRouter fallback.' */
  message?: string
  fallback_used?: boolean
  latency_s?: number | null
  /** The configured fallback chain, in order. */
  chain?: string[]
  attempts?: LlmAttempt[]
  usage?: Record<string, number | null>
}

export interface LlmStatus {
  enabled: boolean
  chain: string[]
  chain_labels: string[]
  providers: LlmProviderStatus[]
  models: Record<string, string>
  timeout_s: number
  max_retries: number
  note: string
}

export interface Finding {
  severity: 'critical' | 'warning' | 'info' | 'ok'
  code: string
  title: string
  explanation: string
  evidence: Record<string, unknown>
  suggestions: string[]
}

export interface DiagnosisResult {
  run_id: string
  context: Record<string, unknown>
  findings: Finding[]
  llm: LlmInfo
}

// ---- activations ------------------------------------------------------------
export interface ActivationCheck {
  ok: boolean
  error?: string
  formula?: string
  range?: number[]
  xs: number[]
  ys: (number | null)[]
  dys: (number | null)[]
  stats?: {
    min?: number | null; max?: number | null; mean?: number | null
    max_abs_derivative?: number | null
    fraction_finite: number
    derivative_defined_fraction: number
  } | null
  issues: string[]
}

export interface SavedActivation { name: string; formula: string; notes: string; created_at?: string }

// ---- projects / experiments --------------------------------------------------
export interface ProjectMeta { id: string; name: string; created_at: string; updated_at: string }

export interface Experiment {
  id: string; name: string; created_at: string
  config: Record<string, unknown>
  runs: { id: string; created_at: string; status: string; duration_s?: number; final?: FinalEval | null; error?: string | null }[]
}

export interface RunRecord {
  id: string; created_at: string; status: string
  network: NetworkSpec
  config: TrainingConfig
  history: EpochMetric[]
  snapshots: Snapshot[]
  data_summary: DataSummary | null
  final: FinalEval | null
  experiment_id?: string | null
  duration_s?: number
}

export interface Preset {
  id: string; name: string; description: string
  network: NetworkSpec
  dataset: DatasetSelection
  config: Partial<TrainingConfig>
  experiment_name?: string
}
