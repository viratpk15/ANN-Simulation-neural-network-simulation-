// Central application store (zustand). Owns the graph, dataset selection,
// training job state, inspection data and UI state.
import { create } from 'zustand'
import {
  addEdge, applyEdgeChanges, applyNodeChanges, updateEdge,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange,
} from 'reactflow'
import { api } from '../api/client'
import { connectTrainingWS, type TrainingEvent } from '../api/ws'
import { timeStr, uid } from '../lib/format'
import type {
  ActivationCheck,
  DataSummary,
  DatasetSelection,
  DatasetSummary,
  DiagnosisResult,
  EpochMetric,
  Experiment,
  Finding,
  JobState,
  LayerCatalogCategory,
  LayerKind,
  LayerParams,
  LayerSpec,
  LlmStatus,
  NetworkSpec,
  PredictionDetails,
  Preset,
  ProjectMeta,
  RunRecord,
  SavedActivation,
  Snapshot,
  TraceLayer,
  TrainingConfig,
  ValidationResult,
} from '../types'

export interface LayerNodeData {
  kind: LayerKind
  label: string
  params: LayerParams
}

/** Complete default parameter set. Every field is always present so the graph,
 *  the spec sent to the backend and the config panel never disagree. */
const BASE_DEFAULTS: LayerParams = {
  features: undefined,
  neurons: undefined,
  activation: 'relu',
  use_bias: true,
  init: 'he',
  dropout_rate: 0.5,
  dropout_mode: 'train_only',
  input_shape: undefined,
  momentum: 0.1,
  eps: 1e-5,
  affine: true,
  filters: 8,
  kernel_size: 3,
  stride: 1,
  padding: 0,
  padding_mode: 'valid',
  pool_size: 2,
  pool_stride: 2,
  pool_padding: 0,
  task: 'auto',
}

const defaultParams = (kind: LayerKind): LayerParams => ({
  ...BASE_DEFAULTS,
  features: kind === 'input' ? 2 : undefined,
  neurons: kind === 'dense' ? 8 : kind === 'output' ? 2 : undefined,
  activation: kind === 'output' ? 'softmax' : 'relu',
  init: kind === 'dense' ? 'he' : 'xavier',
})

/**
 * Merge a (possibly v1, possibly partial) parameter object onto the defaults.
 * This is what keeps projects saved before the new layers existed loading
 * without loss: unknown/missing fields simply take their default.
 */
export function withDefaults(kind: LayerKind, raw?: Partial<LayerParams> | null): LayerParams {
  const d = defaultParams(kind)
  if (!raw) return d
  const merged = { ...d, ...raw } as LayerParams
  // A field explicitly saved as null/undefined falls back to its default.
  // (The cast is needed because TS cannot correlate a keyof-typed index write
  // with the union of value types.)
  const target = merged as unknown as Record<string, unknown>
  const fallback = d as unknown as Record<string, unknown>
  for (const k of Object.keys(target)) {
    if (target[k] === undefined || target[k] === null) target[k] = fallback[k]
  }
  if (kind === 'input' && !merged.input_shape && !merged.features) merged.features = 2
  if ((kind === 'dense' || kind === 'output') && !merged.neurons) {
    merged.neurons = kind === 'output' ? 2 : 8
  }
  return merged
}

const kindLabel: Record<LayerKind, string> = {
  input: 'Input', dense: 'Dense', activation: 'Activation', dropout: 'Dropout',
  batchnorm: 'BatchNorm', flatten: 'Flatten', conv2d: 'Conv2D',
  maxpool: 'MaxPooling2D', avgpool: 'AveragePooling2D',
  globalavgpool: 'GlobalAvgPool', output: 'Output',
  embedding: 'Embedding', lstm: 'LSTM', gru: 'GRU',
}

/**
 * Offline fallback palette, mirroring `backend/app/ml/shapes.py`. The backend
 * catalogue replaces it on the first successful `GET /api/networks/layers`; this
 * copy only keeps the sidebar usable while the API is unreachable.
 */
const FALLBACK_CATALOG: LayerCatalogCategory[] = [
  {
    category: 'BASIC',
    layers: [
      { kind: 'input', name: 'Input', icon: '⬇', description: 'Entry point — declares the number of dataset features, or an image shape.', formula: 'x ∈ ℝᴺ', supported: true, coming_soon_note: '', summary: 'no parameters' },
      { kind: 'dense', name: 'Dense', icon: '⬡', description: 'Fully connected layer — every input connects to every neuron.', formula: 'z = Wx + b,   a = f(z)', supported: true, coming_soon_note: '', summary: '(inputs × neurons) + neurons' },
      { kind: 'activation', name: 'Activation', icon: 'ƒ', description: 'Applies a non-linear mathematical function element-wise.', formula: 'a = f(z)', supported: true, coming_soon_note: '', summary: 'no parameters' },
      { kind: 'dropout', name: 'Dropout', icon: '✂', description: 'Randomly disables activations during training (regularization).', formula: 'aᵢ = 0 w.p. p, else aᵢ / (1 − p)', supported: true, coming_soon_note: '', summary: 'no parameters' },
      { kind: 'batchnorm', name: 'BatchNorm', icon: '⊞', description: 'Normalizes intermediate activations, then rescales and shifts them.', formula: 'x̂ = (x − μ) / √(σ² + ε),   y = γx̂ + β', supported: true, coming_soon_note: '', summary: '2 parameters per channel' },
      { kind: 'flatten', name: 'Flatten', icon: '▤', description: 'Converts a multi-dimensional feature map into a flat vector.', formula: '28 × 28 × 1  →  784', supported: true, coming_soon_note: '', summary: 'no parameters' },
      { kind: 'output', name: 'Output', icon: '⬆', description: 'Final layer — neurons must match the task (classes / 1 real value).', formula: 'ŷ = softmax(Wx + b)', supported: true, coming_soon_note: '', summary: '(inputs × neurons) + neurons' },
    ],
  },
  {
    category: 'CNN',
    layers: [
      { kind: 'conv2d', name: 'Conv2D', icon: '▦', description: 'Learns spatial features by sliding learned filters over the input.', formula: 'y = f( Σ w ⊛ x + b )', supported: true, coming_soon_note: '', summary: '(k × k × C_in + bias) × filters' },
      { kind: 'maxpool', name: 'MaxPooling2D', icon: '▩', description: 'Downsamples feature maps by keeping the maximum of each window.', formula: 'out = max(window)', supported: true, coming_soon_note: '', summary: 'no parameters' },
      { kind: 'avgpool', name: 'AveragePooling2D', icon: '▨', description: 'Downsamples feature maps by averaging each window.', formula: 'out = mean(window)', supported: true, coming_soon_note: '', summary: 'no parameters' },
      { kind: 'globalavgpool', name: 'GlobalAveragePooling2D', icon: '▧', description: 'Reduces each feature map to one value by averaging all its positions.', formula: 'out_c = (1 / H·W) Σ x_c', supported: true, coming_soon_note: '', summary: 'no parameters' },
    ],
  },
  {
    category: 'ADVANCED',
    layers: [
      { kind: 'embedding', name: 'Embedding', icon: '❑', description: 'Maps integer indices to dense vectors (text / categorical input).', formula: 'e_i = W[i]', supported: false, coming_soon_note: 'Needs an integer-token input pipeline and sequence-shaped data.', summary: '' },
      { kind: 'lstm', name: 'LSTM', icon: '↻', description: 'Recurrent layer with gated memory, for sequence and time-series data.', formula: 'h_t, c_t = LSTM(x_t, h_prev, c_prev)', supported: false, coming_soon_note: 'Needs a recurrent data pipeline and a sequence-aware trainer.', summary: '' },
      { kind: 'gru', name: 'GRU', icon: '⇄', description: 'Simplified recurrent layer with fewer gates than LSTM.', formula: 'h_t = GRU(x_t, h_prev)', supported: false, coming_soon_note: 'Needs a recurrent data pipeline and a sequence-aware trainer.', summary: '' },
    ],
  },
]

/**
 * Render a shape for display: `[784]` → `784`, `[28, 28, 1]` → `28 × 28 × 1`.
 * The backend sends channels-last (H, W, C), which is the convention used in
 * papers and therefore the one the UI shows.
 */
export function shapeText(shape?: number[] | null): string {
  if (!shape || !shape.length) return '—'
  return shape.length === 1 ? String(shape[0]) : shape.join(' × ')
}

function newLayerNode(kind: LayerKind, position: { x: number; y: number }): Node<LayerNodeData> {
  const id = uid(kind)
  return {
    id, type: 'layerNode', position,
    data: { kind, label: `${kindLabel[kind]} ${id.slice(-4)}`, params: defaultParams(kind) },
  }
}

export function specFrom(nodes: Node<LayerNodeData>[], edges: Edge[], name: string, customs: Record<string, string>): NetworkSpec {
  return {
    name,
    layers: nodes.map((n) => ({
      id: n.id, kind: n.data.kind, params: n.data.params,
      position: { x: n.position.x, y: n.position.y }, label: n.data.label,
    })),
    edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    custom_activations: customs,
  }
}

export const EMPTY_JOB: JobState = {
  id: null, state: 'idle', history: [], snapshots: [],
  final: null, dataSummary: null, duration: null, error: null,
}

type Theme = 'dark' | 'light'
export type BottomTab =
  | 'training' | 'simulation' | 'math' | 'weights' | 'activations' | 'gradients'
  | 'diagnosis' | 'experiments' | 'predict' | 'dataset' | 'activlab' | 'console'

interface AppState {
  // --- ui -------------------------------------------------------------------
  theme: Theme
  activeTab: BottomTab
  backendOk: boolean | null
  toggleTheme: () => void
  setTab: (t: BottomTab) => void
  checkHealth: () => Promise<void>

  // --- project / graph --------------------------------------------------------
  projectName: string
  projectId: string | null
  nodes: Node<LayerNodeData>[]
  edges: Edge[]
  selectedNodeId: string | null
  customActivations: Record<string, string>
  savedCustoms: SavedActivation[]
  validation: ValidationResult | null
  validating: boolean
  setProjectName: (n: string) => void
  onNodesChange: (c: NodeChange[]) => void
  onEdgesChange: (c: EdgeChange[]) => void
  onConnect: (c: Connection) => void
  onEdgeUpdate: (oldEdge: Edge, conn: Connection) => void
  addLayer: (kind: LayerKind, pos?: { x: number; y: number }) => void
  duplicateNode: (id: string) => void
  updateNodeParams: (id: string, patch: Partial<LayerParams>) => void
  renameNode: (id: string, label: string) => void
  setSelected: (id: string | null) => void
  clearCanvas: () => void
  setGraph: (spec: NetworkSpec) => void
  validateNow: () => Promise<void>

  // --- datasets ----------------------------------------------------------------
  datasets: DatasetSummary[]
  dataset: DatasetSelection
  loadDatasets: () => Promise<void>
  setDatasetSelection: (patch: Partial<DatasetSelection>) => void
  setPreprocessing: (patch: Partial<DatasetSelection['preprocessing']>) => void
  refreshAfterUpload: () => Promise<void>

  // --- training ------------------------------------------------------------------
  trainConfig: TrainingConfig
  experimentName: string
  setTrainConfig: (patch: Partial<TrainingConfig>) => void
  setExperimentName: (n: string) => void
  job: JobState
  lastRunId: string | null
  trainingBusy: boolean
  startTraining: () => Promise<void>
  pauseJob: () => Promise<void>
  resumeJob: () => Promise<void>
  stopJob: () => Promise<void>
  resetJob: () => void
  loadRun: (runId: string) => Promise<void>

  // --- inspection ----------------------------------------------------------------
  probeInput: (number | string)[]
  setProbeInput: (v: (number | string)[]) => void
  forwardTrace: TraceLayer[] | null
  tracePrediction: PredictionDetails | null
  fetchTrace: (overrideFeatures?: (number | string)[]) => Promise<void>
  backpropExample: Record<string, number | string | number[]> | null
  fetchBackprop: (target: number | string) => Promise<void>
  weights: Record<string, import('../types').WeightLayer> | null
  fetchWeights: () => Promise<void>
  predictResult: Record<string, unknown> | null
  predict: (features: (number | string)[]) => Promise<void>

  // --- diagnosis -----------------------------------------------------------------
  diagnosis: DiagnosisResult | null
  diagnosisBusy: boolean
  runDiagnosis: () => Promise<void>
  /** Health of the configured provider fallback chain (never contains keys). */
  llmStatus: LlmStatus | null
  fetchLlmStatus: () => Promise<LlmStatus | null>

  // --- experiments -----------------------------------------------------------------
  experiments: Experiment[]
  compareRuns: RunRecord[] | null
  loadExperiments: () => Promise<void>
  fetchCompare: (ids: string[]) => Promise<void>

  // --- activation library -----------------------------------------------------------
  activationLibrary: { id: string; name: string; hint: string }[]
  activationCheck: ActivationCheck | null
  loadActivationLibrary: () => Promise<void>
  checkFormula: (formula: string) => Promise<ActivationCheck | null>
  saveCustom: (name: string, formula: string) => Promise<boolean>
  deleteCustom: (name: string) => Promise<void>

  // --- layer palette -------------------------------------------------------------
  /** The palette catalogue, fetched from the backend so the UI can never offer
   *  a layer the engine does not implement. Falls back to a local copy offline. */
  layerCatalog: LayerCatalogCategory[]
  loadLayerCatalog: () => Promise<void>
  isSupportedKind: (kind: LayerKind) => boolean

  // --- projects ------------------------------------------------------------------------
  projects: ProjectMeta[]
  loadProjects: () => Promise<void>
  saveProject: () => Promise<void>
  loadProject: (id: string) => Promise<void>
  deleteProject: (id: string) => Promise<void>
  exportProject: () => Promise<void>
  importProject: (file: File) => Promise<void>

  // --- presets ---------------------------------------------------------------------------
  loadPreset: (id?: string) => Promise<void>

  // --- workspace layout & visibility -----------------------------------------------------
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  toggleSidebar: () => void
  inspectorOpen: boolean
  setInspectorOpen: (open: boolean) => void
  toggleInspector: () => void
  bottomPanelSize: 'compact' | 'normal' | 'expanded' | 'collapsed'
  setBottomPanelSize: (size: 'compact' | 'normal' | 'expanded' | 'collapsed') => void

  // --- console ------------------------------------------------------------------------------
  consoleLog: string[]
  log: (msg: string) => void
}

let wsRef: WebSocket | null = null
let validateTimer: ReturnType<typeof setTimeout> | null = null

function getExpectedFeatureCount(s: AppState): number {
  if (s.job.dataSummary?.feature_names_in?.length) {
    return s.job.dataSummary.feature_names_in.length
  }
  const dsMeta = s.datasets.find((d) =>
    (s.dataset.kind === 'builtin' && d.kind === 'builtin' && d.id === s.dataset.name) ||
    (s.dataset.kind === 'upload' && d.kind === 'upload' && d.id === s.dataset.upload_id)
  )
  if (dsMeta?.n_features) return dsMeta.n_features
  if (s.validation?.input_dim) return s.validation.input_dim
  const inputNode = s.nodes.find((n) => n.data.kind === 'input')
  if (inputNode?.data.params?.features) return Number(inputNode.data.params.features)
  return 2
}

export const useAppStore = create<AppState>((set, get) => {
  const scheduleValidate = () => {
    if (validateTimer) clearTimeout(validateTimer)
    validateTimer = setTimeout(() => void get().validateNow(), 350)
  }

  const closeWs = () => {
    try { wsRef?.close() } catch { /* ignore */ }
    wsRef = null
  }

  return {
    theme: 'dark',
    activeTab: 'training',
    backendOk: null,
    sidebarOpen: true,
    setSidebarOpen: (open) => set({ sidebarOpen: open }),
    toggleSidebar: () => set({ sidebarOpen: !get().sidebarOpen }),
    inspectorOpen: true,
    setInspectorOpen: (open) => set({ inspectorOpen: open }),
    toggleInspector: () => set({ inspectorOpen: !get().inspectorOpen }),
    bottomPanelSize: 'normal',
    setBottomPanelSize: (size) => set({ bottomPanelSize: size }),
    toggleTheme: () => {
      const next = get().theme === 'dark' ? 'light' : 'dark'
      document.documentElement.classList.toggle('dark', next === 'dark')
      document.documentElement.classList.toggle('light', next === 'light')
      set({ theme: next })
    },
    setTab: (t) => set({ activeTab: t }),
    checkHealth: async () => {
      try {
        await api.get('/api/health')
        if (!get().backendOk) get().log('Connected to backend API.')
        set({ backendOk: true })
      } catch {
        set({ backendOk: false })
      }
    },

    // ---------------------------------------------------------------- graph
    projectName: 'Untitled Project',
    projectId: null,
    nodes: [],
    edges: [],
    selectedNodeId: null,
    customActivations: {},
    savedCustoms: [],
    validation: null,
    validating: false,
    setProjectName: (n) => set({ projectName: n }),
    onNodesChange: (changes) => {
      set({ nodes: applyNodeChanges(changes, get().nodes) })
      if (changes.some((c) => c.type === 'remove' || c.type === 'position')) scheduleValidate()
      const removed = changes.filter((c) => c.type === 'remove').map((c) => c.id)
      if (removed.includes(get().selectedNodeId ?? '')) set({ selectedNodeId: null })
    },
    onEdgesChange: (changes) => {
      set({ edges: applyEdgeChanges(changes, get().edges) })
      if (changes.some((c) => c.type === 'remove')) scheduleValidate()
    },
    onConnect: (conn) => {
      set({ edges: addEdge({ ...conn, id: uid('e') }, get().edges) })
      scheduleValidate()
    },
    onEdgeUpdate: (oldEdge, conn) => {
      set({ edges: updateEdge(oldEdge, conn, get().edges) })
      scheduleValidate()
    },
    duplicateNode: (id) => {
      const src = get().nodes.find((n) => n.id === id)
      if (!src) return
      const nid = uid(src.data.kind)
      const node: Node<LayerNodeData> = {
        ...src,
        id: nid,
        selected: false,
        position: { x: src.position.x + 56, y: src.position.y + 56 },
        data: {
          ...src.data,
          label: `${src.data.label} copy`,
          params: { ...src.data.params },
        },
      }
      set({ nodes: [...get().nodes, node], selectedNodeId: nid })
      get().log(`Duplicated layer '${src.data.label}'.`)
      scheduleValidate()
    },
    addLayer: (kind, pos) => {
      // Hard guard: a layer the engine cannot actually train must never enter
      // the graph, so nobody can build a network that pretends to work.
      const entry = get().layerCatalog.flatMap((c) => c.layers).find((l) => l.kind === kind)
      if (entry && !entry.supported) {
        get().log(`"${entry.name}" is not available yet — ${entry.coming_soon_note}`)
        return
      }
      const node = newLayerNode(kind, pos ?? { x: 80 + Math.random() * 120, y: 80 + Math.random() * 160 })
      set({ nodes: [...get().nodes, node], selectedNodeId: node.id })
      scheduleValidate()
    },
    updateNodeParams: (id, patch) => {
      set({
        nodes: get().nodes.map((n) => n.id === id
          ? { ...n, data: { ...n.data, params: { ...n.data.params, ...patch } } }
          : n),
      })
      scheduleValidate()
    },
    renameNode: (id, label) => {
      set({ nodes: get().nodes.map((n) => n.id === id ? { ...n, data: { ...n.data, label } } : n) })
    },
    setSelected: (id) => {
      // Only write when the selection actually changed. ReactFlow re-fires
      // onSelectionChange whenever its wrapper re-renders; an unconditional
      // set() here notified subscribers again → infinite update loop that
      // hit React's max-update-depth limit and unmounted the tree (blank page).
      if (get().selectedNodeId !== id) set({ selectedNodeId: id })
    },
    clearCanvas: () => {
      closeWs()
      set({
        nodes: [], edges: [], selectedNodeId: null, validation: null,
        job: EMPTY_JOB, lastRunId: null, forwardTrace: null, weights: null,
        diagnosis: null, predictResult: null, backpropExample: null, projectId: null,
      })
    },
    setGraph: (spec) => {
      set({
        projectName: spec.name,
        // withDefaults fills in every newly-added parameter, so a project saved
        // by an older version (Input/Dense/Dropout/Output) loads unchanged.
        nodes: spec.layers.map((l) => ({
          id: l.id, type: 'layerNode',
          position: { x: l.position?.x ?? 0, y: l.position?.y ?? 0 },
          data: { kind: l.kind, label: l.label || l.id, params: withDefaults(l.kind, l.params) },
        })),
        edges: spec.edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
        customActivations: spec.custom_activations || {},
      })
      scheduleValidate()
    },
    validateNow: async () => {
      const s = get()
      if (!s.nodes.length) { set({ validation: null }); return }
      set({ validating: true })
      try {
        const res = await api.post<ValidationResult>('/api/networks/validate', specFrom(s.nodes, s.edges, s.projectName, s.customActivations))
        set({ validation: res })
        const errs = res.errors.length
        if (!res.ok) s.log(`Validation: ${errs} error(s) — ${res.errors[0].message}`)
      } catch (e) {
        s.log(`Validation request failed: ${(e as Error).message}`)
      } finally {
        set({ validating: false })
      }
    },

    // ---------------------------------------------------------------- palette
    layerCatalog: FALLBACK_CATALOG,
    loadLayerCatalog: async () => {
      try {
        const res = await api.get<{ categories: LayerCatalogCategory[] }>('/api/networks/layers')
        if (res.categories?.length) set({ layerCatalog: res.categories })
      } catch {
        // Offline / backend down: keep the built-in catalogue so the palette
        // still renders. The backend remains the authority once reachable.
      }
    },
    isSupportedKind: (kind) => {
      const e = get().layerCatalog.flatMap((c) => c.layers).find((l) => l.kind === kind)
      return e ? e.supported : true
    },

    // ---------------------------------------------------------------- datasets
    datasets: [],
    dataset: {
      kind: 'builtin', name: 'xor',
      preprocessing: { scale: 'none', test_split: 0.2, val_split: 0.2, seed: 42 },
    },
    loadDatasets: async () => {
      try {
        const res = await api.get<{ datasets: DatasetSummary[] }>('/api/datasets')
        set({ datasets: res.datasets })
      } catch (e) {
        get().log(`Failed to load datasets: ${(e as Error).message}`)
      }
    },
    setDatasetSelection: (patch) => {
      const next = { ...get().dataset, ...patch }
      set({ dataset: next })
      const dsMeta = get().datasets.find((d) =>
        (next.kind === 'builtin' && d.kind === 'builtin' && d.id === next.name) ||
        (next.kind === 'upload' && d.kind === 'upload' && d.id === next.upload_id)
      )
      if (dsMeta?.n_features) {
        set({ probeInput: Array(dsMeta.n_features).fill(0) })
      }
      void get().validateNow()
    },
    setPreprocessing: (patch) => set({
      dataset: { ...get().dataset, preprocessing: { ...get().dataset.preprocessing, ...patch } },
    }),
    refreshAfterUpload: async () => { await get().loadDatasets() },

    // ---------------------------------------------------------------- training
    trainConfig: {
      epochs: 300, batch_size: 16, learning_rate: 0.05,
      optimizer: 'adam', loss: 'cross_entropy', l2: 0, seed: 42,
      shuffle: true, snapshot_every: 15,
    },
    experimentName: '',
    setTrainConfig: (patch) => set({ trainConfig: { ...get().trainConfig, ...patch } }),
    setExperimentName: (n) => set({ experimentName: n }),
    job: EMPTY_JOB,
    lastRunId: null,
    trainingBusy: false,
    startTraining: async () => {
      const s = get()
      await s.validateNow()
      const v = get().validation
      if (!v?.ok) {
        s.log('Cannot start training — fix network validation errors first.')
        set({ activeTab: 'console' })
        return
      }
      set({ trainingBusy: true, job: { ...EMPTY_JOB, state: 'queued' } })
      try {
        const req = {
          network: specFrom(s.nodes, s.edges, s.projectName, s.customActivations),
          dataset: s.dataset,
          config: s.trainConfig,
          experiment_name: s.experimentName || null,
        }
        const res = await api.post<{ job_id: string }>('/api/training/start', req)
        const jobId = res.job_id
        s.log(`Training started (job ${jobId}).`)
        set({ job: { ...EMPTY_JOB, id: jobId, state: 'queued' } })
        closeWs()
        wsRef = connectTrainingWS(jobId, (ev: TrainingEvent) => {
          const cur = get().job
          if (ev.type === 'epoch') {
            const { type: _t, ...rest } = ev
            set({ job: { ...cur, state: cur.state === 'paused' ? 'paused' : 'running', history: [...cur.history, rest as unknown as EpochMetric] } })
          } else if (ev.type === 'snapshot') {
            set({ job: { ...get().job, snapshots: [...get().job.snapshots, ev.snapshot as Snapshot] } })
          } else if (ev.type === 'status') {
            set({ job: { ...get().job, state: ev.state as JobState['state'] } })
          } else if (ev.type === 'done') {
            const j = get().job
            const evSummary = ev.data_summary as DataSummary | null
            const inDim = evSummary?.feature_names_in?.length
            if (inDim && get().probeInput.length !== inDim) {
              set({ probeInput: Array(inDim).fill(0) })
            }
            set({
              job: {
                ...j,
                state: (ev.state as JobState['state']) ?? 'finished',
                final: (ev.final as JobState['final']) ?? null,
                dataSummary: evSummary ?? null,
                duration: (ev.duration_s as number) ?? null,
                error: (ev.error as string | null) ?? null,
              },
              lastRunId: (ev.run_id as string) ?? j.id,
            })
            get().log(ev.state === 'finished'
              ? `Training finished in ${fmtDur(ev.duration_s as number)}. Run id: ${ev.run_id}.`
              : `Training ended: ${ev.state}${ev.error ? ` — ${ev.error}` : ''}.`)
            closeWs()
          } else if (ev.type === 'error') {
            get().log(`Training error: ${ev.message}`)
          }
        }, () => {
          const j = get().job
          if (j.state === 'running' || j.state === 'paused' || j.state === 'queued') {
            // server closed unexpectedly; fetch final status
            void api.get<{ state: JobState['state']; error?: string }>(`/api/training/${j.id}/status`)
              .then((st) => set({ job: { ...get().job, state: st.state, error: st.error ?? null } }))
              .catch(() => set({ job: { ...get().job, state: 'failed', error: 'Lost connection to training stream.' } }))
          }
        })
      } catch (e) {
        set({ job: { ...EMPTY_JOB, state: 'failed', error: (e as Error).message } })
        get().log(`Training failed to start: ${(e as Error).message}`)
      } finally {
        set({ trainingBusy: false })
      }
    },
    pauseJob: async () => {
      const id = get().job.id
      if (id) await api.post(`/api/training/${id}/pause`)
    },
    resumeJob: async () => {
      const id = get().job.id
      if (id) await api.post(`/api/training/${id}/resume`)
    },
    stopJob: async () => {
      const id = get().job.id
      if (!id) return
      await api.post(`/api/training/${id}/stop`)
      get().log('Stop requested — finishing current epoch…')
    },
    resetJob: () => { closeWs(); set({ job: EMPTY_JOB }) },
    loadRun: async (runId) => {
      try {
        const m = await api.get<{
          state: JobState['state']; history: EpochMetric[]; snapshots: Snapshot[]
          final: JobState['final']; data_summary: DataSummary | null; duration_s?: number
        }>(`/api/training/${runId}/metrics`)
        set({
          job: {
            id: runId, state: m.state, history: m.history ?? [], snapshots: m.snapshots ?? [],
            final: m.final ?? null, dataSummary: m.data_summary ?? null,
            duration: m.duration_s ?? null, error: null,
          },
          lastRunId: runId,
        })
        get().log(`Loaded run ${runId} (${m.history?.length ?? 0} epochs).`)
      } catch (e) {
        get().log(`Failed to load run: ${(e as Error).message}`)
      }
    },

    // ---------------------------------------------------------------- inspection
    probeInput: [0, 0],
    setProbeInput: (v) => set({ probeInput: v }),
    forwardTrace: null,
    tracePrediction: null,
    fetchTrace: async (overrideFeatures) => {
      const s = get()
      const runId = s.lastRunId ?? s.job.id
      if (!runId) { s.log('Train the network first to inspect a forward pass.'); return }

      const expectedDim = getExpectedFeatureCount(s)
      let features = overrideFeatures ?? s.probeInput
      if (!Array.isArray(features) || features.length !== expectedDim) {
        features = Array.from({ length: expectedDim }, (_, i) => (features && features[i] !== undefined ? features[i] : 0))
        set({ probeInput: features })
      }

      try {
        const res = await api.post<{ trace: TraceLayer[]; prediction: PredictionDetails }>(
          '/api/inspect/forward-trace', { run_id: runId, features })
        set({ forwardTrace: res.trace, tracePrediction: res.prediction })
      } catch (e) {
        s.log(`Forward trace failed: ${(e as Error).message}`)
      }
    },
    backpropExample: null,
    fetchBackprop: async (target) => {
      const s = get()
      const runId = s.lastRunId ?? s.job.id
      if (!runId) { s.log('Train the network first to inspect backpropagation.'); return }

      const expectedDim = getExpectedFeatureCount(s)
      let features = s.probeInput
      if (!Array.isArray(features) || features.length !== expectedDim) {
        features = Array.from({ length: expectedDim }, (_, i) => (features && features[i] !== undefined ? features[i] : 0))
        set({ probeInput: features })
      }

      try {
        const res = await api.post<Record<string, number | string | number[]>>(
          '/api/inspect/backprop', { run_id: runId, features, target })
        set({ backpropExample: res })
      } catch (e) {
        s.log(`Backprop inspection failed: ${(e as Error).message}`)
      }
    },
    weights: null,
    fetchWeights: async () => {
      const s = get()
      const runId = s.lastRunId ?? s.job.id
      if (!runId) return
      try {
        const res = await api.get<{ layers: NonNullable<AppState['weights']> }>(`/api/inspect/weights/${runId}`)
        set({ weights: res.layers })
      } catch (e) {
        s.log(`Weight fetch failed: ${(e as Error).message}`)
      }
    },
    predictResult: null,
    predict: async (features) => {
      const s = get()
      const runId = s.lastRunId ?? s.job.id
      if (!runId) { s.log('Train the network first, then run predictions.'); return }
      try {
        const res = await api.post<Record<string, unknown>>('/api/predict', { run_id: runId, features })
        set({ predictResult: res })
        s.log(`Prediction → ${JSON.stringify((res.prediction as PredictionDetails)?.label ?? (res.prediction as PredictionDetails)?.value)}`)
      } catch (e) {
        set({ predictResult: { error: (e as Error).message } })
      }
    },

    // ---------------------------------------------------------------- diagnosis
    diagnosis: null,
    diagnosisBusy: false,
    runDiagnosis: async () => {
      const s = get()
      const runId = s.lastRunId ?? s.job.id
      if (!runId) { s.log('Train first, then run AI diagnosis.'); return }
      set({ diagnosisBusy: true })
      try {
        const res = await api.post<DiagnosisResult>('/api/diagnostics/analyze', { run_id: runId, use_llm: true })
        set({ diagnosis: res })
        const n = res.findings.filter((f: Finding) => f.severity !== 'ok').length
        s.log(`Diagnosis complete: ${res.findings.length} finding(s), ${n} potential issue(s).`)
        if (res.llm?.available) {
          s.log(res.llm.message || `Explanation generated using ${res.llm.label ?? res.llm.provider}.`)
        } else {
          s.log('LLM explanation unavailable - deterministic diagnosis only.')
        }
      } catch (e) {
        s.log(`Diagnosis failed: ${(e as Error).message}`)
      } finally {
        set({ diagnosisBusy: false })
      }
    },
    llmStatus: null,
    fetchLlmStatus: async () => {
      try {
        const res = await api.get<LlmStatus>('/api/diagnostics/llm-status')
        set({ llmStatus: res })
        return res
      } catch {
        // Provider status is informational only; never block the panel on it.
        return null
      }
    },

    // ---------------------------------------------------------------- experiments
    experiments: [],
    compareRuns: null,
    loadExperiments: async () => {
      try {
        const res = await api.get<{ experiments: Experiment[] }>('/api/experiments')
        set({ experiments: res.experiments })
      } catch { /* silent */ }
    },
    fetchCompare: async (ids) => {
      try {
        const res = await api.get<{ runs: RunRecord[] }>(`/api/experiments/compare?run_ids=${ids.join(',')}`)
        set({ compareRuns: res.runs })
      } catch (e) {
        get().log(`Compare failed: ${(e as Error).message}`)
      }
    },

    // ---------------------------------------------------------------- activations
    activationLibrary: [],
    activationCheck: null,
    loadActivationLibrary: async () => {
      try {
        const res = await api.get<{ builtin: { id: string; name: string; hint: string }[]; custom: SavedActivation[] }>('/api/activations/library')
        set({ activationLibrary: res.builtin, savedCustoms: res.custom })
        const customs: Record<string, string> = { ...get().customActivations }
        for (const c of res.custom) customs[c.name] = c.formula
        set({ customActivations: customs })
      } catch { /* silent */ }
    },
    checkFormula: async (formula) => {
      try {
        const res = await api.post<ActivationCheck>('/api/activations/validate', { formula })
        set({ activationCheck: res })
        return res
      } catch (e) {
        set({ activationCheck: { ok: false, error: (e as Error).message, xs: [], ys: [], dys: [], issues: [(e as Error).message] } })
        return null
      }
    },
    saveCustom: async (name, formula) => {
      try {
        await api.post('/api/activations', { name, formula, notes: '' })
        set({ customActivations: { ...get().customActivations, [name]: formula } })
        get().log(`Custom activation '${name}' saved — select it as 'custom::${name}' on any Dense layer.`)
        await get().loadActivationLibrary()
        return true
      } catch (e) {
        get().log(`Could not save activation: ${(e as Error).message}`)
        return false
      }
    },
    deleteCustom: async (name) => {
      await api.del(`/api/activations/${name}`)
      const customs = { ...get().customActivations }
      delete customs[name]
      set({ customActivations: customs })
      await get().loadActivationLibrary()
    },

    // ---------------------------------------------------------------- projects
    projects: [],
    loadProjects: async () => {
      try {
        const res = await api.get<{ projects: ProjectMeta[] }>('/api/projects')
        set({ projects: res.projects })
      } catch { /* silent */ }
    },
    saveProject: async () => {
      const s = get()
      try {
        const res = await api.post<{ project_id: string }>('/api/projects', {
          name: s.projectName,
          network: specFrom(s.nodes, s.edges, s.projectName, s.customActivations),
          dataset: s.dataset,
          training_config: s.trainConfig,
          run_id: s.lastRunId,
          custom_activations: s.customActivations,
        })
        set({ projectId: res.project_id })
        s.log(`Project saved (id ${res.project_id}).`)
        await s.loadProjects()
      } catch (e) {
        s.log(`Save failed: ${(e as Error).message}`)
      }
    },
    loadProject: async (id) => {
      const s = get()
      try {
        const rec = await api.get<{
          payload: {
            name: string; network: NetworkSpec; dataset?: DatasetSelection
            training_config?: TrainingConfig; custom_activations?: Record<string, string>
            run_id?: string | null
          }
        }>(`/api/projects/${id}`)
        const p = rec.payload
        s.setGraph(p.network)
        if (p.dataset) set({ dataset: p.dataset })
        if (p.training_config) set({ trainConfig: { ...get().trainConfig, ...p.training_config } })
        if (p.custom_activations) set({ customActivations: p.custom_activations })
        set({ projectId: id, projectName: p.name })
        if (p.run_id) void get().loadRun(p.run_id)
        s.log(`Project '${p.name}' loaded.`)
      } catch (e) {
        s.log(`Load failed: ${(e as Error).message}`)
      }
    },
    deleteProject: async (id) => {
      await api.del(`/api/projects/${id}`)
      await get().loadProjects()
    },
    exportProject: async () => {
      const s = get()
      const doc = {
        format: 'neurosim-lab/project', version: 1,
        payload: {
          name: s.projectName,
          network: specFrom(s.nodes, s.edges, s.projectName, s.customActivations),
          dataset: s.dataset,
          training_config: s.trainConfig,
          run_id: s.lastRunId,
          custom_activations: s.customActivations,
        },
      }
      const { downloadText } = await import('../lib/format')
      downloadText(`${s.projectName.replace(/\s+/g, '-').toLowerCase()}.nnsim`, JSON.stringify(doc, null, 2))
      s.log('Project exported as .nnsim file.')
    },
    importProject: async (file) => {
      try {
        const text = await file.text()
        const doc = JSON.parse(text)
        if (doc.format !== 'neurosim-lab/project') throw new Error('Not a .nnsim project file.')
        const p = doc.payload
        get().setGraph(p.network)
        if (p.dataset) set({ dataset: p.dataset })
        if (p.training_config) set({ trainConfig: { ...get().trainConfig, ...p.training_config } })
        if (p.custom_activations) set({ customActivations: p.custom_activations })
        set({ projectName: p.name || 'Imported Project' })
        if (p.run_id) void get().loadRun(p.run_id)
        get().log(`Imported project '${p.name}'.`)
      } catch (e) {
        get().log(`Import failed: ${(e as Error).message}`)
      }
    },

    // ---------------------------------------------------------------- presets
    loadPreset: async (id = 'xor-demo') => {
      const s = get()
      try {
        const res = await api.get<{ presets: Preset[] }>('/api/networks/presets')
        const p = res.presets.find((x) => x.id === id) ?? res.presets[0]
        s.setGraph(p.network as NetworkSpec)
        const inputLayer = p.network.layers.find((l) => l.kind === 'input')
        const inputDim = (inputLayer?.params as unknown as Record<string, unknown>)?.features as number ?? 2
        set({
          dataset: p.dataset as DatasetSelection,
          trainConfig: { ...get().trainConfig, ...p.config },
          experimentName: p.experiment_name ?? '',
          job: EMPTY_JOB, lastRunId: null, forwardTrace: null, diagnosis: null,
          predictResult: null, weights: null, backpropExample: null,
          projectName: p.network.name,
          probeInput: Array(inputDim).fill(0),
        })
        s.log(`Example loaded: ${p.name}. Press "Train" to run it.`)
      } catch (e) {
        s.log(`Failed to load example: ${(e as Error).message}`)
      }
    },

    // ---------------------------------------------------------------- console
    consoleLog: ['Welcome to NeuroSim Lab. Click "Try Example" to load the XOR demo.'],
    log: (msg) => set({ consoleLog: [...get().consoleLog.slice(-400), `[${timeStr()}] ${msg}`] }),
  }
})

function fmtDur(d?: number): string {
  if (d === undefined || d === null || Number.isNaN(d)) return '—'
  return `${d.toFixed(1)}s`
}
