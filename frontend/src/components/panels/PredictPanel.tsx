import { useEffect, useMemo, useState } from 'react'
import { fmt, pct } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { MathBlock } from '../common/MathBlock'
import { Empty } from '../common/Stat'

/** Run one sample through the trained network and inspect *why* it answered
 * what it did (educational activation inspection — not formal XAI). */
export function PredictPanel() {
  const s = useAppStore()
  const runId = s.lastRunId ?? s.job.id

  const featureNames = useMemo(() => {
    if (s.job.dataSummary?.feature_names_in?.length) return s.job.dataSummary.feature_names_in
    const dsMeta = s.datasets.find((d) =>
      (s.dataset.kind === 'builtin' && d.kind === 'builtin' && d.id === s.dataset.name) ||
      (s.dataset.kind === 'upload' && d.kind === 'upload' && d.id === s.dataset.upload_id))
    if (dsMeta?.feature_names?.length) return dsMeta.feature_names
    const dim = s.validation?.input_dim ?? 2
    return Array.from({ length: dim }, (_, i) => `x${i + 1}`)
  }, [s.job.dataSummary, s.datasets, s.dataset, s.validation])

  const [values, setValues] = useState<(number | string)[]>([])
  const [showRawInputs, setShowRawInputs] = useState(false)

  useEffect(() => {
    setValues((v) => {
      const next = featureNames.map((_, i) => v[i] ?? 0)
      s.setProbeInput(next)
      return next
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [featureNames])

  const updateValues = (next: (number | string)[]) => {
    setValues(next)
    s.setProbeInput(next)
  }

  // Quick shapes for 8x8 (shapes8)
  const apply8x8Shape = (type: 'vertical' | 'horizontal' | 'box' | 'zeros' | 'random') => {
    const arr = new Array(64).fill(0)
    if (type === 'vertical') {
      for (let r = 1; r < 7; r++) { arr[r * 8 + 3] = 1; arr[r * 8 + 4] = 1 }
    } else if (type === 'horizontal') {
      for (let c = 1; c < 7; c++) { arr[3 * 8 + c] = 1; arr[4 * 8 + c] = 1 }
    } else if (type === 'box') {
      for (let i = 1; i < 7; i++) {
        arr[1 * 8 + i] = 1; arr[6 * 8 + i] = 1
        arr[i * 8 + 1] = 1; arr[i * 8 + 6] = 1
      }
    } else if (type === 'random') {
      for (let i = 0; i < 64; i++) { arr[i] = Math.random() > 0.75 ? 1 : 0 }
    }
    updateValues(arr)
  }

  // Quick presets for Iris
  const applyIrisPreset = (spec: number[]) => {
    updateValues(spec)
  }

  if (!runId) {
    return <Empty title="Train the network first">Once trained, enter feature values here to see the model's prediction, class probabilities, and which neurons fired most strongly (educational inspection).</Empty>
  }

  const res = s.predictResult as {
    prediction?: { type: string; label?: string; value?: number; prediction?: number }
    probabilities?: number[] | null
    top_neurons?: { layer: string; layer_id: string; top: { neuron: number; value: number }[] }[]
    final_layer_calculation?: { layer: string; neuron: number; weights: number[]; weighted_inputs: number[]; bias: number; z: number; a: number; truncated: boolean; n_inputs: number } | null
    note?: string
    error?: string
  } | null
  const classes = s.job.dataSummary?.class_names
  const isImage8x8 = featureNames.length === 64

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      <div className="w-84 shrink-0 panel p-3 space-y-3 overflow-y-auto">
        <div className="flex items-center justify-between">
          <div className="panel-title">Input features ({featureNames.length})</div>
          {isImage8x8 && (
            <button className="text-[10px] text-sky-400 hover:underline" onClick={() => setShowRawInputs(!showRawInputs)}>
              {showRawInputs ? 'Hide numbers' : 'Show numbers'}
            </button>
          )}
        </div>

        {/* 8x8 Interactive Visual Canvas */}
        {isImage8x8 ? (
          <div className="space-y-2">
            <div className="text-[11px] text-slate-400">Click or drag pixels to draw 8×8 image:</div>
            <div className="flex justify-center p-2 bg-lab-950/80 rounded border border-lab-700/60">
              <div className="grid grid-cols-8 gap-1 w-48 h-48">
                {values.map((v, i) => {
                  const active = Number(v) > 0.4
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => {
                        const next = [...values]
                        next[i] = active ? 0 : 1
                        updateValues(next)
                      }}
                      className={`rounded-sm transition-colors border ${
                        active
                          ? 'bg-sky-400 border-sky-300 shadow-[0_0_6px_rgba(56,189,248,0.5)]'
                          : 'bg-lab-800/80 border-lab-700/40 hover:bg-lab-700/60'
                      }`}
                      title={`Pixel ${i} (Row ${Math.floor(i / 8)}, Col ${i % 8}): ${v}`}
                    />
                  )
                })}
              </div>
            </div>

            <div className="flex flex-wrap gap-1">
              <button className="btn-ghost !py-0.5 !px-1.5 !text-[10px]" onClick={() => apply8x8Shape('vertical')}>| Vertical</button>
              <button className="btn-ghost !py-0.5 !px-1.5 !text-[10px]" onClick={() => apply8x8Shape('horizontal')}>— Horiz</button>
              <button className="btn-ghost !py-0.5 !px-1.5 !text-[10px]" onClick={() => apply8x8Shape('box')}>▢ Box</button>
              <button className="btn-ghost !py-0.5 !px-1.5 !text-[10px]" onClick={() => apply8x8Shape('random')}>🎲 Rand</button>
              <button className="btn-ghost !py-0.5 !px-1.5 !text-[10px] text-rose-400" onClick={() => apply8x8Shape('zeros')}>✕ Clear</button>
            </div>
          </div>
        ) : featureNames.length === 4 ? (
          <div className="space-y-1.5">
            <div className="text-[10px] text-slate-400">Quick Iris Samples:</div>
            <div className="flex gap-1">
              <button className="btn-ghost !py-0.5 !px-2 !text-[10px] text-emerald-400" onClick={() => applyIrisPreset([5.1, 3.5, 1.4, 0.2])}>Setosa</button>
              <button className="btn-ghost !py-0.5 !px-2 !text-[10px] text-sky-400" onClick={() => applyIrisPreset([5.9, 3.0, 4.2, 1.5])}>Versicolor</button>
              <button className="btn-ghost !py-0.5 !px-2 !text-[10px] text-violet-400" onClick={() => applyIrisPreset([6.5, 3.0, 5.2, 2.0])}>Virginica</button>
            </div>
          </div>
        ) : null}

        {/* Numeric inputs */}
        {(!isImage8x8 || showRawInputs) && (
          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {featureNames.map((name, i) => (
              <label key={name} className="flex items-center gap-2 text-xs">
                <span className="w-28 truncate text-slate-400 text-[11px]" title={name}>{name}</span>
                <input
                  className="input-num flex-1 !py-0.5 text-xs"
                  value={values[i] ?? ''}
                  onChange={(e) => {
                    const raw = e.target.value
                    const num = parseFloat(raw)
                    const next = values.map((v, k) => (k === i ? (raw !== '' && Number.isNaN(num) ? raw : raw === '' ? '' : num) : v))
                    updateValues(next)
                  }}
                />
              </label>
            ))}
          </div>
        )}

        <div className="flex gap-2 pt-1">
          <button
            className="btn-primary flex-1 !py-1 text-xs font-semibold"
            onClick={() => void s.predict(values.map((v) => (v === '' ? 0 : v)))}
          >
            ▶ Predict
          </button>
          <button
            className="btn-ghost !py-1 text-xs"
            title="Inspect forward activations and layer maths for this sample"
            onClick={() => {
              const cleaned = values.map((v) => (v === '' ? 0 : v))
              s.setProbeInput(cleaned)
              s.setTab('math')
              void s.fetchTrace(cleaned)
            }}
          >
            📐 To Math
          </button>
        </div>
        <div className="text-[10px] text-slate-500 leading-tight">
          Values go through the exact preprocessing pipeline fitted during training.
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto min-w-0">
        {res?.error && <div className="panel border-rose-500/50 p-3 text-sm text-rose-300">❌ {res.error}</div>}
        {res && !res.error && res.prediction && (
          <>
            <div className="panel p-4 flex items-center gap-6 flex-wrap">
              <div>
                <div className="panel-title">Prediction</div>
                <div className="text-2xl font-bold text-lab-accent">
                  {res.prediction.type === 'regression' ? fmt(res.prediction.value ?? null) : res.prediction.label}
                </div>
              </div>
              {res.probabilities && (
                <div className="flex-1 min-w-[220px] space-y-1">
                  {res.probabilities.map((p, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="w-24 truncate text-slate-400">{classes?.[i] ?? `class ${i}`}</span>
                      <div className="flex-1 h-3 rounded bg-lab-800 overflow-hidden">
                        <div className="h-full rounded bg-gradient-to-r from-sky-500 to-emerald-400"
                          style={{ width: `${Math.max(1, p * 100)}%` }} />
                      </div>
                      <span className="w-14 text-right font-mono">{pct(p)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {res.final_layer_calculation && (
              <div className="panel p-3 space-y-1">
                <div className="panel-title">Winning neuron's calculation (final layer)</div>
                <MathBlock
                  tex={`z = ${res.final_layer_calculation.weighted_inputs.map((w) => (w ?? 0).toFixed(3)).join(' + ')} ${res.final_layer_calculation.truncated ? ' + \\cdots' : ''} + ${res.final_layer_calculation.bias.toFixed(3)} = ${res.final_layer_calculation.z.toFixed(4)}`} />
                <MathBlock tex={`a = f(z) = ${res.final_layer_calculation.a.toFixed(4)}`} />
              </div>
            )}

            <div className="panel p-3">
              <div className="panel-title mb-2">Strongest activations per layer (educational view)</div>
              <div className="flex gap-3 flex-wrap">
                {(res.top_neurons ?? []).map((l) => (
                  <div key={l.layer_id} className="panel !bg-lab-850 p-2 min-w-[150px]">
                    <div className="text-[11px] font-semibold text-violet-300 mb-1">{l.layer}</div>
                    {l.top.map((t) => (
                      <div key={t.neuron} className="flex justify-between text-[11px] font-mono text-slate-300">
                        <span>neuron {t.neuron + 1}</span>
                        <span className="text-emerald-300">{t.value >= 0 ? '+' : ''}{t.value}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="text-[10px] text-slate-500 mt-2">⚠ {res.note}</div>
            </div>
          </>
        )}
        {!res && <Empty title="No prediction yet">Fill in the feature values and press Predict.</Empty>}
      </div>
    </div>
  )
}
