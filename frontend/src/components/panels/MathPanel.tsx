import { useState } from 'react'
import { fmt } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { MathBlock } from '../common/MathBlock'
import { Empty, Field, Select } from '../common/Stat'
import type { TraceLayer } from '../../types'

function num(v: number): string {
  const r = Math.abs(v) >= 1000 || (Math.abs(v) < 1e-4 && v !== 0) ? v.toExponential(2) : v.toFixed(4)
  return v < 0 ? `(${r})` : r
}

function actTex(name?: string): string {
  if (!name) return 'f'
  const clean = name.replace('custom::', '').replace(/[^a-zA-Z0-9_]/g, '')
  const special: Record<string, string> = {
    relu: '\\mathrm{ReLU}', leaky_relu: '\\mathrm{LeakyReLU}', sigmoid: '\\sigma',
    tanh: '\\tanh', softmax: '\\mathrm{softmax}', gelu: '\\mathrm{GELU}', linear: '\\mathrm{id}',
  }
  return special[clean] ?? `\\mathrm{${clean}}`
}

/** Small helper: render a numeric matrix as a monospace grid. */
function Grid({ rows }: { rows: (number | null)[][] }) {
  return (
    <div className="inline-grid gap-[1px] font-mono text-[9px]"
      style={{ gridTemplateColumns: `repeat(${rows[0]?.length ?? 1}, minmax(0, 1fr))` }}>
      {rows.flat().map((v, i) => (
        <div key={i}
          className={`px-1 py-0.5 text-center rounded-sm ${
            v === null ? 'bg-slate-700/30 text-slate-600'
              : (v as number) < 0 ? 'bg-sky-900/40 text-sky-300'
                : 'bg-amber-900/30 text-amber-200'}`}>
          {v === null ? '—' : fmt(v as number, 2)}
        </div>
      ))}
    </div>
  )
}

function Stat({ label, values, accent = 'text-slate-300' }:
{ label: string; values: number[]; accent?: string }) {
  return (
    <div className="panel px-2 py-1.5">
      <div className="panel-title">{label}</div>
      <div className={`font-mono ${accent} break-all`}>
        {values.length ? values.map((v) => fmt(v, 3)).join(', ') : '—'}
      </div>
    </div>
  )
}

/** Conv2D: the worked patch ⊛ filter → products → sum + bias, from the real weights. */
function ConvSection({ layer }: { layer: TraceLayer }) {
  const c = layer.convolution
  if (!c) return null
  return (
    <div className="panel p-3 space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h3 className="font-semibold text-indigo-300">{layer.label}</h3>
        <span className="text-xs text-slate-400">
          Conv2D — output value at position ({c.output_position.join(', ')})
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <div className="panel-title mb-1">Input patch</div>
          <Grid rows={c.input_patch} />
        </div>
        <div>
          <div className="panel-title mb-1">Learned filter</div>
          <Grid rows={c.filter} />
        </div>
        <div>
          <div className="panel-title mb-1">Element-wise products</div>
          <Grid rows={c.products} />
        </div>
      </div>
      <div>
        <MathBlock tex={`z = \\sum (w \\odot x) + b = ${c.sum_of_products.toFixed(4)} + ${c.bias.toFixed(4)} = ${c.z.toFixed(4)}`} />
        <p className="text-xs text-slate-500 mt-1 leading-relaxed">
          This is the real arithmetic for one output position of one filter. Repeating it for
          every position and every filter builds the whole output feature map.
        </p>
      </div>
      {layer.input_grid && layer.feature_map && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="panel-title mb-1">Input feature map (first channel)</div>
            <Grid rows={layer.input_grid} />
          </div>
          <div>
            <div className="panel-title mb-1">Output feature map (first channel)</div>
            <Grid rows={layer.feature_map} />
          </div>
        </div>
      )}
      {layer.note && <p className="text-xs text-slate-400 leading-relaxed">{layer.note}</p>}
    </div>
  )
}

/** Pooling: the first window, the reduction and where the winning value came from. */
function PoolSection({ layer }: { layer: TraceLayer }) {
  const w = layer.pooling
  if (!w) return null
  const isMax = layer.kind === 'maxpool'
  return (
    <div className="panel p-3 space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h3 className={`font-semibold ${isMax ? 'text-cyan-300' : 'text-blue-300'}`}>{layer.label}</h3>
        <span className="text-xs text-slate-400">
          {isMax ? 'MaxPooling2D' : 'AveragePooling2D'} — first {w.window_size.join('×')} window, stride {w.stride}
        </span>
      </div>
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <div className="panel-title mb-1">Window values</div>
          <Grid rows={w.window} />
        </div>
        <div className="text-2xl text-slate-600">→</div>
        <div>
          <div className="panel-title mb-1">{isMax ? 'max of window' : 'mean of window'}</div>
          <div className={`text-2xl font-mono font-semibold ${isMax ? 'text-cyan-300' : 'text-blue-300'}`}>
            {fmt(w.value, 4)}
          </div>
          {isMax && w.argmax_position && (
            <div className="text-[10px] text-slate-500 mt-1">
              came from position ({w.argmax_position.join(', ')}) inside the window
            </div>
          )}
        </div>
      </div>
      {layer.feature_map && (
        <div>
          <div className="panel-title mb-1">Pooled output feature map (first channel)</div>
          <Grid rows={layer.feature_map} />
        </div>
      )}
      {layer.note && <p className="text-xs text-slate-400 leading-relaxed">{layer.note}</p>}
    </div>
  )
}

/** BatchNorm: μ, σ², x̂, γ, β and y, recomputed for the probe input. */
function BatchNormSection({ layer }: { layer: TraceLayer }) {
  const b = layer.batchnorm
  if (!b) return null
  return (
    <div className="panel p-3 space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h3 className="font-semibold text-teal-300">{layer.label}</h3>
        <span className="text-xs text-slate-400">BatchNorm — {b.mode}</span>
      </div>
      <MathBlock tex="\\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}}, \\qquad y = \\gamma\\hat{x} + \\beta" />
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px] font-mono">
        <Stat label="mean μ" values={b.mean} />
        <Stat label="variance σ²" values={b.variance} />
        <Stat label="1 / √(σ²+ε)" values={b.inv_std} />
        <Stat label="x̂ (normalized)" values={b.x_hat_sample} accent="text-sky-300" />
        <Stat label="scale γ" values={b.scale_gamma} accent="text-emerald-300" />
        <Stat label="shift β" values={b.shift_beta} accent="text-amber-300" />
        <Stat label="output y" values={b.y_sample} accent="text-teal-300" />
        <Stat label="running mean" values={b.running_mean} />
        <Stat label="running var" values={b.running_var} />
      </div>
      <p className="text-xs text-slate-400 leading-relaxed">{layer.note}</p>
    </div>
  )
}

/** Standalone Activation layer: the incoming z and the applied f(z). */
function ActivationSection({ layer }: { layer: TraceLayer }) {
  return (
    <div className="panel p-3 space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h3 className="font-semibold text-blue-300">{layer.label}</h3>
        <span className="text-xs text-slate-400">
          Activation — <MathBlock tex={actTex(layer.activation)} block={false} /> applied element-wise
        </span>
      </div>
      <div>
        <div className="panel-title mb-1">z — values arriving from the previous layer</div>
        <div className="font-mono text-slate-300 break-all text-xs">
          [{(layer.z ?? []).map((v) => fmt(v, 3)).join(', ')}]
        </div>
      </div>
      <div>
        <div className="panel-title mb-1">a = f(z) — after the activation</div>
        <div className="font-mono text-emerald-300 break-all text-xs">
          [{(layer.a ?? []).map((v) => fmt(v, 3)).join(', ')}]
        </div>
      </div>
      <p className="text-xs text-slate-400 leading-relaxed">{layer.note}</p>
    </div>
  )
}

/** Flatten and GlobalAveragePooling2D: the shape transform itself. */
function ReshapeSection({ layer }: { layer: TraceLayer }) {
  const isGlobal = layer.kind === 'globalavgpool'
  return (
    <div className="panel p-3 space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h3 className="font-semibold text-slate-200">{layer.label}</h3>
        <span className="text-xs text-slate-400">{isGlobal ? 'GlobalAveragePooling2D' : 'Flatten'}</span>
      </div>
      <div className="font-mono text-sm text-sky-200">
        {layer.in_shape?.join(' × ')} → {layer.out_shape?.join(' × ')}
      </div>
      {isGlobal && layer.channel_means && (
        <div>
          <div className="panel-title mb-1">
            Average of each feature map (showing {layer.channel_means.length})
          </div>
          <div className="font-mono text-emerald-300 text-xs break-all">
            [{layer.channel_means.map((v) => fmt(v, 4)).join(', ')}]
          </div>
        </div>
      )}
      <p className="text-xs text-slate-400 leading-relaxed">{layer.note}</p>
    </div>
  )
}

/** Step-by-step mathematical view of the REAL forward & backward passes. */
export function MathPanel() {
  const s = useAppStore()
  const [layerIdx, setLayerIdx] = useState(0)
  const [target, setTarget] = useState<string>('')

  const trace = s.forwardTrace
  const runId = s.lastRunId ?? s.job.id
  const classes = s.job.dataSummary?.class_names ?? null
  const task = s.job.dataSummary?.task ?? null

  if (!runId && !trace) {
    return <Empty title="Train the network first">After training, this tab shows every weighted sum, bias addition and activation — computed from the model's real parameters, not illustrations.</Empty>
  }

  const layers = trace ?? []
  // Every layer that has real explainable maths gets its own section. Input and
  // dropout are pass-throughs, so they are not listed.
  const computed = layers.filter((l) =>
    ['dense', 'output', 'conv2d', 'maxpool', 'avgpool', 'batchnorm', 'activation',
      'flatten', 'globalavgpool'].includes(l.kind))
  const current = layers[Math.min(layerIdx, Math.max(0, layers.length - 1))]

  return (
    <div className="h-full flex flex-col p-3 gap-3 overflow-y-auto">
      <div className="flex items-center gap-2 flex-wrap">
        <button className="btn-primary !py-1" onClick={() => void s.fetchTrace()} disabled={!runId}>
          ⟳ Compute forward pass
        </button>
        {trace && computed.length > 0 && (
          <div className="w-72">
            <Select value={current?.id ?? ''} onChange={(id) => setLayerIdx(layers.findIndex((l) => l.id === id))}
              options={computed.map((l, i) => ({ value: l.id, label: `${i + 1}. ${l.label} (${l.kind})` }))} />
          </div>
        )}
        {s.tracePrediction && (
          <span className="badge-ok ml-auto">
            output ŷ = {s.tracePrediction.type === 'regression' ? fmt(s.tracePrediction.value ?? null) : s.tracePrediction.label}
          </span>
        )}
      </div>

      {!trace && <div className="text-xs text-slate-500">Press “Compute forward pass” to trace the probe input (set values in the Predict tab).</div>}

      {current?.kind === 'conv2d' && <ConvSection layer={current} />}
      {(current?.kind === 'maxpool' || current?.kind === 'avgpool') && <PoolSection layer={current} />}
      {current?.kind === 'batchnorm' && <BatchNormSection layer={current} />}
      {current?.kind === 'activation' && <ActivationSection layer={current} />}
      {(current?.kind === 'flatten' || current?.kind === 'globalavgpool') && <ReshapeSection layer={current} />}

      {(current?.kind === 'dense' || current?.kind === 'output') && (
        <div className="panel p-3 space-y-3">
          <div className="flex items-baseline gap-3 flex-wrap">
            <h3 className="font-semibold text-lab-accent">{current.label}</h3>
            <span className="text-xs text-slate-400">activation <MathBlock tex={actTex(current.activation)} block={false} /></span>
            <span className="text-xs text-slate-500">{current.n} neurons {current.n_shown && current.n_shown < current.n ? `· showing first ${current.n_shown}` : ''}</span>
          </div>

          {current.inputs && (
            <div>
              <div className="panel-title mb-1">Inputs from previous layer {current.inputs.length < (current.neurons?.[0]?.n_inputs ?? 0) ? `(first ${current.inputs.length} shown)` : ''}</div>
              <MathBlock tex={`a_{prev} = [ ${current.inputs.map((v) => fmt(v, 3)).join(',\\,')} ]`} />
            </div>
          )}

          <div className="space-y-2">
            {(current.neurons ?? []).slice(0, 3).map((nr) => {
              const terms = (current.inputs ?? nr.weights.map(() => 0)).slice(0, nr.weights.length)
                .map((inp, k) => `${num(inp)} \\times ${num(nr.weights[k])}`)
              return (
                <div key={nr.index} className="rounded-lg border border-lab-700/60 bg-lab-850/60 p-2.5">
                  <div className="text-xs font-semibold text-violet-300 mb-1">Neuron {nr.index + 1}</div>
                  <MathBlock
                    tex={`z_{${nr.index + 1}} = ${terms.join(' + ')} ${nr.truncated ? ' + \\cdots' : ''} + ${num(nr.bias)} = ${num(nr.z)}`}
                  />
                  <MathBlock tex={`a_{${nr.index + 1}} = ${actTex(current.activation)}\\!\\left(z_{${nr.index + 1}}\\right) = ${num(nr.a)}`} />
                </div>
              )
            })}
            {(current.neurons?.length ?? 0) > 3 && (
              <details className="text-xs text-slate-400">
                <summary className="cursor-pointer hover:text-slate-200">Show all {current.neurons!.length} displayed neurons…</summary>
                <div className="mt-2 grid grid-cols-2 lg:grid-cols-4 gap-2">
                  {current.neurons!.slice(3).map((nr) => (
                    <div key={nr.index} className="rounded border border-lab-700/50 p-2 font-mono text-[10px]">
                      <div className="text-violet-300">N{nr.index + 1}</div>
                      <div>z = {fmt(nr.z, 4)}</div>
                      <div>a = {fmt(nr.a, 4)}</div>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div>
              <div className="panel-title mb-1">Pre-activations z</div>
              <div className="font-mono text-slate-300 break-all">[{(current.z ?? []).map((v) => fmt(v, 3)).join(', ')}]</div>
            </div>
            <div>
              <div className="panel-title mb-1">Activations a = f(z)</div>
              <div className="font-mono text-emerald-300 break-all">[{(current.a ?? []).map((v) => fmt(v, 3)).join(', ')}]</div>
            </div>
            <div className="text-slate-500 leading-relaxed">
              Every neuron computes <MathBlock tex="z = \sum_i w_i x_i + b" block={false} /> then applies{' '}
              <MathBlock tex={`a = ${actTex(current.activation)}(z)`} block={false} />. All numbers above come from
              the trained parameters of the current run.
            </div>
          </div>
        </div>
      )}

      {/* ---------------- backpropagation ---------------- */}
      <div className="panel p-3 space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <h3 className="font-semibold text-amber-300">Backpropagation & weight update</h3>
          {task === 'classification' && classes && (
            <div className="w-56">
              <Field label="True class for the probe input">
                <Select value={target || classes[0]} onChange={setTarget}
                  options={classes.map((c, i) => ({ value: String(i), label: `${c} (class ${i})` }))} />
              </Field>
            </div>
          )}
          {task === 'regression' && (
            <div className="w-40">
              <Field label="True target value">
                <input className="input-num" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="e.g. 3.2" />
              </Field>
            </div>
          )}
          <button className="btn !py-1 mt-3" disabled={!runId}
            onClick={() => {
              const t2 = task === 'classification' ? Number(target || 0) : parseFloat(target || '0')
              void s.fetchBackprop(Number.isNaN(t2) ? 0 : t2)
            }}>
            ⟳ Compute one real update
          </button>
        </div>
        <p className="text-xs text-slate-500 max-w-3xl">
          This performs one genuine forward + backward pass on a copy of the trained network (your weights are not
          altered) and reports the resulting update of the first weight of the first trainable layer:
        </p>
        {s.backpropExample ? (
          <div className="space-y-1">
            <MathBlock tex={`\\mathcal{L} = ${num(Number(s.backpropExample.loss ?? 0))}`} />
            <MathBlock tex={`\\frac{\\partial \\mathcal{L}}{\\partial w} = ${num(Number(s.backpropExample.gradient ?? 0))}`} />
            <MathBlock tex={`w_{new} = w_{old} - \\eta \\; \\frac{\\partial \\mathcal{L}}{\\partial w} = ${num(Number(s.backpropExample.w_old ?? 0))} - ${num(Number(s.backpropExample.learning_rate ?? 0))} \\times ${num(Number(s.backpropExample.gradient ?? 0))} = ${num(Number(s.backpropExample.w_new_computed ?? 0))}`} />
            <div className="text-[11px] text-slate-500 font-mono">
              verified against the actual optimiser step: {fmt(Number(s.backpropExample.w_new_actual ?? 0), 6)}
              {' '}· layer {String(s.backpropExample.layer_id)} · weight index [{String((s.backpropExample.index as number[])?.join(', '))}]
            </div>
          </div>
        ) : (
          <div className="text-xs text-slate-500">Choose the true class/target and press the button.</div>
        )}
      </div>
    </div>
  )
}
