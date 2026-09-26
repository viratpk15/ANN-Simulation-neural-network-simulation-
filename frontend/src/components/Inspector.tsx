import { useMemo } from 'react'
import { shapeText, useAppStore } from '../store/useAppStore'
import { Checkbox, Field, NumInput, Select, TextInput } from './common/Stat'

const INIT_HINTS: Record<string, string> = {
  xavier: 'Xavier/Glorot — variance-preserving; good default for tanh/sigmoid.',
  he: 'He/Kaiming — designed for ReLU-family activations.',
  normal: 'Small random normal values (σ=0.05).',
  uniform: 'Small random uniform values in [-0.1, 0.1].',
}

const ACTIVATION_HINTS: Record<string, string> = {
  relu: 'f(z) = max(0, z) — the usual choice for hidden layers.',
  sigmoid: 'f(z) = 1 / (1 + e^-z) — squashes to (0, 1); binary outputs.',
  tanh: 'f(z) = tanh(z) — squashes to (-1, 1) and is zero-centred.',
  gelu: 'f(z) = z · Φ(z) — smooth ReLU used in transformers.',
  softmax: 'probabilities over classes — output layer only.',
  linear: 'f(z) = z — no non-linearity; regression outputs.',
  'custom::': 'Your own formula, validated by a safe expression parser. '
    + 'It is compiled to torch ops, so gradients work automatically — no code is executed.',
}

export function Inspector() {
  const s = useAppStore()
  const node = s.nodes.find((n) => n.id === s.selectedNodeId)
  const v = s.validation

  const activationOptions = useMemo(() => {
    const builtin = s.activationLibrary.map((a) => ({ value: a.id, label: `${a.name} — ${a.hint}` }))
    const custom = s.savedCustoms.map((c) => ({ value: `custom::${c.name}`, label: `✎ ${c.name} (custom)` }))
    return [...builtin, ...custom]
  }, [s.activationLibrary, s.savedCustoms])

  return (
    <aside className="w-72 shrink-0 border-l border-lab-700/60 bg-lab-900/60 overflow-y-auto">
      <div className="p-3 space-y-4">
        <div>
          <div className="panel-title mb-2 flex items-center justify-between">
            <span>Selected layer</span>
            <button
              onClick={() => s.toggleInspector()}
              className="text-slate-400 hover:text-slate-200 text-xs px-1.5 py-0.5 rounded hover:bg-lab-800"
              title="Collapse Inspector"
            >
              ▶
            </button>
          </div>
          {!node && <div className="text-xs text-slate-500">Click a layer on the canvas to configure it. Shift-click multi-selects; Delete removes.</div>}
          {node && (
            <div className="panel p-3 space-y-3">
              <Field label="Display name">
                <TextInput value={node.data.label} onChange={(lbl) => s.renameNode(node.id, lbl)} />
              </Field>

              {node.data.kind === 'input' && (
                <>
                  <Field label="Number of features" hint="Must equal the number of dataset feature columns (after one-hot encoding). Leave the image shape below empty for tabular data.">
                    <NumInput value={node.data.params.features ?? ''} min={1} max={200000} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { features: Math.max(1, Math.round(v2)) })} />
                  </Field>
                  <Field label="Image shape (H × W × C)"
                    hint="Optional. Set this for image data so convolutional layers receive a 3D tensor (e.g. 28 × 28 × 1 for MNIST-style data). It must contain exactly 'features' values in total.">
                    <div className="flex gap-1.5">
                      {(['H', 'W', 'C'] as const).map((label, i) => {
                        const shp = node.data.params.input_shape
                        return (
                          <div key={label} className="flex-1">
                            <div className="text-[9px] text-slate-500 mb-0.5 text-center">{label}</div>
                            <NumInput
                              value={shp?.[i] ?? ''} min={1} max={512} step={1}
                              onChange={(v2) => {
                                const next = [...(shp ?? [8, 8, 1])]
                                next[i] = Math.max(1, Math.round(v2))
                                s.updateNodeParams(node.id, {
                                  input_shape: next,
                                  features: next[0] * next[1] * next[2],
                                })
                              }} />
                          </div>
                        )
                      })}
                    </div>
                  </Field>
                  {node.data.params.input_shape && (
                    <button className="btn-ghost w-full !py-1 text-[11px]"
                      onClick={() => s.updateNodeParams(node.id, { input_shape: undefined })}>
                      Clear image shape (treat input as a flat vector)
                    </button>
                  )}
                </>
              )}

              {node.data.kind === 'activation' && (
                <>
                  <Field label="Activation function"
                    hint={node.data.params.activation.startsWith('custom::')
                      ? ACTIVATION_HINTS['custom::']
                      : ACTIVATION_HINTS[node.data.params.activation]}>
                    <Select value={node.data.params.activation} options={activationOptions}
                      onChange={(a) => s.updateNodeParams(node.id, { activation: a })} />
                  </Field>
                  {node.data.params.activation.startsWith('custom::') && (
                    <div className="text-[11px] text-slate-400 bg-lab-800/60 rounded p-2 space-y-1">
                      <div className="font-semibold text-slate-300">
                        Custom function “{node.data.params.activation.replace('custom::', '')}”
                      </div>
                      <div className="font-mono text-emerald-300 break-all">
                        f(x) = {s.customActivations[node.data.params.activation.replace('custom::', '')] ?? '(formula not found)'}
                      </div>
                      <p className="text-[10px] text-slate-500 leading-snug">
                        Open the <b>Activation Lab</b> tab to edit or validate the formula. It is
                        parsed with a safe expression compiler — no Python or JavaScript is ever
                        executed — and gradients come from autograd automatically.
                      </p>
                    </div>
                  )}
                </>
              )}

              {(node.data.kind === 'dense' || node.data.kind === 'output') && (
                <>
                  <Field label={node.data.kind === 'output' ? 'Output neurons' : 'Neurons'}
                    hint={node.data.kind === 'output'
                      ? 'Classification (cross-entropy): one neuron per class. Binary (BCE): 1 neuron + sigmoid. Regression: 1 neuron + linear.'
                      : 'Width of this hidden layer. More neurons = more capacity (and more overfitting risk).'}>
                    <NumInput value={node.data.params.neurons ?? ''} min={1} max={512} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { neurons: Math.max(1, Math.round(v2)) })} />
                  </Field>
                  <Field label="Activation function" hint="Non-linearity applied after z = Wx + b. Create your own in the Activation Lab tab.">
                    <Select value={node.data.params.activation} options={activationOptions}
                      onChange={(a) => s.updateNodeParams(node.id, { activation: a })} />
                  </Field>
                  <Field label="Weight initialization" hint={INIT_HINTS[node.data.params.init]}>
                    <Select value={node.data.params.init} options={[
                      { value: 'xavier', label: 'Xavier / Glorot' },
                      { value: 'he', label: 'He / Kaiming' },
                      { value: 'normal', label: 'Normal (0, 0.05)' },
                      { value: 'uniform', label: 'Uniform [-0.1, 0.1]' },
                    ]} onChange={(i) => s.updateNodeParams(node.id, { init: i as never })} />
                  </Field>
                  <Checkbox checked={node.data.params.use_bias} label="Use bias vector (b)"
                    onChange={(b) => s.updateNodeParams(node.id, { use_bias: b })} />
                  {node.data.kind === 'output' && (
                    <Field label="Task type"
                      hint="Informational — the dataset determines the real task. Classification with cross-entropy needs one neuron per class and usually softmax; binary BCE needs 1 neuron with sigmoid; regression needs 1 neuron with linear.">
                      <Select value={node.data.params.task} options={[
                        { value: 'auto', label: 'Auto (from dataset)' },
                        { value: 'classification', label: 'Classification (softmax + cross-entropy)' },
                        { value: 'binary', label: 'Binary (1 neuron + sigmoid + BCE)' },
                        { value: 'regression', label: 'Regression (1 neuron + linear + MSE/MAE)' },
                      ]} onChange={(t) => s.updateNodeParams(node.id, { task: t as never })} />
                    </Field>
                  )}
                </>
              )}

              {node.data.kind === 'dropout' && (
                <>
                  <Field label="Dropout rate" hint="Probability of zeroing each activation in training. Typical 0.2–0.5.">
                    <NumInput value={node.data.params.dropout_rate} min={0.05} max={0.95} step={0.05}
                      onChange={(v2) => s.updateNodeParams(node.id, { dropout_rate: Math.min(0.95, Math.max(0.05, v2)) })} />
                  </Field>
                  <Field label="Training mode"
                    hint="'Training only' is standard: dropout is a no-op at inference, so predictions are deterministic. 'Always' applies it even during simulation, which lets you observe its effect but makes predictions non-deterministic.">
                    <Select value={node.data.params.dropout_mode} options={[
                      { value: 'train_only', label: 'Training only (standard)' },
                      { value: 'always', label: 'Always (also at inference)' },
                    ]} onChange={(m) => s.updateNodeParams(node.id, { dropout_mode: m as never })} />
                  </Field>
                </>
              )}

              {node.data.kind === 'batchnorm' && (
                <>
                  <Field label="Momentum" hint="How fast the running mean/variance track the batch statistics (PyTorch default 0.1). Values near 1 adapt very slowly.">
                    <NumInput value={node.data.params.momentum} min={0} max={0.999} step={0.01}
                      onChange={(v2) => s.updateNodeParams(node.id, { momentum: Math.min(0.999, Math.max(0, v2)) })} />
                  </Field>
                  <Field label="Epsilon (ε)" hint="Added inside the square root to avoid dividing by zero when the variance is tiny. Standard value 1e-5.">
                    <NumInput value={node.data.params.eps} min={0.00000001} max={1} step={0.00001}
                      onChange={(v2) => s.updateNodeParams(node.id, { eps: Math.max(1e-8, v2) })} />
                  </Field>
                  <Checkbox checked={node.data.params.affine} label="Learnable scale γ and shift β"
                    onChange={(b) => s.updateNodeParams(node.id, { affine: b })} />
                  <div className="text-[11px] text-slate-400 bg-lab-800/60 rounded p-2 leading-relaxed">
                    <div className="font-mono text-sky-200">x̂ = (x − μ) / √(σ² + ε)</div>
                    <div className="font-mono text-emerald-200">y = γx̂ + β</div>
                    <p className="text-[10px] text-slate-500 mt-1">
                      With affine on, each channel learns 2 parameters (γ and β). Turn it off to
                      normalize without learnable parameters. The live μ, σ², x̂, γ and β for the
                      probe input appear in the Math tab.
                    </p>
                  </div>
                </>
              )}

              {node.data.kind === 'flatten' && (
                <div className="text-[11px] text-slate-400 bg-lab-800/60 rounded p-2 leading-relaxed">
                  <div className="font-semibold text-slate-300 mb-1">No settings needed</div>
                  Flatten has no parameters — it only re-orders memory, turning the incoming
                  feature map into a flat vector for the next Dense layer. The input and output
                  shapes are shown below and on the node.
                </div>
              )}

              {node.data.kind === 'globalavgpool' && (
                <div className="text-[11px] text-slate-400 bg-lab-800/60 rounded p-2 leading-relaxed">
                  <div className="font-semibold text-slate-300 mb-1">No settings needed</div>
                  Averages every spatial position of each channel independently, collapsing
                  H × W × C into just C numbers. It has no parameters and needs no Flatten
                  before a following Dense layer.
                </div>
              )}

              {node.data.kind === 'conv2d' && (
                <>
                  <Field label="Filters" hint="Number of learned filters (output channels). Each filter scans the whole input and produces one feature map, so this sets the channel count of the output.">
                    <NumInput value={node.data.params.filters} min={1} max={512} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { filters: Math.max(1, Math.min(512, Math.round(v2))) })} />
                  </Field>
                  <Field label="Kernel size" hint="Side length of the square sliding window (k × k). Smaller kernels (3) are the usual choice; larger ones see more context but need more parameters.">
                    <NumInput value={node.data.params.kernel_size} min={1} max={15} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { kernel_size: Math.max(1, Math.min(15, Math.round(v2))) })} />
                  </Field>
                  <Field label="Stride" hint="How many pixels the window moves each step. Stride 1 keeps most detail; stride 2 downsamples while convolving.">
                    <NumInput value={node.data.params.stride} min={1} max={8} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { stride: Math.max(1, Math.round(v2)) })} />
                  </Field>
                  <Field label="Padding mode"
                    hint="'valid' = no padding, the output shrinks. 'same' = pad so the spatial size is preserved (requires stride 1).">
                    <Select value={node.data.params.padding_mode} options={[
                      { value: 'valid', label: "Valid (no padding)" },
                      { value: 'same', label: "Same (preserve size)" },
                    ]} onChange={(m) => s.updateNodeParams(node.id, { padding_mode: m as never })} />
                  </Field>
                  {node.data.params.padding_mode === 'valid' && (
                    <Field label="Padding (pixels)" hint="Zero-padding added around the input. Must be smaller than the kernel size.">
                      <NumInput value={node.data.params.padding} min={0} max={14} step={1}
                        onChange={(v2) => s.updateNodeParams(node.id, { padding: Math.max(0, Math.round(v2)) })} />
                    </Field>
                  )}
                  <Field label="Activation" hint="Non-linearity applied to the convolution result before the next layer.">
                    <Select value={node.data.params.activation} options={activationOptions}
                      onChange={(a) => s.updateNodeParams(node.id, { activation: a })} />
                  </Field>
                  <Field label="Weight initialization" hint={INIT_HINTS[node.data.params.init]}>
                    <Select value={node.data.params.init} options={[
                      { value: 'xavier', label: 'Xavier / Glorot' },
                      { value: 'he', label: 'He / Kaiming' },
                      { value: 'normal', label: 'Normal (0, 0.05)' },
                      { value: 'uniform', label: 'Uniform [-0.1, 0.1]' },
                    ]} onChange={(i) => s.updateNodeParams(node.id, { init: i as never })} />
                  </Field>
                  <Checkbox checked={node.data.params.use_bias} label="Use bias per filter"
                    onChange={(b) => s.updateNodeParams(node.id, { use_bias: b })} />
                </>
              )}

              {(node.data.kind === 'maxpool' || node.data.kind === 'avgpool') && (
                <>
                  <Field label="Pool size" hint="Side length of the square window (k × k). The window is reduced to a single number.">
                    <NumInput value={node.data.params.pool_size} min={1} max={16} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { pool_size: Math.max(1, Math.min(16, Math.round(v2))) })} />
                  </Field>
                  <Field label="Stride" hint="How far the window moves each step. Equal to the pool size gives non-overlapping windows.">
                    <NumInput value={node.data.params.pool_stride} min={1} max={16} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { pool_stride: Math.max(1, Math.min(16, Math.round(v2))) })} />
                  </Field>
                  <Field label="Padding" hint="Optional zero-padding at the border. Most pooling layers use 0.">
                    <NumInput value={node.data.params.pool_padding} min={0} max={8} step={1}
                      onChange={(v2) => s.updateNodeParams(node.id, { pool_padding: Math.max(0, Math.round(v2)) })} />
                  </Field>
                </>
              )}

              {v && (
                <div className="text-[11px] font-mono text-slate-400 border-t border-lab-700/60 pt-2 space-y-0.5">
                  {(() => {
                    const sh = v.layers.find((l) => l.id === node.id)
                    if (!sh) return <div className="text-slate-500">Not in the active chain.</div>
                    return (
                      <>
                        <div>Input shape: {shapeText(sh.in_shape)}</div>
                        <div>Output shape: <b className="text-sky-200">{shapeText(sh.out_shape)}</b></div>
                        <div>Parameters: <span className="text-lab-accent">{sh.params.toLocaleString()}</span></div>
                        {sh.note && <div className="text-slate-500 leading-snug pt-1 normal-case">{sh.note}</div>}
                      </>
                    )
                  })()}
                </div>
              )}

              <div className="flex gap-2 pt-1 border-t border-lab-700/60">
                <button className="btn-ghost flex-1 !py-1 text-xs" onClick={() => s.duplicateNode(node.id)}
                  title="Create an independent copy with the same configuration">
                  ⧉ Duplicate
                </button>
                <button className="btn-ghost flex-1 !py-1 text-xs !text-rose-400"
                  onClick={() => s.onNodesChange([{ type: 'remove', id: node.id }])}
                  title="Delete this layer (keyboard: Delete)">
                  🗑 Delete
                </button>
              </div>
              <div className="text-[10px] text-slate-500">
                Tip: drag an edge's end onto another layer to reconnect it.
              </div>
            </div>
          )}
        </div>

        <div>
          <div className="panel-title mb-2">Architecture validation</div>
          {!v && <div className="text-xs text-slate-500">Add layers to see live validation.</div>}
          {v && (
            <div className="space-y-2">
              <div className={`panel p-2 text-sm font-medium ${v.ok ? 'text-emerald-400' : 'text-rose-400'}`}>
                {v.ok
                  ? `✓ Network can train — chain of ${v.order.length} layers, ${v.total_params.toLocaleString()} params (${v.input_dim} → ${v.output_dim})${v.is_cnn ? ' · image (CNN) architecture' : ''}`
                  : '✗ Network cannot train yet'}
              </div>
              {v.errors.map((e, i) => (
                <div key={`e${i}`} className="panel border-rose-500/40 p-2 text-xs">
                  <div className="text-rose-400 font-semibold">❌ {e.message}</div>
                  {e.suggestion && <div className="text-slate-400 mt-1">💡 {e.suggestion}</div>}
                </div>
              ))}
              {v.warnings.map((w, i) => (
                <div key={`w${i}`} className="panel border-amber-500/40 p-2 text-xs">
                  <div className="text-amber-400 font-semibold">⚠ {w.message}</div>
                  {w.suggestion && <div className="text-slate-400 mt-1">💡 {w.suggestion}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}
