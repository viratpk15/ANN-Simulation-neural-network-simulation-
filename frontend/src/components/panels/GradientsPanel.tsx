import { useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmt } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { MathBlock } from '../common/MathBlock'
import { Empty, Select, Stat } from '../common/Stat'
import { tooltipStyle } from './TrainingPanel'

export function GradientsPanel() {
  const s = useAppStore()
  const snaps = s.job.snapshots
  const [snapIdx, setSnapIdx] = useState(0)

  if (!snaps.length) {
    return <Empty title="No gradient statistics yet">
      Gradient norms are captured from a real backward pass every few epochs during training (and stored with the run).
    </Empty>
  }
  const snap = snaps[snaps.length - 1 - snapIdx] ?? snaps[snaps.length - 1]
  const grads = snap.gradients.layers
  const norms = grads.map((g) => g.w_grad_norm)
  const data = grads.map((g) => ({ layer: g.label, norm: Number(g.w_grad_norm.toExponential(3)), meanAbs: g.w_grad_mean_abs }))
  const first = norms[0] ?? 0
  const lastN = norms[norms.length - 1] ?? 0
  const exploding = norms.some((n) => n > 1e3)
  const vanishing = first < 1e-7 || (lastN > 0 && first / lastN < 1e-2 && first < 1e-5)
  const exG = grads[0]
  const ex = exG?.example_weight

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      <div className="w-72 shrink-0 space-y-2 overflow-y-auto">
        <div className="panel p-3 space-y-2">
          <div className="panel-title">Snapshot</div>
          <Select value={String(snapIdx)} onChange={(v) => setSnapIdx(Number(v))}
            options={snaps.map((sn, i) => ({ value: String(snaps.length - 1 - i), label: `epoch ${sn.epoch}${i === 0 ? ' (latest)' : ''}` })).reverse()} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Stat label="Global grad norm" hint="√Σ‖∇θ‖² over every parameter"
            value={fmt(snap.gradients.total_norm, 3)} accent={exploding ? 'text-rose-400' : 'text-emerald-400'} />
          <Stat label="Loss (snapshot batch)" value={fmt(snap.gradients.loss_on_batch)} />
        </div>
        {exploding && (
          <div className="panel border-rose-500/50 p-2 text-xs text-rose-300">
            ⚠ Possible exploding gradients — a layer norm exceeds 1000. Consider lowering the learning rate. See AI Diagnosis.
          </div>
        )}
        {vanishing && !exploding && (
          <div className="panel border-amber-500/50 p-2 text-xs text-amber-300">
            ⚠ Possible vanishing gradients — the first trainable layer receives almost no gradient ({first.toExponential(2)}).
          </div>
        )}
        {ex && (
          <div className="panel p-3 space-y-1">
            <div className="panel-title">One weight's update (this batch)</div>
            <MathBlock tex={`w_{new} = w_{old} - \\eta \\, \\frac{\\partial L}{\\partial w}`} />
            <div className="font-mono text-[11px] text-slate-300 space-y-0.5">
              <div>w_old = {fmt(ex.w, 6)}</div>
              <div>∂L/∂w = {fmt(ex.grad, 6)}</div>
              <div>η = {fmt(s.trainConfig.learning_rate, 5)}</div>
              <div>⇒ w_new = {fmt(ex.w - s.trainConfig.learning_rate * ex.grad, 6)}</div>
            </div>
            <div className="text-[10px] text-slate-500">layer {exG?.label}, weight [{ex.index.join(', ')}]</div>
          </div>
        )}
      </div>

      <div className="flex-1 panel p-2 min-w-0">
        <div className="panel-title px-1 pt-1">Layer-wise weight-gradient norms ‖∇W‖ — epoch {snap.epoch} (log scale)</div>
        <ResponsiveContainer width="100%" height="88%">
          <BarChart data={data} margin={{ top: 10, right: 16, bottom: 20, left: 4 }}>
            <XAxis dataKey="layer" fontSize={10} stroke="#64748b" angle={-18} textAnchor="end" height={44} />
            <YAxis fontSize={10} stroke="#64748b" scale="log" domain={['auto', 'auto']} tickFormatter={(v) => Number(v).toExponential(0)} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => Number(v).toExponential(3)} />
            <Bar dataKey="norm" isAnimationActive={false} radius={[3, 3, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.norm > 1e3 ? '#f87171' : d.norm < 1e-5 ? '#fbbf24' : '#38bdf8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
