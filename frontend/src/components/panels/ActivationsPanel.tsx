import { useEffect, useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmt, histogram, pct } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { Empty, Select, Stat } from '../common/Stat'
import { tooltipStyle } from './TrainingPanel'

export function ActivationsPanel() {
  const s = useAppStore()
  const snaps = s.job.snapshots
  const [snapIdx, setSnapIdx] = useState(0)
  const [layerId, setLayerId] = useState<string>('')

  const snap = snaps[snaps.length - 1 - snapIdx] ?? snaps[snaps.length - 1]
  useEffect(() => {
    if (!layerId && snap?.activations.length) setLayerId(snap.activations[0].layer_id)
  }, [snap, layerId])

  if (!snaps.length) {
    return <Empty title="No activation statistics yet">
      During training, the backend captures real activation statistics every few epochs
      (see “Snapshot every” in the Training tab). Train the network to populate this view.
    </Empty>
  }
  const stat = snap.activations.find((a) => a.layer_id === layerId) ?? snap.activations[0]
  const hist = histogram(stat.sample, 24).map((b) => ({ bin: fmt(b.x0, 2), count: b.count }))

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      <div className="w-72 shrink-0 space-y-2 overflow-y-auto">
        <div className="panel p-3 space-y-2">
          <div className="panel-title">Snapshot</div>
          <Select value={String(snapIdx)} onChange={(v) => setSnapIdx(Number(v))}
            options={snaps.map((sn, i) => ({ value: String(snaps.length - 1 - i), label: `epoch ${sn.epoch}${i === 0 ? ' (latest)' : ''}` })).reverse()} />
          <div className="panel-title">Layer</div>
          <Select value={stat.layer_id} onChange={setLayerId}
            options={snap.activations.map((a) => ({ value: a.layer_id, label: `${a.label} (${a.activation})` }))} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Stat label="Mean" value={fmt(stat.mean)} />
          <Stat label="Std" value={fmt(stat.std)} />
          <Stat label="Min / Max" value={`${fmt(stat.min, 2)} / ${fmt(stat.max, 2)}`} />
          <Stat label="≈0 outputs" hint="Fraction of activations with |a| < 0.001" value={pct(stat.frac_near_zero)} />
          {stat.dead_neuron_frac != null && (
            <Stat label="Dead neurons" hint="Neurons that output ≈0 for EVERY sample in the batch — classic dead ReLU symptom."
              value={pct(stat.dead_neuron_frac)} accent={stat.dead_neuron_frac >= 0.5 ? 'text-rose-400' : 'text-emerald-400'} />
          )}
          {stat.frac_saturated != null && (
            <Stat label="Saturated" hint="Activations in the flat tails of sigmoid/tanh where gradients ≈ 0."
              value={pct(stat.frac_saturated)} accent={stat.frac_saturated >= 0.3 ? 'text-amber-400' : 'text-emerald-400'} />
          )}
        </div>
      </div>
      <div className="flex-1 panel p-2 min-w-0">
        <div className="panel-title px-1 pt-1">Activation distribution — {stat.label} ({stat.activation}) · epoch {snap.epoch}</div>
        <ResponsiveContainer width="100%" height="88%">
          <BarChart data={hist} margin={{ top: 10, right: 16, bottom: 0, left: -18 }}>
            <XAxis dataKey="bin" fontSize={9} stroke="#64748b" interval={3} />
            <YAxis fontSize={10} stroke="#64748b" />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="count" fill="#a78bfa" radius={[2, 2, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
