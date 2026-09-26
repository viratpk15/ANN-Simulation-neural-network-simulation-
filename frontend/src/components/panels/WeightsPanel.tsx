import { useEffect, useMemo, useState } from 'react'
import { fmt, weightColor } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { Empty, Select } from '../common/Stat'

export function WeightsPanel() {
  const s = useAppStore()
  const [layerId, setLayerId] = useState<string>('')
  const runId = s.lastRunId ?? s.job.id

  useEffect(() => {
    if (runId && !s.weights) void s.fetchWeights()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  const layers = useMemo(() => Object.entries(s.weights ?? {}), [s.weights])
  useEffect(() => {
    if (!layerId && layers.length) setLayerId(layers[0][0])
  }, [layers, layerId])

  if (!runId) {
    return <Empty title="Train first">Weight matrices are extracted from the trained model — every number is a real learned parameter.</Empty>
  }
  const current = layers.find(([id]) => id === layerId)?.[1]
  const labelFor = (id: string) => s.nodes.find((n) => n.id === id)?.data.label ?? id

  return (
    <div className="h-full flex gap-3 p-3 overflow-auto">
      <div className="w-64 shrink-0 space-y-2">
        <div className="panel p-3 space-y-2">
          <div className="panel-title">Layer</div>
          <Select value={layerId} onChange={setLayerId}
            options={layers.map(([id, w]) => ({ value: id, label: `${labelFor(id)} — W ${w.shape[0]}×${w.shape[1]}` }))} />
          <button className="btn-ghost w-full !py-1" onClick={() => void s.fetchWeights()}>⟳ Refresh from model</button>
        </div>
        {current && (
          <div className="panel p-3 text-xs font-mono space-y-1">
            <div className="panel-title mb-1">W statistics</div>
            <div className="flex justify-between"><span>min</span><span className="text-sky-400">{fmt(current.stats.min)}</span></div>
            <div className="flex justify-between"><span>max</span><span className="text-rose-400">{fmt(current.stats.max)}</span></div>
            <div className="flex justify-between"><span>mean</span><span>{fmt(current.stats.mean)}</span></div>
            <div className="flex justify-between"><span>std</span><span>{fmt(current.stats.std)}</span></div>
            <div className="text-slate-500 pt-1">
              Colour: <span className="text-sky-400">■ negative</span> → <span className="text-rose-400">■ positive</span>; intensity ∝ |w|.
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 space-y-3 min-w-0">
        {current ? (
          <div className="panel p-3 overflow-auto">
            {current.display_note && <div className="text-[11px] text-amber-400 mb-1">⚠ {current.display_note}</div>}
            <div className="panel-title mb-2">Weight matrix W (rows = output neurons, cols = inputs)</div>
            <div className="inline-block">
              <div className="inline-grid gap-[1px]" style={{ gridTemplateColumns: `repeat(${current.weights[0]?.length ?? 1}, minmax(10px, 18px))` }}>
                {current.weights.flatMap((row, r) =>
                  row.map((v, c) => (
                    <div key={`${r}-${c}`} className="aspect-square rounded-[2px]"
                      style={{ background: weightColor(v, current.stats.min, current.stats.max, s.theme === 'dark') }}
                      title={`W[${r}][${c}] = ${v}`} />
                  )),
                )}
              </div>
            </div>
            {current.bias && (
              <div className="mt-3">
                <div className="panel-title mb-1">Bias vector b</div>
                <div className="inline-flex gap-[2px]">
                  {current.bias.map((b, i) => (
                    <div key={i} className="w-4 h-4 rounded-[2px]"
                      style={{ background: weightColor(b, current.stats.min, current.stats.max, s.theme === 'dark') }}
                      title={`b[${i}] = ${b}`} />
                  ))}
                </div>
              </div>
            )}
            <div className="mt-3">
              <WeightTable weights={current.weights} bias={current.bias ?? null} />
            </div>
          </div>
        ) : (
          <Empty title="No trainable layers found">Dense/Output layers have weight matrices; input/dropout don't.</Empty>
        )}
      </div>
    </div>
  )
}

function WeightTable({ weights, bias }: { weights: number[][]; bias: number[] | null }) {
  const [open, setOpen] = useState(false)
  if (weights.length > 24 || (weights[0]?.length ?? 0) > 12) return null
  return (
    <div>
      <button className="btn-ghost !py-0.5 !px-2 text-[11px]" onClick={() => setOpen(!open)}>
        {open ? 'Hide' : 'Show'} exact numeric table
      </button>
      {open && (
        <table className="mt-2 text-[10px] font-mono border-collapse">
          <thead>
            <tr>
              <th className="border border-lab-700 px-1.5 py-0.5 text-slate-500">W</th>
              {weights[0].map((_, c) => <th key={c} className="border border-lab-700 px-1.5 py-0.5 text-slate-500">x{c + 1}</th>)}
              {bias && <th className="border border-lab-700 px-1.5 py-0.5 text-amber-400">b</th>}
            </tr>
          </thead>
          <tbody>
            {weights.map((row, r) => (
              <tr key={r}>
                <td className="border border-lab-700 px-1.5 py-0.5 text-slate-500">n{r + 1}</td>
                {row.map((v, c) => (
                  <td key={c} className="border border-lab-700 px-1.5 py-0.5"
                    style={{ color: v >= 0 ? '#f87171' : '#60a5fa' }}>{fmt(v, 4)}</td>
                ))}
                {bias && <td className="border border-lab-700 px-1.5 py-0.5 text-amber-300">{fmt(bias[r], 4)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
