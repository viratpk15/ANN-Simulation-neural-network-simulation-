import { useEffect, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { pct } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { api } from '../../api/client'
import { Empty } from '../common/Stat'
import { tooltipStyle } from './TrainingPanel'

const COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15']

/** Research mode: named experiments = recorded runs with config, metrics,
 * curves, activation & gradient stats, timestamps — comparable side by side. */
export function ExperimentsPanel() {
  const s = useAppStore()
  const [selected, setSelected] = useState<string[]>([])

  useEffect(() => {
    void s.loadExperiments()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.job.state === 'finished'])

  const toggle = (id: string) => {
    setSelected((cur) => cur.includes(id) ? cur.filter((x) => x !== id) : cur.length < 6 ? [...cur, id] : cur)
  }

  const runs = s.compareRuns
  const chartData = (() => {
    if (!runs) return []
    const maxLen = Math.max(...runs.map((r) => r.history.length))
    const rows: Record<string, number | undefined>[] = []
    for (let e = 0; e < maxLen; e++) {
      const row: Record<string, number | undefined> = { epoch: e + 1 }
      runs.forEach((r, i) => { row[`loss_${i}`] = r.history[e]?.train_loss })
      rows.push(row)
    }
    return rows
  })()

  const allRuns = s.experiments.flatMap((e) => e.runs.map((r) => ({ ...r, expName: e.name, expId: e.id })))

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      <div className="w-[380px] shrink-0 space-y-2 overflow-y-auto">
        <div className="panel p-2.5 flex items-center gap-2">
          <button className="btn-ghost !py-1 flex-1" onClick={() => void s.loadExperiments()}>⟳ Refresh</button>
          <button className="btn-primary !py-1 flex-[2]" disabled={selected.length < 2}
            onClick={() => void s.fetchCompare(selected)}>
            Compare {selected.length} run{selected.length === 1 ? '' : 's'} →
          </button>
        </div>
        <div className="text-[11px] text-slate-500 px-1">
          To record an experiment: set an <b>Experiment name</b> in the Training tab (e.g. “iris-relu-lr0.001”),
          then train. Vary one thing at a time (activation / lr / architecture) and compare here.
        </div>
        {s.experiments.length === 0 && (
          <Empty title="No experiments yet">Finished runs tagged with an experiment name appear here with their curves, stats and timestamps.</Empty>
        )}
        {s.experiments.map((exp) => (
          <div key={exp.id} className="panel p-2.5">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-lab-accent2">{exp.name}</span>
              <span className="text-[10px] text-slate-500 ml-auto">{new Date(exp.created_at).toLocaleString()}</span>
              <button className="text-rose-400 text-xs px-1" title="Delete experiment"
                onClick={() => void api.del(`/api/experiments/${exp.id}`).then(() => s.loadExperiments())}>✕</button>
            </div>
            {(exp.runs ?? []).map((r) => {
              const sel = selected.includes(r.id)
              const tm = r.final?.test_metrics
              const summary = tm
                ? ('accuracy' in tm ? pct(tm.accuracy) : `R² ${tm.r2?.toFixed(3)}`)
                : r.status
              return (
                <label key={r.id} className={`flex items-center gap-2 mt-1.5 px-1.5 py-1 rounded text-xs cursor-pointer ${sel ? 'bg-sky-600/25' : 'hover:bg-lab-800'}`}>
                  <input type="checkbox" checked={sel} onChange={() => toggle(r.id)} className="accent-sky-500" />
                  <span className="font-mono text-slate-400">{r.id.slice(0, 8)}</span>
                  <span className="text-slate-300">{summary}</span>
                  <button className="ml-auto text-[10px] text-sky-400 hover:underline"
                    onClick={(e) => { e.preventDefault(); void s.loadRun(r.id); s.setTab('training') }}>
                    load
                  </button>
                </label>
              )
            })}
          </div>
        ))}
      </div>

      <div className="flex-1 min-w-0 space-y-3 overflow-y-auto">
        {runs ? (
          <>
            <div className="panel p-2 h-56">
              <div className="panel-title px-1 pt-1">Training loss overlay</div>
              <ResponsiveContainer width="100%" height="88%">
                <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                  <XAxis dataKey="epoch" fontSize={10} stroke="#64748b" />
                  <YAxis fontSize={10} stroke="#64748b" />
                  <Tooltip contentStyle={tooltipStyle} />
                  {runs.map((r, i) => (
                    <Line key={r.id} type="monotone" dataKey={`loss_${i}`} name={r.id.slice(0, 8)}
                      stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.6} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="panel p-3 overflow-auto">
              <div className="panel-title mb-2">Side-by-side metrics (held-out test set)</div>
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-slate-500 text-left">
                    <th className="pr-3 py-1">run</th>
                    <th className="pr-3 py-1">seed</th>
                    <th className="pr-3 py-1">lr</th>
                    <th className="pr-3 py-1">opt</th>
                    <th className="pr-3 py-1">epochs</th>
                    <th className="pr-3 py-1">time</th>
                    {runs[0]?.final && Object.keys(runs[0].final.test_metrics).map((m) => (
                      <th key={m} className="pr-3 py-1">{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r, i) => (
                    <tr key={r.id} className="border-t border-lab-800">
                      <td className="pr-3 py-1" style={{ color: COLORS[i % COLORS.length] }}>{r.id.slice(0, 8)}</td>
                      <td className="pr-3 py-1">{r.config.seed}</td>
                      <td className="pr-3 py-1">{r.config.learning_rate}</td>
                      <td className="pr-3 py-1">{r.config.optimizer}</td>
                      <td className="pr-3 py-1">{r.config.epochs}</td>
                      <td className="pr-3 py-1">{r.duration_s?.toFixed(1)}s</td>
                      {r.final && Object.values(r.final.test_metrics).map((v, mi) => (
                        <td key={mi} className="pr-3 py-1 text-emerald-300">
                          {'accuracy' === Object.keys(r.final!.test_metrics)[mi] ||
                            ['precision_macro', 'recall_macro', 'f1_macro'].includes(Object.keys(r.final!.test_metrics)[mi])
                            ? pct(v) : v.toFixed(4)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-[10px] text-slate-500 mt-2">
                Every run also stores its full history, activation & gradient snapshots — load one to inspect it in the other tabs.
              </div>
            </div>
          </>
        ) : (
          <Empty title="Select 2–6 runs and press Compare">
            Overlay loss curves and compare accuracy / precision / recall / F1 / R² side by side.
          </Empty>
        )}
      </div>
    </div>
  )
}
