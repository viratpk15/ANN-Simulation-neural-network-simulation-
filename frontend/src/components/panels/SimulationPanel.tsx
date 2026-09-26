import { useEffect, useMemo, useRef, useState } from 'react'
import { activationColor, fmt } from '../../lib/format'
import { shapeText, useAppStore } from '../../store/useAppStore'
import { Empty, Select } from '../common/Stat'

const MAX_NEURONS = 12

/** Animated data-flow view. Neuron colours come from the *real* forward
 * trace of the currently trained run (Math tab shows the same numbers). */
export function SimulationPanel() {
  const s = useAppStore()
  const [playing, setPlaying] = useState(false)
  const [mode, setMode] = useState<'auto' | 'step'>('auto')
  const [speed, setSpeed] = useState(600) // ms per layer hop
  const [phase, setPhase] = useState(-1)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)

  const chain = useMemo(() => {
    const v = s.validation
    if (!v?.ok) return []
    const label = (id: string) => s.nodes.find((n) => n.id === id)?.data.label ?? id
    return v.order.map((id) => {
      const sh = v.layers.find((l) => l.id === id)
      return {
        id,
        label: label(id),
        kind: sh?.kind ?? 'dense',
        n: sh?.out_features ?? 1,
        in: sh?.in_features ?? null,
        shape: sh?.out_shape ?? null,
        // image layers have a rank-3 output; show "H × W × C" instead of a count
        isImage: (sh?.out_shape?.length ?? 1) === 3,
      }
    })
  }, [s.validation, s.nodes])

  // values from the real forward trace
  const traceByLayer = useMemo(() => {
    const m = new Map<string, number[]>()
    for (const l of s.forwardTrace ?? []) {
      if (l.kind === 'input' || l.kind === 'dropout') m.set(l.id, (l.values ?? []) as number[])
      else if (l.a) m.set(l.id, l.a as number[])
    }
    return m
  }, [s.forwardTrace])

  useEffect(() => {
    if (playing && chain.length) {
      timer.current = setInterval(() => setPhase((p) => (p + 1) % (chain.length + 1)), speed)
    }
    return () => { if (timer.current) clearInterval(timer.current) }
  }, [playing, speed, chain.length])

  const stepForward = () => {
    if (!s.forwardTrace && (s.lastRunId ?? s.job.id)) { void s.fetchTrace() }
    if (mode !== 'step') setMode('step')
    setPlaying(false)
    setPhase((p) => Math.min(p + 1, Math.max(0, chain.length - 1)))
  }
  const stepBack = () => setPhase((p) => Math.max(-1, p - 1))

  if (!chain.length) {
    return <Empty title="Build & validate a network first">The simulation view shows data flowing through your validated feed-forward chain, layer by layer.</Empty>
  }

  const W = Math.max(680, chain.length * 170)
  const H = 260
  const colX = (i: number) => 70 + i * ((W - 140) / Math.max(1, chain.length - 1))
  const activeIdx = phase >= chain.length ? chain.length - 1 : phase
  const showFlows = playing || mode === 'step'

  return (
    <div className="h-full flex flex-col p-3 gap-2 overflow-auto">
      <div className="flex items-center gap-2 flex-wrap">
        <button className="btn-primary !py-1" onClick={() => void s.fetchTrace()}
          disabled={!(s.lastRunId ?? s.job.id)} title="Compute a real forward pass with the probe input (set it in the Predict tab)">
          ⟳ Run forward pass
        </button>
        <button className="btn !py-1" onClick={() => { setPlaying(!playing); setMode('auto') }}
          disabled={!s.forwardTrace} title="Animate signals flowing Input → Output">
          {playing ? '⏸ Pause flow' : '▶ Animate flow'}
        </button>
        <button className="btn !py-1" onClick={stepBack} disabled={!s.forwardTrace}>◀ Layer back</button>
        <button className="btn !py-1" onClick={stepForward} title="Step the forward pass one layer at a time">Step forward ▶</button>
        <div className="w-44">
          <Select value={String(speed)} onChange={(v) => setSpeed(Number(v))} options={[
            { value: '1100', label: '🐢 Slow (1.1s/layer)' },
            { value: '600', label: '🚶 Normal' },
            { value: '220', label: '🏃 Fast' },
            { value: '80', label: '⚡ Very fast' },
          ]} />
        </div>
        {s.tracePrediction && (
          <span className="badge-ok ml-auto">
            ŷ = {s.tracePrediction.type === 'regression'
              ? fmt(s.tracePrediction.value ?? null)
              : `${s.tracePrediction.label} (${s.tracePrediction.probabilities?.map((p) => (p * 100).toFixed(0) + '%').join(' / ')})`}
          </span>
        )}
        {!s.forwardTrace && <span className="text-xs text-slate-500 ml-auto">Train a model, set a probe input in the Predict tab, then “Run forward pass”.</span>}
      </div>

      <div className="flex-1 overflow-x-auto">
        <svg width={W} height={H} className="select-none">
          {/* edges */}
          {showFlows && chain.slice(0, -1).map((l, i) => {
            const aCount = Math.min(l.n, MAX_NEURONS)
            const bCount = Math.min(chain[i + 1].n, MAX_NEURONS)
            const active = i === activeIdx - 1
            const segEdges = []
            for (let a = 0; a < aCount; a++) {
              for (let b = 0; b < bCount; b++) {
                segEdges.push(
                  <line key={`${i}-${a}-${b}`}
                    x1={colX(i)} y1={neuronY(a, aCount, H)}
                    x2={colX(i + 1)} y2={neuronY(b, bCount, H)}
                    stroke={active ? '#38bdf8' : '#3a4a70'}
                    strokeWidth={active ? 1.4 : 0.7}
                    className={active && playing ? 'flow-edge' : ''}
                    style={active && playing ? { animationDuration: `${speed / 500}s` } : undefined}
                    opacity={active ? 0.9 : 0.35}
                  />,
                )
              }
            }
            return segEdges
          })}
          {/* neurons */}
          {chain.map((l, i) => {
            const vals = traceByLayer.get(l.id)
            const count = Math.min(l.n, MAX_NEURONS)
            const shown = vals?.slice(0, count)
            const vmin = shown?.length ? Math.min(...shown) : 0
            const vmax = shown?.length ? Math.max(...shown) : 1
            const active = i === activeIdx
            return (
              <g key={l.id}>
                <text x={colX(i)} y={18} textAnchor="middle" fontSize={11}
                  className={active ? 'fill-sky-300 font-bold' : 'fill-slate-400'}>
                  {l.label} ({shapeText(l.shape)})
                </text>
                {Array.from({ length: count }, (_, j) => {
                  const v = shown?.[j]
                  return (
                    <g key={j} className={active && playing ? 'pulsing' : ''}>
                      <circle cx={colX(i)} cy={neuronY(j, count, H)} r={count > 8 ? 9 : 12}
                        fill={v === undefined ? '#1e2a47' : activationColor(v, vmin, vmax)}
                        stroke={active ? '#38bdf8' : '#4a5a80'} strokeWidth={active ? 2 : 1}
                      />
                      {v !== undefined && count <= 8 && (
                        <text x={colX(i)} y={neuronY(j, count, H) + (count > 8 ? 16 : 18)} textAnchor="middle" fontSize={8.5} className="fill-slate-400 font-mono">
                          {fmt(v, 2)}
                        </text>
                      )}
                    </g>
                  )
                })}
                {l.n > MAX_NEURONS && (
                  <text x={colX(i)} y={H - 12} textAnchor="middle" fontSize={9} className="fill-slate-500">
                    +{l.n - MAX_NEURONS} more…
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>
      <div className="text-[11px] text-slate-500">
        Neuron colour & numbers show real activations from the forward trace. Blue highlighted edges are the
        connections currently “carrying the signal”. Open the <b>Math</b> tab to see the exact z = Wx + b computations
        for the highlighted computation.
      </div>
    </div>
  )
}

function neuronY(j: number, count: number, H: number): number {
  if (count <= 1) return H / 2
  const pad = 42
  return pad + (j * (H - 2 * pad)) / (count - 1)
}
