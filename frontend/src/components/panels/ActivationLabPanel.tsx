import { useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmt, pct } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { Field, Stat, TextInput } from '../common/Stat'
import { tooltipStyle } from './TrainingPanel'

const EXAMPLES = [
  { name: 'Swish', f: 'x / (1 + exp(-x))' },
  { name: 'Mish', f: 'x * tanh(log(1 + exp(x)))' },
  { name: 'LeakySquash', f: 'max(0, x) + 0.01*x' },
  { name: 'ScaledTanh', f: '1.7159 * tanh(2*x/3)' },
  { name: 'SoftCube', f: 'x^3 / (1 + abs(x^3))' },
]

export function ActivationLabPanel() {
  const s = useAppStore()
  const [name, setName] = useState('my_activation')
  const [formula, setFormula] = useState('x / (1 + exp(-x))')
  const [busy, setBusy] = useState(false)
  const res = s.activationCheck

  const check = async () => {
    setBusy(true)
    await s.checkFormula(formula)
    setBusy(false)
  }

  const curveData = res?.ok
    ? res.xs.map((x, i) => ({ x: Number(x.toFixed(3)), 'f(x)': res.ys[i] ?? undefined, "f'(x)": res.dys[i] ?? undefined }))
    : []

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      <div className="w-[330px] shrink-0 space-y-3 overflow-y-auto">
        <div className="panel p-3 space-y-2.5">
          <div className="panel-title">Define a custom activation</div>
          <Field label="Function name" hint="Letters, digits, underscores. It will appear as custom::<name> in layer settings.">
            <TextInput value={name} onChange={setName} />
          </Field>
          <Field label="Formula f(x)" hint="Variable: x. Operators: + - * / ^ . Functions: exp log sqrt sin cos tan abs tanh sigmoid relu softplus min max pow clip … constants pi, e.">
            <TextInput className="input font-mono" value={formula} onChange={setFormula}
              placeholder="e.g. x / (1 + exp(-x))" />
          </Field>
          <div className="flex gap-2">
            <button className="btn flex-1" disabled={busy} onClick={() => void check()}>
              {busy ? 'Testing…' : '🧪 Test & plot'}
            </button>
            <button className="btn-ok flex-1" disabled={!res?.ok || !name}
              onClick={() => void s.saveCustom(name, formula)}>
              💾 Save to library
            </button>
          </div>
          <div className="text-[10px] text-slate-500 leading-snug">
            🔒 Safety: formulas are parsed to a restricted AST and compiled into whitelisted tensor ops —
            no arbitrary code is ever executed. Gradients come from autograd, so no hand-written derivative is needed.
          </div>
        </div>

        <div className="panel p-3">
          <div className="panel-title mb-1.5">Examples</div>
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLES.map((e) => (
              <button key={e.name} className="btn-ghost !py-0.5 !px-2 text-[11px]" title={e.f}
                onClick={() => { setFormula(e.f); setName(e.name.toLowerCase()) }}>
                {e.name}
              </button>
            ))}
          </div>
        </div>

        <div className="panel p-3">
          <div className="panel-title mb-1.5">Saved library</div>
          {s.savedCustoms.length === 0 && <div className="text-xs text-slate-500">Nothing saved yet.</div>}
          {s.savedCustoms.map((c) => (
            <div key={c.name} className="flex items-center gap-2 text-xs py-1 border-b border-lab-800 last:border-0">
              <span className="font-mono text-violet-300">{c.name}</span>
              <span className="font-mono text-slate-500 truncate flex-1" title={c.formula}>{c.formula}</span>
              <button className="text-rose-400" title="Delete" onClick={() => void s.deleteCustom(c.name)}>✕</button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 space-y-3 min-w-0 overflow-y-auto">
        {res && (
          <div className="flex gap-2 flex-wrap">
            <Stat label="Status" value={res.ok ? '✓ valid & finite on [-5, 5]' : '✗ problems found'}
              accent={res.ok ? 'text-emerald-400' : 'text-rose-400'} />
            {res.stats && (
              <>
                <Stat label="f range" value={`${fmt(res.stats.min ?? null, 2)} … ${fmt(res.stats.max ?? null, 2)}`} />
                <Stat label="max |f'(x)|" hint="Large derivatives can destabilise training" value={fmt(res.stats.max_abs_derivative ?? null, 3)} />
                <Stat label="derivative defined" value={pct(res.stats.derivative_defined_fraction)} />
              </>
            )}
          </div>
        )}
        {res?.error && <div className="panel border-rose-500/50 p-2.5 text-xs text-rose-300">❌ {res.error}</div>}
        {res && res.issues.filter((i) => i !== res.error).map((i, k) => (
          <div key={k} className="panel border-amber-500/40 p-2.5 text-xs text-amber-300">⚠ {i}</div>
        ))}
        {res?.ok && curveData.length > 0 && (
          <div className="panel p-2 h-64">
            <div className="panel-title px-1 pt-1">f(x) and autograd derivative f'(x), x ∈ [-5, 5]</div>
            <ResponsiveContainer width="100%" height="88%">
              <LineChart data={curveData} margin={{ top: 8, right: 12, bottom: 0, left: -14 }}>
                <XAxis dataKey="x" fontSize={10} stroke="#64748b" />
                <YAxis fontSize={10} stroke="#64748b" />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="f(x)" stroke="#38bdf8" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line type="monotone" dataKey="f'(x)" stroke="#a78bfa" dot={false} strokeWidth={1.4} strokeDasharray="5 3" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {!res && (
          <div className="text-xs text-slate-500 max-w-lg">
            Write a formula above and press <b>Test & plot</b>. If it passes validation, save it — then select it on any
            Dense/Output layer as <b>custom::name</b>, retrain, and compare against standard activations in
            <b> Research Mode</b>.
          </div>
        )}
      </div>
    </div>
  )
}
