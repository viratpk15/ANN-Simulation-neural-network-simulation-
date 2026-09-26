import type { ReactNode } from 'react'

export function Stat({ label, value, hint, accent }: { label: string; value: ReactNode; hint?: string; accent?: string }) {
  return (
    <div className="panel px-3 py-2 min-w-[110px]" title={hint}>
      <div className="panel-title flex items-center gap-1">
        {label}
        {hint && <InfoDot text={hint} />}
      </div>
      <div className={`text-lg font-mono font-semibold mt-0.5 ${accent ?? ''}`}>{value}</div>
    </div>
  )
}

export function InfoDot({ text }: { text: string }) {
  return (
    <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-lab-600 text-[9px] text-slate-300 cursor-help"
      title={text}>?</span>
  )
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="panel-title flex items-center gap-1 mb-1">
        {label}
        {hint && <InfoDot text={hint} />}
      </span>
      {children}
    </label>
  )
}

export function NumInput({ value, onChange, step = 'any', min, max, disabled }:
  { value: number | string; onChange: (v: number) => void; step?: number | string; min?: number; max?: number; disabled?: boolean }) {
  return (
    <input type="number" className="input-num" value={value} step={step} min={min} max={max} disabled={disabled}
      onChange={(e) => {
        const v = parseFloat(e.target.value)
        if (!Number.isNaN(v)) onChange(v)
      }} />
  )
}

export function TextInput({ value, onChange, placeholder, className = 'input' }:
  { value: string; onChange: (v: string) => void; placeholder?: string; className?: string }) {
  return <input className={className} value={value} placeholder={placeholder}
    onChange={(e) => onChange(e.target.value)} />
}

export function Select({ value, onChange, options, disabled }:
  { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; disabled?: boolean }) {
  return (
    <select className="input" value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

export function Checkbox({ checked, onChange, label }:
  { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
      <input type="checkbox" className="accent-sky-500" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[180px] text-center gap-2 text-slate-500">
      <div className="text-3xl">🧠</div>
      <div className="font-medium">{title}</div>
      {children && <div className="text-xs max-w-md">{children}</div>}
    </div>
  )
}
