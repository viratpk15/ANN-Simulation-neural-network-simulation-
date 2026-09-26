// Small shared utilities.
export function uid(prefix = 'n'): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 8)}${Date.now().toString(36).slice(-4)}`
}

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  if (Math.abs(v) >= 10000 || (Math.abs(v) < 0.001 && v !== 0)) return v.toExponential(2)
  return v.toFixed(digits)
}

export function pct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(1)}%`
}

export function timeStr(): string {
  return new Date().toLocaleTimeString()
}

/** Bucket real values into a histogram. */
export function histogram(values: number[], bins = 24): { x0: number; x1: number; count: number }[] {
  if (!values.length) return []
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (min === max) { min -= 0.5; max += 0.5 }
  const width = (max - min) / bins
  const out = Array.from({ length: bins }, (_, i) => ({ x0: min + i * width, x1: min + (i + 1) * width, count: 0 }))
  for (const v of values) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)))
    out[i].count++
  }
  return out
}

/** Diverging blue→white→red weight colour. */
export function weightColor(v: number, min: number, max: number, dark = true): string {
  const range = Math.max(1e-9, Math.max(Math.abs(min), Math.abs(max)))
  const t = Math.max(-1, Math.min(1, v / range))
  const opacity = 0.15 + 0.85 * Math.abs(t)
  if (t >= 0) return dark ? `rgba(248,113,113,${opacity})` : `rgba(220,38,38,${opacity})`
  return dark ? `rgba(96,165,250,${opacity})` : `rgba(37,99,235,${opacity})`
}

/** Activation intensity colour (0..1). */
export function activationColor(v: number, min: number, max: number): string {
  const t = max > min ? (v - min) / (max - min) : 0.5
  // dark slate → sky → emerald ramp
  const r = Math.round(30 + (52 - 30) * t + (t > 0.7 ? (120 - 52) * (t - 0.7) * 3 : 0))
  const g = Math.round(45 + (190 - 45) * t)
  const b = Math.round(90 + (200 - 90) * t * 0.8)
  return `rgb(${r},${g},${b})`
}

export function downloadText(filename: string, text: string, mime = 'application/json'): void {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
