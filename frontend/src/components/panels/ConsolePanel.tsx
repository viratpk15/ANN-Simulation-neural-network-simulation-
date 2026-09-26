import { useEffect, useRef } from 'react'
import { useAppStore } from '../../store/useAppStore'

export function ConsolePanel() {
  const logs = useAppStore((s) => s.consoleLog)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight })
  }, [logs])
  return (
    <div ref={ref} className="h-full overflow-y-auto p-3 font-mono text-xs space-y-0.5 text-slate-300">
      {logs.map((l, i) => (
        <div key={i} className={l.includes('error') || l.includes('failed') || l.includes('✗') || l.includes('Cannot') ? 'text-rose-400' : ''}>
          {l}
        </div>
      ))}
    </div>
  )
}
