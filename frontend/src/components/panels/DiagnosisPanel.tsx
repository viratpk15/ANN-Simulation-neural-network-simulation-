import { useEffect, useState } from 'react'
import { useAppStore } from '../../store/useAppStore'
import type { Finding, LlmProviderStatus, LlmStatus, ProviderState } from '../../types'
import { Empty } from '../common/Stat'

const SEV: Record<Finding['severity'], { bar: string; chip: string; icon: string }> = {
  critical: { bar: 'border-l-rose-500', chip: 'badge-err', icon: '⛔' },
  warning: { bar: 'border-l-amber-400', chip: 'badge-warn', icon: '⚠' },
  info: { bar: 'border-l-sky-400', chip: 'badge-info', icon: 'ℹ' },
  ok: { bar: 'border-l-emerald-400', chip: 'badge-ok', icon: '✓' },
}

const STATE_DOT: Record<ProviderState, { dot: string; text: string }> = {
  active: { dot: 'bg-emerald-400', text: 'Active' },
  available: { dot: 'bg-emerald-400', text: 'Available' },
  not_configured: { dot: 'bg-slate-500', text: 'API key missing' },
  failed: { dot: 'bg-rose-400', text: 'Failed' },
  skipped: { dot: 'bg-amber-400', text: 'Skipped' },
  disabled: { dot: 'bg-slate-500', text: 'Disabled' },
}

/** Small pill showing which provider answered, and whether it was a fallback. */
function ProviderBadge({ provider, model, fallback }: {
  provider?: string | null; model?: string | null; fallback?: boolean
}) {
  if (!provider) return null
  return (
    <span className={`badge ${fallback ? 'badge-warn' : 'badge-ok'} flex items-center gap-1`}
      title={model ? `model: ${model}` : undefined}>
      <span className={`h-1.5 w-1.5 rounded-full ${fallback ? 'bg-amber-300' : 'bg-emerald-300'}`} />
      {fallback ? 'fallback · ' : ''}{provider}
    </span>
  )
}

/** The configured chain and each provider's health. Keys are never shown. */
function ProviderList({ status }: { status: LlmStatus }) {
  if (!status.providers.length) {
    return <p className="text-[11px] text-slate-500">
      No providers configured (LLM_PROVIDER_ORDER is empty).
    </p>
  }
  return (
    <div className="space-y-1">
      {status.providers.map((p: LlmProviderStatus) => {
        const s = STATE_DOT[p.state] ?? STATE_DOT.disabled
        return (
          <div key={p.name} className="flex items-center gap-2 text-[11px]"
            title={p.model ? `model: ${p.model}` : undefined}>
            <span className={`h-2 w-2 shrink-0 rounded-full ${s.dot}`} />
            <span className="text-slate-300">{p.label}</span>
            <span className="text-slate-500">
              {p.state === 'not_configured' && p.detail ? p.detail : s.text}
            </span>
            {p.latency_s != null && (
              <span className="ml-auto font-mono text-slate-600">{p.latency_s.toFixed(2)}s</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function DiagnosisPanel() {
  const s = useAppStore()
  const runId = s.lastRunId ?? s.job.id
  const d = s.diagnosis
  const [showProviders, setShowProviders] = useState(false)
  const llmStatus = s.llmStatus

  useEffect(() => { void s.fetchLlmStatus() }, [])

  if (!runId) {
    return <Empty title="Train first, then diagnose">
      The deterministic diagnostic engine analyses real loss curves, gradient norms, activation statistics,
      weight statistics, architecture and dataset balance — and (optionally, if an LLM provider is configured)
      explains the findings in natural language.
    </Empty>
  }

  return (
    <div className="h-full flex flex-col p-3 gap-3 overflow-y-auto">
      <div className="flex items-center gap-2">
        <button className="btn-primary !py-1.5" disabled={s.diagnosisBusy} onClick={() => void s.runDiagnosis()}>
          {s.diagnosisBusy ? '🔍 Analyzing run…' : '🔍 Analyze this run'}
        </button>
        {d && typeof d.context.epochs_run === 'number' && (
          <span className="text-xs text-slate-400">
            run {d.run_id} · {String(d.context.task)} · {String(d.context.epochs_run)} epochs · {String(d.context.total_params)} params
          </span>
        )}
      </div>

      {d && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {d.findings.map((f, i) => {
              const st = SEV[f.severity]
              return (
                <div key={i} className={`panel border-l-4 ${st.bar} p-3 space-y-1.5`}>
                  <div className="flex items-center gap-2">
                    <span>{st.icon}</span>
                    <span className="font-semibold text-sm">{f.title}</span>
                    <span className={`${st.chip} ml-auto`}>{f.code}</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{f.explanation}</p>
                  {f.suggestions.length > 0 && (
                    <div>
                      <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-0.5">Consider</div>
                      <ul className="text-xs text-slate-300 space-y-0.5 list-disc list-inside">
                        {f.suggestions.map((sg, j) => <li key={j}>{sg}</li>)}
                      </ul>
                    </div>
                  )}
                  {Object.keys(f.evidence ?? {}).length > 0 && (
                    <details className="text-[10px] text-slate-500">
                      <summary className="cursor-pointer hover:text-slate-300">evidence (real numbers)</summary>
                      <pre className="mt-1 p-1.5 rounded bg-lab-950/70 overflow-x-auto">{JSON.stringify(f.evidence, null, 2)}</pre>
                    </details>
                  )}
                </div>
              )
            })}
          </div>

          {/* ---------------- AI explanation (optional LLM layer) ---------------- */}
          <div className="panel p-3 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="panel-title">AI explanation</span>
              <ProviderBadge provider={d.llm.label ?? d.llm.provider}
                model={d.llm.model} fallback={d.llm.fallback_used} />
              {llmStatus && (
                <button className="btn-ghost !py-0.5 !px-2 text-[10px] ml-auto"
                  onClick={() => setShowProviders((v) => !v)}>
                  {showProviders ? '▾' : '▸'} providers
                </button>
              )}
            </div>

            {d.llm.message && (
              <div className="text-[11px] text-slate-400">{d.llm.message}</div>
            )}

            {d.llm.explanation ? (
              <p className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
                {d.llm.explanation}
              </p>
            ) : (
              <p className="text-xs text-slate-500">
                {d.llm.error ??
                  'No LLM provider is configured. Add a key in .env (see README §LLM Architecture) ' +
                  'to have an LLM rephrase these findings. The deterministic findings above always work.'}
              </p>
            )}

            {d.llm.chain && d.llm.chain.length > 0 && (
              <div className="text-[10px] text-slate-500 font-mono">
                fallback chain: {d.llm.chain.join(' → ')}
              </div>
            )}

            {showProviders && llmStatus && (
              <div className="mt-2 pt-2 border-t border-lab-700/60 space-y-2">
                <ProviderList status={llmStatus} />
                <div className="text-[10px] text-slate-600">
                  timeout {llmStatus.timeout_s}s · max {llmStatus.max_retries} retries per provider
                  · API keys are never sent to the browser.
                </div>
              </div>
            )}
          </div>
        </>
      )}
      {!d && <div className="text-xs text-slate-500">Findings appear here after analysis.</div>}
    </div>
  )
}
