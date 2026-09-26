import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../../api/client'
import { useAppStore } from '../../store/useAppStore'
import { Field, NumInput, Select } from '../common/Stat'

interface ColDetail { name: string; dtype: string; numeric: boolean; missing: number; unique: number; sample_values: unknown[] }
interface Analysis {
  rows: number; columns: number; numeric_columns: number; categorical_columns: number
  total_missing: number; columns_detail: ColDetail[]; suggested_target: string
  suggested_task: string; suggested_classes?: number | null; preview: Record<string, unknown>[]
  upload_id?: string; original_name?: string
}

export function DatasetPanel() {
  const s = useAppStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [features, setFeatures] = useState<string[]>([])
  const [target, setTarget] = useState<string>('')

  const selectedDS = useMemo(() => s.datasets.find((d) =>
    (s.dataset.kind === 'builtin' && d.kind === 'builtin' && d.id === s.dataset.name) ||
    (s.dataset.kind === 'upload' && d.kind === 'upload' && d.id === s.dataset.upload_id)),
    [s.datasets, s.dataset])

  const loadAnalysis = async () => {
    try {
      if (s.dataset.kind === 'builtin' && s.dataset.name) {
        const res = await api.get<{ analysis: Analysis }>(`/api/datasets/builtin/${s.dataset.name}/preview`)
        setAnalysis(res.analysis)
        setTarget(res.analysis.suggested_target)
        setFeatures(res.analysis.columns_detail.filter((c) => c.name !== res.analysis.suggested_target).map((c) => c.name))
      } else if (s.dataset.kind === 'upload' && s.dataset.upload_id) {
        const res = await api.get<Analysis>(`/api/datasets/upload/${s.dataset.upload_id}`)
        setAnalysis(res)
        setTarget(res.suggested_target)
        setFeatures(res.columns_detail.filter((c) => c.name !== res.suggested_target).map((c) => c.name))
      }
    } catch (e) {
      s.log(`Dataset analysis failed: ${(e as Error).message}`)
    }
  }

  useEffect(() => { setAnalysis(null) }, [s.dataset.kind, s.dataset.name, s.dataset.upload_id])

  const doUpload = async (f: File) => {
    setUploading(true)
    try {
      const meta = await api.upload<Analysis & { upload_id: string }>('/api/datasets/upload', f)
      await s.loadDatasets()
      s.setDatasetSelection({ kind: 'upload', upload_id: meta.upload_id, name: undefined })
      setAnalysis(meta)
      setTarget(meta.suggested_target)
      setFeatures(meta.columns_detail.filter((c) => c.name !== meta.suggested_target).map((c) => c.name))
      s.log(`Uploaded '${f.name}': ${meta.rows} rows × ${meta.columns} cols, ${meta.total_missing} missing values → task suggestion: ${meta.suggested_task}.`)
    } catch (e) {
      s.log(`Upload failed: ${(e as Error).message}`)
    } finally { setUploading(false) }
  }

  const applyColumns = () => {
    s.setDatasetSelection({ feature_columns: features, target_column: target || undefined })
    s.log(`Dataset columns applied: ${features.length} feature(s), target '${target}'.`)
  }

  const builtins = s.datasets.filter((d) => d.kind === 'builtin')
  const uploads = s.datasets.filter((d) => d.kind === 'upload')

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      {/* dataset pickers */}
      <div className="w-72 shrink-0 space-y-3 overflow-y-auto">
        <div className="panel p-3">
          <div className="panel-title mb-2">Built-in educational datasets</div>
          <div className="space-y-1.5">
            {builtins.map((d) => (
              <button key={d.id}
                className={`w-full text-left px-2.5 py-2 rounded-lg border text-xs transition-colors ${selectedDS?.id === d.id && s.dataset.kind === 'builtin' ? 'border-sky-500/70 bg-sky-600/10' : 'border-lab-700/60 hover:border-lab-600'}`}
                onClick={() => s.setDatasetSelection({ kind: 'builtin', name: d.id, upload_id: undefined, feature_columns: undefined, target_column: undefined })}
                title={d.description}>
                <div className="font-semibold">{d.name}</div>
                <div className="text-[10px] text-slate-400">{d.samples} samples · {d.n_features} features{d.n_classes ? ` · ${d.n_classes} classes` : ' · regression'}</div>
              </button>
            ))}
          </div>
        </div>
        <div className="panel p-3">
          <div className="panel-title mb-2">Your CSV datasets</div>
          <button className="btn-primary w-full !py-1.5" disabled={uploading} onClick={() => fileRef.current?.click()}>
            {uploading ? 'Uploading…' : '⬆ Upload CSV'}
          </button>
          <input ref={fileRef} type="file" accept=".csv" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void doUpload(f); e.target.value = '' }} />
          <div className="space-y-1 mt-2">
            {uploads.map((d) => (
              <button key={d.id}
                className={`w-full text-left px-2 py-1.5 rounded text-xs ${selectedDS?.id === d.id && s.dataset.kind === 'upload' ? 'bg-sky-600/25' : 'hover:bg-lab-800'}`}
                onClick={() => s.setDatasetSelection({ kind: 'upload', upload_id: d.id, name: undefined })}>
                {d.name} <span className="text-slate-500">({d.samples} rows)</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* analysis + preprocessing */}
      <div className="flex-1 min-w-0 space-y-3 overflow-y-auto">
        {selectedDS ? (
          <>
            <div className="panel p-3">
              <div className="flex items-center gap-3 flex-wrap">
                <h3 className="font-semibold text-lab-accent">{selectedDS.name}</h3>
                <button className="btn !py-0.5" onClick={() => void loadAnalysis()}>
                  {analysis ? '⟳ Re-analyse' : '🔍 Analyse & preview'}
                </button>
                <span className="text-xs text-slate-400 flex-1 min-w-[260px]">{selectedDS.description}</span>
              </div>
              {analysis && (
                <div className="grid grid-cols-3 lg:grid-cols-6 gap-2 mt-3 text-center">
                  {[
                    ['rows', analysis.rows], ['columns', analysis.columns],
                    ['numeric', analysis.numeric_columns], ['categorical', analysis.categorical_columns],
                    ['missing', analysis.total_missing],
                    ['suggested task', analysis.suggested_task + (analysis.suggested_classes ? ` (${analysis.suggested_classes})` : '')],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="panel !bg-lab-850 p-2">
                      <div className="text-[10px] text-slate-500 uppercase">{k}</div>
                      <div className="font-mono text-sm">{String(v ?? '—')}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {analysis && (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <div className="panel p-3 space-y-2">
                  <div className="panel-title">Feature selection (used for training)</div>
                  <div className="max-h-44 overflow-y-auto space-y-1">
                    {analysis.columns_detail.filter((c) => c.name !== target).map((c) => (
                      <label key={c.name} className="flex items-center gap-2 text-xs cursor-pointer">
                        <input type="checkbox" className="accent-sky-500"
                          checked={features.includes(c.name)}
                          onChange={(e) => setFeatures(e.target.checked ? [...features, c.name] : features.filter((f) => f !== c.name))} />
                        <span className="font-mono">{c.name}</span>
                        <span className="text-slate-500">{c.numeric ? 'num' : 'cat'} · {c.unique} unique{c.missing ? ` · ${c.missing} missing` : ''}</span>
                      </label>
                    ))}
                  </div>
                  <Field label="Target column" hint="What the network learns to predict (built-ins have a fixed target).">
                    <Select value={target} onChange={setTarget} disabled={s.dataset.kind === 'builtin'}
                      options={analysis.columns_detail.map((c) => ({ value: c.name, label: `${c.name} (${c.numeric ? 'numeric' : 'categorical'}, ${c.unique} unique)` }))} />
                  </Field>
                  <button className="btn-primary !py-1 w-full" onClick={applyColumns} disabled={!features.length}>
                    Apply {features.length} feature(s) + target '{target}'
                  </button>
                </div>

                <div className="panel p-3 space-y-2">
                  <div className="panel-title">Preprocessing</div>
                  <Field label="Feature scaling" hint="Standardization: zero mean / unit variance (recommended when feature scales differ). Min-Max: squash to [0,1]. Fitted on the train split only — never on test data.">
                    <Select value={s.dataset.preprocessing.scale} onChange={(v) => s.setPreprocessing({ scale: v as never })}
                      options={[
                        { value: 'standard', label: 'Standardize (z-score) — recommended' },
                        { value: 'minmax', label: 'Min-Max normalize to [0, 1]' },
                        { value: 'none', label: 'None (raw values)' },
                      ]} />
                  </Field>
                  <div className="grid grid-cols-2 gap-2">
                    <Field label={`Test split — ${(s.dataset.preprocessing.test_split * 100).toFixed(0)}%`} hint="Held out completely; used only for the final honest evaluation.">
                      <input type="range" min={0.05} max={0.5} step={0.05} className="w-full accent-sky-500"
                        value={s.dataset.preprocessing.test_split}
                        onChange={(e) => s.setPreprocessing({ test_split: Number(e.target.value) })} />
                    </Field>
                    <Field label={`Validation split — ${(s.dataset.preprocessing.val_split * 100).toFixed(0)}%`} hint="Used for val curves/diagnosis during training.">
                      <input type="range" min={0} max={0.4} step={0.05} className="w-full accent-sky-500"
                        value={s.dataset.preprocessing.val_split}
                        onChange={(e) => s.setPreprocessing({ val_split: Number(e.target.value) })} />
                    </Field>
                  </div>
                  <Field label="Split seed">
                    <NumInput value={s.dataset.preprocessing.seed} min={0} step={1}
                      onChange={(v) => s.setPreprocessing({ seed: Math.max(0, Math.round(v)) })} />
                  </Field>
                  <div className="text-[10px] text-slate-500">
                    Splits are stratified for classification when possible. Categorical features are one-hot encoded;
                    missing values are imputed (numeric: mean, categorical: explicit "__missing__" bucket).
                  </div>
                </div>
              </div>
            )}

            {analysis && (
              <div className="panel p-3">
                <div className="panel-title mb-2">Preview (first {analysis.preview.length} rows)</div>
                <div className="overflow-x-auto">
                  <table className="text-[11px] font-mono border-collapse">
                    <thead>
                      <tr>{Object.keys(analysis.preview[0] ?? {}).map((k) => (
                        <th key={k} className="border border-lab-700 px-2 py-1 text-slate-400">{k}</th>
                      ))}</tr>
                    </thead>
                    <tbody>
                      {analysis.preview.map((row, i) => (
                        <tr key={i}>{Object.values(row).map((v, j) => (
                          <td key={j} className="border border-lab-700 px-2 py-0.5">{v === null ? <span className="text-rose-400">∅</span> : String(v)}</td>
                        ))}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-xs text-slate-500">Select a dataset on the left or upload a CSV.</div>
        )}
      </div>
    </div>
  )
}
