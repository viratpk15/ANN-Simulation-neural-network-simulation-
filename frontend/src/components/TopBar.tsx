import { useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/useAppStore'

export function TopBar() {
  const s = useAppStore()
  const [projectsOpen, setProjectsOpen] = useState(false)
  const [presetsOpen, setPresetsOpen] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void s.checkHealth()
    void s.loadDatasets()
    void s.loadActivationLibrary()
    const t = setInterval(() => void s.checkHealth(), 15000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const job = s.job
  const running = job.state === 'running' || job.state === 'paused' || job.state === 'queued'

  return (
    <header className="flex items-center gap-2 px-3 h-12 border-b border-lab-700/60 bg-lab-900/80 backdrop-blur z-20 relative">
      <div className="flex items-center gap-2 pr-3 border-r border-lab-700/60">
        <img src="/brain.svg" alt="NeuroSim Lab" className="h-7 w-7" />
        <div className="leading-tight">
          <div className="font-bold text-sm">NeuroSim Lab</div>
          <div className="text-[9px] text-slate-400 -mt-0.5 hidden lg:block">Neural Network Simulation · Visualization · AI Diagnosis</div>
        </div>
      </div>

      <input
        className="input !w-44 !py-1 text-sm"
        value={s.projectName}
        onChange={(e) => s.setProjectName(e.target.value)}
        title="Project name"
      />

      <button className="btn-ghost !py-1" onClick={() => s.clearCanvas()} title="New empty project">New</button>
      <div className="relative">
        <button className="btn-ghost !py-1" onClick={() => { setProjectsOpen(!projectsOpen); void s.loadProjects() }}>Projects</button>
        {projectsOpen && (
          <div className="absolute top-9 left-0 w-72 panel p-2 z-30 max-h-80 overflow-auto">
            <div className="flex gap-1 mb-2">
              <button className="btn-primary !py-1 flex-1" onClick={() => void s.saveProject()}>Save current</button>
              <button className="btn-ghost !py-1 flex-1" onClick={() => void s.exportProject()}>Export .nnsim</button>
            </div>
            <button className="btn-ghost !py-1 w-full mb-2" onClick={() => fileRef.current?.click()}>Import .nnsim…</button>
            <input ref={fileRef} type="file" accept=".nnsim,.json" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void s.importProject(f); e.target.value = '' }} />
            {s.projects.length === 0 && <div className="text-xs text-slate-500 p-2">No saved projects yet.</div>}
            {s.projects.map((p) => (
              <div key={p.id} className="flex items-center justify-between gap-1 px-1 py-1 rounded hover:bg-lab-800 text-sm">
                <button className="text-left flex-1 truncate" onClick={() => { void s.loadProject(p.id); setProjectsOpen(false) }}
                  title={`Updated ${p.updated_at}`}>{p.name}</button>
                <button className="text-rose-400 text-xs px-1" title="Delete"
                  onClick={() => void s.deleteProject(p.id)}>✕</button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="h-6 border-l border-lab-700/60 mx-1" />

      <div className="relative">
        <button
          className="btn-primary !py-1 flex items-center gap-1.5"
          onClick={() => setPresetsOpen(!presetsOpen)}
          title="Load preconfigured demo networks (XOR, Iris, CNN)"
        >
          <span>⚡ Presets</span>
          <span className="text-[10px]">▾</span>
        </button>
        {presetsOpen && (
          <div className="absolute top-9 left-0 w-80 panel p-2 z-30 space-y-1 shadow-xl border border-lab-700 bg-lab-900">
            <div className="text-[10px] font-semibold text-slate-400 px-2 py-1 uppercase tracking-wider">Example Architectures</div>
            <button
              className="w-full text-left p-2 rounded hover:bg-lab-800 transition-colors"
              onClick={() => { void s.loadPreset('xor-demo'); setPresetsOpen(false) }}
            >
              <div className="text-xs font-semibold text-sky-400">⚡ XOR Demo (Try Example)</div>
              <div className="text-[10px] text-slate-400">2 → 8 → 8 → 2 feedforward network on XOR</div>
            </button>
            <button
              className="w-full text-left p-2 rounded hover:bg-lab-800 transition-colors"
              onClick={() => { void s.loadPreset('iris-research'); setPresetsOpen(false) }}
            >
              <div className="text-xs font-semibold text-emerald-400">🌸 Iris Research Experiment</div>
              <div className="text-[10px] text-slate-400">4 → 16 → 8 → 3 multi-class classifier</div>
            </button>
            <button
              className="w-full text-left p-2 rounded hover:bg-lab-800 transition-colors"
              onClick={() => { void s.loadPreset('cnn-shapes'); setPresetsOpen(false) }}
            >
              <div className="text-xs font-semibold text-violet-400">🖼️ CNN Demo (Conv + Pool + BatchNorm)</div>
              <div className="text-[10px] text-slate-400">8×8×1 → Conv2D(8) → MaxPool → Flatten → Dense 16 → 3 on shapes8</div>
            </button>
          </div>
        )}
      </div>

      <button className="btn-ghost !py-1" onClick={() => void s.validateNow()} disabled={s.validating || !s.nodes.length}>
        {s.validating ? 'Validating…' : 'Validate'}
      </button>
      {s.validation && (
        <span className={s.validation.ok ? 'badge-ok' : 'badge-err'}>
          {s.validation.ok ? `✓ ${s.validation.total_params.toLocaleString()} params` : `✗ ${s.validation.errors.length} error(s)`}
        </span>
      )}

      <div className="h-6 border-l border-lab-700/60 mx-1" />

      {!running ? (
        <button className="btn-ok !py-1" onClick={() => { void s.startTraining(); s.setTab('training') }}
          disabled={!s.nodes.length || s.trainingBusy || s.backendOk === false}
          title="Start training the current network on the selected dataset">
          ▶ Train
        </button>
      ) : (
        <div className="flex items-center gap-1">
          {job.state === 'running' && <button className="btn-ghost !py-1" onClick={() => void s.pauseJob()}>⏸ Pause</button>}
          {job.state === 'paused' && <button className="btn-ok !py-1" onClick={() => void s.resumeJob()}>⏵ Resume</button>}
          <button className="btn-danger !py-1" onClick={() => void s.stopJob()}>■ Stop</button>
          <span className="badge-info animate-pulse">{job.state}{job.history.length ? ` · ep ${job.history[job.history.length - 1].epoch}` : ''}</span>
        </div>
      )}

      <div className="flex-1" />

      <div className="flex items-center gap-1 px-1 py-0.5 rounded bg-lab-850/80 border border-lab-700/60 hidden md:flex">
        <button
          onClick={s.toggleSidebar}
          className={`px-2 py-0.5 rounded text-xs transition-colors ${s.sidebarOpen ? 'bg-sky-500/20 text-sky-300 font-medium' : 'text-slate-400 hover:text-slate-200'}`}
          title="Toggle Left Palette & Data Sidebar"
        >
          ◧ Palette
        </button>
        <button
          onClick={s.toggleInspector}
          className={`px-2 py-0.5 rounded text-xs transition-colors ${s.inspectorOpen ? 'bg-sky-500/20 text-sky-300 font-medium' : 'text-slate-400 hover:text-slate-200'}`}
          title="Toggle Right Layer Inspector"
        >
          ◨ Inspector
        </button>
        <button
          onClick={() => s.setBottomPanelSize(s.bottomPanelSize === 'collapsed' ? 'normal' : 'collapsed')}
          className={`px-2 py-0.5 rounded text-xs transition-colors ${s.bottomPanelSize !== 'collapsed' ? 'bg-sky-500/20 text-sky-300 font-medium' : 'text-slate-400 hover:text-slate-200'}`}
          title="Toggle Bottom Analysis Panel"
        >
          ◫ Panels
        </button>
      </div>

      <span
        className={s.backendOk === null ? 'badge-warn' : s.backendOk ? 'badge-ok' : 'badge-err'}
        title={s.backendOk ? 'Backend API reachable' : 'Backend not reachable — start it per README (uvicorn on :8000)'}>
        {s.backendOk === null ? '…checking API' : s.backendOk ? '● API online' : '○ API offline'}
      </span>
      <button className="btn-ghost !py-1" onClick={s.toggleTheme} title="Toggle dark / light mode">
        {s.theme === 'dark' ? '☀️' : '🌙'}
      </button>
    </header>
  )
}
