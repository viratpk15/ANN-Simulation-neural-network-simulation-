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
    <header className="flex items-center gap-2.5 px-4 h-14 border-b border-lab-700 bg-lab-900 z-20 relative shadow-subtle">
      <div className="flex items-center gap-2.5 pr-3 border-r border-lab-750">
        <div className="relative">
          <img src="/brain.svg" alt="NeuroSim Lab" className="h-8 w-8 filter drop-shadow-sm" />
          <span className="absolute -top-0.5 -right-0.5 flex h-2 w-2">
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
        </div>
        <div className="leading-tight">
          <div className="font-bold text-base text-white tracking-tight">
            NeuroSim Lab
          </div>
          <div className="text-[10px] text-zinc-400 font-mono hidden xl:block">
            Neural Simulation · Visuals · AI Diagnosis
          </div>
        </div>
      </div>

      <input
        className="input !w-44 !py-1 text-xs font-medium"
        value={s.projectName}
        onChange={(e) => s.setProjectName(e.target.value)}
        title="Project name"
      />

      <button className="btn-ghost !py-1" onClick={() => s.clearCanvas()} title="New empty project">New</button>
      <div className="relative">
        <button className="btn-ghost !py-1" onClick={() => { setProjectsOpen(!projectsOpen); void s.loadProjects() }}>Projects</button>
        {projectsOpen && (
          <div className="absolute top-10 left-0 w-72 panel p-2.5 z-30 max-h-80 overflow-auto border-lab-600 bg-lab-900 shadow-elevated">
            <div className="flex gap-1.5 mb-2">
              <button className="btn-primary !py-1 flex-1 text-xs" onClick={() => void s.saveProject()}>Save current</button>
              <button className="btn-ghost !py-1 flex-1 text-xs" onClick={() => void s.exportProject()}>Export .nnsim</button>
            </div>
            <button className="btn-ghost !py-1 w-full mb-2 text-xs" onClick={() => fileRef.current?.click()}>Import .nnsim…</button>
            <input ref={fileRef} type="file" accept=".nnsim,.json" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void s.importProject(f); e.target.value = '' }} />
            {s.projects.length === 0 && <div className="text-xs text-zinc-500 p-2">No saved projects yet.</div>}
            {s.projects.map((p) => (
              <div key={p.id} className="flex items-center justify-between gap-1 px-1.5 py-1 rounded hover:bg-lab-800 text-xs">
                <button className="text-left flex-1 truncate text-zinc-200" onClick={() => { void s.loadProject(p.id); setProjectsOpen(false) }}
                  title={`Updated ${p.updated_at}`}>{p.name}</button>
                <button className="text-rose-400 text-xs px-1 hover:text-rose-300" title="Delete"
                  onClick={() => void s.deleteProject(p.id)}>✕</button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="h-6 border-l border-lab-750 mx-1" />

      <div className="relative">
        <button
          className="btn-preset !py-1.5 !px-3 flex items-center gap-1.5 text-xs font-semibold"
          onClick={() => setPresetsOpen(!presetsOpen)}
          title="Load preconfigured demo networks (XOR, Iris, CNN)"
        >
          <span>⚡ Presets</span>
          <span className="text-[10px]">▾</span>
        </button>
        {presetsOpen && (
          <div className="absolute top-10 left-0 w-84 panel p-2.5 z-30 space-y-1.5 shadow-elevated border border-lab-600 bg-lab-900">
            <div className="text-[10px] font-bold text-zinc-400 px-2 py-0.5 uppercase tracking-wider">Example Architectures</div>
            <button
              className="w-full text-left p-2 rounded-lg hover:bg-lab-800 transition-colors border border-transparent hover:border-blue-500/40"
              onClick={() => { void s.loadPreset('xor-demo'); setPresetsOpen(false) }}
            >
              <div className="text-xs font-bold text-blue-400">⚡ XOR Demo (Try Example)</div>
              <div className="text-[10px] text-zinc-400">2 → 8 → 8 → 2 feedforward network on XOR</div>
            </button>
            <button
              className="w-full text-left p-2 rounded-lg hover:bg-lab-800 transition-colors border border-transparent hover:border-emerald-500/40"
              onClick={() => { void s.loadPreset('iris-research'); setPresetsOpen(false) }}
            >
              <div className="text-xs font-bold text-emerald-400">🌸 Iris Research Experiment</div>
              <div className="text-[10px] text-zinc-400">4 → 16 → 8 → 3 multi-class classifier</div>
            </button>
            <button
              className="w-full text-left p-2 rounded-lg hover:bg-lab-800 transition-colors border border-transparent hover:border-indigo-500/40"
              onClick={() => { void s.loadPreset('cnn-shapes'); setPresetsOpen(false) }}
            >
              <div className="text-xs font-bold text-indigo-400">🖼️ CNN Demo (Conv + Pool + BatchNorm)</div>
              <div className="text-[10px] text-zinc-400">8×8×1 → Conv2D(8) → MaxPool → Flatten → Dense 16 → 3 on shapes8</div>
            </button>
          </div>
        )}
      </div>

      <button className="btn-ghost !py-1 text-xs" onClick={() => void s.validateNow()} disabled={s.validating || !s.nodes.length}>
        {s.validating ? 'Validating…' : 'Validate'}
      </button>
      {s.validation && (
        <span className={s.validation.ok ? 'badge-ok' : 'badge-err'}>
          {s.validation.ok ? `✓ ${s.validation.total_params.toLocaleString()} params` : `✗ ${s.validation.errors.length} error(s)`}
        </span>
      )}

      <div className="h-6 border-l border-lab-750 mx-1" />

      {!running ? (
        <button className="btn-train !py-1.5 !px-4 text-xs font-bold" onClick={() => { void s.startTraining(); s.setTab('training') }}
          disabled={!s.nodes.length || s.trainingBusy || s.backendOk === false}
          title="Start training the current network on the selected dataset">
          ▶ Train
        </button>
      ) : (
        <div className="flex items-center gap-1.5">
          {job.state === 'running' && <button className="btn-ghost !py-1 text-xs" onClick={() => void s.pauseJob()}>⏸ Pause</button>}
          {job.state === 'paused' && <button className="btn-ok !py-1 text-xs" onClick={() => void s.resumeJob()}>⏵ Resume</button>}
          <button className="btn-danger !py-1 text-xs" onClick={() => void s.stopJob()}>■ Stop</button>
          <span className="badge-info animate-pulse">{job.state}{job.history.length ? ` · ep ${job.history[job.history.length - 1].epoch}` : ''}</span>
        </div>
      )}

      <div className="flex-1" />

      <div className="flex items-center gap-1 px-1.5 py-0.5 rounded-lg bg-lab-850 border border-lab-700 hidden md:flex">
        <button
          onClick={s.toggleSidebar}
          className={`px-2 py-0.5 rounded text-xs transition-colors ${s.sidebarOpen ? 'bg-zinc-700 text-white font-medium' : 'text-zinc-400 hover:text-zinc-200'}`}
          title="Toggle Left Palette & Data Sidebar"
        >
          ◧ Palette
        </button>
        <button
          onClick={s.toggleInspector}
          className={`px-2 py-0.5 rounded text-xs transition-colors ${s.inspectorOpen ? 'bg-zinc-700 text-white font-medium' : 'text-zinc-400 hover:text-zinc-200'}`}
          title="Toggle Right Layer Inspector"
        >
          ◨ Inspector
        </button>
        <button
          onClick={() => s.setBottomPanelSize(s.bottomPanelSize === 'collapsed' ? 'normal' : 'collapsed')}
          className={`px-2 py-0.5 rounded text-xs transition-colors ${s.bottomPanelSize !== 'collapsed' ? 'bg-zinc-700 text-white font-medium' : 'text-zinc-400 hover:text-zinc-200'}`}
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
      <button className="btn-ghost !py-1 text-xs" onClick={s.toggleTheme} title="Toggle dark / light mode">
        {s.theme === 'dark' ? '☀️' : '🌙'}
      </button>
    </header>
  )
}
