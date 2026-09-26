import { useAppStore, type BottomTab } from '../../store/useAppStore'
import { ActivationLabPanel } from './ActivationLabPanel'
import { ActivationsPanel } from './ActivationsPanel'
import { ConsolePanel } from './ConsolePanel'
import { DatasetPanel } from './DatasetPanel'
import { DiagnosisPanel } from './DiagnosisPanel'
import { ExperimentsPanel } from './ExperimentsPanel'
import { GradientsPanel } from './GradientsPanel'
import { MathPanel } from './MathPanel'
import { PredictPanel } from './PredictPanel'
import { SimulationPanel } from './SimulationPanel'
import { TrainingPanel } from './TrainingPanel'
import { WeightsPanel } from './WeightsPanel'

const TABS: { id: BottomTab; label: string; icon: string }[] = [
  { id: 'training', label: 'Training', icon: '📈' },
  { id: 'simulation', label: 'Simulation', icon: '🎬' },
  { id: 'math', label: 'Math', icon: '📐' },
  { id: 'predict', label: 'Predict', icon: '🔍' },
  { id: 'weights', label: 'Weights', icon: '🧠' },
  { id: 'activations', label: 'Activations', icon: '⚡' },
  { id: 'gradients', label: 'Gradients', icon: '📉' },
  { id: 'diagnosis', label: 'AI Diagnosis', icon: '🤖' },
  { id: 'activlab', label: 'Activation Lab', icon: '🧪' },
  { id: 'experiments', label: 'Research', icon: '🔬' },
  { id: 'dataset', label: 'Dataset', icon: '📁' },
  { id: 'console', label: 'Console', icon: '💻' },
]

export function BottomTabs() {
  const s = useAppStore()
  const size = s.bottomPanelSize

  const heightClass =
    size === 'collapsed' ? 'h-10' :
    size === 'compact' ? 'h-64' :
    size === 'expanded' ? 'h-[520px]' :
    'h-[330px]' // normal

  const handleTabClick = (tabId: BottomTab) => {
    if (s.activeTab === tabId) {
      s.setBottomPanelSize(size === 'collapsed' ? 'normal' : 'collapsed')
    } else {
      s.setTab(tabId)
      if (size === 'collapsed') s.setBottomPanelSize('normal')
    }
  }

  return (
    <div className={`${heightClass} shrink-0 border-t border-lab-700 bg-lab-900 flex flex-col transition-all duration-150 z-10`}>
      <div className="flex items-center justify-between px-2 border-b border-lab-700 bg-lab-950/60 min-h-10">
        <div className="flex items-center gap-0.5 overflow-x-auto py-0.5">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab-btn flex items-center gap-1.5 !py-1 !px-2.5 ${s.activeTab === t.id && size !== 'collapsed' ? 'active' : ''}`}
              onClick={() => handleTabClick(t.id)}
              title={`Switch to ${t.label} (Click again to collapse/expand)`}
            >
              <span className="text-xs">{t.icon}</span>
              <span className="text-xs font-medium">{t.label}</span>
              {t.id === 'console' && (
                <span className="ml-1 text-[9px] px-1 py-0.2 rounded bg-zinc-800 text-zinc-400">
                  {s.consoleLog.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Panel size controls */}
        <div className="flex items-center gap-1 pl-2 border-l border-lab-700 shrink-0">
          <button
            onClick={() => s.setBottomPanelSize(size === 'collapsed' ? 'normal' : 'collapsed')}
            className={`px-1.5 py-0.5 rounded text-[11px] text-zinc-400 hover:text-white hover:bg-lab-800 ${size === 'collapsed' ? 'bg-zinc-700 text-white font-medium' : ''}`}
            title="Collapse bottom panel to give canvas maximum space"
          >
            {size === 'collapsed' ? '▲ Show' : '_ Hide'}
          </button>
          <button
            onClick={() => s.setBottomPanelSize('compact')}
            className={`px-1.5 py-0.5 rounded text-[11px] text-zinc-400 hover:text-white hover:bg-lab-800 ${size === 'compact' ? 'bg-zinc-700 text-white font-medium' : ''}`}
            title="Compact height (256px)"
          >
            ▫ Compact
          </button>
          <button
            onClick={() => s.setBottomPanelSize('normal')}
            className={`px-1.5 py-0.5 rounded text-[11px] text-zinc-400 hover:text-white hover:bg-lab-800 ${size === 'normal' ? 'bg-zinc-700 text-white font-medium' : ''}`}
            title="Normal height (330px)"
          >
            ◽ Normal
          </button>
          <button
            onClick={() => s.setBottomPanelSize('expanded')}
            className={`px-1.5 py-0.5 rounded text-[11px] text-zinc-400 hover:text-white hover:bg-lab-800 ${size === 'expanded' ? 'bg-zinc-700 text-white font-medium' : ''}`}
            title="Expand height (520px)"
          >
            ⤢ Max
          </button>
        </div>
      </div>

      {size !== 'collapsed' && (
        <div className="flex-1 overflow-hidden min-h-0">
          {s.activeTab === 'training' && <TrainingPanel />}
          {s.activeTab === 'simulation' && <SimulationPanel />}
          {s.activeTab === 'math' && <MathPanel />}
          {s.activeTab === 'predict' && <PredictPanel />}
          {s.activeTab === 'weights' && <WeightsPanel />}
          {s.activeTab === 'activations' && <ActivationsPanel />}
          {s.activeTab === 'gradients' && <GradientsPanel />}
          {s.activeTab === 'diagnosis' && <DiagnosisPanel />}
          {s.activeTab === 'activlab' && <ActivationLabPanel />}
          {s.activeTab === 'experiments' && <ExperimentsPanel />}
          {s.activeTab === 'dataset' && <DatasetPanel />}
          {s.activeTab === 'console' && <ConsolePanel />}
        </div>
      )}
    </div>
  )
}
