import { useEffect } from 'react'
import { Inspector } from './components/Inspector'
import { Sidebar } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import { NetworkCanvas } from './components/canvas/NetworkCanvas'
import { BottomTabs } from './components/panels/BottomTabs'
import { useAppStore } from './store/useAppStore'

export default function App() {
  const checkHealth = useAppStore((s) => s.checkHealth)
  const sidebarOpen = useAppStore((s) => s.sidebarOpen)
  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)
  const toggleInspector = useAppStore((s) => s.toggleInspector)

  useEffect(() => {
    void checkHealth()
  }, [checkHealth])

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-lab-950 text-slate-100">
      <TopBar />
      <div className="flex-1 flex min-h-0 relative">
        {sidebarOpen ? (
          <Sidebar />
        ) : (
          <button
            onClick={toggleSidebar}
            className="absolute top-2 left-2 z-10 btn-ghost !py-1 !px-2.5 !text-xs bg-lab-900/90 border border-lab-700/80 shadow-lg backdrop-blur hover:bg-lab-800"
            title="Open Layer Palette & Datasets"
          >
            ▶ Palette & Data
          </button>
        )}
        <main className="flex-1 flex min-w-0 relative">
          <NetworkCanvas />
          {inspectorOpen ? (
            <Inspector />
          ) : (
            <button
              onClick={toggleInspector}
              className="absolute top-2 right-2 z-10 btn-ghost !py-1 !px-2.5 !text-xs bg-lab-900/90 border border-lab-700/80 shadow-lg backdrop-blur hover:bg-lab-800"
              title="Open Layer Inspector"
            >
              ⚙ Inspector ◀
            </button>
          )}
        </main>
      </div>
      <BottomTabs />
    </div>
  )
}
