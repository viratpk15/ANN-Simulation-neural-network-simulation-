import { useCallback, useMemo, useRef } from 'react'
import ReactFlow, {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlowProvider,
  useReactFlow, type Node,
} from 'reactflow'
import { useAppStore, type LayerNodeData } from '../../store/useAppStore'
import type { LayerKind } from '../../types'
import LayerNode from './LayerNode'

const nodeTypes = { layerNode: LayerNode }

function CanvasInner() {
  const s = useAppStore()
  const wrapper = useRef<HTMLDivElement>(null)
  const rf = useReactFlow()

  // Both callbacks below are intentionally dependency-free (except `rf`) and
  // read the store via getState(), so their identity is stable across store
  // updates. ReactFlow's memoized SelectionListener re-runs its effect when
  // the onSelectionChange prop changes identity; a callback that changed on
  // every store update therefore created an infinite update loop
  // (render → onSelectionChange → set() → render → …) that crashed React.
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const kind = e.dataTransfer.getData('application/neurosim-layer') as LayerKind
    if (!kind) return
    const bounds = wrapper.current?.getBoundingClientRect()
    const pos = rf.screenToFlowPosition({ x: e.clientX - (bounds?.left ?? 0), y: e.clientY - (bounds?.top ?? 0) })
    useAppStore.getState().addLayer(kind, pos)
  }, [rf])

  const onSelectionChange = useCallback(({ nodes }: { nodes: Node<LayerNodeData>[] }) => {
    const id = nodes.length === 1 ? nodes[0].id : null
    const { selectedNodeId, setSelected } = useAppStore.getState()
    if (selectedNodeId !== id) setSelected(id)
  }, [])

  const defaultEdgeOptions = useMemo(() => ({
    animated: s.job.state === 'running',
    style: {
      stroke: s.job.state === 'running' ? '#ec4899' : '#8b5cf6',
      strokeWidth: 2.5,
      filter: s.job.state === 'running' ? 'drop-shadow(0 0 6px rgba(236, 72, 153, 0.8))' : 'drop-shadow(0 0 3px rgba(139, 92, 246, 0.4))',
    },
  }), [s.job.state])

  return (
    <div ref={wrapper} className="flex-1 relative bg-lab-950" onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }}>
      <ReactFlow
        nodes={s.nodes}
        edges={s.edges}
        nodeTypes={nodeTypes}
        onNodesChange={s.onNodesChange}
        onEdgesChange={s.onEdgesChange}
        onConnect={s.onConnect}
        onEdgeUpdate={s.onEdgeUpdate}
        edgeUpdaterRadius={18}
        onSelectionChange={onSelectionChange}
        defaultEdgeOptions={defaultEdgeOptions}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode={['Shift', 'Meta', 'Control']}
        fitView
        minZoom={0.2}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1.8}
          color="#381d69" />
        <Controls position="bottom-left" />
        <MiniMap pannable zoomable className="!h-24 !w-36"
          nodeColor={(n) => (n.data as LayerNodeData)?.kind === 'output' ? '#ec4899'
            : (n.data as LayerNodeData)?.kind === 'input' ? '#a855f7'
              : (n.data as LayerNodeData)?.kind === 'dropout' ? '#f59e0b' : '#7c3aed'} />
      </ReactFlow>

      {s.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center panel p-8 max-w-md border-purple-500/40 shadow-neon-purple space-y-3 backdrop-blur-2xl bg-lab-900/90">
            <div className="text-5xl animate-bounce">🧠</div>
            <div className="text-lg font-bold bg-gradient-to-r from-pink-400 via-purple-300 to-indigo-300 bg-clip-text text-transparent">
              Neural Network Architecture Canvas
            </div>
            <div className="text-xs text-purple-200/80 leading-relaxed">
              Drag layers from the left palette onto this grid, then connect output handles to input handles.
            </div>
            <div className="pt-2">
              <span className="text-xs text-pink-300 bg-pink-500/20 border border-pink-500/40 px-3 py-1 rounded-full font-mono">
                ⚡ Presets ▾ in top bar to load instantly
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export function NetworkCanvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  )
}
