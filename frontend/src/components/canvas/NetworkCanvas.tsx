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
      stroke: s.job.state === 'running' ? '#3b82f6' : '#52525b',
      strokeWidth: 2.5,
      filter: s.job.state === 'running' ? 'drop-shadow(0 0 6px rgba(59, 130, 246, 0.8))' : 'none',
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
        <Background variant={BackgroundVariant.Dots} gap={24} size={1.5}
          color="#27272a" />
        <Controls position="bottom-left" />
        <MiniMap pannable zoomable className="!h-24 !w-36"
          nodeColor={(n) => (n.data as LayerNodeData)?.kind === 'output' ? '#f43f5e'
            : (n.data as LayerNodeData)?.kind === 'input' ? '#10b981'
              : (n.data as LayerNodeData)?.kind === 'dropout' ? '#f97316' : '#3b82f6'} />
      </ReactFlow>

      {s.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center panel p-8 max-w-md border-zinc-700 space-y-3 bg-lab-900/95 shadow-elevated">
            <div className="text-4xl">🧠</div>
            <div className="text-lg font-bold text-white tracking-tight">
              Neural Network Architecture Canvas
            </div>
            <div className="text-xs text-zinc-400 leading-relaxed">
              Drag layers from the left palette onto this grid, then connect output handles to input handles.
            </div>
            <div className="pt-2">
              <span className="text-xs text-blue-300 bg-blue-950/70 border border-blue-700/60 px-3 py-1 rounded-full font-mono">
                ⚡ Click Presets in top bar to load instantly
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
