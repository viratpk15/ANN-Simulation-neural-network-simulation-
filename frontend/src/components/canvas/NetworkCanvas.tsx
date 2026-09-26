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
    style: { stroke: s.theme === 'dark' ? '#5b6b8c' : '#94a3b8', strokeWidth: 2 },
  }), [s.job.state, s.theme])

  return (
    <div ref={wrapper} className="flex-1 relative" onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }}>
      <ReactFlow
        nodes={s.nodes}
        edges={s.edges}
        nodeTypes={nodeTypes}
        onNodesChange={s.onNodesChange}
        onEdgesChange={s.onEdgesChange}
        onConnect={s.onConnect}
        onEdgeUpdate={s.onEdgeUpdate}
        edgeUpdaterRadius={16}
        onSelectionChange={onSelectionChange}
        defaultEdgeOptions={defaultEdgeOptions}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode={['Shift', 'Meta', 'Control']}
        fitView
        minZoom={0.2}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.4}
          color={s.theme === 'dark' ? '#233052' : '#cbd5e1'} />
        <Controls position="bottom-left" />
        <MiniMap pannable zoomable className="!h-24 !w-36"
          nodeColor={(n) => (n.data as LayerNodeData)?.kind === 'output' ? '#34d399'
            : (n.data as LayerNodeData)?.kind === 'input' ? '#38bdf8'
              : (n.data as LayerNodeData)?.kind === 'dropout' ? '#fbbf24' : '#a78bfa'} />
      </ReactFlow>

      {s.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center text-slate-500 space-y-2">
            <div className="text-5xl">🕸️</div>
            <div className="font-semibold">Build a neural network here</div>
            <div className="text-sm">Drag layers from the left panel, connect them left-to-right.</div>
            <div className="text-xs">…or press <b>⚡ Try Example</b> in the top bar.</div>
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
