import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import { shapeText, useAppStore, type LayerNodeData } from '../../store/useAppStore'
import type { LayerKind } from '../../types'

const KIND_STYLE: Record<LayerKind, { ring: string; chip: string; title: string }> = {
  input: { ring: 'border-emerald-500/50 hover:border-emerald-400', chip: 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/50', title: 'Input layer' },
  dense: { ring: 'border-blue-500/50 hover:border-blue-400', chip: 'bg-blue-950/70 text-blue-300 border border-blue-700/50', title: 'Fully-connected (dense) layer' },
  activation: { ring: 'border-amber-500/50 hover:border-amber-400', chip: 'bg-amber-950/70 text-amber-300 border border-amber-700/50', title: 'Activation layer' },
  dropout: { ring: 'border-orange-500/50 hover:border-orange-400', chip: 'bg-orange-950/70 text-orange-300 border border-orange-700/50', title: 'Dropout regularization layer' },
  batchnorm: { ring: 'border-violet-500/50 hover:border-violet-400', chip: 'bg-violet-950/70 text-violet-300 border border-violet-700/50', title: 'Batch normalization layer' },
  flatten: { ring: 'border-slate-500/50 hover:border-slate-400', chip: 'bg-slate-900 text-slate-300 border border-slate-700/50', title: 'Flatten layer' },
  conv2d: { ring: 'border-indigo-500/50 hover:border-indigo-400', chip: 'bg-indigo-950/70 text-indigo-300 border border-indigo-700/50', title: '2-D convolutional layer' },
  maxpool: { ring: 'border-cyan-500/50 hover:border-cyan-400', chip: 'bg-cyan-950/70 text-cyan-300 border border-cyan-700/50', title: 'Max pooling layer' },
  avgpool: { ring: 'border-sky-500/50 hover:border-sky-400', chip: 'bg-sky-950/70 text-sky-300 border border-sky-700/50', title: 'Average pooling layer' },
  globalavgpool: { ring: 'border-teal-500/50 hover:border-teal-400', chip: 'bg-teal-950/70 text-teal-300 border border-teal-700/50', title: 'Global average pooling layer' },
  output: { ring: 'border-rose-500/50 hover:border-rose-400', chip: 'bg-rose-950/70 text-rose-300 border border-rose-700/50', title: 'Output layer' },
  embedding: { ring: 'border-zinc-700', chip: 'bg-zinc-800 text-zinc-400', title: 'Embedding (coming soon)' },
  lstm: { ring: 'border-zinc-700', chip: 'bg-zinc-800 text-zinc-400', title: 'LSTM (coming soon)' },
  gru: { ring: 'border-zinc-700', chip: 'bg-zinc-800 text-zinc-400', title: 'GRU (coming soon)' },
}

/** One-line configuration summary shown under the layer name. */
function subtitleOf(kind: LayerKind, p: LayerNodeData['params']): string {
  switch (kind) {
    case 'input':
      return p.input_shape?.length === 3
        ? `${shapeText(p.input_shape)} image`
        : `${p.features ?? '?'} features`
    case 'dense':
    case 'output':
      return `${p.neurons ?? '?'} neurons · ${p.activation}`
    case 'activation':
      return `${p.activation} · element-wise`
    case 'dropout':
      return `p = ${p.dropout_rate}${p.dropout_mode === 'always' ? ' · always' : ''}`
    case 'batchnorm':
      return `mom ${p.momentum} · ε ${p.eps}${p.affine ? '' : ' · no affine'}`
    case 'flatten':
      return 'feature map → vector'
    case 'conv2d':
      return `${p.filters} filters · ${p.kernel_size}×${p.kernel_size} · s${p.stride} · ${p.padding_mode}`
    case 'maxpool':
    case 'avgpool':
      return `${p.pool_size}×${p.pool_size} · stride ${p.pool_stride}`
    case 'globalavgpool':
      return 'H × W × C → C'
    default:
      return 'coming soon'
  }
}

function LayerNode({ id, data, selected }: NodeProps<LayerNodeData>) {
  const validation = useAppStore((s) => s.validation)
  const style = KIND_STYLE[data.kind]
  const p = data.params
  const issue = validation?.errors.find((e) => e.layer_id === id) ?? validation?.warnings.find((w) => w.layer_id === id)
  const shape = validation?.layers.find((l) => l.id === id)
  const subtitle = subtitleOf(data.kind, p)

  return (
    <div
      className={`rounded-xl border ${style.ring} bg-lab-900 shadow-subtle min-w-[170px] px-3.5 py-2.5 transition-all duration-150 ${
        selected ? 'ring-2 ring-blue-500 shadow-elevated scale-[1.02]' : 'hover:scale-[1.01]'
      }`}
      title={[
        style.title,
        shape?.note,
        issue ? `⚠ ${issue.message}` : '',
        issue?.suggestion ? `💡 ${issue.suggestion}` : '',
      ].filter(Boolean).join('\n')}
    >
      {data.kind !== 'input' && (
        <Handle
          type="target"
          position={Position.Left}
          className="!bg-blue-500 !border-2 !border-zinc-950 !w-3 !h-3"
        />
      )}
      <div className="flex items-center gap-2">
        <span className={`badge ${style.chip}`}>{data.kind}</span>
        {shape && shape.params > 0 && (
          <span className="ml-auto text-[10px] font-mono text-zinc-400 font-medium" title="Trainable parameters">
            {shape.params.toLocaleString()} θ
          </span>
        )}
      </div>
      <div className="font-bold text-sm mt-1 truncate max-w-[200px] text-white">{data.label}</div>
      <div className="text-xs text-zinc-400 font-mono mt-0.5">{subtitle}</div>
      {/* Shape propagation: show the real inferred input → output shape. */}
      {shape?.out_shape && (
        <div className="text-[11px] text-zinc-300 font-mono mt-1 flex items-center gap-1">
          <span className="text-zinc-500">{shape.in_shape ? `${shapeText(shape.in_shape)} → ` : ''}</span>
          <b className="text-blue-300 font-medium">{shapeText(shape.out_shape)}</b>
        </div>
      )}
      {issue && <div className="text-xs text-rose-400 mt-1.5 max-w-[200px] leading-tight font-medium">⚠ {issue.message}</div>}
      {data.kind !== 'output' && (
        <Handle
          type="source"
          position={Position.Right}
          className="!bg-blue-500 !border-2 !border-zinc-950 !w-3 !h-3"
        />
      )}
    </div>
  )
}

export default memo(LayerNode)
