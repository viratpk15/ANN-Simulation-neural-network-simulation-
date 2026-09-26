import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import { shapeText, useAppStore, type LayerNodeData } from '../../store/useAppStore'
import type { LayerKind } from '../../types'

const KIND_STYLE: Record<LayerKind, { ring: string; chip: string; title: string }> = {
  input: { ring: 'border-pink-500/70 hover:border-pink-400', chip: 'bg-pink-500/20 text-pink-300 border border-pink-500/40', title: 'Input layer' },
  dense: { ring: 'border-purple-500/70 hover:border-purple-400', chip: 'bg-purple-500/20 text-purple-300 border border-purple-500/40', title: 'Fully-connected (dense) layer' },
  activation: { ring: 'border-fuchsia-500/70 hover:border-fuchsia-400', chip: 'bg-fuchsia-500/20 text-fuchsia-300 border border-fuchsia-500/40', title: 'Activation layer' },
  dropout: { ring: 'border-amber-500/70 hover:border-amber-400', chip: 'bg-amber-500/20 text-amber-300 border border-amber-500/40', title: 'Dropout regularization layer' },
  batchnorm: { ring: 'border-violet-400/70 hover:border-violet-300', chip: 'bg-violet-500/20 text-violet-300 border border-violet-400/40', title: 'Batch normalization layer' },
  flatten: { ring: 'border-indigo-400/70 hover:border-indigo-300', chip: 'bg-indigo-500/20 text-indigo-300 border border-indigo-400/40', title: 'Flatten layer' },
  conv2d: { ring: 'border-pink-400/70 hover:border-pink-300', chip: 'bg-pink-600/20 text-pink-300 border border-pink-400/40', title: '2-D convolutional layer' },
  maxpool: { ring: 'border-purple-400/70 hover:border-purple-300', chip: 'bg-purple-600/20 text-purple-300 border border-purple-400/40', title: 'Max pooling layer' },
  avgpool: { ring: 'border-violet-500/70 hover:border-violet-400', chip: 'bg-violet-600/20 text-violet-300 border border-violet-500/40', title: 'Average pooling layer' },
  globalavgpool: { ring: 'border-fuchsia-400/70 hover:border-fuchsia-300', chip: 'bg-fuchsia-600/20 text-fuchsia-300 border border-fuchsia-400/40', title: 'Global average pooling layer' },
  output: { ring: 'border-rose-400/80 hover:border-rose-300', chip: 'bg-rose-500/20 text-rose-300 border border-rose-400/40', title: 'Output layer' },
  embedding: { ring: 'border-slate-600/60', chip: 'bg-slate-600/30 text-slate-400', title: 'Embedding (coming soon)' },
  lstm: { ring: 'border-slate-600/60', chip: 'bg-slate-600/30 text-slate-400', title: 'LSTM (coming soon)' },
  gru: { ring: 'border-slate-600/60', chip: 'bg-slate-600/30 text-slate-400', title: 'GRU (coming soon)' },
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
      className={`rounded-xl border-2 ${style.ring} bg-lab-900/90 backdrop-blur-md shadow-glass min-w-[170px] px-3.5 py-2.5 transition-all duration-200 ${
        selected ? 'ring-2 ring-pink-500 shadow-neon-pink scale-[1.02]' : 'hover:shadow-neon-purple hover:scale-[1.01]'
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
          className="!bg-pink-500 !border-2 !border-purple-300 !w-3 !h-3 shadow-[0_0_8px_rgba(236,72,153,0.9)]"
        />
      )}
      <div className="flex items-center gap-2">
        <span className={`badge ${style.chip}`}>{data.kind}</span>
        {shape && shape.params > 0 && (
          <span className="ml-auto text-[10px] font-mono text-purple-300/80 font-medium" title="Trainable parameters">
            {shape.params.toLocaleString()} θ
          </span>
        )}
      </div>
      <div className="font-bold text-sm mt-1 truncate max-w-[200px] text-slate-100">{data.label}</div>
      <div className="text-xs text-purple-300/70 font-mono mt-0.5">{subtitle}</div>
      {/* Shape propagation: show the real inferred input → output shape. */}
      {shape?.out_shape && (
        <div className="text-[11px] text-pink-300/90 font-mono mt-1 flex items-center gap-1">
          <span className="text-slate-400">{shape.in_shape ? `${shapeText(shape.in_shape)} → ` : ''}</span>
          <b className="text-pink-300 font-semibold">{shapeText(shape.out_shape)}</b>
        </div>
      )}
      {issue && <div className="text-xs text-rose-400 mt-1.5 max-w-[200px] leading-tight font-medium">⚠ {issue.message}</div>}
      {data.kind !== 'output' && (
        <Handle
          type="source"
          position={Position.Right}
          className="!bg-purple-500 !border-2 !border-pink-300 !w-3 !h-3 shadow-[0_0_8px_rgba(168,85,247,0.9)]"
        />
      )}
    </div>
  )
}

export default memo(LayerNode)
