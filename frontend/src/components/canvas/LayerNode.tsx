import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import { shapeText, useAppStore, type LayerNodeData } from '../../store/useAppStore'
import type { LayerKind } from '../../types'

const KIND_STYLE: Record<LayerKind, { ring: string; chip: string; title: string }> = {
  input: { ring: 'border-sky-500/70', chip: 'bg-sky-500/20 text-sky-300', title: 'Input layer' },
  dense: { ring: 'border-violet-500/70', chip: 'bg-violet-500/20 text-violet-300', title: 'Fully-connected (dense) layer' },
  activation: { ring: 'border-fuchsia-500/70', chip: 'bg-fuchsia-500/20 text-fuchsia-300', title: 'Activation layer' },
  dropout: { ring: 'border-amber-500/70', chip: 'bg-amber-500/20 text-amber-300', title: 'Dropout regularization layer' },
  batchnorm: { ring: 'border-teal-500/70', chip: 'bg-teal-500/20 text-teal-300', title: 'Batch normalization layer' },
  flatten: { ring: 'border-slate-400/70', chip: 'bg-slate-400/20 text-slate-300', title: 'Flatten layer' },
  conv2d: { ring: 'border-indigo-500/70', chip: 'bg-indigo-500/20 text-indigo-300', title: '2-D convolutional layer' },
  maxpool: { ring: 'border-cyan-500/70', chip: 'bg-cyan-500/20 text-cyan-300', title: 'Max pooling layer' },
  avgpool: { ring: 'border-blue-500/70', chip: 'bg-blue-500/20 text-blue-300', title: 'Average pooling layer' },
  globalavgpool: { ring: 'border-emerald-500/60', chip: 'bg-emerald-500/15 text-emerald-300', title: 'Global average pooling layer' },
  output: { ring: 'border-emerald-500/70', chip: 'bg-emerald-500/20 text-emerald-300', title: 'Output layer' },
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
      className={`rounded-lg border-2 ${style.ring} bg-lab-850/95 shadow-lg min-w-[150px] px-3 py-2 ${selected ? 'outline outline-2 outline-sky-400/60' : ''}`}
      title={[
        style.title,
        shape?.note,
        issue ? `⚠ ${issue.message}` : '',
        issue?.suggestion ? `💡 ${issue.suggestion}` : '',
      ].filter(Boolean).join('\n')}
    >
      {data.kind !== 'input' && <Handle type="target" position={Position.Left} className="!bg-slate-400 !w-2.5 !h-2.5" />}
      <div className="flex items-center gap-2">
        <span className={`badge ${style.chip}`}>{data.kind}</span>
        {shape && shape.params > 0 && (
          <span className="ml-auto text-[9px] font-mono text-slate-400" title="Trainable parameters">
            {shape.params.toLocaleString()} θ
          </span>
        )}
      </div>
      <div className="font-semibold text-sm mt-1 truncate max-w-[180px]">{data.label}</div>
      <div className="text-[11px] text-slate-400 font-mono">{subtitle}</div>
      {/* Shape propagation: show the real inferred input → output shape. */}
      {shape?.out_shape && (
        <div className="text-[10px] text-sky-300/80 font-mono mt-0.5">
          {shape.in_shape ? `${shapeText(shape.in_shape)} → ` : ''}
          <b className="text-sky-200">{shapeText(shape.out_shape)}</b>
        </div>
      )}
      {issue && <div className="text-[10px] text-rose-400 mt-1 max-w-[190px] leading-tight">⚠ {issue.message}</div>}
      {data.kind !== 'output' && <Handle type="source" position={Position.Right} className="!bg-slate-400 !w-2.5 !h-2.5" />}
    </div>
  )
}

export default memo(LayerNode)
