import { useMemo } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { fmt, pct } from '../../lib/format'
import { useAppStore } from '../../store/useAppStore'
import { Checkbox, Field, NumInput, Select, Stat, TextInput } from '../common/Stat'

const OPT_HINTS: Record<string, string> = {
  sgd: 'Plain stochastic gradient descent: w ← w − η·∇L.',
  momentum: 'SGD + momentum 0.9 — accelerates through valleys, damps oscillations.',
  adam: 'Adaptive per-parameter learning rates (momentum + RMS scaling). Great default.',
  rmsprop: 'Adapts the learning rate by a moving average of squared gradients.',
}
const LOSS_HINTS: Record<string, string> = {
  cross_entropy: 'Multi-class classification (softmax output, one neuron per class).',
  bce: 'Binary classification (exactly 1 sigmoid output neuron).',
  mse: 'Regression — mean squared error, punishes large errors strongly.',
  mae: 'Regression — mean absolute error, more robust to outliers.',
}

export function TrainingPanel() {
  const s = useAppStore()
  const job = s.job
  const running = job.state === 'running' || job.state === 'paused' || job.state === 'queued'
  const last = job.history[job.history.length - 1]
  const chartData = useMemo(() => job.history.map((h) => ({
    epoch: h.epoch,
    'train loss': h.train_loss,
    'val loss': h.val_loss ?? undefined,
    [`train ${h.metric_name}`]: h.train_metric,
    [`val ${h.metric_name}`]: h.val_metric ?? undefined,
  })), [job.history])
  const metricName = last?.metric_name ?? 'accuracy'
  const isCls = metricName === 'accuracy'

  return (
    <div className="h-full flex gap-3 p-3 overflow-hidden">
      {/* configuration column */}
      <div className="w-80 shrink-0 overflow-y-auto space-y-3">
        <div className="panel p-3 space-y-2.5">
          <div className="panel-title">Training configuration</div>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Epochs" hint="One epoch = one full pass over the training set.">
              <NumInput value={s.trainConfig.epochs} min={1} max={2000} step={1} disabled={running}
                onChange={(v) => s.setTrainConfig({ epochs: Math.max(1, Math.round(v)) })} />
            </Field>
            <Field label="Batch size" hint="Samples per weight update.">
              <NumInput value={s.trainConfig.batch_size} min={1} max={1024} step={1} disabled={running}
                onChange={(v) => s.setTrainConfig({ batch_size: Math.max(1, Math.round(v)) })} />
            </Field>
            <Field label="Learning rate η" hint="Controls how large each weight update is during training.">
              <NumInput value={s.trainConfig.learning_rate} min={0.000001} step={0.001} disabled={running}
                onChange={(v) => s.setTrainConfig({ learning_rate: v })} />
            </Field>
            <Field label="Random seed" hint="Same seed = reproducible results.">
              <NumInput value={s.trainConfig.seed} min={0} step={1} disabled={running}
                onChange={(v) => s.setTrainConfig({ seed: Math.max(0, Math.round(v)) })} />
            </Field>
          </div>
          <Field label="Optimizer" hint={OPT_HINTS[s.trainConfig.optimizer]}>
            <Select value={s.trainConfig.optimizer} disabled={running}
              options={Object.keys(OPT_HINTS).map((k) => ({ value: k, label: k.toUpperCase() }))}
              onChange={(v) => s.setTrainConfig({ optimizer: v as never })} />
          </Field>
          <Field label="Loss function" hint={LOSS_HINTS[s.trainConfig.loss]}>
            <Select value={s.trainConfig.loss} disabled={running}
              options={Object.entries(LOSS_HINTS).map(([k, h]) => ({ value: k, label: `${k === 'bce' ? 'Binary Cross-Entropy' : k === 'cross_entropy' ? 'Cross-Entropy' : k.toUpperCase()} — ${h.split(' —')[0].split('(')[0]}` }))}
              onChange={(v) => s.setTrainConfig({ loss: v as never })} />
          </Field>
          <div className="grid grid-cols-2 gap-2">
            <Field label="L2 regularization λ" hint="Adds λ·Σw² to the loss to discourage large weights (0 = off).">
              <NumInput value={s.trainConfig.l2} min={0} step={0.0001} disabled={running}
                onChange={(v) => s.setTrainConfig({ l2: Math.max(0, v) })} />
            </Field>
            <Field label="Snapshot every (epochs)" hint="How often activation & gradient statistics are captured.">
              <NumInput value={s.trainConfig.snapshot_every} min={1} max={200} step={1} disabled={running}
                onChange={(v) => s.setTrainConfig({ snapshot_every: Math.max(1, Math.round(v)) })} />
            </Field>
          </div>
          <Checkbox checked={s.trainConfig.shuffle} label="Shuffle batches each epoch"
            onChange={(v) => s.setTrainConfig({ shuffle: v })} />
          <Field label="Experiment name (optional)" hint="Set a name to record this run under Research Mode for comparison.">
            <TextInput value={s.experimentName} onChange={s.setExperimentName}
              placeholder="e.g. xor-relu-vs-tanh-v1" />
          </Field>

          <div className="flex gap-2 pt-1">
            {!running ? (
              <button className="btn-ok flex-1" onClick={() => void s.startTraining()} disabled={s.trainingBusy || s.backendOk === false}>
                ▶ Train
              </button>
            ) : (
              <>
                {job.state === 'running' && <button className="btn flex-1" onClick={() => void s.pauseJob()}>⏸ Pause</button>}
                {job.state === 'paused' && <button className="btn-ok flex-1" onClick={() => void s.resumeJob()}>⏵ Resume</button>}
                <button className="btn-danger flex-1" onClick={() => void s.stopJob()}>■ Stop</button>
              </>
            )}
            {(job.history.length > 0 || job.state === 'failed') && !running && (
              <button className="btn-ghost" onClick={s.resetJob} title="Clear training state & graphs">↺ Reset</button>
            )}
          </div>
        </div>

        {job.final && (
          <div className="panel p-3">
            <div className="panel-title mb-1.5">Final test-set evaluation</div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
              {Object.entries(job.final.test_metrics).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-slate-400">{k}</span>
                  <span className="text-lab-accent">{isCls ? pct(v) : fmt(v)}</span>
                </div>
              ))}
              <div className="flex justify-between"><span className="text-slate-400">epochs run</span><span>{job.final.total_epochs_run}</span></div>
              <div className="flex justify-between"><span className="text-slate-400">duration</span><span>{job.duration?.toFixed(1)}s</span></div>
            </div>
            {job.final.confusion_matrix && (
              <div className="mt-2">
                <div className="panel-title mb-1">Confusion matrix (rows=true)</div>
                <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${job.final.confusion_matrix.length}, minmax(28px, auto))` }}>
                  {job.final.confusion_matrix.flatMap((row, r) => row.map((c, i) => (
                    <div key={`${r}-${i}`}
                      className="text-center text-[10px] font-mono rounded px-1 py-0.5"
                      style={{
                        background: r === i ? 'rgba(52,211,153,0.25)' : c > 0 ? 'rgba(248,113,113,0.25)' : 'rgba(100,116,139,0.12)',
                      }}
                      title={`true ${r} → predicted ${i}: ${c}`}>{c}</div>
                  )))}
                </div>
              </div>
            )}
          </div>
        )}
        {job.state === 'failed' && job.error && (
          <div className="panel border-rose-500/50 p-3 text-xs text-rose-300">❌ {job.error}</div>
        )}
      </div>

      {/* live stats + charts */}
      <div className="flex-1 flex flex-col gap-3 min-w-0 overflow-y-auto">
        <div className="flex gap-2 flex-wrap">
          <Stat label="Epoch" hint="Current epoch / total" value={last ? `${last.epoch} / ${last.total_epochs ?? s.trainConfig.epochs}` : '—'} />
          <Stat label="Batch" hint="Batches per epoch" value={last?.batches != null ? `${last.batches}` : '—'} />
          <Stat label="Train loss" hint="Average loss over training batches" value={fmt(last?.train_loss)} />
          <Stat label={isCls ? 'Train acc' : 'Train MAE'} value={isCls ? pct(last?.train_metric) : fmt(last?.train_metric)}
            accent="text-emerald-400" />
          <Stat label="Val loss" hint="Loss on the held-out validation split" value={fmt(last?.val_loss ?? null)} />
          <Stat label={isCls ? 'Val acc' : 'Val MAE'} hint="Validation metric — the one to trust" value={isCls ? pct(last?.val_metric ?? null) : fmt(last?.val_metric ?? null)}
            accent="text-sky-400" />
          <Stat label="LR" hint="Learning rate" value={fmt(s.trainConfig.learning_rate, 5)} />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 flex-1 min-h-[220px]">
          <div className="panel p-2">
            <div className="panel-title px-1 pt-1">Loss curves</div>
            <ResponsiveContainer width="100%" height="88%">
              <LineChart data={chartData} margin={{ top: 8, right: 10, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233052" />
                <XAxis dataKey="epoch" fontSize={10} stroke="#64748b" />
                <YAxis fontSize={10} stroke="#64748b" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="train loss" stroke="#f87171" dot={false} strokeWidth={1.6} isAnimationActive={false} />
                <Line type="monotone" dataKey="val loss" stroke="#fb923c" dot={false} strokeWidth={1.6} strokeDasharray="5 3" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="panel p-2">
            <div className="panel-title px-1 pt-1">{isCls ? 'Accuracy curves' : 'Metric curves (MAE)'}</div>
            <ResponsiveContainer width="100%" height="88%">
              <LineChart data={chartData} margin={{ top: 8, right: 10, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233052" />
                <XAxis dataKey="epoch" fontSize={10} stroke="#64748b" />
                <YAxis fontSize={10} stroke="#64748b" domain={isCls ? [0, 1] : ['auto', 'auto']} tickFormatter={isCls ? (v: number) => `${Math.round(v * 100)}%` : undefined} />
                <Tooltip contentStyle={tooltipStyle} formatter={isCls ? (v) => pct(v as number) : undefined} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey={`train ${metricName}`} stroke="#34d399" dot={false} strokeWidth={1.6} isAnimationActive={false} />
                <Line type="monotone" dataKey={`val ${metricName}`} stroke="#38bdf8" dot={false} strokeWidth={1.6} strokeDasharray="5 3" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

export const tooltipStyle = {
  background: '#131a2e', border: '1px solid #2f3f69', borderRadius: 8, fontSize: 11,
} as const
