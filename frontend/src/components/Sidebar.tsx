import { useEffect, useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import type { LayerCatalogCategory, LayerKind } from '../types'

/** Accent colour per category, so BASIC / CNN / ADVANCED read differently. */
const CATEGORY_STYLE: Record<string, { chip: string; ring: string }> = {
  BASIC: { chip: 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/50', ring: 'text-emerald-400' },
  CNN: { chip: 'bg-indigo-950/70 text-indigo-300 border border-indigo-700/50', ring: 'text-indigo-400' },
  ADVANCED: { chip: 'bg-blue-950/70 text-blue-300 border border-blue-700/50', ring: 'text-blue-400' },
}

function LayerCard({ entry, category }: { entry: LayerCatalogCategory['layers'][number]; category: string }) {
  const addLayer = useAppStore((s) => s.addLayer)
  const disabled = !entry.supported
  const accent = CATEGORY_STYLE[category] ?? CATEGORY_STYLE.BASIC

  return (
    <div
      draggable={!disabled}
      onDragStart={disabled ? undefined : (e) => {
        e.dataTransfer.setData('application/neurosim-layer', entry.kind)
        e.dataTransfer.effectAllowed = 'copy'
      }}
      onDoubleClick={disabled ? undefined : () => addLayer(entry.kind as LayerKind)}
      className={[
        'panel p-2.5 select-none transition-all duration-150',
        disabled
          ? 'opacity-40 cursor-not-allowed'
          : 'card-hover cursor-grab active:cursor-grabbing hover:bg-lab-800',
      ].join(' ')}
      title={[
        entry.formula,
        '',
        entry.description,
        entry.summary ? `Parameters: ${entry.summary}` : '',
        disabled ? `Coming soon — ${entry.coming_soon_note}` : 'Drag onto the canvas (or double-click to add)',
      ].filter(Boolean).join('\n')}
    >
      <div className="flex items-center gap-1.5">
        <span className={accent.ring}>{entry.icon}</span>
        <span className="font-bold text-xs leading-tight text-white">{entry.name}</span>
        {disabled && (
          <span className="ml-auto text-[9px] uppercase tracking-wide bg-zinc-800 text-zinc-400 border border-zinc-700 px-1.5 py-[1px] rounded">
            soon
          </span>
        )}
        {!disabled && <span className="ml-auto text-[10px] text-zinc-500">⠿</span>}
      </div>
      <p className="text-[11px] text-zinc-400 mt-1 leading-snug line-clamp-2">
        {disabled ? entry.coming_soon_note : entry.description}
      </p>
    </div>
  )
}

function CategorySection({ cat, defaultOpen }: { cat: LayerCatalogCategory; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const accent = CATEGORY_STYLE[cat.category] ?? CATEGORY_STYLE.BASIC
  const supported = cat.layers.filter((l) => l.supported).length

  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-lab-800 transition-colors"
        aria-expanded={open}
      >
        <span className={`text-[10px] transition-transform ${open ? 'rotate-90 text-blue-400' : 'text-zinc-500'}`}>▶</span>
        <span className={`text-xs font-bold tracking-wider ${accent.ring}`}>{cat.category}</span>
        <span className="text-[10px] text-zinc-500 ml-auto font-mono">
          {supported}/{cat.layers.length}
        </span>
      </button>
      {open && (
        <div className="space-y-1.5 pl-1.5 mt-1.5">
          {cat.layers.map((l) => <LayerCard key={l.kind} entry={l} category={cat.category} />)}
        </div>
      )}
    </div>
  )
}

export function Sidebar() {
  const s = useAppStore()
  const loadLayerCatalog = useAppStore((st) => st.loadLayerCatalog)
  const catalog = s.layerCatalog

  useEffect(() => { void loadLayerCatalog() }, [loadLayerCatalog])

  return (
    <aside className="w-64 shrink-0 border-r border-lab-700 bg-lab-900 flex flex-col overflow-hidden">
      {/* Layer palette: collapsible so the long list never forces endless scrolling. */}
      <div className="p-3.5 overflow-y-auto flex-1 min-h-0">
        <div className="panel-title mb-2.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-zinc-200">Layers</span>
            <span className="text-[10px] font-normal text-zinc-500 normal-case font-mono">
              ({catalog.reduce((n, c) => n + c.layers.filter((l) => l.supported).length, 0)} available)
            </span>
          </div>
          <button
            onClick={() => s.toggleSidebar()}
            className="text-zinc-400 hover:text-white text-xs px-1.5 py-0.5 rounded hover:bg-lab-800 transition-colors"
            title="Collapse Sidebar"
          >
            ◀
          </button>
        </div>
        {catalog.map((c) => (
          <CategorySection key={c.category} cat={c} defaultOpen={c.category === 'BASIC'} />
        ))}
        <p className="text-[10px] text-zinc-500 leading-snug mt-2 px-1">
          Greyed-out layers are for architectural preview and cannot be added yet.
        </p>
      </div>

      <div className="p-3.5 border-t border-lab-700 bg-lab-950/40">
        <div className="panel-title mb-2 text-zinc-300">Datasets</div>
        <div className="space-y-1.5 max-h-56 overflow-auto pr-1">
          {s.datasets.map((d) => {
            const active = (d.kind === 'builtin' && s.dataset.kind === 'builtin' && s.dataset.name === d.id) ||
              (d.kind === 'upload' && s.dataset.upload_id === d.id)
            return (
              <button
                key={`${d.kind}-${d.id}`}
                className={`w-full text-left px-2.5 py-2 rounded-lg text-xs transition-all border ${
                  active
                    ? 'bg-blue-950/60 text-blue-200 border-blue-500/50 shadow-sm'
                    : 'hover:bg-lab-800 text-zinc-300 border-transparent hover:border-zinc-700'
                }`}
                onClick={() => {
                  if (d.kind === 'builtin') s.setDatasetSelection({ kind: 'builtin', name: d.id, upload_id: undefined })
                  else s.setDatasetSelection({ kind: 'upload', upload_id: d.id, name: undefined })
                  s.log(`Dataset selected: ${d.name} (${d.samples} samples, ${d.n_features} features, ${d.task}).`)
                }}
                title={d.description}
              >
                <div className="font-semibold truncate">{d.name}</div>
                <div className="text-[10px] text-zinc-400 font-mono mt-0.5">{d.samples} · {d.n_features}f · {d.task}{d.n_classes ? ` · ${d.n_classes}cls` : ''}</div>
              </button>
            )
          })}
        </div>
        <button className="btn-ghost w-full mt-2.5 !py-1.5 text-xs text-center justify-center font-medium" onClick={() => s.setTab('dataset')}>
          Dataset Manager →
        </button>
      </div>

      <div className="mt-auto p-3 border-t border-lab-700 text-[11px] text-zinc-400 leading-snug bg-lab-950/60">
        <b className="text-white">Quick start:</b> Drag nodes, connect handles, pick a dataset, then click <b className="text-blue-400">▶ Train</b>.
      </div>
    </aside>
  )
}
