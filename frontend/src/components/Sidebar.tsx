import { useEffect, useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import type { LayerCatalogCategory, LayerKind } from '../types'

/** Accent colour per category, so BASIC / CNN / ADVANCED read differently. */
const CATEGORY_STYLE: Record<string, { chip: string; ring: string }> = {
  BASIC: { chip: 'bg-sky-500/15 text-sky-300', ring: 'text-sky-400/80' },
  CNN: { chip: 'bg-violet-500/15 text-violet-300', ring: 'text-violet-400/80' },
  ADVANCED: { chip: 'bg-slate-500/15 text-slate-400', ring: 'text-slate-500' },
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
        'panel px-2.5 py-1.5 select-none transition-colors',
        disabled
          ? 'opacity-45 cursor-not-allowed'
          : 'card-hover cursor-grab active:cursor-grabbing',
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
        <span className="font-semibold text-[12px] leading-tight">{entry.name}</span>
        {disabled && (
          <span className="ml-auto text-[8px] uppercase tracking-wide bg-slate-600/50 text-slate-300 px-1.5 py-[1px] rounded">
            soon
          </span>
        )}
        {!disabled && <span className="ml-auto text-[9px] text-slate-600">⠿</span>}
      </div>
      <p className="text-[10px] text-slate-400 mt-0.5 leading-snug line-clamp-2">
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
    <div className="mb-1.5">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-1.5 px-1.5 py-1 rounded hover:bg-lab-800/70 transition-colors"
        aria-expanded={open}
      >
        <span className={`text-[9px] transition-transform ${open ? 'rotate-90' : ''}`}>▶</span>
        <span className={`text-[10px] font-bold tracking-wider ${accent.ring}`}>{cat.category}</span>
        <span className="text-[9px] text-slate-600">
          {supported}/{cat.layers.length}
        </span>
      </button>
      {open && (
        <div className="space-y-1.5 pl-1 mt-1">
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
    <aside className="w-60 shrink-0 border-r border-lab-700/60 bg-lab-900/60 flex flex-col overflow-hidden">
      {/* Layer palette: collapsible so the long list never forces endless scrolling. */}
      <div className="p-3 overflow-y-auto flex-1 min-h-0">
        <div className="panel-title mb-2 flex items-center justify-between">
          <div className="flex items-center gap-1">
            <span>Layers</span>
            <span className="text-[9px] font-normal text-slate-500 normal-case">
              ({catalog.reduce((n, c) => n + c.layers.filter((l) => l.supported).length, 0)} available)
            </span>
          </div>
          <button
            onClick={() => s.toggleSidebar()}
            className="text-slate-400 hover:text-slate-200 text-xs px-1.5 py-0.5 rounded hover:bg-lab-800"
            title="Collapse Sidebar"
          >
            ◀
          </button>
        </div>
        {catalog.map((c) => (
          <CategorySection key={c.category} cat={c} defaultOpen={c.category === 'BASIC'} />
        ))}
        <p className="text-[10px] text-slate-600 leading-snug mt-2 px-1">
          Greyed-out layers are shown for orientation only — they are not implemented
          end-to-end yet, so they cannot be added to a network.
        </p>
      </div>

      <div className="p-3 border-t border-lab-700/60">
        <div className="panel-title mb-2">Datasets</div>
        <div className="space-y-1 max-h-56 overflow-auto pr-1">
          {s.datasets.map((d) => {
            const active = (d.kind === 'builtin' && s.dataset.kind === 'builtin' && s.dataset.name === d.id) ||
              (d.kind === 'upload' && s.dataset.upload_id === d.id)
            return (
              <button
                key={`${d.kind}-${d.id}`}
                className={`w-full text-left px-2 py-1.5 rounded-md text-xs transition-colors ${active ? 'bg-sky-600/30 text-sky-300' : 'hover:bg-lab-800'}`}
                onClick={() => {
                  if (d.kind === 'builtin') s.setDatasetSelection({ kind: 'builtin', name: d.id, upload_id: undefined })
                  else s.setDatasetSelection({ kind: 'upload', upload_id: d.id, name: undefined })
                  s.log(`Dataset selected: ${d.name} (${d.samples} samples, ${d.n_features} features, ${d.task}).`)
                }}
                title={d.description}
              >
                <div className="font-medium truncate">{d.name}</div>
                <div className="text-[10px] text-slate-500">{d.samples} · {d.n_features}f · {d.task}{d.n_classes ? ` · ${d.n_classes}cls` : ''}</div>
              </button>
            )
          })}
        </div>
        <button className="btn-ghost w-full mt-2 !py-1" onClick={() => s.setTab('dataset')}>
          Open Dataset Manager →
        </button>
      </div>

      <div className="mt-auto p-3 border-t border-lab-700/60 text-[10px] text-slate-500 leading-snug">
        <b className="text-slate-400">Quick start:</b> drag Input → Dense → Dense → Output, connect them,
        open <b>Dataset</b> & pick one, open <b>Training</b> and press <b>▶ Train</b>.
      </div>
    </aside>
  )
}
