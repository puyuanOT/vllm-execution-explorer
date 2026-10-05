import type { ArchitectureProjection, KvCacheProjection } from "../../../replay/src/learning.ts"
import { architectureFocusLabel, type ArchitectureFocus } from "../architecture/model.ts"
import { cn } from "../lib/cn.ts"
import type { Selection } from "./types.ts"

const control = "inline-flex min-h-7 items-center rounded-none border border-line-strong bg-raised px-2.5 py-0 font-mono text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink hover:border-slate-600 aria-[current=true]:border-accent-400/50 aria-[current=true]:text-accent"
const select = "h-7 w-auto rounded-none border-line-strong bg-surface py-0 pl-2 pr-6 font-mono text-[11px] text-ink"
const field = "flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.08em] text-faint"

export function ArchitectureNavigation({ architecture, focus, onExplore, onSelect, progress, onProgress }: {
  architecture: ArchitectureProjection
  focus: ArchitectureFocus | null
  onExplore: (focus: ArchitectureFocus) => void
  onSelect: (selection: Selection) => void
  progress?: number
  onProgress?: (value: number) => void
}) {
  if (!architecture.supported || !architecture.layerCount) return <p className="m-0 text-xs text-muted">{architecture.reason}</p>
  const layer = focus && focus.kind !== "model" ? focus.layerIndex : 0
  const current = (kind: ArchitectureFocus["kind"]) => (focus?.kind ?? "model") === kind ? "true" : undefined
  return (
    <div className="grid gap-2">
      <nav className="flex flex-wrap items-center gap-1.5" aria-label="Model architecture navigation">
        <button type="button" className={control} aria-current={current("model")} onClick={() => onExplore({ kind: "model" })}>Model</button>
        <label className={cn(field, "ml-1")}>
          Layer
          <select aria-label="Model layer" className={select} value={layer} onChange={(event) => {
            const layerIndex = Number(event.target.value)
            onExplore({ kind: "layer", layerIndex }); onSelect({ kind: "layer", layerIndex })
          }}>
            {Array.from({ length: architecture.layerCount }, (_, index) => <option key={index} value={index}>Layer {index + 1}</option>)}
          </select>
        </label>
        <button type="button" className={control} aria-current={current("layer")} onClick={() => { onExplore({ kind: "layer", layerIndex: layer }); onSelect({ kind: "layer", layerIndex: layer }) }}>Explore layer</button>
        <button type="button" className={control} aria-current={focus?.kind === "attention" || focus?.kind === "head" ? "true" : undefined} onClick={() => { onExplore({ kind: "attention", layerIndex: layer }); onSelect({ kind: "module", layerIndex: layer, module: "attention" }) }}>Attention</button>
        <button type="button" className={control} aria-current={current("ffn")} onClick={() => { onExplore({ kind: "ffn", layerIndex: layer }); onSelect({ kind: "module", layerIndex: layer, module: "ffn" }) }}>Feed-forward</button>
        {focus?.kind === "attention" || focus?.kind === "head" ? (["query", "kv"] as const).map((headKind) => (
          <label key={headKind} className={cn(field, "ml-1")}>
            {headKind === "query" ? "Query head" : "KV head"}
            <select aria-label={headKind === "query" ? "Query head" : "KV head"} className={select} value={focus.kind === "head" && focus.headKind === headKind ? focus.headIndex : ""} onChange={(event) => {
              if (!event.target.value) return
              const headIndex = Number(event.target.value)
              onExplore({ kind: "head", layerIndex: layer, headKind, headIndex }); onSelect({ kind: "head", layerIndex: layer, headKind, headIndex })
            }}>
              <option value="">Select</option>
              {Array.from({ length: (headKind === "query" ? architecture.queryHeads : architecture.kvHeads) ?? 0 }, (_, index) => <option key={index} value={index}>Head {index + 1}</option>)}
            </select>
          </label>
        )) : null}
      </nav>
      <p className="m-0 text-pretty text-[11.5px] text-faint" aria-live="polite">{architectureFocusLabel(focus)}. Shapes represent model structure; numerical activations and individual head durations were not captured.</p>
      {progress != null && onProgress ? <label className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-muted">
        <span>Layer walkthrough · explanatory stage {Math.min(architecture.layerCount, Math.floor(progress * architecture.layerCount) + 1)} of {architecture.layerCount}</span>
        <input className="min-w-40 flex-1 accent-[#9d90ec]" aria-label="Layer walkthrough stage" type="range" min={0} max={architecture.layerCount - 1} step={1} value={Math.min(architecture.layerCount - 1, Math.floor(progress * architecture.layerCount))} onChange={(event) => onProgress(Number(event.target.value) / architecture.layerCount!)} />
      </label> : null}
    </div>
  )
}

export function PoolNavigation({ cache, onSelect }: { cache: KvCacheProjection; onSelect: (selection: Selection) => void }) {
  return <div className="grid gap-2">
    <p className="m-0 text-pretty text-[11.5px] text-faint">{cache.reason ?? "Each cell is one physical block. A cached block with zero references is available for reassignment."}</p>
    {cache.pools.map((pool) => <form key={pool.pool} className="flex flex-wrap items-center gap-2" onSubmit={(event) => {
      event.preventDefault()
      const raw = new FormData(event.currentTarget).get("block")
      const blockId = Number(raw)
      if (raw === "" || !Number.isSafeInteger(blockId) || blockId < 0 || blockId >= pool.capacity) return
      const block = pool.blocks[blockId]
      if (block) onSelect({ kind: "block", key: block.key })
    }}>
      <label className={field}>
        <span className="normal-case tracking-normal text-muted">{pool.pool} · {pool.capacity.toLocaleString()} blocks · {pool.tokenOccupancyStatus !== "observed" ? "token-slot layout unavailable" : pool.blockSize == null ? "group-specific token geometry" : `${pool.blockSize} token positions each`}</span>
        <input className="h-7 w-40 rounded-none border-line-strong bg-surface py-0 font-mono text-[11px] text-ink" name="block" type="number" min={0} max={pool.capacity - 1} step={1} required placeholder="Physical block ID" aria-label={`Physical block ID in ${pool.pool}`} />
      </label>
      <button type="submit" className={control}>Inspect block</button>
    </form>)}
  </div>
}
