import type { ArchitectureProjection, KvCacheProjection } from "../../../replay/src/learning.ts"
import { architectureFocusLabel, type ArchitectureFocus } from "../architecture/model.ts"
import type { Selection } from "./types.ts"

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
  const button = "rounded-md border border-line bg-raised px-3 py-1.5 text-sm text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
  return (
    <div className="space-y-3">
      <nav className="flex flex-wrap items-center gap-2" aria-label="Model architecture navigation">
        <button type="button" className={button} onClick={() => onExplore({ kind: "model" })}>Model</button>
        <label className="flex min-w-36 items-center gap-2 text-xs text-muted">
          Layer
          <select aria-label="Model layer" className="rounded-md border border-line bg-raised p-1.5 text-sm text-ink" value={layer} onChange={(event) => {
            const layerIndex = Number(event.target.value)
            onExplore({ kind: "layer", layerIndex }); onSelect({ kind: "layer", layerIndex })
          }}>
            {Array.from({ length: architecture.layerCount }, (_, index) => <option key={index} value={index}>Layer {index + 1}</option>)}
          </select>
        </label>
        <button type="button" className={button} onClick={() => { onExplore({ kind: "layer", layerIndex: layer }); onSelect({ kind: "layer", layerIndex: layer }) }}>Explore layer</button>
        <button type="button" className={button} onClick={() => { onExplore({ kind: "attention", layerIndex: layer }); onSelect({ kind: "module", layerIndex: layer, module: "attention" }) }}>Attention</button>
        <button type="button" className={button} onClick={() => { onExplore({ kind: "ffn", layerIndex: layer }); onSelect({ kind: "module", layerIndex: layer, module: "ffn" }) }}>Feed-forward</button>
      </nav>
      {focus?.kind === "attention" || focus?.kind === "head" ? (
        <div className="flex flex-wrap gap-2">
          {(["query", "kv"] as const).map((headKind) => <label key={headKind} className="flex items-center gap-2 text-xs text-muted">
            {headKind === "query" ? "Query head" : "KV head"}
            <select aria-label={headKind === "query" ? "Query head" : "KV head"} value={focus.kind === "head" && focus.headKind === headKind ? focus.headIndex : ""} onChange={(event) => {
              if (!event.target.value) return
              const headIndex = Number(event.target.value)
              onExplore({ kind: "head", layerIndex: layer, headKind, headIndex }); onSelect({ kind: "head", layerIndex: layer, headKind, headIndex })
            }}>
              <option value="">Select a head</option>
              {Array.from({ length: (headKind === "query" ? architecture.queryHeads : architecture.kvHeads) ?? 0 }, (_, index) => <option key={index} value={index}>Head {index + 1}</option>)}
            </select>
          </label>)}
        </div>
      ) : null}
      <p className="m-0 text-xs text-muted" aria-live="polite">{architectureFocusLabel(focus)}. Shapes represent model structure; numerical activations and individual head durations were not captured.</p>
      {progress != null && onProgress ? <label className="block text-xs text-muted">
        Layer walkthrough · explanatory stage {Math.min(architecture.layerCount, Math.floor(progress * architecture.layerCount) + 1)} of {architecture.layerCount}
        <input aria-label="Layer walkthrough stage" type="range" min={0} max={architecture.layerCount - 1} step={1} value={Math.min(architecture.layerCount - 1, Math.floor(progress * architecture.layerCount))} onChange={(event) => onProgress(Number(event.target.value) / architecture.layerCount!)} />
      </label> : null}
    </div>
  )
}

export function PoolNavigation({ cache, onSelect }: { cache: KvCacheProjection; onSelect: (selection: Selection) => void }) {
  return <div className="space-y-2">
    <p className="m-0 text-xs text-muted">{cache.reason ?? "Each cell is one physical block. A cached block with zero references is available for reassignment."}</p>
    {cache.pools.map((pool) => <form key={pool.pool} className="flex flex-wrap items-end gap-2" onSubmit={(event) => {
      event.preventDefault()
      const raw = new FormData(event.currentTarget).get("block")
      const blockId = Number(raw)
      if (raw === "" || !Number.isSafeInteger(blockId) || blockId < 0 || blockId >= pool.capacity) return
      const block = pool.blocks[blockId]
      if (block) onSelect({ kind: "block", key: block.key })
    }}>
      <label className="min-w-40 flex-1 text-xs text-muted">{pool.pool} · {pool.capacity.toLocaleString()} blocks · {pool.tokenOccupancyStatus !== "observed" ? "token-slot layout unavailable" : pool.blockSize == null ? "group-specific token geometry" : `${pool.blockSize} token positions each`}
        <input name="block" type="number" min={0} max={pool.capacity - 1} step={1} required placeholder="Physical block ID" aria-label={`Physical block ID in ${pool.pool}`} />
      </label>
      <button type="submit">Inspect block</button>
    </form>)}
  </div>
}
