import { useMemo, useState } from "react"
import { projectLayerTokenMatrix } from "../../../replay/src/kv.ts"
import { projectArchitecture } from "../../../replay/src/learning.ts"
import type { ProfileKvObservations } from "../../../replay/src/profiling.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { requestLabel } from "../explore/model.ts"
import { cn } from "../lib/cn.ts"
import { EvidenceTag } from "../ui/controls.tsx"
import type { Selection } from "./types.ts"

const STATES = {
  scheduled: { label: "Scheduled", mark: "S", tone: "text-muted bg-raised" },
  operation_active: { label: "KV write operation active", mark: "W", tone: "text-amber-200 bg-amber-400/15" },
  completed: { label: "KV write operation completed", mark: "C", tone: "text-accent bg-accent/10" },
  reused_existing_kv: { label: "Existing prefix KV reused", mark: "R", tone: "text-emerald-200 bg-emerald-400/10" },
  not_participating: { label: "Outside this step", mark: "—", tone: "text-faint" },
  unknown: { label: "Unknown", mark: "?", tone: "text-faint cp-hatch" },
} as const

export function LayerTokenMatrix({ state, manifest, profile, timeUs, stepId, requestId, tokenPosition, labels, onSelect, onClearTracking, onLayerTimeSelect, onParticipantSelect }: {
  state: ReplayState; manifest: RecordingManifest | null; profile?: ProfileKvObservations | null; timeUs?: number | null
  requestId?: string | null; tokenPosition?: number | null
  stepId?: string | null
  labels?: ReadonlyMap<string, string>
  onSelect: (selection: Selection) => void
  onClearTracking?: () => void
  onLayerTimeSelect?: (layerIndex: number) => void
  onParticipantSelect?: (layerIndex: number, requestId: string, position: number) => void
}) {
  const [page, setPage] = useState(0)
  const matrix = useMemo(() => projectLayerTokenMatrix(state, manifest, profile ?? null, { stepId, requestId, tokenPosition, timeUs }), [state, manifest, profile, stepId, requestId, tokenPosition, timeUs])
  const architecture = useMemo(() => projectArchitecture(manifest), [manifest])
  const recurrentLayers = new Set(architecture.layerTypes.flatMap((type, index) => type === "linear_attention" ? [index] : []))
  const mappedBatch = !profile || stepId != null
  const scheduledCounts = useMemo(() => {
    const counts = new Map<string, number>()
    if (profile) {
      const fullStep = profile.steps.find((step) => step.stepId === matrix.stepId)
      if (fullStep?.mappingStatus !== "observed") return null
      for (const row of fullStep.packedRows) {
        if (!row.isPadding && row.requestId != null && row.tokenPosition != null) counts.set(row.requestId, (counts.get(row.requestId) ?? 0) + 1)
      }
    } else {
      const fullStep = state.steps.find((step) => step.id === matrix.stepId)
      if (!fullStep) return null
      for (const request of fullStep.requests) counts.set(request.requestId, (counts.get(request.requestId) ?? 0) + request.rangeLength)
    }
    return counts
  }, [profile, matrix.stepId, state.steps])
  const totalScheduled = scheduledCounts ? [...scheduledCounts].reduce((count, [id, positions]) => count + (requestId == null || id === requestId ? positions : 0), 0) : null
  const pageSize = 8
  const columns = useMemo(() => !mappedBatch ? [] : requestId ? matrix.tokens.map((token) => ({ key: `${token.requestId}:${token.position}`, requestId: token.requestId, positions: [token.position], scheduledCount: Number(token.scheduled), label: `Position ${token.position}`, fragment: token.tokenPiece, tokenId: token.tokenId }))
    : [...new Set(matrix.tokens.map((token) => token.requestId))].map((id) => {
      const tokens = matrix.tokens.filter((token) => token.requestId === id)
      return { key: id, requestId: id, positions: tokens.map((token) => token.position), scheduledCount: tokens.filter((token) => token.scheduled).length, label: requestLabel(labels, id), fragment: null, tokenId: null }
    }), [matrix.tokens, requestId, labels, mappedBatch])
  const currentPage = Math.min(page, Math.max(0, Math.ceil(columns.length / pageSize) - 1))
  const visible = columns.slice(currentPage * pageSize, (currentPage + 1) * pageSize)
  const cells = useMemo(() => new Map(matrix.cells.map((cell) => [`${cell.layerIndex}:${cell.requestId}:${cell.position}`, cell])), [matrix.cells])
  const selectLayer = (layerIndex: number) => onLayerTimeSelect ? onLayerTimeSelect(layerIndex) : onSelect({ kind: "layer", layerIndex })
  return <section className="grid gap-2 border-t border-line bg-[#060a11] px-3.5 py-3" aria-label="Layer and token participation">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 className="m-0 text-sm font-semibold text-ink">Layer × {requestId ? "token positions" : "requests"} · {mappedBatch ? matrix.stepId ?? "no step" : "outside mapped batch"}</h2>
      <EvidenceTag kind={matrix.evidenceKind === "fixture" ? "synthetic" : profile ? "observed" : "derived"}>{matrix.evidenceKind === "fixture" ? "Synthetic" : profile ? "Profiler evidence" : "Host step boundaries"}</EvidenceTag>
    </div>
    {requestId ? <div className="flex flex-wrap gap-2">{tokenPosition != null ? <button type="button" className="text-xs" onClick={() => onSelect({ kind: "request", id: requestId })}>Request token positions</button> : null}{onClearTracking ? <button type="button" className="text-xs" onClick={onClearTracking}>Whole batch</button> : null}</div> : null}
    <p className="m-0 text-pretty text-xs text-muted">{!mappedBatch ? "The profiler playhead is outside an observed batch interval. Select a profiled batch to inspect its participants." : matrix.reason ?? "Write activity applies to the participating batch operation. It does not measure an individual token’s duration."}</p>
    {profile && !profile.cacheWrites.length ? <p className="m-0 text-pretty text-xs text-muted">Per-layer cache-write operations were not captured in this recording. Unknown cells distinguish this missing evidence from scheduled host participation.</p> : null}
    {recurrentLayers.size ? <p className="m-0 text-pretty text-xs text-muted">Gated DeltaNet layers use recurrent state, rather than per-token attention KV blocks. Their cells show GDN; select a layer to inspect its captured recurrent activity.</p> : null}
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted" aria-label="Layer matrix legend">
      {Object.values(STATES).map((item) => <span key={item.mark}><b className={cn("mr-1 inline-grid size-4 place-items-center font-mono", item.tone)}>{item.mark}</b>{item.label}</span>)}
      {matrix.truncated ? <span>Cell counts cover shown positions only.</span> : null}
    </div>
    {columns.length ? <>
      <div className="flex items-center gap-2 text-xs tabular-nums text-muted">
        <button type="button" disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>Previous columns</button>
        <span>{currentPage * pageSize + 1}–{Math.min(columns.length, (currentPage + 1) * pageSize)} of {columns.length}</span>
        <button type="button" disabled={(currentPage + 1) * pageSize >= columns.length} onClick={() => setPage(currentPage + 1)}>Next columns</button>
        {requestId ? <span className="ml-auto">Following {requestLabel(labels, requestId)}{tokenPosition == null ? "" : ` · selected position ${tokenPosition}`}</span> : <span className="ml-auto">Select a request to inspect its tokens</span>}
      </div>
      <div className="max-h-56 overflow-auto border border-line">
        <table className="w-full border-collapse text-left text-[11px] tabular-nums">
          <caption className="sr-only">Layer cache-write status for scheduled requests and token positions; individual token timings are unavailable.</caption>
          <thead className="sticky top-0 z-10 bg-[#0b1320] text-muted"><tr><th scope="col" className="px-2 py-1">Layer</th>{visible.map((column) => <th key={column.key} scope="col" className="min-w-24 border-l border-line px-1 py-1">
            <button type="button" className={cn("w-full min-h-7 border-0 bg-transparent px-1 py-0 text-left text-xs", column.positions.length === 1 && column.positions[0] === tokenPosition && "text-accent")} onClick={() => onSelect(requestId ? { kind: "token", requestId: column.requestId, position: column.positions[0] } : { kind: "request", id: column.requestId })}>
              <span className="block">{column.label}</span>
              <span className={cn("block text-[10px] font-normal text-muted", requestId && "max-w-32 truncate")}>{requestId ? `${column.fragment ?? "Fragment not captured"}${column.tokenId == null ? "" : ` · ID ${column.tokenId}`}` : matrix.truncated ? `${column.scheduledCount} of ${scheduledCounts?.get(column.requestId) ?? (scheduledCounts ? 0 : "unknown")} scheduled shown` : `${column.scheduledCount} scheduled`}</span>
              {!requestId && column.positions.length > column.scheduledCount ? <span className="block text-[10px] font-normal text-muted">{column.positions.length - column.scheduledCount} prior positions</span> : null}
            </button>
          </th>)}</tr></thead>
          <tbody>{matrix.layers.map((layerIndex) => <tr key={layerIndex}>
            <th scope="row" className="px-1"><button type="button" className="min-h-6 border-0 bg-transparent px-1 py-0 font-mono text-[11px] text-muted" onClick={() => selectLayer(layerIndex)}>Layer {layerIndex + 1}{recurrentLayers.has(layerIndex) ? " · GDN" : ""}</button></th>
            {visible.map((column) => {
              if (recurrentLayers.has(layerIndex)) return <td key={column.key} className="border-l border-t border-line p-0.5"><button type="button" className="min-h-6 w-full border-0 bg-raised px-1 py-0 text-[10px] text-muted" aria-label={`${column.label}, layer ${layerIndex + 1}: Gated DeltaNet recurrent state; no attention KV block`} onClick={() => selectLayer(layerIndex)}>GDN</button></td>
              const observations = column.positions.map((position) => cells.get(`${layerIndex}:${column.requestId}:${position}`)).filter((cell) => cell != null)
              const statuses = [...new Set(observations.map((cell) => cell.state))]
              const status = statuses.length === 1 ? statuses[0] : statuses.includes("operation_active") ? "operation_active" : "unknown"
              const presentation = STATES[status ?? "unknown"]
              const counts = statuses.map((value) => `${observations.filter((cell) => cell.state === value).length} ${STATES[value].label.toLowerCase()}`).join(", ")
              const label = `${column.label}, layer ${layerIndex + 1}: ${counts || "unknown"}${matrix.truncated ? " (shown positions only)" : ""}`
              return <td key={column.key} className="border-l border-t border-line p-0.5"><button type="button" className={cn("flex w-full min-h-6 flex-wrap items-center justify-center gap-1 rounded-none border-0 px-1 py-0 text-[10px]", statuses.length === 1 && presentation.tone)} aria-label={label} title={`${label}. ${observations[0]?.reason ?? ""}`} onClick={() => requestId && column.positions.length === 1 && onParticipantSelect ? onParticipantSelect(layerIndex, column.requestId, column.positions[0]) : selectLayer(layerIndex)}>{statuses.length ? statuses.map((value) => <span key={value} className={cn("px-0.5", STATES[value].tone)}>{STATES[value].mark}{column.positions.length > 1 ? ` ${observations.filter((cell) => cell.state === value).length}` : ""}</span>) : "?"}</button></td>
            })}
          </tr>)}</tbody>
        </table>
      </div>
      {matrix.truncated ? <p className="m-0 text-xs text-muted">Showing {matrix.tokens.length} token positions in this bounded matrix; {totalScheduled == null ? "total scheduled participation is unknown" : `${totalScheduled} scheduled input positions in the ${requestId ? "request" : "iteration"}`}. Prior positions can also occupy the preview. Select a request or token to inspect a smaller scope.</p> : null}
    </> : <p className="m-0 text-xs text-muted">No participating positions are available. Step forward or choose a captured profiler batch.</p>}
  </section>
}
