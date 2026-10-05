import { useMemo, useState } from "react"
import { projectArchitecture, projectKvCache, projectRequestTokens, projectScheduler } from "../../../replay/src/learning.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { kvHeadForQuery, mappedBlockSlot } from "../architecture/model.ts"
import { requestLabel } from "../explore/model.ts"
import type { Selection } from "./types.ts"

export function LearningDetails({ selection, state, manifest, labels, onSelect }: {
  selection: Selection | null; state: ReplayState | null; manifest: RecordingManifest | null
  labels?: ReadonlyMap<string, string>
  onSelect: (selection: Selection) => void
}) {
  const nameOf = (requestId: string) => requestLabel(labels, requestId)
  const architecture = useMemo(() => projectArchitecture(manifest), [manifest])
  const cache = useMemo(() => state ? projectKvCache(state, manifest) : null, [state, manifest])
  const requestId = selection?.kind === "request" ? selection.id : selection?.kind === "token" ? selection.requestId : null
  const tokens = useMemo(() => state && requestId ? projectRequestTokens(state, requestId) : [], [state, requestId])
  if (!selection || !state) return null
  if (selection.kind === "layer" || selection.kind === "module" || selection.kind === "head") {
    const layerIndex = selection.layerIndex
    const step = state.steps.at(-1)
    const queryHead = selection.kind === "head" && selection.headKind === "query" ? selection.headIndex : null
    const sharedKv = queryHead != null && architecture.queryHeads && architecture.kvHeads
      ? kvHeadForQuery(queryHead, architecture.queryHeads, architecture.kvHeads) : null
    return <section className="mt-4 space-y-3">
      <h3>Layer {layerIndex + 1}{selection.kind === "module" ? ` / ${selection.module === "attention" ? "Attention" : "Feed-forward"}` : selection.kind === "head" ? ` / ${selection.headKind === "query" ? "Query" : "KV"} head ${selection.headIndex + 1}` : ""}</h3>
      <p className="text-sm text-muted">This structure view explains the host replay batch. Teaching progression is explanatory; the separate profiler panel shows captured batch and layer measurements when available. Individual token and head durations remain unknown.</p>
      <dl className="facts">
        <div><dt>Hidden dimension</dt><dd className="tabular-nums">{architecture.hiddenSize ?? "Unknown"}</dd></div>
        <div><dt>Feed-forward dimension</dt><dd className="tabular-nums">{architecture.intermediateSize ?? "Unknown"}</dd></div>
        <div><dt>Query / KV heads</dt><dd className="tabular-nums">{architecture.queryHeads ?? "Unknown"} / {architecture.kvHeads ?? "Unknown"}</dd></div>
        <div><dt>Head dimension</dt><dd className="tabular-nums">{architecture.headDim ?? "Unknown"}</dd></div>
        {sharedKv != null ? <div><dt>Shared keys and values</dt><dd>KV head {sharedKv + 1}</dd></div> : null}
      </dl>
      <p className="text-sm">{selection.kind === "module" && selection.module === "ffn" ? architecture.modelType === "qwen3" ? `The gated feed-forward network projects the residual stream into gate and up branches, applies ${architecture.activation ?? "the configured activation"} gating, then projects back to the hidden dimension.` : `The feed-forward network expands the hidden representation, applies ${architecture.activation ?? "the configured activation"}, then projects back to the hidden dimension.` : "Queries for the scheduled positions attend to keys and values at allowed earlier positions. Matching cached prefixes supply previously computed keys and values; the remaining scheduled positions compute new ones."}</p>
      <h3>Host replay batch {step?.id ?? "unavailable"}</h3>
      {step?.requests.map((request) => <button className="block w-full text-left" key={request.requestId} type="button" onClick={() => onSelect({ kind: "request", id: request.requestId })}>{nameOf(request.requestId)}: positions {request.rangeStart}–{request.rangeStart + request.rangeLength - 1} · {request.promptTokens} prefill / {request.generationTokens} decode</button>)}
    </section>
  }
  if (selection.kind === "block") {
    const block = cache?.pools.flatMap((pool) => pool.blocks).find((candidate) => candidate.key === selection.key || `${candidate.pool}:${candidate.blockId}` === selection.key)
    const pool = cache?.pools.find((candidate) => candidate.pool === block?.pool)
    if (!block || !pool) return <p className="mt-4 text-sm text-muted">Full physical block contents were not captured in this recording.</p>
    return <section className="mt-4 space-y-3">
      <h3>{block.pool} · Physical block {block.blockId}</h3>
      <dl className="facts">
        <div><dt>Allocation generation</dt><dd>{block.generation}</dd></div>
        <div><dt>References</dt><dd>{block.trust === "unknown" ? "Unknown" : block.refCount}</dd></div>
        <div><dt>Prefix cached</dt><dd>{block.trust === "unknown" ? "Unknown" : block.hasHash ? "Yes" : "No"}</dd></div>
        <div><dt>Available for allocation</dt><dd>{block.trust === "unknown" ? "Unknown" : block.isNull ? "Reserved null block" : block.refCount === 0 ? "Yes" : "No"}</dd></div>
        <div><dt>Computed token positions</dt><dd>{block.validTokenCount == null ? "Unknown" : `${block.validTokenCount} / ${pool.blockSize ?? "group-specific capacity"}`}</dd></div>
      </dl>
      {block.isNull ? <p className="text-sm text-muted">Reserved placeholder; this cell cannot hold request KV contents.</p> : null}
      <h3>Token positions within the block</h3>
      {block.tokenOccupancyStatus !== "observed" || pool.groups.length > 1 || pool.blockSize == null ? <p className="text-sm text-muted">{block.tokenOccupancyReason ?? "Multiple cache groups have separate slot contents; a combined token-slot view is unavailable."}</p> : <ol className="grid list-none grid-cols-4 gap-2 p-0" aria-label="KV block token positions">
        {Array.from({ length: pool.blockSize ?? 0 }, (_, slot) => {
          const { span, binding, position, requestId, sourceRequestId, tokenId, tokenPiece } = mappedBlockSlot(block, pool.blockSize!, slot)
          const known = block.trust === "known"
          return <li key={slot}><button className="w-full text-left" type="button" disabled={position == null || requestId == null || !known || block.isNull} aria-label={`Slot ${slot}: ${!known ? "unknown" : block.isNull ? "reserved" : span ? `computed token ${tokenId ?? "ID unavailable"}, ${requestId ? nameOf(requestId) : "unknown request"} position ${position}; content source ${sourceRequestId ? nameOf(sourceRequestId) : "unknown"}` : binding ? `reserved for ${requestId ? nameOf(requestId) : "unknown request"} position ${position}, computation unobserved` : "contents unobserved"}`} onClick={() => { if (requestId && position != null) onSelect({ kind: "token", requestId, position }) }}><span className="block text-xs text-muted">Slot {slot}</span><span className="block truncate text-sm tabular-nums">{!known ? "Unknown" : tokenPiece ?? tokenId ?? (span ? "Computed" : binding ? "Reserved" : "—")}</span></button></li>
        })}
      </ol>}
      <h3>Request and prefix bindings</h3>
      {block.bindings.map((binding) => <button className="block w-full text-left" type="button" key={`${binding.requestId}-${binding.cacheGroup}-${binding.tokenStart}`} onClick={() => onSelect({ kind: "request", id: binding.requestId })}>{nameOf(binding.requestId)} · cache group {binding.cacheGroup} · positions {binding.tokenStart}–{binding.tokenEnd - 1}</button>)}
      {block.refCount === 0 && block.hasHash && block.trust === "known" ? <p className="text-sm text-muted">The prefix remains cached after release. Another request can reuse it; reassignment removes the old occupant.</p> : null}
    </section>
  }
  if (selection.kind === "component" && selection.id === "scheduler" || selection.kind === "step") {
    const scheduler = projectScheduler(state, selection.kind === "step" ? selection.id : null)
    return <section className="mt-4 space-y-3">
      <h3>Scheduler queues · {scheduler.stepId ?? "no step"}</h3>
      <p className="text-sm text-muted">{scheduler.reason ?? `Recorded policy: ${scheduler.policy ?? "unknown"}. Queue order is observed; a skipped request's decision reason remains unknown unless captured.`}</p>
      {(["before", "after"] as const).map((boundary) => scheduler[boundary] ? <div key={boundary}><h3>{boundary === "before" ? "Before scheduling" : "After scheduling"}</h3>{(["running", "waiting", "skippedWaiting"] as const).map((queue) => <div key={queue}><p className="text-xs text-muted">{queue === "skippedWaiting" ? "Skipped waiting" : queue}</p><ol className="list-decimal space-y-1 pl-5">{scheduler[boundary]![queue].map((entry) => <li key={entry.requestId}><button type="button" onClick={() => onSelect({ kind: "request", id: entry.requestId })}>{nameOf(entry.requestId)} · priority {entry.priority ?? "unknown"} · scheduler cursor {entry.computedTokens ?? "unknown"}</button></li>)}</ol>{scheduler[boundary]![queue].length === 0 ? <p className="text-xs text-muted">Empty</p> : null}</div>)}</div> : null)}
    </section>
  }
  if (requestId) {
    const selectedToken = selection.kind === "token" ? tokens.find((token) => token.position === selection.position) : null
    return <section className="mt-4 space-y-3">
      <h3>{selectedToken ? `Token at position ${selectedToken.position}` : "Request tokens"}</h3>
      {selectedToken ? <><dl className="facts"><div><dt>Token ID</dt><dd>{selectedToken.tokenId ?? "Content omitted"}</dd></div><div><dt>Decoded fragment</dt><dd>{selectedToken.tokenPiece ?? "Unavailable"}</dd></div><div><dt>KV state now</dt><dd>{selectedToken.kvState.replaceAll("_", " ")}</dd></div><div><dt>Previously computed</dt><dd>{selectedToken.computedEver == null ? "Unknown" : selectedToken.computedEver ? "Yes" : "No"}</dd></div></dl>{selectedToken.locations.map((location) => <button type="button" className="block w-full text-left" key={`${location.key}-${location.cacheGroup}`} onClick={() => onSelect({ kind: "block", key: location.key })}>{location.pool} block {location.blockId} · generation {location.generation} · group {location.cacheGroup} · slot {location.slot}</button>)}</> : null}
      {selectedToken?.kvReason ? <p className="text-sm text-muted">{selectedToken.kvReason}</p> : null}
      {selectedToken && selectedToken.cacheGroups.length > 1 ? <ul className="list-none space-y-1 p-0 text-xs text-muted">{selectedToken.cacheGroups.map((group) => <li key={group.cacheGroup}>Group {group.cacheGroup}: {group.kvState.replaceAll("_", " ")}{group.reason ? ` · ${group.reason}` : ""}</li>)}</ul> : null}
      <p className="text-xs text-muted">A sampled output token may not yet have been processed by the model. Reserved slots are distinct from computed KV data.</p>
      <TokenList key={requestId} tokens={tokens} requestId={requestId} onSelect={onSelect} />
    </section>
  }
  return null
}

function TokenList({ tokens, requestId, onSelect }: { tokens: ReturnType<typeof projectRequestTokens>; requestId: string; onSelect: (selection: Selection) => void }) {
  const [page, setPage] = useState(0)
  const maxPage = Math.max(0, Math.ceil(tokens.length / 64) - 1)
  const currentPage = Math.min(page, maxPage)
  const visible = tokens.slice(currentPage * 64, currentPage * 64 + 64)
  return <div className="space-y-2">
    <div className="flex items-center justify-between gap-2"><button type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous tokens</button><span className="text-xs tabular-nums">{tokens.length ? `${currentPage * 64}–${currentPage * 64 + visible.length - 1} of ${tokens.length}` : "Tokens unavailable"}</span><button type="button" disabled={currentPage >= maxPage} onClick={() => setPage(currentPage + 1)}>Next tokens</button></div>
    <ol className="grid list-none grid-cols-4 gap-2 p-0" aria-label="Request token positions">{visible.map((token) => <li key={token.position}><button type="button" className="w-full text-left" aria-label={`Position ${token.position}, ${token.kind}, token ${token.tokenId ?? "unavailable"}, KV ${token.kvState}`} onClick={() => onSelect({ kind: "token", requestId, position: token.position })}><span className="block text-xs text-muted">{token.position}</span><span className="block truncate text-sm">{token.tokenPiece ?? token.tokenId ?? "—"}</span><span className="block truncate text-xs text-muted">{token.kvState.replaceAll("_", " ")}</span></button></li>)}</ol>
  </div>
}
