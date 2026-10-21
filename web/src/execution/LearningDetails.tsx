import { useMemo, useState } from "react"
import { projectArchitecture, projectKvCache, projectRequestTokens, projectScheduler } from "../../../replay/src/learning.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { kvHeadForQuery, mappedBlockSlot } from "../architecture/model.ts"
import { requestLabel } from "../explore/model.ts"
import type { Selection } from "./types.ts"
import type { ProfileKvObservations } from "../../../replay/src/profiling.ts"
import { KvTensorDetails, type TensorCaptureTarget } from "./KvTensorDetails.tsx"
import { projectKvConfiguration } from "../../../replay/src/kv.ts"
import { physicalBlockState, physicalBlockStateLabel } from "../architecture/model.ts"
import { tokenKvLabel, tokenSlotOwnership } from "./tokenSemantics.ts"

export function LearningDetails({ selection, state, manifest, labels, measuredProfile, onSelect, onCaptureTensor }: {
  selection: Selection | null; state: ReplayState | null; manifest: RecordingManifest | null
  labels?: ReadonlyMap<string, string>
  measuredProfile?: ProfileKvObservations | null
  onCaptureTensor?: (target: TensorCaptureTarget) => void
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
    const recurrent = architecture.layerTypes[layerIndex] === "linear_attention"
    const step = state.steps.at(-1)
    const queryHead = selection.kind === "head" && selection.headKind === "query" ? selection.headIndex : null
    const sharedKv = queryHead != null && architecture.queryHeads && architecture.kvHeads
      ? kvHeadForQuery(queryHead, architecture.queryHeads, architecture.kvHeads) : null
    return <section className="mt-4 space-y-3">
      <h3>Layer {layerIndex + 1}{selection.kind === "module" ? ` / ${selection.module === "attention" ? recurrent ? "Gated DeltaNet" : "Attention" : "Feed-forward"}` : selection.kind === "head" ? ` / ${selection.headKind === "query" ? "Query" : "KV"} head ${selection.headIndex + 1}` : recurrent ? " / Gated DeltaNet" : ""}</h3>
      <p className="text-sm text-muted">The layer matrix identifies the participating requests and tokens. GPU cache-write activity appears only when a batch operation has correlated profiler evidence. Teaching progression remains explanatory; individual token and head durations are unknown.</p>
      <dl className="facts">
        <div><dt>Hidden dimension</dt><dd className="tabular-nums">{architecture.hiddenSize ?? "Unknown"}</dd></div>
        <div><dt>Feed-forward dimension</dt><dd className="tabular-nums">{architecture.intermediateSize ?? "Unknown"}</dd></div>
        <div><dt>{recurrent ? "Recurrent key / value heads" : "Query / KV heads"}</dt><dd className="tabular-nums">{recurrent ? `${architecture.recurrent?.keyHeads ?? "Unknown"} / ${architecture.recurrent?.valueHeads ?? "Unknown"}` : `${architecture.queryHeads ?? "Unknown"} / ${architecture.kvHeads ?? "Unknown"}`}</dd></div>
        <div><dt>{recurrent ? "Recurrent key / value head dimension" : "Head dimension"}</dt><dd className="tabular-nums">{recurrent ? `${architecture.recurrent?.keyHeadDim ?? "Unknown"} / ${architecture.recurrent?.valueHeadDim ?? "Unknown"}` : architecture.headDim ?? "Unknown"}</dd></div>
        {recurrent ? <div><dt>State storage</dt><dd>Convolution history and recurrent matrix; separate from attention KV token blocks.</dd></div> : null}
        {sharedKv != null ? <div><dt>Shared keys and values</dt><dd>KV head {sharedKv + 1}</dd></div> : null}
      </dl>
      <details><summary className="cursor-pointer text-xs text-muted">Explains {selection.kind === "module" && selection.module === "ffn" ? "the feed-forward network" : recurrent ? "Gated DeltaNet state" : "attention and prefix reuse"}</summary><p className="mt-2 text-pretty text-sm">{selection.kind === "module" && selection.module === "ffn" ? `The gated feed-forward network projects the residual stream into gate and up branches, applies ${architecture.activation ?? "the configured activation"} gating, then projects back to the hidden dimension.` : recurrent ? "Gated DeltaNet updates a recurrent summary and convolution history. It does not retain one full attention K/V pair per past token. Attention layers in this hybrid model use a separate KV cache." : "Queries for the scheduled positions attend to keys and values at allowed earlier positions. Matching cached prefixes supply previously computed keys and values; the remaining scheduled positions compute new ones."}</p></details>
      <h3>Host replay batch {step?.id ?? "unavailable"}</h3>
      {step?.requests.map((request) => <button className="block w-full text-left" key={request.requestId} type="button" onClick={() => onSelect({ kind: "request", id: request.requestId })}>{nameOf(request.requestId)}: positions {request.rangeStart}–{request.rangeStart + request.rangeLength - 1} · {request.promptTokens} prefill / {request.generationTokens} decode</button>)}
    </section>
  }
  if (selection.kind === "block") {
    const block = cache?.pools.flatMap((pool) => pool.blocks).find((candidate) => candidate.key === selection.key || `${candidate.pool}:${candidate.blockId}` === selection.key)
    const pool = cache?.pools.find((candidate) => candidate.pool === block?.pool)
    if (!block || !pool) return <p className="mt-4 text-sm text-muted">Full physical block contents were not captured in this recording.</p>
    const configuration = projectKvConfiguration(state, manifest)
    const configuredPool = configuration.pools.find((candidate) => candidate.pool === pool.pool)
    const holders = [...new Set(block.bindings.map((binding) => nameOf(binding.requestId)))]
    const writers = [...new Set(block.contents.map((span) => nameOf(span.requestId)))]
    return <section className="mt-4 space-y-3">
      <h3>{configuredPool?.devices.join(", ") || "Device not captured"} / {block.pool} / Block {block.blockId}</h3>
      <dl className="facts">
        <div><dt>Allocation state</dt><dd>{physicalBlockStateLabel(physicalBlockState(block))}</dd></div>
        <div><dt>Actual KV datatype</dt><dd>{configuredPool?.dtypes.join(", ") || "Not captured"}</dd></div>
        <div><dt>Current request holders</dt><dd>{block.trust === "unknown" ? "Unknown" : holders.join(", ") || (block.refCount > 0 && !block.isNull ? "Binding identities not captured" : "None")}</dd></div>
        <div><dt>Original KV writers</dt><dd>{block.trust === "unknown" ? "Unknown" : writers.join(", ") || (block.isNull ? "None — null placeholder" : "No recorded contents")}</dd></div>
        <div><dt>Prefix cached</dt><dd>{block.trust === "unknown" ? "Unknown" : block.hasHash ? "Yes" : "No"}</dd></div>
        <div><dt>Available for allocation</dt><dd>{block.trust === "unknown" ? "Unknown" : block.isNull ? "Reserved null block" : block.refCount === 0 ? "Yes" : "No"}</dd></div>
        <div><dt>Recorded computed slots</dt><dd>{block.isNull ? "Not applicable" : block.validTokenCount == null ? "Unknown" : `${block.validTokenCount} / ${pool.blockSize ?? "group-specific capacity"}`}</dd></div>
      </dl>
      {block.isNull ? <p className="text-sm text-muted">Reserved placeholder; this cell cannot hold request KV contents.</p> : null}
      <h3>Token positions within the block</h3>
      {block.isNull ? <p className="text-sm text-muted">No token slots: this is the reserved null placeholder.</p> : block.tokenOccupancyStatus !== "observed" || pool.groups.length > 1 || pool.blockSize == null ? <p className="text-sm text-muted">{block.tokenOccupancyReason ?? "Multiple cache groups have separate slot contents; a combined token-slot view is unavailable."}</p> : <ol className="grid list-none grid-cols-4 gap-2 p-0" aria-label="KV block token positions">
        {Array.from({ length: pool.blockSize ?? 0 }, (_, slot) => {
          const { span, binding, position, requestId, sourceRequestId, tokenId, tokenPiece } = mappedBlockSlot(block, pool.blockSize!, slot)
          const known = block.trust === "known"
          return <li key={slot}><button className="w-full text-left" type="button" disabled={position == null || requestId == null || !known || block.isNull} aria-label={`Slot ${slot}: ${!known ? "unknown" : block.isNull ? "reserved" : span ? `computed token ${tokenId ?? "ID unavailable"}, ${requestId ? nameOf(requestId) : "unknown request"} position ${position}; content source ${sourceRequestId ? nameOf(sourceRequestId) : "unknown"}` : binding ? `reserved for ${requestId ? nameOf(requestId) : "unknown request"} position ${position}, computation unobserved` : "contents unobserved"}`} onClick={() => { if (requestId && position != null) onSelect({ kind: "token", requestId, position }) }}><span className="block text-xs text-muted">Slot {slot}</span><span className="block truncate text-sm tabular-nums">{!known ? "Unknown" : tokenPiece ?? (span ? "Computed" : binding ? "Reserved" : "Unwritten")}</span>{position != null ? <span className="block text-[10px] text-muted">Position {position}</span> : null}{tokenId != null ? <span className="block text-[10px] tabular-nums text-muted">ID {tokenId}</span> : null}</button></li>
        })}
      </ol>}
      <h3>Current request bindings</h3>
      {block.trust === "known" ? block.bindings.map((binding) => <button className="block w-full text-left" type="button" key={`${binding.requestId}-${binding.cacheGroup}-${binding.tokenStart}`} onClick={() => onSelect({ kind: "request", id: binding.requestId })}>{nameOf(binding.requestId)} · cache group {binding.cacheGroup} · positions {binding.tokenStart}–{binding.tokenEnd - 1}</button>) : null}
      {block.trust === "unknown" || !block.bindings.length ? <p className="text-xs text-muted">{block.trust === "unknown" ? "Current bindings are unknown after a trace gap." : block.refCount > 0 && !block.isNull ? "Holder identities have not been captured." : "No current request binding."}</p> : null}
      {block.refCount === 0 && block.hasHash && block.trust === "known" ? <p className="text-sm text-muted">The prefix remains cached after release. Another request can reuse it; reassignment invalidates the old token mapping.</p> : null}
      {block.refCount === 0 && !block.isNull && block.trust === "known" ? <p className="text-xs text-muted">Available for allocation can still contain old recorded tensor contents. Allocation state and memory erasure are different.</p> : null}
      <details><summary className="cursor-pointer text-xs text-muted">Explains ownership, reservations and KV validity</summary><p className="mt-2 text-pretty text-xs text-muted">A holder references this allocation; a KV writer identifies the request that originally computed its contents. They can differ when a prefix is shared. A bound but unwritten slot is reserved, while a recorded write establishes computed contents. Zero references release ownership without implying that bytes were erased.</p></details>
      <details><summary className="cursor-pointer text-xs text-muted">Raw allocation metadata</summary><dl className="facts mt-2"><div><dt>Allocation generation</dt><dd className="tabular-nums">{block.generation}</dd></div><div><dt>Reference count</dt><dd className="tabular-nums">{block.trust === "unknown" ? "Unknown" : block.refCount}</dd></div><div><dt>Associated layers</dt><dd>{configuredPool?.layerIndices.length ? configuredPool.layerIndices.map((index) => index + 1).join(", ") : "Not captured"}</dd></div><div><dt>Cache groups</dt><dd>{pool.groups.map((group) => group.cacheGroup).join(", ") || "Not captured"}</dd></div></dl></details>
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
    const ownership = selectedToken ? tokenSlotOwnership(selectedToken, cache) : null
    return <section className="mt-4 space-y-3">
      <h3>{selectedToken ? `Token at position ${selectedToken.position}` : "Request tokens"}</h3>
      {selectedToken ? <><dl className="facts"><div><dt>Token ID</dt><dd className="tabular-nums">{selectedToken.tokenId ?? "Content omitted"}</dd></div><div><dt>Decoded fragment</dt><dd>{selectedToken.tokenPiece ?? "Unavailable"}</dd></div><div><dt>Current KV mapping</dt><dd>{tokenKvLabel(selectedToken)}</dd></div><div><dt>Previously computed</dt><dd>{selectedToken.computedEver == null ? "Unknown" : selectedToken.computedEver ? "Yes" : "No"}</dd></div><div><dt>Current slot holders</dt><dd>{!ownership?.known ? "Unknown / mapping not captured" : ownership.holders.map(nameOf).join(", ") || "None observed"}</dd></div><div><dt>Original KV writers</dt><dd>{state.trust === "unknown" ? "Unknown" : ownership?.writers.map(nameOf).join(", ") || (selectedToken.computedEver == null ? "Writer mapping not captured" : selectedToken.computedEver ? "No retained writer mapping" : "No recorded write")}</dd></div></dl>{selectedToken.locations.map((location) => <button type="button" className="block w-full text-left" key={`${location.key}-${location.cacheGroup}`} onClick={() => onSelect({ kind: "block", key: location.key })}>{location.pool} block {location.blockId} · slot {location.slot} · group {location.cacheGroup}</button>)}</> : null}
      {selectedToken?.kvReason ? <p className="text-sm text-muted">{selectedToken.kvReason}</p> : null}
      {selectedToken && selectedToken.cacheGroups.length > 1 ? <ul className="list-none space-y-1 p-0 text-xs text-muted">{selectedToken.cacheGroups.map((group) => <li key={group.cacheGroup}>Group {group.cacheGroup}: {group.kvState.replaceAll("_", " ")}{group.reason ? ` · ${group.reason}` : ""}</li>)}</ul> : null}
      {selectedToken?.kvState === "reserved" ? <p className="text-pretty text-xs text-muted">A slot is reserved for this position, but its KV write has not been observed yet.</p> : selectedToken?.kind === "output" && selectedToken.kvState === "not_computed" && selectedToken.computedEver === false ? <p className="text-pretty text-xs text-muted">This sampled output has not yet been processed as model input. Sampling alone does not populate its KV slot.</p> : null}
      {selectedToken && selection.kind === "token" ? <KvTensorDetails key={`${requestId}:${selectedToken.position}`} state={state} manifest={manifest} profile={measuredProfile} requestId={requestId} position={selectedToken.position} selection={selection} labels={labels} onSelect={onSelect} onCaptureTensor={onCaptureTensor} /> : null}
      <details><summary className="cursor-pointer text-xs text-muted">Explains token processing and KV ownership</summary><p className="mt-2 text-pretty text-xs text-muted">A sampled output token becomes input to a later decode iteration. Reservation assigns a slot; a recorded KV write establishes computed contents. Prefix reuse can bind the same contents to another holder while preserving the original writer. Released or reassigned storage changes ownership independently of whether this token was computed earlier.</p></details>
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
    <ol className="grid list-none grid-cols-4 gap-2 p-0" aria-label="Request token positions">{visible.map((token) => <li key={token.position}><button type="button" className="w-full text-left" aria-label={`Position ${token.position}, ${token.kind}, token ${token.tokenId ?? "unavailable"}, KV ${tokenKvLabel(token)}`} onClick={() => onSelect({ kind: "token", requestId, position: token.position })}><span className="block text-xs text-muted">{token.position}</span><span className="block truncate text-sm">{token.tokenPiece ?? token.tokenId ?? "—"}</span><span className="block truncate text-xs text-muted">{tokenKvLabel(token, true)}</span></button></li>)}</ol>
  </div>
}
