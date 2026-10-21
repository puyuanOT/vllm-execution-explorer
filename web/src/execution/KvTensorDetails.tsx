import { useMemo, useState } from "react"
import { projectKvConfiguration, projectKvTensorPreview } from "../../../replay/src/kv.ts"
import type { ProfileKvObservations } from "../../../replay/src/profiling.ts"
import { projectRequestTokens } from "../../../replay/src/learning.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { requestLabel } from "../explore/model.ts"
import { EvidenceTag, Segmented } from "../ui/controls.tsx"
import { KvNumericalHeatmap } from "./KvNumericalHeatmap.tsx"
import { numericalCells, numericalFrames } from "./numericalHeatmap.ts"
import { namedKvDimensions } from "./kvDimensions.ts"
import type { Selection } from "./types.ts"

export interface TensorCaptureTarget { requestId: string; tokenPosition: number; layerIndex: number; kvHead: number }

export function KvTensorDetails({ state, manifest, profile, requestId, position, selection, labels, onSelect, onCaptureTensor }: {
  state: ReplayState; manifest: RecordingManifest | null; profile?: ProfileKvObservations | null
  requestId: string; position: number; labels?: ReadonlyMap<string, string>
  selection: Extract<Selection, { kind: "token" }>
  onSelect: (selection: Selection) => void; onCaptureTensor?: (target: TensorCaptureTarget) => void
}) {
  const selectedLayer = selection.kvLayerIndex ?? 0
  const selectedHead = selection.kvHead ?? 0
  const kind = selection.kvKind ?? "K"
  const [frameKey, setFrameKey] = useState("")
  const configuration = useMemo(() => projectKvConfiguration(state, manifest), [state, manifest])
  const token = useMemo(() => projectRequestTokens(state, requestId).find((token) => token.position === position), [state, requestId, position])
  const mappedLayers = configuration.layers.filter((layer) => token?.locations.some((location) => location.pool === layer.pool && location.cacheGroup === layer.cacheGroup))
  const layers = mappedLayers.length ? mappedLayers : configuration.layers
  const layer = layers.find((layer) => layer.layerIndex === selectedLayer) ?? layers[0]
  const kvHead = Math.min(selectedHead, Math.max(0, (layer?.numKvHeads ?? 1) - 1))
  const location = token?.locations.find((location) => location.pool === layer?.pool && location.cacheGroup === layer?.cacheGroup) ?? token?.locations[0]
  const currentBlock = location ? configuration.pools.flatMap((pool) => pool.blocks).find((block) => block.key === location.key) : null
  const sourceSpan = location ? currentBlock?.contents.find((span) => span.cacheGroup === location.cacheGroup && location.slot >= span.slotStart && location.slot < span.slotEnd) : null
  const writerRequestId = sourceSpan?.requestId ?? requestId
  const writerPosition = sourceSpan && location ? sourceSpan.tokenStart + location.slot - sourceSpan.slotStart : position
  const hasCurrentHolder = currentBlock?.bindings.some((binding) => binding.requestId === requestId && position >= binding.tokenStart && position < binding.tokenEnd) === true
  const preview = useMemo(() => projectKvTensorPreview(manifest, profile, { requestId: writerRequestId, tokenPosition: writerPosition, layerIndex: layer?.layerIndex ?? selectedLayer, kvHead, kind, pool: location?.pool, cacheGroup: location?.cacheGroup, blockId: location?.blockId, generation: location?.generation, slot: location?.slot }), [manifest, profile, writerRequestId, writerPosition, layer?.layerIndex, selectedLayer, kvHead, kind, location?.pool, location?.cacheGroup, location?.blockId, location?.generation, location?.slot])
  const headGridPreview = useMemo(() => projectKvTensorPreview(manifest, profile, { requestId: writerRequestId, tokenPosition: writerPosition, layerIndex: layer?.layerIndex ?? selectedLayer, kind, pool: location?.pool, cacheGroup: location?.cacheGroup, blockId: location?.blockId, generation: location?.generation, slot: location?.slot }), [manifest, profile, writerRequestId, writerPosition, layer?.layerIndex, selectedLayer, kind, location?.pool, location?.cacheGroup, location?.blockId, location?.generation, location?.slot])
  const frames = useMemo(() => numericalFrames(headGridPreview.snapshots), [headGridPreview.snapshots])
  const frame = frames.find((frame) => frame.key === frameKey) ?? frames.at(-1)
  const selectedSnapshot = frame?.snapshots.find((snapshot) => snapshot.kvHead === kvHead)
  const snapshot = selectedSnapshot ?? frame?.snapshots[0]
  const headValueCount = useMemo(() => [...numericalCells(frame, layer?.numKvHeads ?? 0, layer?.headDim ?? 0).cells.values()].filter((cell) => cell.head === kvHead).length, [frame, layer?.numKvHeads, layer?.headDim, kvHead])
  const partialHead = selectedSnapshot && layer?.headDim != null && headValueCount < layer.headDim
  const controlsAvailable = layers.length > 0
  const tensorDimensions = layer ? namedKvDimensions(layer) : null
  function selectSlice(patch: Partial<Extract<Selection, { kind: "token" }>>) {
    if (patch.kvLayerIndex != null || patch.kvKind != null) setFrameKey("")
    onSelect({ ...selection, kvLayerIndex: layer?.layerIndex ?? selectedLayer, kvHead, kvKind: kind, kvBlockKey: location?.key, ...patch })
  }
  return <section className="grid gap-3 border-t border-line pt-3" aria-label="Token KV tensor inspection">
    <h3 className="m-0 text-sm font-semibold text-ink">KV tensors for token position {position}</h3>
    <nav className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-muted" aria-label="KV tensor breadcrumb">
      <button type="button" className="min-h-6 border-0 bg-transparent p-0 text-accent" onClick={() => onSelect({ kind: "component", id: "kv-cache" })}>{layer?.device ?? "Device not captured"}</button>
      <span aria-hidden="true">/</span><span>{location?.pool ?? layer?.pool ?? "Pool not mapped"}{location ? ` · group ${location.cacheGroup}` : ""}</span>
      {location ? <><span aria-hidden="true">/</span><button type="button" className="min-h-6 border-0 bg-transparent p-0 text-accent" onClick={() => onSelect({ kind: "block", key: location.key })}>Block {location.blockId}</button><span aria-hidden="true">/</span><span>Slot {location.slot}</span></> : null}
      <span aria-hidden="true">/</span><span>Token position {position}</span><span aria-hidden="true">/</span><span>Layer {(layer?.layerIndex ?? selectedLayer) + 1}</span><span aria-hidden="true">/</span><span>KV head {kvHead}</span><span aria-hidden="true">/</span><span aria-current="page">{kind}</span>
    </nav>
    <div className="grid grid-cols-2 gap-2">
      <label className="grid gap-1 text-xs text-muted">Layer slice<select aria-label="KV tensor layer" value={layer?.layerIndex ?? selectedLayer} disabled={!controlsAvailable} onChange={(event) => selectSlice({ kvLayerIndex: Number(event.target.value), kvBlockKey: undefined })}>
        {layers.length ? layers.map((layer) => <option key={layer.layerIndex} value={layer.layerIndex}>Layer {layer.layerIndex + 1}</option>) : <option value={0}>Not captured</option>}
      </select></label>
      <label className="grid gap-1 text-xs text-muted">KV head index<select aria-label="KV tensor head" value={kvHead} disabled={!layer?.numKvHeads} onChange={(event) => selectSlice({ kvHead: Number(event.target.value) })}>
        {Array.from({ length: layer?.numKvHeads ?? 1 }, (_, head) => <option key={head} value={head}>{layer?.numKvHeads ? `Head ${head} (zero-based)` : "Not captured"}</option>)}
      </select></label>
    </div>
    <Segmented<"K" | "V"> label="KV tensor kind" value={kind} onChange={(value) => selectSlice({ kvKind: value })} options={[{ value: "K", label: "Keys (K)" }, { value: "V", label: "Values (V)" }]} />
    <dl className="facts text-sm">
      <div><dt>Request / token</dt><dd>{requestLabel(labels, requestId)} · {token?.tokenPiece ?? "Fragment not captured"} · ID {token?.tokenId ?? "not captured"}</dd></div>
      <div><dt>Current holder</dt><dd>{state.trust === "unknown" ? "Unknown after trace gap" : hasCurrentHolder ? requestLabel(labels, requestId) : "No mapped allocation"}</dd></div>
      <div><dt>Contents written by</dt><dd>{sourceSpan ? `${requestLabel(labels, sourceSpan.requestId)} · token position ${writerPosition}` : snapshot ? `${requestLabel(labels, snapshot.requestId)} · token position ${snapshot.tokenPosition} (historical snapshot)` : "Not observed"}</dd></div>
      <div><dt>Actual storage datatype</dt><dd>{layer?.dtype ?? "Not captured"}{snapshot?.quantization ? ` · ${snapshot.quantization.format} encoding` : ""}</dd></div>
      <div><dt>Actual quantization</dt><dd>{layer?.quantization?.mode ?? "Not captured"}</dd></div>
      <div><dt>Per-layer tensor axes</dt><dd>{tensorDimensions ?? "Axis names not established for this layout"}</dd></div>
      <div><dt>Features per KV head</dt><dd>{layer?.headDim == null ? "Not captured" : `${layer.headDim} per key vector or value vector`}</dd></div>
    </dl>
    <details><summary className="cursor-pointer text-xs text-muted">Raw tensor layout and allocation metadata</summary><dl className="facts mt-2 text-xs">
      <div><dt>Backing shape</dt><dd className="font-mono">{layer?.shape.length ? `[${layer.shape.join(", ")}]` : "Not captured"}</dd></div>
      <div><dt>Tensor layout</dt><dd>{layer?.layout ?? "Not captured"}</dd></div>
      <div><dt>Backing allocation ID</dt><dd className="break-all">{layer?.storageId ?? "Not captured"}</dd></div>
      <div><dt>Manager / kernel block size</dt><dd>{layer?.managerBlockSize ?? "?"} / {layer?.kernelBlockSize ?? "?"} slots</dd></div>
      <div><dt>Allocation generation</dt><dd>{location?.generation ?? "Not mapped"}</dd></div>
      <div><dt>{kind} quantization scale</dt><dd>{layer?.quantization?.scales[kind]?.status === "observed" ? `${layer.quantization.scales[kind].dtype} · [${layer.quantization.scales[kind].shape.join(", ")}] · ${layer.quantization.scales[kind].device}` : "Not captured"}</dd></div>
    </dl></details>
    {layer?.layoutStatus !== "qualified" ? <p className="m-0 text-pretty text-xs text-muted">{layer?.layoutReason ?? "Layer-specific tensor layout was not captured. Physical token occupancy does not establish a numerical tensor slice."}</p> : null}
    {!location ? <p className="m-0 text-pretty text-xs text-muted">This token has no mapped cache allocation at the host playhead. The structural fields describe configured layer storage; they do not establish that this token has been written.</p> : null}
    <p className="m-0 text-pretty text-xs text-muted">Layer/head slices share the block’s allocation. They do not increase the pool’s block count. These keys and values are attention state, separate from the token ID.</p>
    <div className="flex items-center justify-between gap-2"><h4 className="m-0 text-xs font-medium text-ink">Recorded K/V values</h4><EvidenceTag kind={headGridPreview.evidenceKind === "fixture" ? "synthetic" : snapshot ? "observed" : "unknown"}>{headGridPreview.evidenceKind === "fixture" ? "Synthetic" : snapshot ? "Captured snapshot" : "Not captured"}</EvidenceTag></div>
    <p className="m-0 text-pretty text-xs text-muted">{headGridPreview.reason}</p>
    {snapshot ? <>
      <label className="grid gap-1 text-xs text-muted">Snapshot boundary<select aria-label="KV numerical snapshot" value={frame?.key} onChange={(event) => setFrameKey(event.target.value)}>{frames.map((frame) => { const snapshot = frame.snapshots[0]; return <option key={frame.key} value={frame.key}>{snapshot.stepId} · {snapshot.operationId} · generation {snapshot.allocationGeneration ?? "?"}</option> })}</select></label>
      <dl className="facts text-xs"><div><dt>Snapshot original writer</dt><dd>{requestLabel(labels, snapshot.requestId)} · token position {snapshot.tokenPosition} · {snapshot.stepId}</dd></div><div><dt>Observed device / datatype</dt><dd>{snapshot.device} / {snapshot.dtype}</dd></div><div><dt>Snapshot location</dt><dd>{snapshot.pool ?? "Pool unknown"}, block {snapshot.blockId ?? "?"}, slot {snapshot.slot ?? "?"}, generation {snapshot.allocationGeneration ?? "?"}</dd></div><div><dt>Observation boundary</dt><dd>{snapshot.observationBoundary}</dd></div></dl>
      {snapshot.quantization ? <dl className="facts text-xs"><div><dt>Recorded FP8 encoding</dt><dd>{snapshot.quantization.format} · raw bytes retained</dd></div><div><dt>Captured {kind} scale</dt><dd>{snapshot.quantization.scale.value.toString()} · {snapshot.quantization.scale.mode.replace("_", " ")} · {snapshot.quantization.scale.dtype}</dd></div><div><dt>Reconstruction</dt><dd>Decoded FP8 × captured scale; quantization error remains.</dd></div></dl> : null}
      <p className="m-0 text-xs text-muted">Historical snapshot at its stated boundary. It does not claim that the current allocation still contains these values.</p>
    </> : null}
    <KvNumericalHeatmap key={`${writerRequestId}:${writerPosition}`} frame={frame} heads={layer?.numKvHeads ?? null} dimensions={layer?.headDim ?? null} selectedHead={kvHead} kind={kind} onHeadChange={(head) => selectSlice({ kvHead: head })} />
    {!selectedSnapshot || partialHead ? <>
      {!selectedSnapshot && snapshot ? <p className="m-0 text-pretty text-xs text-muted">The grid has retained values for other heads, but head {kvHead} was not captured at this boundary.</p> : null}
      <p className="m-0 text-pretty text-xs text-muted">{partialHead ? `${headValueCount} of ${layer?.headDim} ${kind} dimensions were retained for head ${kvHead}. Uncaptured dimensions require a new capture.` : preview.reason}</p>
      {writerRequestId !== requestId ? <p className="m-0 text-pretty text-xs text-muted">This holder reuses prefix state written by {requestLabel(labels, writerRequestId)}. It does not write that prefix again; a numerical rerun targets the original writer.</p> : null}
      <p className="m-0 text-pretty text-xs text-muted">Target a new diagnostic capture: {requestLabel(labels, writerRequestId)}, token position {writerPosition}, layer {(layer?.layerIndex ?? selectedLayer) + 1}, KV head {kvHead}. Choose a small dimension slice and byte limit before recording.</p>
      {onCaptureTensor ? <button type="button" className="justify-self-start text-xs" onClick={() => onCaptureTensor({ requestId: writerRequestId, tokenPosition: writerPosition, layerIndex: layer?.layerIndex ?? selectedLayer, kvHead })}>Prepare targeted rerun</button> : null}
    </> : null}
  </section>
}
