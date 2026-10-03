import { projectArchitecture } from "../../../replay/src/learning.ts"
import type { RecordingManifest, ReplayState, StepState } from "../../../replay/src/reduce.ts"
import type { ProfilePackedRow } from "../../../replay/src/profiling.ts"
import type { KvBlockProjection } from "../../../replay/src/learning.ts"

export { projectArchitecture }

export type ArchitectureFocus =
  | { kind: "model" }
  | { kind: "layer"; layerIndex: number }
  | { kind: "attention" | "ffn"; layerIndex: number }
  | { kind: "head"; layerIndex: number; headIndex: number; headKind: "query" | "kv" }

export type ModelArchitecture = ReturnType<typeof projectArchitecture>

export interface MeasuredSceneProfile {
  activeLayers: number[]
  packedRows: ProfilePackedRow[]
  stepId: string | null
  activeModules?: Array<{ layerIndex: number; module: "attention" | "ffn" }>
  evidenceKind?: "gpu" | "fixture" | "unknown"
}

export function architectureLabel(manifest?: RecordingManifest | null): string {
  const model = manifest?.model
  if (model && typeof model === "object" && "id" in model && typeof model.id === "string") {
    return model.id.split("/").at(-1) ?? model.id
  }
  return "Recorded model"
}

export function architectureFocusLabel(focus: ArchitectureFocus | null): string {
  if (!focus || focus.kind === "model") return "Model architecture"
  const layer = `Layer ${focus.layerIndex + 1}`
  if (focus.kind === "layer") return layer
  if (focus.kind === "attention") return `${layer} / Attention`
  if (focus.kind === "ffn") return `${layer} / Feed-forward`
  if (focus.kind === "head") return `${layer} / ${focus.headKind === "query" ? "Query" : "KV"} head ${focus.headIndex + 1}`
  return layer
}

/** A query head shares one KV head in the supported contiguous GQA layout. */
export function kvHeadForQuery(queryHead: number, queryHeads: number, kvHeads: number): number | null {
  if (!Number.isInteger(queryHead) || queryHead < 0 || queryHead >= queryHeads
    || !Number.isInteger(queryHeads) || !Number.isInteger(kvHeads)
    || kvHeads <= 0 || queryHeads <= 0 || queryHeads % kvHeads !== 0) return null
  return Math.floor(queryHead / (queryHeads / kvHeads))
}

export function activeTeachingStep(state: ReplayState, stepId?: string | null): StepState | null {
  return (stepId ? state.steps.find((step) => step.id === stepId) : state.steps.at(-1)) ?? null
}

export interface ScheduledTokenPosition {
  requestId: string
  position: number
  kind: "prefill" | "decode" | "unclassified"
  row?: number
}

/** The geometry includes scheduled inputs only, never the next sampled output. */
export function scheduledTokenPositions(step: Pick<StepState, "requests">): ScheduledTokenPosition[] {
  return step.requests.flatMap((request) => Array.from({ length: request.rangeLength }, (_, offset) => {
    const position = request.rangeStart + offset
    return { requestId: request.requestId, position, kind: position < request.numPromptTokens ? "prefill" as const : "decode" as const }
  }))
}

/** Native profiler row identity does not depend on the host replay cursor. */
export function observedProfileTokenPositions(rows: ProfilePackedRow[]): ScheduledTokenPosition[] {
  return rows.filter((row) => !row.isPadding && row.requestId != null && row.tokenPosition != null)
    .sort((a, b) => a.row - b.row)
    .map((row) => ({ requestId: row.requestId!, position: row.tokenPosition!, row: row.row, kind: "unclassified" as const }))
}

/** Reused physical contents keep source provenance while selection follows the current request binding. */
export function mappedBlockSlot(block: Pick<KvBlockProjection, "contents" | "bindings">, blockSize: number, slot: number, activeRequestId?: string | null) {
  const span = block.contents.find((item) => slot >= item.slotStart && slot < item.slotEnd)
  const coversSlot = (item: KvBlockProjection["bindings"][number]) => slot >= item.tokenStart % blockSize
    && slot < Math.min(blockSize, item.tokenStart % blockSize + item.tokenEnd - item.tokenStart)
  const binding = block.bindings.find((item) => item.requestId === activeRequestId && coversSlot(item)) ?? block.bindings.find(coversSlot)
  return {
    span, binding, sourceRequestId: span?.requestId ?? null,
    requestId: binding?.requestId ?? span?.requestId ?? null,
    position: binding ? binding.tokenStart + slot - binding.tokenStart % blockSize : span ? span.tokenStart + slot - span.slotStart : null,
    tokenId: span?.tokenIds?.[slot - span.slotStart] ?? null,
    tokenPiece: span?.tokenPieces?.[slot - span.slotStart] ?? null,
  }
}

/** Shared by all objects so modifier-based camera gestures cannot select data. */
export function sceneClick(event: { delta: number; nativeEvent: { button: number; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean } }): boolean {
  return event.delta <= 4 && event.nativeEvent.button === 0
    && !event.nativeEvent.ctrlKey && !event.nativeEvent.metaKey && !event.nativeEvent.shiftKey
}

/** This is a logical display clock; it is deliberately unrelated to GPU duration. */
export function teachingLayer(progress: number | undefined, layerCount: number): number | null {
  if (progress == null || !Number.isFinite(progress) || !Number.isInteger(layerCount) || layerCount <= 0) return null
  return Math.min(layerCount - 1, Math.floor(Math.max(0, Math.min(1, progress)) * layerCount))
}

export interface PoolGridLayout {
  columns: number
  rows: number
  depths: number
  capacity: number
  spacing: number
}

/** Every physical ID has one stable grid cell; zoom never changes its identity. */
export function poolGridLayout(capacity: number, span = 7): PoolGridLayout {
  const count = Number.isSafeInteger(capacity) && capacity > 0 ? capacity : 0
  const columns = Math.max(1, Math.ceil(Math.cbrt(count)))
  const rows = Math.max(1, Math.ceil(Math.sqrt(Math.ceil(count / columns))))
  const depths = Math.max(1, Math.ceil(count / (columns * rows)))
  return { columns, rows, depths, capacity: count, spacing: span / Math.max(columns, rows, depths) }
}

export function poolCellPosition(index: number, layout: PoolGridLayout): [number, number, number] {
  const x = index % layout.columns
  const y = Math.floor(index / layout.columns) % layout.rows
  const z = Math.floor(index / (layout.columns * layout.rows))
  return [
    (x - (layout.columns - 1) / 2) * layout.spacing,
    ((layout.rows - 1) / 2 - y) * layout.spacing,
    ((layout.depths - 1) / 2 - z) * layout.spacing,
  ]
}
