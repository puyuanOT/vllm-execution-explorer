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
  activeModules?: Array<{ layerIndex: number; module: "attention" | "ffn" | "recurrent" }>
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

export function iterationDisplayLabel(stepId: string): string {
  const match = /^step-(\d+)$/.exec(stepId)
  return match ? `Iteration ${Number(match[1])}` : `Iteration · ${stepId}`
}

/** Only the selected clock's batch determines participation; host replay never fills missing profiler rows. */
export function modelIterationSummary(state: ReplayState, profile?: MeasuredSceneProfile, stepId?: string | null) {
  if (profile) {
    const tokens = observedProfileTokenPositions(profile.packedRows)
    const rows = new Map<string, number[]>()
    for (const token of tokens) {
      const positions = rows.get(token.requestId) ?? []
      positions.push(token.position)
      rows.set(token.requestId, positions)
    }
    return { label: profile.stepId ? iterationDisplayLabel(profile.stepId) : "Profiler batch", positions: tokens.length,
      participants: [...rows].map(([requestId, positions]) => ({ requestId, positions: positions.length, ranges: tokenPositionRanges(positions), prefill: null, decode: null })),
      clock: "profiler" as const, known: tokens.length > 0 }
  }
  const step = activeTeachingStep(state, stepId)
  return { label: step ? iterationDisplayLabel(step.id) : "No iteration yet", positions: step?.totalScheduledTokens ?? 0,
    participants: state.trust === "known" ? (step?.requests ?? []).map((request) => ({ requestId: request.requestId,
      positions: request.rangeLength, ranges: request.rangeLength ? [{ start: request.rangeStart, end: request.rangeStart + request.rangeLength - 1 }] : [],
      prefill: request.promptTokens as number | null, decode: request.generationTokens as number | null })) : [],
    clock: "host" as const, known: state.trust === "known" && Boolean(step) }
}

/** Inclusive ranges retain gaps rather than implying uncaptured token positions. */
export function tokenPositionRanges(positions: number[]): Array<{ start: number; end: number }> {
  const ranges: Array<{ start: number; end: number }> = []
  for (const position of [...new Set(positions)].sort((a, b) => a - b)) {
    const last = ranges.at(-1)
    if (last && position === last.end + 1) last.end = position
    else ranges.push({ start: position, end: position })
  }
  return ranges
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

/** Allocation state is independent of request identity and retained token contents. */
export function physicalBlockState(block: Pick<KvBlockProjection, "isNull" | "trust" | "refCount" | "hasHash"> | undefined): "available" | "cached" | "referenced" | "null" | "unknown" {
  if (!block || block.trust === "unknown") return "unknown"
  if (block.isNull) return "null"
  if (block.refCount > 0) return "referenced"
  return block.hasHash ? "cached" : "available"
}

export const BLOCK_STATE_COLORS = {
  available: "#142034", cached: "#4a5569", referenced: "#647185", null: "#758095", unknown: "#252936",
} as const

export function physicalBlockStateLabel(state: ReturnType<typeof physicalBlockState>): string {
  return { available: "Available", cached: "Reusable prefix", referenced: "Referenced", null: "Reserved null", unknown: "Unknown" }[state]
}

/** Shared by all objects so modifier-based camera gestures cannot select data. */
export function sceneClick(event: { delta: number; nativeEvent: { button: number; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean } }): boolean {
  return event.delta <= 4 && event.nativeEvent.button === 0
    && !event.nativeEvent.ctrlKey && !event.nativeEvent.metaKey && !event.nativeEvent.shiftKey
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

export interface DieLayout {
  columns: number
  rows: number
  capacity: number
  pitch: number
  width: number
  depth: number
}

/** Physical block IDs fill a flat die row by row from the front-left corner; zoom never changes a block's cell. */
export function dieLayout(capacity: number, span = 11, aspect = 1.12): DieLayout {
  const count = Number.isSafeInteger(capacity) && capacity > 0 ? capacity : 0
  const columns = Math.max(1, Math.ceil(Math.sqrt(count * aspect)))
  const rows = Math.max(1, Math.ceil(count / columns))
  const pitch = Math.min(2, span / Math.max(columns, rows))
  return { columns, rows, capacity: count, pitch, width: columns * pitch, depth: rows * pitch }
}

/** Returns the [x, z] floor position of a physical block; row 0 faces the default camera. */
export function dieCellPosition(index: number, layout: DieLayout): [number, number] {
  const column = index % layout.columns
  const row = Math.floor(index / layout.columns)
  return [(column - (layout.columns - 1) / 2) * layout.pitch, ((layout.rows - 1) / 2 - row) * layout.pitch]
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
