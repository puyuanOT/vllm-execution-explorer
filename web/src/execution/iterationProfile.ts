import type { ProfileActivity, ProfileCacheParticipant, ProfileCacheWrite, ProfileKvObservations } from "../../../replay/src/profiling.ts"
import { profileObservationIndex } from "../../../replay/src/profileIndex.ts"
import type { ReplayState } from "../../../replay/src/reduce.ts"
import type { Selection } from "./types.ts"

export interface IterationInterval { write: ProfileCacheWrite; activity: ProfileActivity }

/** Native observations are linked by step/operation IDs, never by host timestamps. */
export function iterationProfile(observations: ProfileKvObservations | null | undefined, stepId: string | null) {
  const index = profileObservationIndex(observations)
  const writes = stepId == null ? [] : [...(index?.writesByStep.get(stepId) ?? [])]
  const intervals: IterationInterval[] = writes.flatMap((write) => write.status !== "observed" ? [] : write.gpuActivityIds.flatMap((id) => {
    const activity = index?.activitiesById.get(id)
    return activity && activity.kind.startsWith("cuda_") && activity.scope.correlation === "runtime_flow"
      && activity.scope.operation === "kv_cache_write" && activity.scope.operationId === write.operationId
      && activity.scope.stepId === stepId && activity.scope.layerIndex === write.layerIndex ? [{ write, activity }] : []
  })).sort((a, b) => a.activity.startUs - b.activity.startUs)
  const native = index?.gpuActivitiesByStep.get(stepId) ?? []
  const extent = native.length ? { startUs: Math.min(...native.map((activity) => activity.startUs)), endUs: Math.max(...native.map((activity) => activity.endUs)) } : null
  const packedRows = stepId == null ? [] : index?.stepsById.get(stepId)?.packedRows ?? []
  return { writes, intervals, extent, packedRows }
}

export function activeIterationIntervals(intervals: IterationInterval[], timeUs: number) {
  return intervals.filter(({ activity }) => activity.startUs <= timeUs && timeUs < activity.endUs)
}

export function participantSelection(state: ReplayState, write: ProfileCacheWrite, participant: ProfileCacheParticipant): { selection: Selection; physical: boolean } {
  const identity: Selection = { kind: "token", requestId: participant.requestId, position: participant.tokenPosition }
  if (state.trust !== "known" || participant.pool == null || participant.blockId == null || participant.allocationGeneration == null || participant.slot == null) return { selection: identity, physical: false }
  const key = `${participant.pool}:${participant.blockId}`
  const block = state.blocks[key]
  const group = state.cacheGroups.find((group) => group.pool === participant.pool && group.cacheGroup === participant.cacheGroup)
  const mapped = block?.generation === participant.allocationGeneration && [...block.contents, ...block.bindings].some((span) =>
    span.requestId === participant.requestId && participant.tokenPosition >= span.tokenStart && participant.tokenPosition < span.tokenEnd
    && ("slotStart" in span ? (span.slotStart as number) + participant.tokenPosition - span.tokenStart
      : group?.blockSize && (group.tokenSlotMapping === "qualified_full_attention" || group.tokenSlotMapping === "authored_fixture") ? participant.tokenPosition % group.blockSize : null) === participant.slot)
  return mapped ? { selection: { ...identity, kvLayerIndex: write.layerIndex, kvBlockKey: key }, physical: true } : { selection: identity, physical: false }
}

export function selectedLayerIndex(selection: Selection | null): number | null {
  if (selection?.kind === "layer" || selection?.kind === "module" || selection?.kind === "head") return selection.layerIndex
  return selection?.kind === "token" ? selection.kvLayerIndex ?? null : null
}

/** Controlled by current selection so an external block choice resets the picker. */
export function participantPickerValue(write: ProfileCacheWrite, selection: Selection | null): string {
  if (selection?.kind !== "token") return ""
  const index = write.participants.findIndex((participant) => participant.requestId === selection.requestId && participant.tokenPosition === selection.position
    && (selection.kvLayerIndex == null || selection.kvLayerIndex === write.layerIndex)
    && (selection.kvBlockKey == null || selection.kvBlockKey === `${participant.pool}:${participant.blockId}`))
  return index < 0 ? "" : String(index)
}

export function nativeWindowAt(extent: { startUs: number; endUs: number }, cursor: number, spanUs: number) {
  const fullSpan = Math.max(0.001, extent.endUs - extent.startUs)
  const width = Math.min(fullSpan, Math.max(0.001, spanUs))
  const startUs = Math.max(extent.startUs, Math.min(extent.startUs + fullSpan - width, cursor - width / 2))
  return { startUs, endUs: startUs + width, cursor: Math.max(startUs, Math.min(startUs + width, cursor)) }
}

export function formatPositions(values: number[]): string {
  const sorted = [...new Set(values)].sort((a, b) => a - b)
  const ranges: string[] = []
  for (let index = 0; index < sorted.length; index++) {
    const start = sorted[index]
    let end = start
    while (index + 1 < sorted.length && sorted[index + 1] === end + 1) end = sorted[++index]
    ranges.push(start === end ? String(start) : `${start}–${end}`)
  }
  return ranges.join(", ") || "none"
}
