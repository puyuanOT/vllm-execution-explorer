import type { Player, ReplayState } from "../../../replay/src/reduce.ts"

export interface TelemetryPoint {
  sequence: number
  requests: number
  steps: number
  referenced: number
  capacity: number | null
  results: number
  submissions: number
  accepted: number
}

export function telemetryPoint(state: ReplayState): TelemetryPoint {
  const blocks = Object.values(state.blocks)
  return {
    sequence: state.lastSequence,
    requests: Object.keys(state.requests).length,
    steps: state.steps.length,
    referenced: blocks.filter((block) => !block.isNull && block.refCount > 0).length,
    capacity: state.cacheInventory ? state.cacheInventory.reduce((sum, pool) => sum + pool.capacity, 0) : null,
    results: state.steps.filter((step) => step.resultSequence != null).length,
    submissions: state.steps.filter((step) => step.submittedSequence != null).length,
    accepted: Object.values(state.requests).reduce((sum, request) => sum + request.acceptedTokenCount, 0),
  }
}

/** Samples the recording at evenly spaced sequences; callers trim it to the playhead before display. */
export function telemetrySeries(player: Pick<Player, "at" | "maxSequence"> | null, buckets = 40): TelemetryPoint[] {
  if (!player || player.maxSequence <= 0) return []
  const count = Math.max(2, Math.min(buckets, player.maxSequence + 1))
  const points: TelemetryPoint[] = []
  let previous = -1
  for (let index = 0; index < count; index += 1) {
    const sequence = Math.round((index / (count - 1)) * player.maxSequence)
    if (sequence === previous) continue
    previous = sequence
    points.push({ ...telemetryPoint(player.at(sequence)), sequence })
  }
  return points
}

/** Keeps only samples recorded before the playhead and ends on the playhead value, so no future count is drawn. */
export function telemetryThrough(series: TelemetryPoint[], current: TelemetryPoint): TelemetryPoint[] {
  return [...series.filter((point) => point.sequence < current.sequence), current]
}
