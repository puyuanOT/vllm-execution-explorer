import type { TraceEvent } from "../../replay/src/reduce.ts"
import type { IndexedStep, TimePoint } from "../../replay/src/timeline.ts"

export function iterationEndpoints(events: TraceEvent[], steps: IndexedStep[]): number[] {
  const ends = new Map(steps.map(step => [step.id, step.sequence]))
  for (const event of events) {
    if (typeof event.step_id === "string" && ends.has(event.step_id)) ends.set(event.step_id, Math.max(ends.get(event.step_id)!, event.sequence))
  }
  return [...new Set([...ends.values()])].sort((a, b) => a - b)
}

export function nextIteration(endpoints: number[], sequence: number, maxSequence: number): number {
  return endpoints.find(end => end > sequence) ?? maxSequence
}

export function previousIteration(endpoints: number[], sequence: number): number {
  return endpoints.filter(end => end < sequence).at(-1) ?? 0
}

export function recordedTimeRange(points: TimePoint[]): { startMs: number; endMs: number } | null {
  let range: { startMs: number; endMs: number } | null = null
  for (const point of points) {
    if (point.elapsedMs == null || !Number.isFinite(point.elapsedMs)) continue
    range ??= { startMs: 0, endMs: 0 }
    range.startMs = Math.min(range.startMs, point.elapsedMs)
    range.endMs = Math.max(range.endMs, point.elapsedMs)
  }
  return range
}

/** Opens demos at a complete observed iteration, including its associated results. */
export function firstIterationSequence(events: TraceEvent[]): number {
  const first = events.find(event => event.event_type === "step_scheduled")
  if (!first) return 0
  if (typeof first.step_id !== "string") return first.sequence
  let end = first.sequence
  for (const event of events) {
    if (event.step_id === first.step_id) end = Math.max(end, event.sequence)
  }
  return end
}
