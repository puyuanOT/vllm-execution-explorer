import type { TraceEvent } from "../../replay/src/reduce.ts"
import type { PlaybackMode } from "./execution/types.ts"

/** Events beyond the first trace gap cannot be played as observed execution. */
export function replayableEvents(events: TraceEvent[], maxSequence: number): TraceEvent[] {
  return events.filter((event) => event.sequence <= maxSequence)
}

export function nextSequence(events: TraceEvent[], current: number): number {
  for (const event of events) {
    if (event.sequence > current) return event.sequence
  }
  return current
}

export function previousSequence(events: TraceEvent[], current: number): number {
  let previous = 0
  for (const event of events) {
    if (event.sequence >= current) break
    previous = event.sequence
  }
  return previous
}

export function eventAt(events: TraceEvent[], sequence: number): TraceEvent | null {
  let found: TraceEvent | null = null
  for (const event of events) {
    if (event.sequence <= sequence) found = event
    else break
  }
  return found
}

export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const

export const RECORDED_SPEEDS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, ...SPEEDS] as const
export const PROFILE_SPEEDS = [0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, ...RECORDED_SPEEDS] as const

export function playbackSpeeds(mode: PlaybackMode): readonly number[] {
  return mode === "measured" ? PROFILE_SPEEDS : mode === "recorded" ? RECORDED_SPEEDS : SPEEDS
}
