import type { TraceEvent } from "../../replay/src/reduce.ts"

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
