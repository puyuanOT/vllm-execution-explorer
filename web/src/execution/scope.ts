export const SCOPE_CHANNELS = [
  { id: "intake", code: "INT", label: "Request intake" },
  { id: "scheduler", code: "SCH", label: "Scheduler" },
  { id: "kv", code: "KV", label: "KV cache" },
  { id: "model", code: "MDL", label: "Model execution" },
  { id: "output", code: "OUT", label: "Output" },
] as const

export type ScopeChannel = typeof SCOPE_CHANNELS[number]["id"]

const CHANNEL_BY_EVENT: Record<string, ScopeChannel> = {
  request_received: "intake",
  request_tokenized: "intake",
  request_queued: "intake",
  request_dispatched: "intake",
  request_dispatch_completed: "intake",
  http_request_received: "intake",
  http_engine_admission_correlated: "intake",
  request_model_route: "intake",
  engine_prompt_observed: "intake",
  engine_admitted: "intake",
  capture_ready: "intake",
  step_scheduled: "scheduler",
  prefix_lookup: "scheduler",
  preemption: "scheduler",
  block_change: "kv",
  cache_inventory: "kv",
  cache_reset: "kv",
  kv_block_binding: "kv",
  kv_block_written: "kv",
  step_submitted: "model",
  step_result_received: "model",
  tokens_accepted: "output",
  text_emitted: "output",
  completion: "output",
  cancellation: "output",
  http_response_completed: "output",
}

export function scopeChannel(eventType: string): ScopeChannel | null {
  return CHANNEL_BY_EVENT[eventType] ?? null
}

export interface ScopeTick {
  position: number
  channel: ScopeChannel
}

export interface ScopeMarker {
  position: number
  label: string
}

export interface ScopeModel {
  columns: number
  bins: Record<ScopeChannel, number[]>
  current: { channel: ScopeChannel; column: number } | null
  markers: Array<{ column: number; label: string }>
  playheadColumn: number
  gapColumn: number | null
}

/**
 * Positions are 0..1 along the recording. Ticks and markers after the playhead are dropped so
 * the scope shows what has been played, not where later activity happens.
 */
export function buildScope({ ticks, markers, playhead, gap = null, columns, currentPosition = null }: {
  ticks: ScopeTick[]
  markers: ScopeMarker[]
  playhead: number
  gap?: number | null
  columns: number
  currentPosition?: number | null
}): ScopeModel {
  const width = Math.max(1, Math.floor(columns))
  const column = (position: number) => Math.min(width - 1, Math.max(0, Math.round(clamp(position) * (width - 1))))
  const head = clamp(playhead)
  const sets: Record<ScopeChannel, Set<number>> = { intake: new Set(), scheduler: new Set(), kv: new Set(), model: new Set(), output: new Set() }
  let current: ScopeModel["current"] = null
  for (const tick of ticks) {
    if (tick.position > head + 1e-9) continue
    const at = column(tick.position)
    sets[tick.channel].add(at)
    if (currentPosition != null && Math.abs(tick.position - currentPosition) < 1e-9) current = { channel: tick.channel, column: at }
  }
  const bins = Object.fromEntries(Object.entries(sets).map(([channel, set]) => [channel, [...set].sort((left, right) => left - right)])) as Record<ScopeChannel, number[]>
  return {
    columns: width,
    bins,
    current,
    markers: markers.filter((marker) => marker.position <= head + 1e-9).map((marker) => ({ column: column(marker.position), label: marker.label })),
    playheadColumn: column(head),
    gapColumn: gap == null ? null : column(gap),
  }
}

export function sequencePosition(sequence: number, maxSequence: number): number {
  if (maxSequence <= 1) return sequence >= maxSequence ? 1 : 0
  return clamp((sequence - 1) / (maxSequence - 1))
}

function clamp(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0
}
