/** Host-clock measurements. Never interpolate or mix native profiler clocks. */
import type { RecordingManifest, TraceEvent } from "./reduce.ts"

export interface DeviceMemory {
  uuid: string
  name: string
  total: number
  used: number
  free: number
  processUsed: number | null
  processStatus: string
}

export interface FrameworkMemory {
  uuid: string | null
  device: string
  pid: number | null
  rank: number | null
  allocated: number
  reserved: number
  peakAllocated: number
  peakReserved: number
  peakScope: string
}

export interface MemorySample {
  sequence: number
  timeNs: bigint
  startNs: bigint
  status: string
  phase: string
  source: string
  reason: string | null
  pid: number | null
  devices: DeviceMemory[]
  framework: FrameworkMemory | null
}

export interface MemoryStorage {
  id: string
  device: string
  bytes: number
  categories: string[]
  owners: string[]
  dtypes: string[]
}

export interface MemoryInventory {
  sequence: number
  timeNs: bigint
  storages: MemoryStorage[]
  coverage: string
  status: string
}

export interface MemoryProjection {
  samples: MemorySample[]
  current: MemorySample | null
  latestAttempt: MemorySample | null
  inventory: MemoryInventory | null
  timeNs: bigint | null
  originNs: bigint | null
  ageMs: number | null
  inventoryAgeMs: number | null
  gap: boolean
  intervalMs: number
  phase: string | null
  reason: string | null
  captureReason: string | null
}

function object(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}
}
function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null
}
function strings(value: unknown): string[] { return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [] }
function ns(value: unknown): bigint | null {
  try { return typeof value === "string" || typeof value === "number" ? BigInt(value) : null } catch { return null }
}

function sample(event: TraceEvent): MemorySample | null {
  if (event.event_type !== "gpu_memory_sample" || event.memory_clock_domain !== "time.monotonic_ns") return null
  const timeNs = ns(event.sample_completed_monotonic_ns)
  const startNs = ns(event.sample_started_monotonic_ns)
  if (timeNs == null || startNs == null || startNs > timeNs) return null
  const devices = (Array.isArray(event.devices) ? event.devices : []).flatMap((value): DeviceMemory[] => {
    const row = object(value), total = finite(row.total_bytes), used = finite(row.used_bytes), free = finite(row.free_bytes)
    return typeof row.gpu_uuid === "string" && total != null && used != null && free != null
      ? [{ uuid: row.gpu_uuid, name: String(row.name ?? "GPU"), total, used, free, processUsed: row.process_status === "observed" ? finite(row.process_used_bytes) : null, processStatus: String(row.process_status ?? "unavailable") }] : []
  })
  const row = object(event.framework), allocated = finite(row.allocated_bytes), reserved = finite(row.reserved_bytes)
  const peakAllocated = finite(row.peak_allocated_bytes), peakReserved = finite(row.peak_reserved_bytes)
  const framework: FrameworkMemory | null = allocated != null && reserved != null && peakAllocated != null && peakReserved != null ? {
    uuid: typeof row.gpu_uuid === "string" ? row.gpu_uuid : null, device: String(row.device ?? "unknown"), pid: finite(row.worker_pid), rank: finite(row.rank),
    allocated, reserved, peakAllocated, peakReserved, peakScope: String(row.peak_scope ?? "unavailable"),
  } : null
  return { sequence: event.sequence, timeNs, startNs, status: String(event.status ?? "unavailable"), phase: String(event.phase ?? "unknown"), source: String(event.source ?? "unknown"),
    reason: typeof event.reason === "string" ? event.reason : null, pid: finite(event.engine_pid), devices, framework }
}

export function projectMemory(events: TraceEvent[], manifest: RecordingManifest | null, sequence: number): MemoryProjection {
  const metadata = object(manifest?.gpu_memory)
  const intervalMs = finite(metadata.sample_interval_ms) ?? 100
  const traceGap = events.find((event) => event.event_type === "trace_gap")
  const usable = traceGap ? events.filter((event) => event.sequence < traceGap.sequence) : events
  const visible = usable.filter((event) => event.sequence <= sequence)
  const timeNs = ns(visible.at(-1)?.monotonic_ns)
  const samples = usable.map(sample).filter((row): row is MemorySample => row !== null)
  const attempts = samples.filter((row) => row.sequence <= sequence && timeNs != null && row.timeNs <= timeNs)
  const current = attempts.filter((row) => row.status === "observed" && row.devices.length > 0).at(-1) ?? null
  const latestAttempt = attempts.at(-1) ?? null
  const phaseEvent = visible.filter((event) => event.event_type === "gpu_memory_phase").at(-1)
  const phase = typeof phaseEvent?.phase === "string" ? phaseEvent.phase : latestAttempt?.phase ?? null
  let inventory: MemoryInventory | null = null
  const event = visible.filter((event) => event.event_type === "gpu_memory_inventory" && event.memory_clock_domain === "time.monotonic_ns").at(-1)
  if (event && phase !== "cleanup") {
    const timestamp = ns(event.monotonic_ns)
    if (timestamp != null) inventory = { sequence: event.sequence, timeNs: timestamp, status: String(event.status ?? "unavailable"), coverage: String(event.coverage ?? event.reason ?? "Coverage unavailable"),
      storages: (Array.isArray(event.storages) ? event.storages : []).flatMap((value): MemoryStorage[] => {
        const row = object(value), bytes = finite(row.byte_count)
        return typeof row.storage_id === "string" && typeof row.device === "string" && bytes != null ? [{ id: row.storage_id, device: row.device, bytes, categories: strings(row.categories), owners: strings(row.owners), dtypes: strings(row.dtypes) }] : []
      }) }
  }
  const ageMs = current && timeNs != null ? Number(timeNs - current.timeNs) / 1e6 : null
  const inventoryAgeMs = inventory && timeNs != null ? Number(timeNs - inventory.timeNs) / 1e6 : null
  const gap = Boolean(traceGap && sequence >= traceGap.sequence) || latestAttempt?.status !== "observed" || ageMs != null && ageMs > Math.max(intervalMs * 2.5, 250)
  const reason = samples.length ? current ? null : "No successful device memory sample was recorded before this playhead." : "GPU memory was not recorded for this run. A new capture is required."
  const firstEventNs = ns(events[0]?.monotonic_ns)
  const firstQueryNs = samples[0]?.startNs ?? null
  const originNs = firstEventNs == null ? firstQueryNs : firstQueryNs != null && firstQueryNs < firstEventNs ? firstQueryNs : firstEventNs
  const captureReason = metadata.status === "partial" ? typeof metadata.reason === "string" ? metadata.reason : "The memory sampler did not finish before the capture closed." : null
  return { samples, current, latestAttempt, inventory, timeNs, originNs, ageMs, inventoryAgeMs, gap, intervalMs, phase, reason, captureReason }
}

export function storageBreakdown(inventory: MemoryInventory | null, device: string): Array<{ category: string; bytes: number; storageCount: number }> {
  if (inventory?.status !== "observed") return []
  const groups = new Map<string, { category: string; bytes: number; storageCount: number }>()
  const seen = new Set<string>()
  for (const row of inventory?.storages ?? []) {
    if (row.device !== device || seen.has(row.id)) continue
    seen.add(row.id)
    const category = row.categories.length === 1 ? row.categories[0] : "shared"
    const group = groups.get(category) ?? { category, bytes: 0, storageCount: 0 }
    group.bytes += row.bytes
    group.storageCount += 1
    groups.set(category, group)
  }
  return [...groups.values()]
}

export function sampledDeviceMaximum(samples: MemorySample[], uuid: string): number | null {
  const values = samples.filter((sample) => sample.status === "observed").flatMap((sample) => sample.devices.filter((row) => row.uuid === uuid).map((row) => row.used))
  return values.length ? Math.max(...values) : null
}
