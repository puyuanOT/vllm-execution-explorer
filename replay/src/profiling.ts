/** Eager diagnostic measurements on their native profiler clock, independent of host replay. */
import type { RecordingManifest } from "./reduce.ts"

export class ProfileError extends Error {}

export interface ProfileClock {
  domain: "torch.profiler.chrome_trace"
  unit: "microseconds"
  origin: "trace_local"
}

export interface ProfileArtifact {
  path: string
  sha256: string
  sizeBytes: number
}

export type ProfileActivityKind = "cpu_scope" | "cuda_kernel" | "cuda_memcpy" | "cuda_memset" | "other"
export type ProfileCorrelation = "explicit_scope" | "runtime_flow" | "unmapped"
export type ProfileIdentity = number | string | null

export interface ProfileScope {
  stepId: string | null
  layerIndex: number | null
  component: "attention" | "ffn" | null
  correlation: ProfileCorrelation
  correlationSource: string | null
}

export interface ProfileActivity {
  id: string
  kind: ProfileActivityKind
  name: string
  device: ProfileIdentity
  stream: ProfileIdentity
  processId: ProfileIdentity
  threadId: ProfileIdentity
  startUs: number
  durationUs: number
  endUs: number
  scope: ProfileScope
  relatedActivityIds: string[]
}

export interface ProfilePackedRow {
  row: number
  requestId: string | null
  tokenPosition: number | null
  isPadding: boolean
}

export interface ProfileStep {
  stepId: string
  mappingStatus: "observed" | "unavailable"
  mappingReason: string | null
  mappingObservationBoundary: string | null
  paddingObservationBoundary: string | null
  paddingStatus: "observed" | "unavailable"
  paddingReason: string | null
  unpaddedTokens: number | null
  paddedTokens: number | null
  packedRows: ProfilePackedRow[]
}

export interface ProfileAttachment {
  contractVersion: 1
  runId: string
  mode: "eager_diagnostic"
  executionMode: "eager"
  backend: "torch.profiler"
  clock: ProfileClock
  activities: ProfileActivity[]
  steps: ProfileStep[]
  normalization: Record<string, unknown>
}

export interface ProfileSelection {
  stepId?: string | null
  layerIndex?: number | null
  component?: "attention" | "ffn" | null
  /** Native trace-local microseconds; never a host monotonic timestamp. */
  timeUs?: number | null
}

export interface ProfileIntervalIndex {
  readonly activityCount: number
  /** Native trace-local time; returned observations retain attachment order. */
  at(timeUs: number): ProfileActivity[]
}

export interface ProfileDeviceSummary {
  device: ProfileIdentity
  startUs: number
  endUs: number
  spanUs: number
  /** Union of device activity intervals, accounting for overlap across streams. */
  busyUs: number | null
  /** Sum of intervals; can exceed span/busy time when streams overlap. */
  summedActivityUs: number
  activityCount: number
  streams: ProfileIdentity[]
}

export interface ProfileLayerSummary {
  stepId: string | null
  layerIndex: number
  component: "attention" | "ffn" | null
  cpuScopeCount: number
  cpuScopeSpanUs: number | null
  devices: ProfileDeviceSummary[]
}

export interface MeasuredProfileProjection {
  status: "observed" | "unavailable"
  reason: string | null
  captureStatus: string | null
  completeness: string | null
  qualification: string | null
  evidenceKind: "gpu" | "fixture" | "unknown"
  runtimeVersions: Record<string, unknown>
  normalization: Record<string, unknown>
  clock: ProfileClock | null
  extent: { startUs: number; endUs: number } | null
  activities: ProfileActivity[]
  activeActivities: ProfileActivity[]
  layers: ProfileLayerSummary[]
  steps: ProfileStep[]
  selectedStep: ProfileStep | null
  mappingStatus: "observed" | "unavailable"
  mappingReason: string | null
  unmappedActivityCount: number
}

/** Returns only a declared, safe relative attachment path; no default filename guesses. */
export function readProfileArtifact(manifest?: RecordingManifest | null): ProfileArtifact | null {
  const metadata = record(manifest?.profiling)
  const artifact = metadata.normalized_artifact
  if (artifact == null) return null
  const value = requiredRecord(artifact, "normalized_artifact")
  const path = requiredString(value.path, "normalized_artifact.path")
  if (path.includes("\\") || path.startsWith("/") || path.split("/").some((part) => !part || part === "." || part === "..")
    || /[%?#:\u0000-\u001f\u007f]/.test(path)) throw new ProfileError("Profile attachment path must be a safe relative file path.")
  const sha256 = requiredString(value.sha256, "normalized_artifact.sha256")
  if (!/^[a-fA-F0-9]{64}$/.test(sha256)) throw new ProfileError("Profile attachment requires a SHA-256 digest.")
  return { path, sha256: sha256.toLowerCase(), sizeBytes: integer(value.size_bytes, "normalized_artifact.size_bytes") }
}

/** Parse a normalized attachment. Transport must independently verify its declared hash/size. */
export function parseProfileAttachment(text: string, manifest: RecordingManifest): ProfileAttachment {
  let parsed: unknown
  try { parsed = JSON.parse(text) } catch { throw new ProfileError("Profile attachment is not valid JSON.") }
  const value = requiredRecord(parsed, "profile attachment")
  const metadata = requiredRecord(manifest.profiling, "manifest.profiling")
  if (metadata.contract_version !== 1 || value.contract_version !== 1) throw new ProfileError("Unsupported measured profile contract version.")
  const runId = requiredString(value.run_id, "run_id")
  if (!manifest.run_id || manifest.run_id !== runId) throw new ProfileError("Profile attachment belongs to a different recording.")
  for (const [field, expected] of [["mode", "eager_diagnostic"], ["execution_mode", "eager"], ["backend", "torch.profiler"]] as const) {
    if (metadata[field] !== expected || value[field] !== expected) throw new ProfileError(`Unsupported measured profile ${field}.`)
  }
  const clock = parseClock(value.clock)
  parseClock(metadata.clock)
  const activities = array(value.activities, "activities").map(parseActivity)
  unique(activities.map((activity) => activity.id), "activity IDs")
  const steps = array(value.steps, "steps").map(parseStep)
  unique(steps.map((step) => step.stepId), "step IDs")
  return { contractVersion: 1, runId, mode: "eager_diagnostic", executionMode: "eager", backend: "torch.profiler", clock, activities, steps,
    normalization: structuredClone(record(value.normalization)) }
}

/** Profile selection uses native trace-local time, without consulting or shifting the host playhead. */
export function projectMeasuredProfile(manifest?: RecordingManifest | null, attachment?: ProfileAttachment | null, selection: ProfileSelection = {}): MeasuredProfileProjection {
  const metadata = record(manifest?.profiling)
  const unavailable = unavailableReason(metadata, attachment)
  const common = {
    captureStatus: string(metadata.status), completeness: string(metadata.completeness), qualification: string(metadata.qualification),
    evidenceKind: manifest?.execution_kind === "gpu" ? "gpu" as const : manifest?.execution_kind === "fixture" ? "fixture" as const : "unknown" as const,
    runtimeVersions: structuredClone(record(metadata.runtime_versions ?? metadata.runtime)),
    normalization: attachment ? structuredClone(attachment.normalization) : {},
  }
  if (unavailable || !attachment) return {
    ...common, status: "unavailable", reason: unavailable, clock: null, extent: null,
    activities: [], activeActivities: [], layers: [], steps: [], selectedStep: null,
    mappingStatus: "unavailable", mappingReason: unavailable, unmappedActivityCount: 0,
  }
  if (manifest?.run_id !== attachment.runId) throw new ProfileError("Profile attachment belongs to a different recording.")
  const activities = attachment.activities.filter((activity) =>
    (selection.stepId == null || activity.scope.stepId === selection.stepId)
    && (selection.layerIndex == null || activity.scope.layerIndex === selection.layerIndex)
    && (selection.component == null || activity.scope.component === selection.component))
    .map((activity) => structuredClone(activity))
  const activeActivities = selection.timeUs == null ? [] : profileActivitiesAt(activities, selection.timeUs)
  const steps = attachment.steps.map((step) => structuredClone(step))
  const selectedStep = selection.stepId == null ? null : steps.find((step) => step.stepId === selection.stepId) ?? null
  const mappingReason = selectedStep?.mappingReason ?? (selection.stepId == null ? "Select a captured batch to inspect packed token rows."
    : selectedStep ? selectedStep.mappingStatus === "observed" ? null : "Packed-row mapping was not captured."
      : "This profile has no packed-row observation for the selected step.")
  return {
    ...common, status: "observed", reason: common.evidenceKind === "fixture" ? "Synthetic profiler fixture; these intervals are not GPU measurements."
      : metadata.status === "partial" || metadata.status === "failed"
      ? "Only the retained profiler intervals are shown; this capture did not complete." : null,
    clock: { ...attachment.clock }, extent: extent(attachment.activities), activities, activeActivities,
    layers: summarizeLayers(activities), steps, selectedStep,
    mappingStatus: selectedStep?.mappingStatus ?? "unavailable", mappingReason,
    unmappedActivityCount: attachment.activities.filter((activity) => activity.scope.correlation === "unmapped").length,
  }
}

/** Lightweight playhead selector over a memoized projection; returned observations are read-only views. */
export function profileActivitiesAt(activities: readonly ProfileActivity[], timeUs: number): ProfileActivity[] {
  if (!Number.isFinite(timeUs)) throw new ProfileError("Profiler playhead must be finite trace-local microseconds.")
  return activities.filter((activity) => activity.startUs <= timeUs
    && (activity.endUs > timeUs || activity.durationUs === 0 && activity.startUs === timeUs))
}

/** Build once per attachment/scope. Queries prune expired subtrees rather than scan the whole trace. */
export function createProfileIntervalIndex(activities: readonly ProfileActivity[]): ProfileIntervalIndex {
  const indices = Array.from({ length: activities.length }, (_, index) => index)
    .sort((a, b) => activities[a].startUs - activities[b].startUs || a - b)
  let size = 1
  while (size < activities.length) size *= 2
  const maxEnds = new Float64Array(size * 2)
  maxEnds.fill(-Infinity)
  for (let index = 0; index < indices.length; index += 1) {
    const activity = activities[indices[index]]
    if (![activity.startUs, activity.endUs, activity.durationUs].every(Number.isFinite) || activity.durationUs < 0) throw new ProfileError("Cannot index an invalid native activity interval.")
    // A point event is active only at its recorded timestamp. The next representable
    // float is a pruning sentinel, not a measured or displayed duration.
    maxEnds[size + index] = activity.durationUs === 0 ? nextFloat(activity.startUs) : activity.endUs
  }
  for (let node = size - 1; node > 0; node -= 1) maxEnds[node] = Math.max(maxEnds[node * 2], maxEnds[node * 2 + 1])
  return {
    activityCount: activities.length,
    at(timeUs) {
      if (!Number.isFinite(timeUs)) throw new ProfileError("Profiler playhead must be finite trace-local microseconds.")
      let lower = 0
      let upper = indices.length
      while (lower < upper) {
        const middle = Math.floor((lower + upper) / 2)
        if (activities[indices[middle]].startUs <= timeUs) lower = middle + 1
        else upper = middle
      }
      const limit = lower
      const matches: number[] = []
      function visit(node: number, start: number, end: number): void {
        if (start >= limit || maxEnds[node] <= timeUs) return
        if (end - start === 1) {
          const index = indices[start]
          const activity = activities[index]
          if (activity.endUs > timeUs || activity.durationUs === 0 && activity.startUs === timeUs) matches.push(index)
          return
        }
        const middle = (start + end) / 2
        visit(node * 2, start, middle)
        visit(node * 2 + 1, middle, end)
      }
      visit(1, 0, size)
      return matches.sort((a, b) => a - b).map((index) => activities[index])
    },
  }
}

const floatView = new DataView(new ArrayBuffer(8))
function nextFloat(value: number): number {
  if (value === 0) return Number.MIN_VALUE
  floatView.setFloat64(0, value)
  floatView.setBigUint64(0, floatView.getBigUint64(0) + (value > 0 ? 1n : -1n))
  return floatView.getFloat64(0)
}

function parseClock(value: unknown): ProfileClock {
  const clock = requiredRecord(value, "clock")
  if (clock.domain !== "torch.profiler.chrome_trace" || clock.unit !== "microseconds" || clock.origin !== "trace_local") {
    throw new ProfileError("Measured profile requires an independent native trace-local microsecond clock.")
  }
  return { domain: "torch.profiler.chrome_trace", unit: "microseconds", origin: "trace_local" }
}

function parseActivity(input: unknown): ProfileActivity {
  const value = requiredRecord(input, "activity")
  const kind = value.kind
  if (!["cpu_scope", "cuda_kernel", "cuda_memcpy", "cuda_memset", "other"].includes(String(kind))) throw new ProfileError("Unsupported profile activity kind.")
  const scope = requiredRecord(value.scope, "activity.scope")
  const correlation = scope.correlation
  if (!["explicit_scope", "runtime_flow", "unmapped"].includes(String(correlation))) throw new ProfileError("Unsupported profile scope correlation.")
  const stepId = nullableString(scope.step_id, "scope.step_id")
  const layerIndex = nullableInteger(scope.layer_index, "scope.layer_index")
  const component = scope.component
  if (component !== null && component !== "attention" && component !== "ffn") throw new ProfileError("Unsupported measured component scope.")
  if (correlation === "unmapped" && (stepId != null || layerIndex != null || component != null)) throw new ProfileError("Unmapped activity cannot claim a batch or layer scope.")
  if (correlation !== "unmapped" && stepId == null && layerIndex == null && component == null) throw new ProfileError("Mapped activity requires an observed scope.")
  if (String(kind).startsWith("cuda_") && correlation === "explicit_scope") throw new ProfileError("CUDA scope requires observed runtime flow correlation.")
  const startUs = timestamp(value.start, "activity.start")
  const durationUs = timestamp(value.duration, "activity.duration")
  if (durationUs < 0 || !Number.isSafeInteger(Math.trunc(startUs + durationUs))) throw new ProfileError("Invalid measured activity duration.")
  const device = identity(value.device, "activity.device")
  const stream = identity(value.stream, "activity.stream")
  if (kind === "cpu_scope" && (device != null || stream != null)) throw new ProfileError("CPU scope cannot claim a CUDA device or stream.")
  return {
    id: requiredString(value.activity_id, "activity_id"), kind: kind as ProfileActivityKind,
    name: requiredString(value.name, "activity.name"), device, stream,
    processId: identity(value.process_id ?? null, "activity.process_id"), threadId: identity(value.thread_id ?? null, "activity.thread_id"),
    startUs, durationUs, endUs: startUs + durationUs,
    scope: { stepId, layerIndex, component, correlation: correlation as ProfileCorrelation,
      correlationSource: nullableString(scope.correlation_source ?? null, "scope.correlation_source") },
    relatedActivityIds: array(value.related_activity_ids ?? [], "related_activity_ids").map((value) => requiredString(value, "related_activity_id")),
  }
}

function parseStep(input: unknown): ProfileStep {
  const value = requiredRecord(input, "profile step")
  if (value.mapping_status !== "observed" && value.mapping_status !== "unavailable") throw new ProfileError("Unsupported packed-row mapping status.")
  const unpaddedTokens = nullableInteger(value.unpadded_tokens, "unpadded_tokens")
  const paddedTokens = nullableInteger(value.padded_tokens, "padded_tokens")
  const packedRows = array(value.packed_rows, "packed_rows").map((input) => {
    const row = requiredRecord(input, "packed row")
    if (typeof row.is_padding !== "boolean") throw new ProfileError("Packed row requires an explicit padding flag.")
    const result = {
      row: integer(row.row, "row"), requestId: nullableString(row.request_id, "row.request_id"),
      tokenPosition: nullableInteger(row.token_position, "row.token_position"), isPadding: row.is_padding,
    }
    if (result.isPadding && (result.requestId != null || result.tokenPosition != null)) throw new ProfileError("Padding cannot be attributed to a request token.")
    if (value.mapping_status === "observed" && !result.isPadding && (result.requestId == null || result.tokenPosition == null)) throw new ProfileError("Observed packed rows require request and position observations.")
    if (value.mapping_status === "unavailable" && (result.requestId != null || result.tokenPosition != null)) throw new ProfileError("Unavailable mapping cannot attribute token rows.")
    return result
  })
  unique(packedRows.map((row) => String(row.row)), "packed row indices")
  if (unpaddedTokens != null && paddedTokens != null && unpaddedTokens > paddedTokens) throw new ProfileError("Unpadded token count exceeds padded count.")
  if (value.mapping_status === "observed") {
    if (unpaddedTokens == null || packedRows.length !== (paddedTokens ?? unpaddedTokens)
      || packedRows.some((row, index) => row.row !== index || row.isPadding !== (index >= unpaddedTokens))) {
      throw new ProfileError("Observed packed rows must cover the ordered real rows and padding exactly.")
    }
    if (!string(value.mapping_observation_boundary)) throw new ProfileError("Observed packed rows require an observation boundary.")
  }
  const reason = nullableString(value.mapping_reason ?? value.reason ?? null, "mapping_reason")
  if (value.mapping_status === "unavailable" && reason == null) throw new ProfileError("Unavailable packed-row mapping requires a reason.")
  if (value.padding_status != null && value.padding_status !== "observed" && value.padding_status !== "unavailable") throw new ProfileError("Unsupported padding observation status.")
  if (value.padding_status === "observed" && paddedTokens == null) throw new ProfileError("Observed padding requires a padded input count.")
  if (value.padding_status === "unavailable" && paddedTokens != null) throw new ProfileError("Unavailable padding cannot claim an observed padded input count.")
  const paddingStatus = value.padding_status === "unavailable" || paddedTokens == null ? "unavailable" : "observed"
  return {
    stepId: requiredString(value.step_id, "step_id"), mappingStatus: value.mapping_status,
    mappingReason: reason, mappingObservationBoundary: nullableString(value.mapping_observation_boundary ?? null, "mapping_observation_boundary"),
    paddingObservationBoundary: nullableString(value.padding_observation_boundary ?? null, "padding_observation_boundary"),
    paddingStatus,
    paddingReason: nullableString(value.padding_reason ?? (paddingStatus === "unavailable" ? value.reason ?? "Padded input dimensions were not observed." : null), "padding_reason"),
    unpaddedTokens, paddedTokens, packedRows,
  }
}

function summarizeLayers(activities: ProfileActivity[]): ProfileLayerSummary[] {
  const groups = new Map<string, ProfileActivity[]>()
  for (const activity of activities) {
    if (activity.scope.layerIndex == null || activity.scope.correlation === "unmapped") continue
    const key = JSON.stringify([activity.scope.stepId, activity.scope.layerIndex, activity.scope.component])
    const group = groups.get(key) ?? []
    group.push(activity)
    groups.set(key, group)
  }
  return [...groups.values()].map((group) => {
    const scope = group[0].scope
    const cpu = group.filter((activity) => activity.kind === "cpu_scope")
    const cpuExtent = extent(cpu)
    const devices = new Map<string, ProfileActivity[]>()
    for (const activity of group) {
      if (!activity.kind.startsWith("cuda_")) continue
      const key = JSON.stringify(activity.device)
      const device = devices.get(key) ?? []
      device.push(activity)
      devices.set(key, device)
    }
    return {
      stepId: scope.stepId, layerIndex: scope.layerIndex!, component: scope.component,
      cpuScopeCount: cpu.length, cpuScopeSpanUs: cpuExtent ? cpuExtent.endUs - cpuExtent.startUs : null,
      devices: [...devices.values()].map((device) => {
        const range = extent(device)!
        return {
          device: device[0].device, ...range, spanUs: range.endUs - range.startUs,
          busyUs: device[0].device == null ? null : unionDuration(device), summedActivityUs: device.reduce((sum, activity) => sum + activity.durationUs, 0),
          activityCount: device.length, streams: [...new Set(device.map((activity) => activity.stream))],
        }
      }),
    }
  }).sort((a, b) => String(a.stepId).localeCompare(String(b.stepId)) || a.layerIndex - b.layerIndex || String(a.component).localeCompare(String(b.component)))
}

function unionDuration(activities: ProfileActivity[]): number {
  let total = 0
  let end = -Infinity
  for (const activity of [...activities].sort((a, b) => a.startUs - b.startUs)) {
    total += Math.max(0, activity.endUs - Math.max(activity.startUs, end))
    end = Math.max(end, activity.endUs)
  }
  return total
}

function extent(activities: ProfileActivity[]): { startUs: number; endUs: number } | null {
  if (!activities.length) return null
  let startUs = Infinity
  let endUs = -Infinity
  for (const activity of activities) { startUs = Math.min(startUs, activity.startUs); endUs = Math.max(endUs, activity.endUs) }
  return { startUs, endUs }
}

function unavailableReason(metadata: Record<string, unknown>, attachment?: ProfileAttachment | null): string | null {
  if (!Object.keys(metadata).length) return "This recording has no measured profile contract."
  if (metadata.mode === "off" || metadata.status === "disabled") return "Measured profiling was disabled for this recording."
  if (metadata.contract_version !== 1 || metadata.mode !== "eager_diagnostic" || metadata.execution_mode !== "eager" || metadata.backend !== "torch.profiler") return "This recording has no supported eager diagnostic profile contract."
  if (metadata.attachment_status === "omitted_by_structural_export") return "Measured profile attachments were omitted by the structural export."
  if (metadata.attachment_status === "unavailable") return "The normalized measured profile attachment is unavailable."
  if (!attachment) return metadata.status === "failed" ? "Measured profiling failed and no normalized intervals are available."
    : "The normalized measured profile attachment is unavailable."
  return null
}

function record(value: unknown): Record<string, unknown> { return value != null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {} }
function requiredRecord(value: unknown, name: string): Record<string, unknown> {
  if (value == null || typeof value !== "object" || Array.isArray(value)) throw new ProfileError(`${name} must be an object.`)
  return value as Record<string, unknown>
}
function array(value: unknown, name: string): unknown[] { if (!Array.isArray(value)) throw new ProfileError(`${name} must be an array.`); return value }
function string(value: unknown): string | null { return typeof value === "string" && value.length ? value : null }
function requiredString(value: unknown, name: string): string { const result = string(value); if (result == null) throw new ProfileError(`${name} must be a nonempty string.`); return result }
function nullableString(value: unknown, name: string): string | null { return value === null ? null : requiredString(value, name) }
function integer(value: unknown, name: string): number { if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw new ProfileError(`${name} must be a nonnegative safe integer.`); return value }
function nullableInteger(value: unknown, name: string): number | null { return value === null ? null : integer(value, name) }
function timestamp(value: unknown, name: string): number { if (typeof value !== "number" || !Number.isFinite(value) || !Number.isSafeInteger(Math.trunc(value))) throw new ProfileError(`${name} must be a finite native microsecond timestamp.`); return value }
function identity(value: unknown, name: string): ProfileIdentity { if (value === null || typeof value === "string" && value.length || typeof value === "number" && Number.isSafeInteger(value)) return value as ProfileIdentity; throw new ProfileError(`${name} must be a number, string, or null.`) }
function unique(values: string[], name: string): void { if (new Set(values).size !== values.length) throw new ProfileError(`Duplicate ${name}.`) }
