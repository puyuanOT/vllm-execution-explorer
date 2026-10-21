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
  component: "attention" | "recurrent" | "ffn" | null
  correlation: ProfileCorrelation
  correlationSource: string | null
  operation?: "kv_cache_write" | null
  operationId?: string | null
}

export interface ProfileCacheParticipant {
  row: number
  requestId: string
  tokenPosition: number
  pool: string | null
  cacheGroup: number | null
  blockId: number | null
  allocationGeneration: number | null
  slot: number | null
  kernelBlockId: number | null
  kernelSlot: number | null
}

export interface ProfileCacheWrite {
  operationId: string
  stepId: string
  layerIndex: number
  layerName: string
  cacheGroup: number | null
  pool: string | null
  device: string | null
  status: "observed" | "unavailable"
  reason: string | null
  participants: ProfileCacheParticipant[]
  gpuActivityIds: string[]
  observationBoundary: string
}

export interface ProfileTensorSnapshot {
  snapshotId: string
  operationId: string
  stepId: string
  requestId: string
  tokenPosition: number
  layerIndex: number
  cacheGroup: number | null
  pool: string | null
  blockId: number | null
  allocationGeneration: number | null
  slot: number | null
  device: string
  kvHead: number
  kind: "K" | "V"
  dtype: string
  shape: number[]
  dimensionStart: number
  values: number[]
  byteCount: number
  observationBoundary: string
  quantization?: {
    format: "fp8_e4m3" | "fp8_e5m2"
    rawBits: number[]
    scale: { value: number; dtype: "float32"; shape: number[]; mode: "per_tensor" | "per_head"; headIndex: number | null; observationBoundary: string }
    reconstructedValues: number[]
    reconstruction: "decoded_fp8_times_recorded_scale"
  }
}

export interface ProfileTensorCapture {
  enabled: boolean
  status: string
  reason: string | null
  maxBytes: number
  retainedBytes: number
  selection: {
    requestId: string | null
    tokenPositions: number[]
    layerIndices: number[]
    headIndices: number[]
    dimensionStart: number
    dimensionCount: number
  }
  observationBoundary: string | null
}

export interface ProfileKvObservations {
  activities: ProfileActivity[]
  steps: ProfileStep[]
  cacheWrites: ProfileCacheWrite[]
  tensorSnapshots: ProfileTensorSnapshot[]
  tensorCapture: ProfileTensorCapture | null
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

export interface ProfileAttachment extends ProfileKvObservations {
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
  component?: "attention" | "recurrent" | "ffn" | null
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
  component: "attention" | "recurrent" | "ffn" | null
  cpuScopeCount: number
  cpuScopeSpanUs: number | null
  devices: ProfileDeviceSummary[]
}

export interface MeasuredProfileProjection extends ProfileKvObservations {
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
  let value = requiredRecord(parsed, "profile attachment")
  const compact = record(record(value.normalization).attachment_encoding).kind === "native_profile_compact_v1"
  if (compact) value = expandCompactProfile(value)
  const metadata = requiredRecord(manifest.profiling, "manifest.profiling")
  if (metadata.contract_version !== 1 || value.contract_version !== 1) throw new ProfileError("Unsupported measured profile contract version.")
  const runId = requiredString(value.run_id, "run_id")
  if (!manifest.run_id || manifest.run_id !== runId) throw new ProfileError("Profile attachment belongs to a different recording.")
  for (const [field, expected] of [["mode", "eager_diagnostic"], ["execution_mode", "eager"], ["backend", "torch.profiler"]] as const) {
    if (metadata[field] !== expected || value[field] !== expected) throw new ProfileError(`Unsupported measured profile ${field}.`)
  }
  const clock = parseClock(value.clock)
  parseClock(metadata.clock)
  const activityNames = compact || value.activity_names == null ? null : array(value.activity_names, "activity_names").map((name) => requiredString(name, "activity name"))
  if (activityNames) {
    unique(activityNames, "activity names")
    const encoding = requiredRecord(record(value.normalization).attachment_encoding, "normalization.attachment_encoding")
    if (encoding.kind !== "activity_name_dictionary_v1" || encoding.unique_name_count !== activityNames.length
      || encoding.encoded_activity_count !== array(value.activities, "activities").length) throw new ProfileError("Invalid activity-name dictionary encoding metadata.")
  } else if (!compact && record(record(value.normalization).attachment_encoding).kind != null) {
    throw new ProfileError("Activity-name dictionary encoding requires its name table.")
  }
  const activities = array(value.activities, "activities").map((activity) => parseActivity(activity, activityNames))
  unique(activities.map((activity) => activity.id), "activity IDs")
  const steps = array(value.steps, "steps").map(parseStep)
  unique(steps.map((step) => step.stepId), "step IDs")
  const cacheWrites = array(value.cache_writes ?? [], "cache_writes").map(parseCacheWrite)
  unique(cacheWrites.map((operation) => operation.operationId), "cache-write operation IDs")
  const tensorSnapshots = array(value.tensor_snapshots ?? [], "tensor_snapshots").map(parseTensorSnapshot)
  unique(tensorSnapshots.map((snapshot) => snapshot.snapshotId), "tensor snapshot IDs")
  const tensorCapture = value.tensor_capture == null ? null : parseTensorCapture(value.tensor_capture)
  validateKvObservations(activities, steps, cacheWrites, tensorSnapshots, tensorCapture)
  return { contractVersion: 1, runId, mode: "eager_diagnostic", executionMode: "eager", backend: "torch.profiler", clock, activities, steps,
    cacheWrites, tensorSnapshots, tensorCapture, normalization: structuredClone(record(value.normalization)) }
}

/** Expand transport dictionaries before the existing observation/link validation. */
function expandCompactProfile(value: Record<string, unknown>): Record<string, unknown> {
  const encoding = requiredRecord(record(value.normalization).attachment_encoding, "compact encoding")
  const names = array(value.activity_names, "activity_names").map(name => requiredString(name, "activity name"))
  unique(names, "activity names")
  const scopes = array(value.activity_scopes, "activity_scopes").map(scope => requiredRecord(scope, "activity scope"))
  const participants = array(value.cache_participants, "cache_participants").map(row => requiredRecord(row, "cache participant"))
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical)
    : value != null && typeof value === "object" ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, field]) => [key, canonical(field)])) : value
  unique(scopes.map(scope => JSON.stringify(canonical(scope))), "activity scopes")
  unique(participants.map(row => JSON.stringify(canonical(row))), "cache participants")
  const wire = array(value.activities, "activities")
  const writes = array(value.cache_writes ?? [], "cache_writes")
  for (const [key, count] of [["unique_name_count", names.length], ["unique_scope_count", scopes.length],
    ["unique_participant_count", participants.length], ["encoded_activity_count", wire.length], ["encoded_cache_write_count", writes.length]] as const) {
    if (encoding[key] !== count) throw new ProfileError(`Invalid compact encoding ${key}.`)
  }
  const lookup = <T,>(table: T[], reference: unknown, label: string): T => {
    const index = integer(reference, label)
    if (index >= table.length) throw new ProfileError(`${label} is outside its dictionary.`)
    return table[index]
  }
  const ids = wire.map(row => requiredString(Array.isArray(row) ? row[0] : requiredRecord(row, "activity").activity_id, "activity_id"))
  unique(ids, "activity IDs")
  const activityRef = (ref: unknown): string => typeof ref === "string" ? requiredString(ref, "activity reference") : lookup(ids, ref, "activity reference")
  const activities = wire.map(row => {
    if (!Array.isArray(row)) {
      const plain = requiredRecord(row, "activity")
      if (plain.name_id !== undefined) throw new ProfileError("Compact fallback activity must contain its exact name.")
      return plain
    }
    if (row.length !== 11) throw new ProfileError("Compact activity must contain exactly eleven fields.")
    return { activity_id: row[0], kind: row[1], name: lookup(names, row[2], "activity name reference"),
      device: row[3], stream: row[4], process_id: row[5], thread_id: row[6], start: row[7], duration: row[8],
      scope: lookup(scopes, row[9], "activity scope reference"),
      related_activity_ids: array(row[10], "related activity references").map(activityRef) }
  })
  const cacheWrites = writes.map(row => {
    const write = requiredRecord(row, "cache write")
    if (write.participant_ids === undefined && write.gpu_activity_refs === undefined) return write
    if (write.participant_ids !== undefined && write.participants !== undefined
      || write.gpu_activity_refs !== undefined && write.gpu_activity_ids !== undefined) throw new ProfileError("Ambiguous compact cache-write fields.")
    const { participant_ids, gpu_activity_refs, ...fields } = write
    return { ...fields,
      ...(participant_ids === undefined ? {} : { participants: array(participant_ids, "participant_ids").map(id => lookup(participants, id, "participant reference")) }),
      ...(gpu_activity_refs === undefined ? {} : { gpu_activity_ids: array(gpu_activity_refs, "gpu_activity_refs").map(activityRef) }) }
  })
  return { ...value, activities, cache_writes: cacheWrites }
}

/** Profile selection uses native trace-local time, without consulting or shifting the host playhead. */
export function projectMeasuredProfile(manifest?: RecordingManifest | null, attachment?: ProfileAttachment | null, selection: ProfileSelection = {}): MeasuredProfileProjection {
  return projectProfile(manifest, attachment, selection, true)
}

/** Immutable published recordings may share their observations with read-only UI
 * consumers. Unlike the default projector, this view must never be mutated. */
export function projectMeasuredProfileView(manifest?: RecordingManifest | null, attachment?: ProfileAttachment | null, selection: ProfileSelection = {}): MeasuredProfileProjection {
  return projectProfile(manifest, attachment, selection, false)
}

function projectProfile(manifest: RecordingManifest | null | undefined, attachment: ProfileAttachment | null | undefined, selection: ProfileSelection, isolate: boolean): MeasuredProfileProjection {
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
    cacheWrites: [], tensorSnapshots: [], tensorCapture: null,
    mappingStatus: "unavailable", mappingReason: unavailable, unmappedActivityCount: 0,
  }
  if (manifest?.run_id !== attachment.runId) throw new ProfileError("Profile attachment belongs to a different recording.")
  const selectedActivities = selection.stepId == null && selection.layerIndex == null && selection.component == null
    ? attachment.activities : attachment.activities.filter((activity) =>
    (selection.stepId == null || activity.scope.stepId === selection.stepId)
    && (selection.layerIndex == null || activity.scope.layerIndex === selection.layerIndex)
    && (selection.component == null || activity.scope.component === selection.component))
  const activities = isolate ? selectedActivities.map(activity => structuredClone(activity)) : selectedActivities
  const activeActivities = selection.timeUs == null ? [] : profileActivitiesAt(activities, selection.timeUs)
  const steps = isolate ? attachment.steps.map(step => structuredClone(step)) : attachment.steps
  const selectedStep = selection.stepId == null ? null : steps.find((step) => step.stepId === selection.stepId) ?? null
  const mappingReason = selectedStep?.mappingReason ?? (selection.stepId == null ? "Select a captured batch to inspect packed token rows."
    : selectedStep ? selectedStep.mappingStatus === "observed" ? null : "Packed-row mapping was not captured."
      : "This profile has no packed-row observation for the selected step.")
  const writes = attachment.cacheWrites.filter(operation =>
    (selection.stepId == null || operation.stepId === selection.stepId)
    && (selection.layerIndex == null || operation.layerIndex === selection.layerIndex))
  const snapshots = attachment.tensorSnapshots.filter(snapshot =>
    (selection.stepId == null || snapshot.stepId === selection.stepId)
    && (selection.layerIndex == null || snapshot.layerIndex === selection.layerIndex))
  return {
    ...common, status: "observed", reason: common.evidenceKind === "fixture" ? "Synthetic profiler fixture; these intervals are not GPU measurements."
      : metadata.status === "partial" || metadata.status === "failed"
      ? string(metadata.reason) ?? "Only the retained profiler intervals are shown; this capture did not complete." : null,
    clock: { ...attachment.clock }, extent: extent(attachment.activities), activities, activeActivities,
    cacheWrites: isolate ? structuredClone(writes) : writes,
    tensorSnapshots: isolate ? structuredClone(snapshots) : snapshots,
    tensorCapture: attachment.tensorCapture ? isolate ? structuredClone(attachment.tensorCapture) : attachment.tensorCapture : null,
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

function parseActivity(input: unknown, activityNames: string[] | null): ProfileActivity {
  const value = requiredRecord(input, "activity")
  let name: string
  if (activityNames) {
    if (value.name !== undefined) throw new ProfileError("Dictionary-encoded activity cannot also declare a name.")
    const nameId = integer(value.name_id, "activity.name_id")
    if (nameId >= activityNames.length) throw new ProfileError("Activity name ID is outside its dictionary.")
    name = activityNames[nameId]
  } else {
    if (value.name_id !== undefined) throw new ProfileError("Activity name ID requires its dictionary.")
    name = requiredString(value.name, "activity.name")
  }
  const kind = value.kind
  if (!["cpu_scope", "cuda_kernel", "cuda_memcpy", "cuda_memset", "other"].includes(String(kind))) throw new ProfileError("Unsupported profile activity kind.")
  const scope = requiredRecord(value.scope, "activity.scope")
  const correlation = scope.correlation
  if (!["explicit_scope", "runtime_flow", "unmapped"].includes(String(correlation))) throw new ProfileError("Unsupported profile scope correlation.")
  const stepId = nullableString(scope.step_id, "scope.step_id")
  const layerIndex = nullableInteger(scope.layer_index, "scope.layer_index")
  const component = scope.component
  if (component !== null && component !== "attention" && component !== "recurrent" && component !== "ffn") throw new ProfileError("Unsupported measured component scope.")
  if (correlation === "unmapped" && (stepId != null || layerIndex != null || component != null)) throw new ProfileError("Unmapped activity cannot claim a batch or layer scope.")
  if (correlation !== "unmapped" && stepId == null && layerIndex == null && component == null) throw new ProfileError("Mapped activity requires an observed scope.")
  if (String(kind).startsWith("cuda_") && correlation === "explicit_scope") throw new ProfileError("CUDA scope requires observed runtime flow correlation.")
  const operation = scope.operation ?? null
  const operationId = nullableString(scope.operation_id ?? null, "scope.operation_id")
  if (operation !== null && operation !== "kv_cache_write") throw new ProfileError("Unsupported measured operation scope.")
  if ((operation === null) !== (operationId === null)) throw new ProfileError("Measured operation requires its operation ID and kind together.")
  if (operation != null && (stepId == null || layerIndex == null || correlation === "unmapped")) throw new ProfileError("Cache-write activity requires a correlated step and layer.")
  const startUs = timestamp(value.start, "activity.start")
  const durationUs = timestamp(value.duration, "activity.duration")
  if (durationUs < 0 || !Number.isSafeInteger(Math.trunc(startUs + durationUs))) throw new ProfileError("Invalid measured activity duration.")
  const device = identity(value.device, "activity.device")
  const stream = identity(value.stream, "activity.stream")
  if (kind === "cpu_scope" && (device != null || stream != null)) throw new ProfileError("CPU scope cannot claim a CUDA device or stream.")
  return {
    id: requiredString(value.activity_id, "activity_id"), kind: kind as ProfileActivityKind,
    name, device, stream,
    processId: identity(value.process_id ?? null, "activity.process_id"), threadId: identity(value.thread_id ?? null, "activity.thread_id"),
    startUs, durationUs, endUs: startUs + durationUs,
    scope: { stepId, layerIndex, component, correlation: correlation as ProfileCorrelation,
      correlationSource: nullableString(scope.correlation_source ?? null, "scope.correlation_source"), operation, operationId },
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

function parseCacheWrite(input: unknown): ProfileCacheWrite {
  const value = requiredRecord(input, "cache write")
  if (value.status !== "observed" && value.status !== "unavailable") throw new ProfileError("Unsupported cache-write observation status.")
  const participants = array(value.participants, "cache_write.participants").map((input): ProfileCacheParticipant => {
    const row = requiredRecord(input, "cache-write participant")
    return {
      row: integer(row.row, "participant.row"), requestId: requiredString(row.request_id, "participant.request_id"),
      tokenPosition: integer(row.token_position, "participant.token_position"),
      pool: nullableString(row.pool ?? null, "participant.pool"), cacheGroup: nullableInteger(row.cache_group ?? null, "participant.cache_group"),
      blockId: nullableInteger(row.block_id ?? null, "participant.block_id"), allocationGeneration: nullableInteger(row.allocation_generation ?? null, "participant.allocation_generation"),
      slot: nullableInteger(row.slot ?? null, "participant.slot"), kernelBlockId: nullableInteger(row.kernel_block_id ?? null, "participant.kernel_block_id"),
      kernelSlot: nullableInteger(row.kernel_slot ?? null, "participant.kernel_slot"),
    }
  })
  unique(participants.map((participant) => String(participant.row)), "cache-write participant rows")
  unique(participants.map((participant) => JSON.stringify([participant.requestId, participant.tokenPosition])), "cache-write participant tokens")
  const gpuActivityIds = array(value.gpu_activity_ids, "cache_write.gpu_activity_ids").map((value) => requiredString(value, "cache_write.gpu_activity_id"))
  unique(gpuActivityIds, "cache-write GPU activity IDs")
  const reason = nullableString(value.reason ?? null, "cache_write.reason")
  if (value.status === "unavailable" && reason == null) throw new ProfileError("Unavailable cache-write observation requires a reason.")
  return {
    operationId: requiredString(value.operation_id, "cache_write.operation_id"), stepId: requiredString(value.step_id, "cache_write.step_id"),
    layerIndex: integer(value.layer_index, "cache_write.layer_index"), layerName: requiredString(value.layer_name, "cache_write.layer_name"),
    cacheGroup: nullableInteger(value.cache_group ?? null, "cache_write.cache_group"), pool: nullableString(value.pool ?? null, "cache_write.pool"),
    device: nullableString(value.device ?? null, "cache_write.device"), status: value.status, reason, participants, gpuActivityIds,
    observationBoundary: requiredString(value.observation_boundary, "cache_write.observation_boundary"),
  }
}

function parseTensorSnapshot(input: unknown): ProfileTensorSnapshot {
  const value = requiredRecord(input, "tensor snapshot")
  if (value.kind !== "K" && value.kind !== "V") throw new ProfileError("Tensor snapshot kind must be K or V.")
  const shape = array(value.shape, "snapshot.shape").map((dimension) => integer(dimension, "snapshot.shape dimension"))
  const values = array(value.values, "snapshot.values").map((value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) throw new ProfileError("Tensor snapshot values must be finite numbers.")
    return value
  })
  if (shape.length !== 1 || shape[0] !== values.length || !values.length) throw new ProfileError("Tensor snapshot shape must describe its nonempty retained dimension slice.")
  const byteCount = integer(value.byte_count, "snapshot.byte_count")
  if (!byteCount) throw new ProfileError("Tensor snapshot must account for its retained bytes.")
  const result: ProfileTensorSnapshot = {
    snapshotId: requiredString(value.snapshot_id, "snapshot.snapshot_id"), operationId: requiredString(value.operation_id, "snapshot.operation_id"),
    stepId: requiredString(value.step_id, "snapshot.step_id"), requestId: requiredString(value.request_id, "snapshot.request_id"),
    tokenPosition: integer(value.token_position, "snapshot.token_position"), layerIndex: integer(value.layer_index, "snapshot.layer_index"),
    cacheGroup: nullableInteger(value.cache_group ?? null, "snapshot.cache_group"), pool: nullableString(value.pool ?? null, "snapshot.pool"),
    blockId: nullableInteger(value.block_id ?? null, "snapshot.block_id"), allocationGeneration: nullableInteger(value.allocation_generation ?? null, "snapshot.allocation_generation"),
    slot: nullableInteger(value.slot ?? null, "snapshot.slot"), device: requiredString(value.device, "snapshot.device"), kvHead: integer(value.kv_head, "snapshot.kv_head"),
    kind: value.kind, dtype: requiredString(value.dtype, "snapshot.dtype"), shape, dimensionStart: integer(value.dimension_start, "snapshot.dimension_start"), values, byteCount,
    observationBoundary: requiredString(value.observation_boundary, "snapshot.observation_boundary"),
  }
  if (value.quantization != null) result.quantization = parseSnapshotQuantization(value.quantization, result)
  return result
}

/** Decodes the observed CUDA encoding; byte storage is never interpreted as integer-valued KV. */
export function decodeFp8Byte(byte: number, format: "fp8_e4m3" | "fp8_e5m2"): number {
  const sign = byte & 128 ? -1 : 1
  const mantissaBits = format === "fp8_e4m3" ? 3 : 2
  const exponent = byte >> mantissaBits & (format === "fp8_e4m3" ? 15 : 31)
  const fraction = byte & (2 ** mantissaBits - 1)
  const bias = format === "fp8_e4m3" ? 7 : 15
  if (format === "fp8_e4m3" && exponent === 15 && fraction === 7) return NaN
  if (format === "fp8_e5m2" && exponent === 31) return fraction === 0 ? sign * Infinity : NaN
  return sign * (exponent === 0 ? fraction / 2 ** mantissaBits * 2 ** (1 - bias) : (1 + fraction / 2 ** mantissaBits) * 2 ** (exponent - bias))
}

function parseSnapshotQuantization(input: unknown, snapshot: ProfileTensorSnapshot): NonNullable<ProfileTensorSnapshot["quantization"]> {
  const value = requiredRecord(input, "snapshot.quantization")
  if (snapshot.dtype !== "uint8" || value.format !== "fp8_e4m3" && value.format !== "fp8_e5m2" || value.reconstruction !== "decoded_fp8_times_recorded_scale") throw new ProfileError("Quantized numerical slices require qualified FP8 byte storage and reconstruction metadata.")
  const rawBits = array(value.raw_bits, "snapshot.raw_bits").map((item) => {
    const byte = integer(item, "snapshot.raw_bit")
    if (byte > 255) throw new ProfileError("Raw FP8 bytes must be in [0, 255].")
    return byte
  })
  if (rawBits.length !== snapshot.values.length || rawBits.some((byte, index) => decodeFp8Byte(byte, value.format as "fp8_e4m3" | "fp8_e5m2") !== snapshot.values[index])) throw new ProfileError("Decoded FP8 values must match the recorded raw byte encoding.")
  const scale = requiredRecord(value.scale, "snapshot.quantization.scale")
  const shape = array(scale.shape, "snapshot.scale.shape").map((dimension) => integer(dimension, "snapshot.scale.dimension"))
  const elements = shape.reduce((count, dimension) => count * dimension, 1)
  if (!Number.isSafeInteger(elements) || elements < 1) throw new ProfileError("FP8 scale shape must describe a finite nonempty tensor.")
  if (typeof scale.value !== "number" || !Number.isFinite(scale.value) || scale.value <= 0 || Math.fround(scale.value) !== scale.value || scale.dtype !== "float32") throw new ProfileError("FP8 reconstruction requires a positive finite observed float32 scale.")
  if (scale.mode !== "per_tensor" && scale.mode !== "per_head") throw new ProfileError("Unsupported FP8 scale mode.")
  const headIndex = nullableInteger(scale.head_index ?? null, "snapshot.scale.head_index")
  if (scale.mode === "per_tensor" && (elements !== 1 || headIndex != null) || scale.mode === "per_head" && (elements <= 1 || elements <= snapshot.kvHead || headIndex !== snapshot.kvHead)) throw new ProfileError("FP8 scale shape and head identity do not match the recorded slice.")
  const reconstructedValues = array(value.reconstructed_values, "snapshot.reconstructed_values").map((item, index) => {
    if (typeof item !== "number" || !Number.isFinite(item) || item !== snapshot.values[index] * (scale.value as number)) throw new ProfileError("Reconstructed FP8 values must equal decoded storage values times the recorded scale.")
    return item
  })
  if (reconstructedValues.length !== snapshot.values.length || snapshot.byteCount !== rawBits.length + 4) throw new ProfileError("Quantized snapshots must account for every retained raw byte and float32 scale.")
  if (scale.observation_boundary !== snapshot.observationBoundary) throw new ProfileError("FP8 bytes and scale must share the captured update boundary.")
  return {
    format: value.format, rawBits, reconstructedValues, reconstruction: value.reconstruction,
    scale: { value: scale.value, dtype: "float32", shape, mode: scale.mode, headIndex, observationBoundary: requiredString(scale.observation_boundary, "snapshot.scale.observation_boundary") },
  }
}

function parseTensorCapture(input: unknown): ProfileTensorCapture {
  const value = requiredRecord(input, "tensor capture")
  if (typeof value.enabled !== "boolean") throw new ProfileError("Tensor capture requires an explicit enabled flag.")
  const selection = requiredRecord(value.selection, "tensor_capture.selection")
  const indices = (value: unknown, name: string) => {
    const result = array(value, name).map((value) => integer(value, name))
    unique(result.map(String), name)
    return result
  }
  const maxBytes = integer(value.max_bytes, "tensor_capture.max_bytes")
  const retainedBytes = integer(value.retained_bytes, "tensor_capture.retained_bytes")
  if (retainedBytes > maxBytes) throw new ProfileError("Tensor capture exceeds its declared byte budget.")
  const dimensionCount = integer(selection.dimension_count, "tensor_capture.dimension_count")
  if (!dimensionCount) throw new ProfileError("Tensor capture dimension count must be positive.")
  return {
    enabled: value.enabled, status: requiredString(value.status, "tensor_capture.status"), reason: nullableString(value.reason ?? null, "tensor_capture.reason"), maxBytes, retainedBytes,
    selection: {
      requestId: nullableString(selection.request_id ?? null, "tensor_capture.request_id"), tokenPositions: indices(selection.token_positions, "tensor_capture.token_positions"),
      layerIndices: indices(selection.layer_indices, "tensor_capture.layer_indices"), headIndices: indices(selection.head_indices, "tensor_capture.head_indices"),
      dimensionStart: integer(selection.dimension_start, "tensor_capture.dimension_start"), dimensionCount,
    }, observationBoundary: nullableString(value.observation_boundary ?? null, "tensor_capture.observation_boundary"),
  }
}

function validateKvObservations(activities: ProfileActivity[], steps: ProfileStep[], writes: ProfileCacheWrite[], snapshots: ProfileTensorSnapshot[], capture: ProfileTensorCapture | null): void {
  const byActivity = new Map(activities.map((activity) => [activity.id, activity]))
  const byStep = new Map(steps.map((step) => [step.stepId, step]))
  const byWrite = new Map(writes.map((write) => [write.operationId, write]))
  for (const write of writes) {
    for (const id of write.gpuActivityIds) {
      const activity = byActivity.get(id)
      if (!activity || !activity.kind.startsWith("cuda_") || activity.scope.correlation !== "runtime_flow"
        || activity.scope.operation !== "kv_cache_write" || activity.scope.operationId !== write.operationId
        || activity.scope.stepId !== write.stepId || activity.scope.layerIndex !== write.layerIndex) {
        throw new ProfileError("Cache-write GPU references must identify correlated CUDA activities of the same operation, step and layer.")
      }
    }
    const step = byStep.get(write.stepId)
    for (const participant of write.participants) {
      if (write.pool != null && participant.pool !== write.pool || write.cacheGroup != null && participant.cacheGroup !== write.cacheGroup) throw new ProfileError("Cache-write participants must belong to the observed operation's pool and group.")
      const row = step?.packedRows.find((row) => row.row === participant.row)
      if (!step || step.mappingStatus !== "observed" || !row || row.isPadding
        || row.requestId !== participant.requestId || row.tokenPosition !== participant.tokenPosition) {
        throw new ProfileError("Cache-write participants must match observed nonpadding packed rows.")
      }
    }
  }
  for (const snapshot of snapshots) {
    const write = byWrite.get(snapshot.operationId)
    const participant = write?.participants.find((participant) => participant.requestId === snapshot.requestId && participant.tokenPosition === snapshot.tokenPosition)
    if (!write || !participant || write.stepId !== snapshot.stepId || write.layerIndex !== snapshot.layerIndex || write.device != null && write.device !== snapshot.device
      || ["pool", "cacheGroup", "blockId", "allocationGeneration", "slot"].some((field) => {
        const key = field as "pool" | "cacheGroup" | "blockId" | "allocationGeneration" | "slot"
        return participant[key] != null && snapshot[key] !== participant[key]
      })) throw new ProfileError("Tensor snapshot must match the observed cache-write token and allocation.")
    if (capture && (capture.selection.requestId != null && snapshot.requestId !== capture.selection.requestId
      || capture.selection.tokenPositions.length && !capture.selection.tokenPositions.includes(snapshot.tokenPosition)
      || capture.selection.layerIndices.length && !capture.selection.layerIndices.includes(snapshot.layerIndex)
      || capture.selection.headIndices.length && !capture.selection.headIndices.includes(snapshot.kvHead)
      || snapshot.dimensionStart < capture.selection.dimensionStart
      || snapshot.dimensionStart + snapshot.values.length > capture.selection.dimensionStart + capture.selection.dimensionCount)) {
      throw new ProfileError("Tensor snapshot lies outside its declared recording-time selection.")
    }
  }
  const retainedBytes = snapshots.reduce((sum, snapshot) => sum + snapshot.byteCount, 0)
  if (snapshots.length && (!capture?.enabled || retainedBytes !== capture.retainedBytes)) throw new ProfileError("Tensor snapshots require enabled capture and exact retained-byte accounting.")
  if (capture && retainedBytes !== capture.retainedBytes) throw new ProfileError("Tensor capture retained bytes do not match its snapshots.")
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
