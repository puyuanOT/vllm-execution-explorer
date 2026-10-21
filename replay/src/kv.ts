/** Cache configuration and diagnostic evidence. Host and profiler clocks never mix. */
import { projectArchitecture, projectKvCache, projectRequestTokens, type KvPoolProjection, type ObservationStatus } from "./learning.ts"
import { ProfileError, type ProfileActivity, type ProfileKvObservations, type ProfileTensorCapture, type ProfileTensorSnapshot } from "./profiling.ts"
import { profileObservationIndex } from "./profileIndex.ts"
import type { ContentRecord, RecordingManifest, ReplayState } from "./reduce.ts"

export type KvEvidenceKind = "gpu" | "fixture" | "unknown"

export interface KvStorageConfiguration {
  storageId: string
  device: string
  dtype: string
  byteCount: number
}

export interface KvLayerConfiguration {
  layerIndex: number
  layerName: string
  pool: string | null
  cacheGroup: number | null
  device: string
  dtype: string
  shape: number[]
  strides: number[]
  storageId: string
  storageOffset: number
  logicalBytes: number
  backend: string
  layout: string
  layoutStatus: "qualified" | "unavailable"
  layoutReason: string | null
  managerBlockSize: number | null
  kernelBlockSize: number | null
  kernelBlocksPerManagerBlock: number | null
  numKernelBlocks: number | null
  numKvHeads: number | null
  headDim: number | null
  quantization: {
    cacheDtype: string | null
    mode: string | null
    scales: Record<"K" | "V", { status: "observed" | "unavailable"; shape: number[]; dtype: string | null; device: string | null }>
  } | null
}

export interface KvConfiguredPool extends KvPoolProjection {
  device: string | null
  devices: string[]
  dtypes: string[]
  layerIndices: number[]
  storageBytes: number | null
  usableCapacity: number
  tokenCapacity: number | null
  referencedBlocks: number | null
  cachedBlocks: number | null
  freeBlocks: number | null
  availableBlocks: number | null
  unknownBlocks: number
}

export interface KvConfigurationProjection {
  status: ObservationStatus
  reason: string | null
  evidenceKind: KvEvidenceKind
  uniqueStorageBytes: number | null
  storages: KvStorageConfiguration[]
  layers: KvLayerConfiguration[]
  pools: KvConfiguredPool[]
  requestedDtype: string | null
  maxModelLength: number | null
  prefixCaching: boolean | null
  observationBoundary: string | null
}

export type LayerTokenState = "scheduled" | "operation_active" | "completed" | "reused_existing_kv" | "not_participating" | "unknown"

export interface LayerTokenSelection {
  stepId?: string | null
  requestId?: string | null
  tokenPosition?: number | null
  layerIndex?: number | null
  /** Independent native profiler clock. Omit for host/teaching playback. */
  timeUs?: number | null
  maxTokens?: number
  content?: ContentRecord[]
}

export interface LayerTokenMatrixProjection {
  status: ObservationStatus
  reason: string | null
  evidenceKind: KvEvidenceKind
  stepId: string | null
  layers: number[]
  tokens: Array<{ requestId: string; position: number; tokenId: number | null; tokenPiece: string | null; phase: "prefill" | "decode"; scheduled: boolean }>
  cells: Array<{ requestId: string; position: number; layerIndex: number; state: LayerTokenState; reason: string | null; operationIds: string[]; gpuActivityIds: string[] }>
  truncated: boolean
}

export interface KvTensorSelection {
  stepId?: string | null
  requestId?: string | null
  tokenPosition?: number | null
  layerIndex?: number | null
  kvHead?: number | null
  kind?: "K" | "V" | null
  pool?: string | null
  cacheGroup?: number | null
  blockId?: number | null
  generation?: number | null
  slot?: number | null
  /** Origin writer, when a shared prefix belongs to another request. */
  sourceRequestId?: string | null
}

export interface KvTensorPreviewProjection {
  status: ObservationStatus
  reason: string | null
  evidenceKind: KvEvidenceKind
  snapshots: ProfileTensorSnapshot[]
  capture: ProfileTensorCapture | null
  resolvedByAllocation: boolean
  requestedRequestId: string | null
}

/** Physical allocations are counted once; layer views never multiply capacity or bytes. */
export function projectKvConfiguration(state: ReplayState, manifest?: RecordingManifest | null, content: ContentRecord[] = []): KvConfigurationProjection {
  const cache = projectKvCache(state, manifest, content)
  const metadata = parseKvConfiguration(manifest)
  const settings = object(manifest?.engine_settings)
  return {
    ...metadata, evidenceKind: evidenceKind(manifest),
    reason: state.trust === "unknown" ? "Configuration is retained; current occupancy is unknown after a trace gap." : metadata.reason,
    requestedDtype: optionalString(settings.kv_cache_dtype), maxModelLength: optionalPositiveInteger(settings.max_model_len),
    prefixCaching: typeof settings.enable_prefix_caching === "boolean" ? settings.enable_prefix_caching : null,
    pools: cache.pools.map((pool) => {
      const layers = metadata.layers.filter((layer) => layer.pool === pool.pool)
      const recordedGroups = object(manifest?.cache_geometry).groups ?? object(manifest?.architecture).kv_groups
      const declaredLayers = (Array.isArray(recordedGroups) ? recordedGroups : []).flatMap((input) => {
        const group = object(input)
        if (group.pool !== pool.pool || !Array.isArray(group.layer_names)) return []
        return group.layer_names.flatMap((name) => {
          const match = typeof name === "string" ? /(?:^|\.)layers\.(\d+)(?:\.|$)/.exec(name) : null
          const index = match ? Number(match[1]) : null
          return index != null && Number.isSafeInteger(index) && index <= 2_000_000 ? [index] : []
        })
      })
      const devices = [...new Set(layers.map((layer) => layer.device))]
      const storageIds = new Set(layers.map((layer) => layer.storageId))
      const usableCapacity = pool.capacity - pool.blocks.filter((block) => block.isNull).length
      const known = pool.blocks.every((block) => block.status !== "unknown")
      return {
        ...pool, device: devices.length === 1 ? devices[0] : null, devices, dtypes: [...new Set(layers.map((layer) => layer.dtype))],
        layerIndices: [...new Set(layers.length ? layers.map((layer) => layer.layerIndex) : declaredLayers)].sort((a, b) => a - b),
        storageBytes: storageIds.size ? metadata.storages.filter((storage) => storageIds.has(storage.storageId)).reduce((sum, storage) => sum + storage.byteCount, 0) : null,
        usableCapacity, tokenCapacity: pool.blockSize == null ? null : usableCapacity * pool.blockSize,
        referencedBlocks: known ? pool.blocks.filter((block) => !block.isNull && block.status === "allocated").length : null,
        cachedBlocks: known ? pool.blocks.filter((block) => !block.isNull && block.status === "cached").length : null,
        freeBlocks: known ? pool.blocks.filter((block) => !block.isNull && block.status === "free").length : null,
        availableBlocks: known ? pool.blocks.filter((block) => !block.isNull && (block.status === "free" || block.status === "cached")).length : null,
        unknownBlocks: pool.blocks.filter((block) => !block.isNull && block.status === "unknown").length,
      }
    }),
  }
}

/** Batch participation is shared by every token in an operation; no per-token duration is invented. */
export function projectLayerTokenMatrix(state: ReplayState, manifest?: RecordingManifest | null, observations?: ProfileKvObservations | null, selection: LayerTokenSelection = {}): LayerTokenMatrixProjection {
  if (selection.timeUs != null && !Number.isFinite(selection.timeUs)) throw new ProfileError("Profiler playhead must be finite trace-local microseconds.")
  const selectedStep = selection.stepId ?? state.steps.at(-1)?.id ?? null
  const step = state.steps.find((step) => step.id === selectedStep)
  const profileIndex = profileObservationIndex(observations)
  const profileStep = selectedStep == null ? undefined : profileIndex?.stepsById.get(selectedStep)
  const configuration = parseKvConfiguration(manifest)
  const architecture = projectArchitecture(manifest)
  const allLayers = architecture.layerCount != null ? Array.from({ length: architecture.layerCount }, (_, index) => index)
    : [...new Set([...configuration.layers.map((layer) => layer.layerIndex), ...(observations?.cacheWrites.map((write) => write.layerIndex) ?? [])])].sort((a, b) => a - b)
  const layers = selection.layerIndex == null ? allLayers : allLayers.filter((layer) => layer === selection.layerIndex)
  const maxTokens = selection.maxTokens ?? 128
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) throw new ProfileError("Layer-token matrix token limit must be a positive integer.")
  let truncated = false
  const positions = new Map<string, { requestId: string; position: number; scheduled: boolean }>()
  const add = (requestId: string, position: number, scheduled: boolean) => {
    if (selection.requestId != null && requestId !== selection.requestId || selection.tokenPosition != null && position !== selection.tokenPosition) return
    const key = JSON.stringify([requestId, position])
    const previous = positions.get(key)
    if (!previous && positions.size >= maxTokens) { truncated = true; return }
    positions.set(key, { requestId, position, scheduled: scheduled || previous?.scheduled === true })
  }
  const addRange = (requestId: string, start: number, end: number, scheduled: boolean) => {
    if (selection.requestId != null && requestId !== selection.requestId) return
    if (selection.tokenPosition != null) {
      if (selection.tokenPosition >= start && selection.tokenPosition < end) add(requestId, selection.tokenPosition, scheduled)
      return
    }
    const limit = Math.min(end, start + maxTokens + 1)
    for (let position = start; position < limit; position += 1) add(requestId, position, scheduled)
    if (limit < end) truncated = true
  }
  for (const request of step?.requests ?? []) {
    addRange(request.requestId, request.rangeStart, request.rangeStart + request.rangeLength, true)
  }
  for (const row of profileStep?.packedRows ?? []) if (!row.isPadding && row.requestId != null && row.tokenPosition != null) add(row.requestId, row.tokenPosition, true)
  // Prioritize scheduled inputs; earlier positions fill remaining space for prefix reuse.
  for (const request of step?.requests ?? []) addRange(request.requestId, 0, request.rangeStart, false)
  if (selection.requestId != null && selection.tokenPosition != null) add(selection.requestId, selection.tokenPosition, positions.get(JSON.stringify([selection.requestId, selection.tokenPosition]))?.scheduled === true)
  const selectedPositions = [...positions.values()].sort((a, b) => a.requestId.localeCompare(b.requestId) || a.position - b.position)
  const tokenProjections = new Map<string, ReturnType<typeof projectRequestTokens>>()
  for (const { requestId } of selectedPositions.slice(0, maxTokens)) if (!tokenProjections.has(requestId)) tokenProjections.set(requestId, projectRequestTokens(state, requestId, selection.content))
  const tokens = selectedPositions.slice(0, maxTokens).map((position) => {
    const token = tokenProjections.get(position.requestId)?.[position.position]
    const request = state.requests[position.requestId]
    return { ...position, tokenId: token?.tokenId ?? null, tokenPiece: token?.tokenPiece ?? null,
      phase: position.position < (request?.promptTokenCount ?? Infinity) ? "prefill" as const : "decode" as const }
  })
  const writes = selectedStep == null ? [] : profileIndex?.writesByStep.get(selectedStep) ?? []
  const cells = tokens.flatMap((token) => layers.map((layerIndex) => {
    const operations = writes.filter((write) => write.layerIndex === layerIndex && write.participants.some((participant) => participant.requestId === token.requestId && participant.tokenPosition === token.position))
    const gpuActivityIds = [...new Set(operations.flatMap((write) => write.gpuActivityIds))]
    const gpu = gpuActivityIds.map((id) => profileIndex?.activitiesById.get(id)).filter((activity): activity is ProfileActivity => activity != null && activity.kind.startsWith("cuda_") && activity.scope.correlation === "runtime_flow")
    let cellState: LayerTokenState = "unknown"
    let reason: string | null = null
    if (state.trust === "unknown") reason = "A trace gap prevents a current linked token/cache claim."
    else if (token.scheduled || operations.length) {
      cellState = "scheduled"
      reason = "Scheduled batch participation; layer cache-write timing was not captured."
      if (selection.timeUs != null && !operations.length) {
        cellState = "unknown"
        reason = "This recording has no cache-write observation for the selected token and layer."
      }
      if (selection.timeUs != null && operations.length) {
        if (operations.some((operation) => operation.status !== "observed") || !gpu.length || operations.some((operation) => !operation.gpuActivityIds.length)) {
          cellState = "unknown"
          reason = operations.find((operation) => operation.reason)?.reason ?? "The CPU write boundary does not establish a correlated GPU completion."
        } else if (gpu.some((activity) => activity.startUs <= selection.timeUs! && activity.endUs > selection.timeUs!)) {
          cellState = "operation_active"
          reason = "This token participates in the active batch cache-write operation; the interval is shared by its participants."
        } else if (gpu.every((activity) => activity.endUs <= selection.timeUs!)) {
          cellState = "completed"
          reason = "All retained correlated GPU cache-write activities for this token/layer operation have ended; no per-token duration is implied."
        } else reason = "Scheduled; the correlated GPU cache-write operation has not started at this profiler time."
      }
    } else {
      const projectedToken = tokenProjections.get(token.requestId)?.[token.position]
      const layer = configuration.layers.find((layer) => layer.layerIndex === layerIndex)
      const group = layer?.cacheGroup != null ? state.cacheGroups.find((group) => group.cacheGroup === layer.cacheGroup && group.pool === layer.pool)
        : state.cacheGroups.find((group) => group.layerNames.some((name) => new RegExp(`^model\\.layers\\.${layerIndex}\\.self_attn(?:\\.attn)?$`).test(name)))
      const computedGroup = projectedToken?.cacheGroups.find((candidate) => candidate.cacheGroup === group?.cacheGroup && candidate.pool === group?.pool)
      const earlier = step?.requests.find((request) => request.requestId === token.requestId)
      const existingContents = projectedToken?.locations.some((location) => {
        const block = state.blocks[location.key]
        return location.valid && location.cacheGroup === group?.cacheGroup && location.pool === group?.pool
          && block?.generation === location.generation && block.contents.some((span) => span.cacheGroup === location.cacheGroup
            && span.stepId !== selectedStep && location.slot >= span.slotStart && location.slot < span.slotEnd)
      })
      if (earlier && token.position < earlier.rangeStart && group
        && (group.tokenSlotMapping === "qualified_full_attention" || group.tokenSlotMapping === "authored_fixture")
        && computedGroup?.mappingStatus === "observed" && computedGroup.kvState === "computed" && existingContents) {
        cellState = "reused_existing_kv"
        reason = "Previously computed KV for this cache group is bound to the request; this position is outside the scheduled suffix."
      } else if (step && projectedToken && (!earlier || token.position >= earlier.rangeStart + earlier.rangeLength)) {
        cellState = "not_participating"
        reason = "This request position is outside the selected step's scheduled input ranges."
      } else reason = "Existing per-layer KV availability was not established for this position."
    }
    return { requestId: token.requestId, position: token.position, layerIndex, state: cellState, reason,
      operationIds: operations.map((operation) => operation.operationId), gpuActivityIds }
  }))
  return {
    status: state.trust === "unknown" ? "unknown" : selectedStep && tokens.length && layers.length ? "observed" : "unavailable",
    reason: state.trust === "unknown" ? "Linked state is frozen before a trace gap."
      : !selectedStep ? "Select a scheduler step to inspect request-token participation."
      : !layers.length ? "Layer structure was not recorded." : !tokens.length ? "No participating request-token positions were recorded for this step."
      : evidenceKind(manifest) === "fixture" ? "Synthetic participation; these are not measured GPU intervals."
      : selection.timeUs == null ? "Host/teaching playback shows scheduled participation. GPU write chronology uses the independent profiler clock." : null,
    evidenceKind: evidenceKind(manifest), stepId: selectedStep, layers, tokens, cells, truncated,
  }
}

/** Snapshots are immutable recording-time slices, never a live read of replayed cache memory. */
export function projectKvTensorPreview(manifest?: RecordingManifest | null, observations?: ProfileKvObservations | null, selection: KvTensorSelection = {}): KvTensorPreviewProjection {
  const capture = observations?.tensorCapture ? structuredClone(observations.tensorCapture) : null
  const metadata = object(manifest?.profiling)
  const omitted = metadata.attachment_status === "omitted_by_structural_export"
  const resolvedByAllocation = selection.pool != null && selection.blockId != null && selection.generation != null && selection.slot != null
  const snapshots = omitted ? [] : (observations?.tensorSnapshots ?? []).filter((snapshot) =>
    (resolvedByAllocation || selection.stepId == null || snapshot.stepId === selection.stepId)
    && (resolvedByAllocation || selection.requestId == null || snapshot.requestId === selection.requestId)
    && (resolvedByAllocation || selection.tokenPosition == null || snapshot.tokenPosition === selection.tokenPosition)
    && (selection.sourceRequestId == null || snapshot.requestId === selection.sourceRequestId)
    && (selection.layerIndex == null || snapshot.layerIndex === selection.layerIndex)
    && (selection.kvHead == null || snapshot.kvHead === selection.kvHead)
    && (selection.kind == null || snapshot.kind === selection.kind)
    && (selection.pool == null || snapshot.pool === selection.pool)
    && (selection.cacheGroup == null || snapshot.cacheGroup === selection.cacheGroup)
    && (selection.blockId == null || snapshot.blockId === selection.blockId)
    && (selection.generation == null || snapshot.allocationGeneration === selection.generation)
    && (selection.slot == null || snapshot.slot === selection.slot)).map((snapshot) => structuredClone(snapshot))
  return {
    status: snapshots.length ? "observed" : "unavailable", evidenceKind: evidenceKind(manifest), snapshots, capture, resolvedByAllocation, requestedRequestId: selection.requestId ?? null,
    reason: omitted ? "Numerical/profile attachments were omitted by the structural export."
      : snapshots.length ? evidenceKind(manifest) === "fixture" ? "Synthetic numerical fixture; these values were not read from a GPU."
        : resolvedByAllocation ? "Recorded slices for this exact pool/block/allocation generation/slot. Original writer and capture step are retained; shared-prefix holders may differ. These are historical observations, not a live cache read."
        : "Recorded slices at the stated observation boundary; they do not represent live cache contents."
      : !observations ? "The diagnostic attachment is unavailable."
      : !capture ? "This recording predates numerical tensor capture."
      : !capture.enabled ? "Numerical tensor capture was disabled for this recording."
      : capture.reason ?? "This token/layer/head slice was not captured. Select targets before recording or run a targeted capture.",
  }
}

function parseKvConfiguration(manifest?: RecordingManifest | null): Omit<KvConfigurationProjection, "evidenceKind" | "pools" | "requestedDtype" | "maxModelLength" | "prefixCaching"> {
  const value = object(manifest?.kv_cache)
  if (!Object.keys(value).length) return { status: "unavailable", reason: "Actual KV tensor placement, dtype and storage were not captured in this recording.", uniqueStorageBytes: null, storages: [], layers: [], observationBoundary: null }
  if (value.contract_version !== 1) return { status: "unavailable", reason: "This recording has an unsupported KV configuration contract.", uniqueStorageBytes: null, storages: [], layers: [], observationBoundary: null }
  if (value.status !== "observed" && value.status !== "unavailable") throw new ProfileError("Unsupported KV configuration status.")
  const storages = list(value.storages ?? [], "kv_cache.storages").map((input): KvStorageConfiguration => {
    const row = requiredObject(input, "KV storage")
    return { storageId: requiredString(row.storage_id, "storage_id"), device: requiredString(row.device, "storage.device"), dtype: requiredString(row.dtype, "storage.dtype"), byteCount: integer(row.byte_count, "storage.byte_count") }
  })
  if (new Set(storages.map((storage) => storage.storageId)).size !== storages.length) throw new ProfileError("KV storage IDs must be unique.")
  const layers = list(value.layers ?? [], "kv_cache.layers").map((input): KvLayerConfiguration => {
    const row = requiredObject(input, "KV layer")
    if (row.layout_status !== "qualified" && row.layout_status !== "unavailable") throw new ProfileError("Unsupported KV layout qualification.")
    const shape = list(row.shape, "layer.shape").map((dimension) => integer(dimension, "layer.shape dimension"))
    const strides = list(row.strides, "layer.strides").map((dimension) => integer(dimension, "layer.stride"))
    if (shape.length !== strides.length || !shape.length) throw new ProfileError("KV tensor shape and strides must have equal nonzero rank.")
    const result = {
      layerIndex: integer(row.layer_index, "layer_index"), layerName: requiredString(row.layer_name, "layer_name"), pool: nullableString(row.pool ?? null, "layer.pool"),
      cacheGroup: nullableInteger(row.cache_group ?? null, "layer.cache_group"), device: requiredString(row.device, "layer.device"), dtype: requiredString(row.dtype, "layer.dtype"), shape, strides,
      storageId: requiredString(row.storage_id, "layer.storage_id"), storageOffset: integer(row.storage_offset, "layer.storage_offset"), logicalBytes: integer(row.logical_bytes, "layer.logical_bytes"),
      backend: requiredString(row.backend, "layer.backend"), layout: requiredString(row.layout, "layer.layout"), layoutStatus: row.layout_status,
      layoutReason: nullableString(row.layout_reason ?? null, "layer.layout_reason"), managerBlockSize: nullablePositiveInteger(row.manager_block_size ?? null, "manager_block_size"),
      kernelBlockSize: nullablePositiveInteger(row.kernel_block_size ?? null, "kernel_block_size"), kernelBlocksPerManagerBlock: nullablePositiveInteger(row.kernel_blocks_per_manager_block ?? null, "kernel_blocks_per_manager_block"),
      numKernelBlocks: nullableInteger(row.num_kernel_blocks ?? null, "num_kernel_blocks"), numKvHeads: nullablePositiveInteger(row.num_kv_heads ?? null, "num_kv_heads"), headDim: nullablePositiveInteger(row.head_dim ?? null, "head_dim"),
      quantization: row.quantization == null ? null : parseQuantization(row.quantization),
    } as KvLayerConfiguration
    const storage = storages.find((storage) => storage.storageId === result.storageId)
    if (!storage || storage.device !== result.device || storage.dtype !== "mixed" && storage.dtype !== result.dtype) throw new ProfileError("KV layer must reference compatible recorded storage.")
    if (result.layoutStatus === "qualified" && [result.managerBlockSize, result.kernelBlockSize, result.kernelBlocksPerManagerBlock, result.numKernelBlocks, result.numKvHeads, result.headDim].some((value) => value == null)) throw new ProfileError("Qualified KV layout requires its block/head dimensions.")
    return result
  })
  if (new Set(layers.map((layer) => layer.layerName)).size !== layers.length) throw new ProfileError("KV layer names must be unique.")
  const uniqueStorageBytes = nullableInteger(value.unique_storage_bytes ?? null, "unique_storage_bytes")
  if (uniqueStorageBytes != null && uniqueStorageBytes !== storages.reduce((sum, storage) => sum + storage.byteCount, 0)) throw new ProfileError("KV unique storage bytes must count every physical storage once.")
  if (value.status === "observed" && (!storages.length || !layers.length || uniqueStorageBytes == null)) throw new ProfileError("Observed KV configuration requires storage and layer observations.")
  return { status: value.status, reason: nullableString(value.reason ?? null, "kv_cache.reason"), uniqueStorageBytes, storages, layers,
    observationBoundary: nullableString(value.observation_boundary ?? null, "kv_cache.observation_boundary") }
}

function parseQuantization(input: unknown): NonNullable<KvLayerConfiguration["quantization"]> {
  const value = requiredObject(input, "layer.quantization")
  const scales = requiredObject(value.scales, "quantization.scales")
  const scale = (kind: "K" | "V") => {
    const row = requiredObject(scales[kind], `quantization.scales.${kind}`)
    if (row.status !== "observed" && row.status !== "unavailable") throw new ProfileError("Unsupported quantization scale observation status.")
    const result = { status: row.status as "observed" | "unavailable", shape: list(row.shape, "scale.shape").map((dimension) => integer(dimension, "scale.shape dimension")),
      dtype: nullableString(row.dtype ?? null, "scale.dtype"), device: nullableString(row.device ?? null, "scale.device") }
    if (result.status === "observed" && (result.dtype == null || result.device == null)) throw new ProfileError("Observed quantization scale requires tensor dtype and device.")
    return result
  }
  return { cacheDtype: nullableString(value.cache_dtype ?? null, "quantization.cache_dtype"), mode: nullableString(value.mode ?? null, "quantization.mode"), scales: { K: scale("K"), V: scale("V") } }
}

function evidenceKind(manifest?: RecordingManifest | null): KvEvidenceKind { return manifest?.execution_kind === "gpu" ? "gpu" : manifest?.execution_kind === "fixture" ? "fixture" : "unknown" }
function object(value: unknown): Record<string, unknown> { return value != null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {} }
function requiredObject(value: unknown, name: string): Record<string, unknown> { if (value == null || typeof value !== "object" || Array.isArray(value)) throw new ProfileError(`${name} must be an object.`); return value as Record<string, unknown> }
function list(value: unknown, name: string): unknown[] { if (!Array.isArray(value)) throw new ProfileError(`${name} must be an array.`); return value }
function optionalString(value: unknown): string | null { return typeof value === "string" && value ? value : null }
function requiredString(value: unknown, name: string): string { const string = optionalString(value); if (string == null) throw new ProfileError(`${name} must be a nonempty string.`); return string }
function nullableString(value: unknown, name: string): string | null { return value === null ? null : requiredString(value, name) }
function integer(value: unknown, name: string): number { if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw new ProfileError(`${name} must be a nonnegative safe integer.`); return value }
function nullableInteger(value: unknown, name: string): number | null { return value === null ? null : integer(value, name) }
function nullablePositiveInteger(value: unknown, name: string): number | null { const result = nullableInteger(value, name); if (result === 0) throw new ProfileError(`${name} must be positive or null.`); return result }
function optionalPositiveInteger(value: unknown): number | null { return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null }
