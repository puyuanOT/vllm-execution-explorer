/** Observed learning projections. These helpers never supply a GPU clock. */
import { attachContent, type CacheGroupInventory, type ContentRecord, type KvBinding, type KvWrittenRange, type RecordingManifest, type ReplayState, type SchedulerQueues } from "./reduce.ts"

export type ObservationStatus = "observed" | "unavailable" | "unknown"

export interface ArchitectureProjection {
  supported: boolean
  reason: string | null
  modelType: string | null
  activation?: string | null
  attentionOutputGate?: boolean
  source: string | null
  layerCount: number | null
  hiddenSize: number | null
  queryHeads: number | null
  kvHeads: number | null
  headDim: number | null
  intermediateSize: number | null
  layerTypes: Array<"full_attention" | "linear_attention">
  recurrent: { keyHeads: number; valueHeads: number; keyHeadDim: number; valueHeadDim: number; convKernelSize: number } | null
}

export interface SchedulerProjection {
  status: ObservationStatus
  reason: string | null
  stepId: string | null
  policy: string | null
  before: SchedulerQueues | null
  after: SchedulerQueues | null
}

export interface KvBlockProjection {
  key: string
  pool: string
  blockId: number
  generation: number
  isNull: boolean
  status: "free" | "allocated" | "cached" | "unknown"
  refCount: number
  hasHash: boolean
  bindings: KvBinding[]
  contents: KvWrittenRange[]
  tokenOccupancyStatus: ObservationStatus
  tokenOccupancyReason: string | null
  validTokenCount: number | null
  validTokenCountsByGroup: Record<string, number | null>
  trust: "known" | "unknown"
}

export interface KvPoolProjection {
  pool: string
  blockSize: number | null
  capacity: number
  blocks: KvBlockProjection[]
  groups: CacheGroupInventory[]
  tokenOccupancyStatus: ObservationStatus
  tokenOccupancyReason: string | null
}

export interface KvCacheProjection {
  status: ObservationStatus
  reason: string | null
  evidenceKind?: "gpu" | "fixture" | "unknown"
  pools: KvPoolProjection[]
  groups: CacheGroupInventory[]
}

export interface TokenLocation {
  key: string
  pool: string
  blockId: number
  generation: number
  cacheGroup: number
  slot: number
  sourceRequestId: string | null
  valid: boolean
}

export interface RequestTokenProjection {
  position: number
  kind: "prompt" | "output"
  tokenId: number | null
  tokenPiece: string | null
  kvState: "computed" | "reserved" | "not_computed" | "unknown"
  computedEver: boolean | null
  kvReason: string | null
  cacheGroups: Array<{
    cacheGroup: number
    pool: string | null
    mappingStatus: ObservationStatus
    reason: string | null
    kvState: RequestTokenProjection["kvState"]
    computedEver: boolean | null
  }>
  locations: TokenLocation[]
}

export function projectArchitecture(manifest?: RecordingManifest | null): ArchitectureProjection {
  const architecture = object(manifest?.architecture)
  const value: ArchitectureProjection = {
    supported: false,
    reason: typeof architecture.reason === "string" ? architecture.reason : null,
    modelType: string(architecture.model_type),
    activation: string(architecture.activation),
    attentionOutputGate: architecture.attention_output_gate === true,
    source: string(architecture.source),
    layerCount: positiveInteger(architecture.num_layers),
    hiddenSize: positiveInteger(architecture.hidden_size),
    queryHeads: positiveInteger(architecture.num_attention_heads),
    kvHeads: positiveInteger(architecture.num_key_value_heads),
    headDim: positiveInteger(architecture.head_dim),
    intermediateSize: positiveInteger(architecture.intermediate_size),
    layerTypes: [],
    recurrent: null,
  }
  value.supported = architecture.supported === true
    && [value.layerCount, value.hiddenSize, value.queryHeads, value.kvHeads, value.headDim, value.intermediateSize].every((dimension) => dimension != null)
  if (value.modelType === "qwen3_5_text") {
    const types = architecture.layer_types
    const recurrent = object(architecture.recurrent)
    const dimensions = ["key_heads", "value_heads", "key_head_dim", "value_head_dim", "conv_kernel_size"].map((key) => positiveInteger(recurrent[key]))
    const valid = Array.isArray(types) && types.length === value.layerCount
      && types.every((kind) => kind === "full_attention" || kind === "linear_attention") && dimensions.every((dimension) => dimension != null)
    value.supported = value.supported && valid
    if (valid) {
      value.layerTypes = types as ArchitectureProjection["layerTypes"]
      value.recurrent = { keyHeads: dimensions[0]!, valueHeads: dimensions[1]!, keyHeadDim: dimensions[2]!, valueHeadDim: dimensions[3]!, convKernelSize: dimensions[4]! }
    }
  } else if (value.supported && value.layerCount != null) {
    value.layerTypes = Array.from({ length: value.layerCount }, () => "full_attention")
  }
  if (!value.supported && !value.reason) value.reason = "This recording has no supported model architecture observation."
  return value
}

export function projectScheduler(state: ReplayState, stepId?: string | null): SchedulerProjection {
  const step = stepId ? state.steps.find((item) => item.id === stepId) : state.steps.at(-1)
  const observed = step?.queueBefore != null && step.queueAfter != null
  return {
    status: state.trust === "unknown" ? "unknown" : observed ? "observed" : "unavailable",
    reason: state.trust === "unknown" ? "A trace gap prevents a current queue claim."
      : observed ? null : "Ordered scheduler queues were not captured at this step.",
    stepId: step?.id ?? null,
    policy: step?.policy ?? null,
    before: step?.queueBefore ? structuredClone(step.queueBefore) : null,
    after: step?.queueAfter ? structuredClone(step.queueAfter) : null,
  }
}

export function projectKvCache(state: ReplayState, manifest?: RecordingManifest | null, content: ContentRecord[] = []): KvCacheProjection {
  const observed = content.length ? attachContent(state, content) : state
  const evidenceKind = manifest?.execution_kind === "gpu" ? "gpu" as const : manifest?.execution_kind === "fixture" ? "fixture" as const : "unknown" as const
  if (!observed.cacheInventory) return {
    evidenceKind,
    status: state.trust === "unknown" ? "unknown" : "unavailable",
    reason: state.trust === "unknown"
      ? "A trace gap ends the trusted prefix before a full KV pool inventory was observed."
      : "No full KV pool inventory has been observed up to this playhead. Captures from the vLLM 0.30 instrumentation record it when the engine starts.",
    pools: [],
    groups: [],
  }
  return {
    evidenceKind,
    status: state.trust === "unknown" ? "unknown" : "observed",
    reason: state.trust === "unknown" ? "KV state is frozen at the last observation before a trace gap."
      : observed.cacheGroups.some((group) => !tokenMappingAvailable(group)) ? "Some cache groups have no qualified token-slot mapping; physical inventory remains observed." : null,
    groups: structuredClone(observed.cacheGroups),
    pools: observed.cacheInventory.map((pool) => {
      const groups = observed.cacheGroups.filter((group) => group.pool === pool.pool)
      const poolMapping = occupancyAvailability(groups, null)
      return {
      ...pool,
      groups: structuredClone(groups),
      tokenOccupancyStatus: state.trust === "unknown" ? "unknown" as const : poolMapping.status,
      tokenOccupancyReason: state.trust === "unknown" ? "A trace gap prevents a current token occupancy claim." : poolMapping.reason,
      blocks: Array.from({ length: pool.capacity }, (_, blockId) => observed.blocks[`${pool.pool}:${blockId}`] ?? {
        key: `${pool.pool}:${blockId}`, pool: pool.pool, blockId, generation: 0,
        refCount: 0, hasHash: false, isNull: false,
        trust: state.trust, bindings: [], contents: [],
      })
        .map((block) => {
          const mapping = occupancyAvailability(groups, block)
          const slots = new Set<number>()
          const byGroup: Record<string, Set<number>> = {}
          for (const span of block.contents) {
            const group = byGroup[span.cacheGroup] ??= new Set()
            for (let slot = span.slotStart; slot < span.slotEnd; slot += 1) {
              slots.add(slot)
              group.add(slot)
            }
          }
          return {
            key: block.key, pool: block.pool, blockId: block.blockId,
            generation: block.generation, isNull: block.isNull,
            status: block.trust === "unknown" ? "unknown" as const
              : block.refCount > 0 ? "allocated" as const : block.hasHash ? "cached" as const : "free" as const,
            refCount: block.refCount, hasHash: block.hasHash,
            bindings: structuredClone(block.bindings), contents: structuredClone(block.contents),
            tokenOccupancyStatus: block.trust === "unknown" ? "unknown" as const : mapping.status,
            tokenOccupancyReason: block.trust === "unknown" ? "Token contents are frozen before a trace gap; current occupancy is unknown." : mapping.reason,
            validTokenCount: block.trust === "known" && mapping.status === "observed" ? slots.size : null,
            validTokenCountsByGroup: Object.fromEntries(groups.map((group) => [group.cacheGroup,
              block.trust === "known" && groupOccupancyAvailability(group, block).status === "observed" ? byGroup[group.cacheGroup]?.size ?? 0 : null])),
            trust: block.trust,
          }
        }),
    }}),
  }
}

export function projectRequestTokens(state: ReplayState, requestId: string, content: ContentRecord[] = []): RequestTokenProjection[] {
  const observed = content.length ? attachContent(state, content) : state
  const request = observed.requests[requestId]
  if (!request || request.promptTokenCount == null) return []
  const promptCount = request.promptTokenCount
  const count = Math.max(promptCount + request.acceptedTokenCount, request.reservedCursor ?? 0)
  const acceptedIds = new Map<number, number>()
  for (const span of request.acceptedSpans) {
    span.tokenIds?.slice(0, span.count).forEach((tokenId, index) => acceptedIds.set(span.start + index, tokenId))
  }
  const locationsByPosition = new Map<number, TokenLocation[]>()
  const piecesByPosition = new Map<number, string>()
  const idsByPosition = new Map<number, number>()
  const addLocation = (position: number, location: TokenLocation, piece?: string, tokenId?: number) => {
    const locations = locationsByPosition.get(position) ?? []
    if (!locations.some((item) => item.key === location.key && item.cacheGroup === location.cacheGroup && item.slot === location.slot)) locations.push(location)
    locationsByPosition.set(position, locations)
    if (piece != null) piecesByPosition.set(position, piece)
    if (tokenId != null) idsByPosition.set(position, tokenId)
  }
  const blockSizes = new Map(observed.cacheInventory?.map((pool) => [pool.pool, pool.blockSize]) ?? [])
  for (const block of Object.values(observed.blocks)) {
    for (const binding of block.bindings) {
      if (binding.requestId !== requestId) continue
      const blockSize = observed.cacheGroups.find((group) => group.pool === block.pool && group.cacheGroup === binding.cacheGroup)?.blockSize ?? blockSizes.get(block.pool)
      if (!blockSize) continue
      for (let position = binding.tokenStart; position < Math.min(binding.tokenEnd, count); position += 1) {
        const slot = position % blockSize
        const span = block.contents.find((item) => item.cacheGroup === binding.cacheGroup && slot >= item.slotStart && slot < item.slotEnd)
        addLocation(position, { key: block.key, pool: block.pool, blockId: block.blockId, generation: block.generation,
          cacheGroup: binding.cacheGroup, slot, sourceRequestId: span?.requestId ?? null, valid: span != null }, span?.tokenPieces?.[slot - span.slotStart], span?.tokenIds?.[slot - span.slotStart])
      }
    }
    for (const span of block.contents) {
      if (span.requestId !== requestId) continue
      for (let position = span.tokenStart; position < Math.min(span.tokenEnd, count); position += 1) {
        const offset = position - span.tokenStart
        addLocation(position, { key: block.key, pool: block.pool, blockId: block.blockId, generation: block.generation,
          cacheGroup: span.cacheGroup, slot: span.slotStart + offset, sourceRequestId: span.requestId, valid: true }, span.tokenPieces?.[offset], span.tokenIds?.[offset])
      }
    }
  }
  return Array.from({ length: count }, (_, position) => {
    const outputPosition = position - promptCount
    const locations = locationsByPosition.get(position) ?? []
    const unknown = observed.trust === "unknown" || observed.cacheInventory == null
    const cacheGroups = observed.cacheGroups.map((group) => {
      const available = tokenMappingAvailable(group)
      const groupLocations = locations.filter((location) => location.cacheGroup === group.cacheGroup && location.pool === group.pool)
      const groupUnknown = unknown || !available
      return {
        cacheGroup: group.cacheGroup, pool: group.pool,
        mappingStatus: groupUnknown ? group.tokenSlotMapping === "unavailable" && !unknown ? "unavailable" as const : "unknown" as const : "observed" as const,
        reason: unknown ? "A trace gap or missing inventory prevents a current token validity claim."
          : available ? group.tokenSlotMappingReason : group.tokenSlotMappingReason ?? "Per-token mapping was not observed for this cache group.",
        kvState: groupUnknown ? "unknown" as const : groupLocations.some((location) => location.valid) ? "computed" as const
          : groupLocations.length ? "reserved" as const : "not_computed" as const,
        computedEver: groupUnknown ? null : (request.computedTokenRangesByGroup[group.cacheGroup] ?? [])
          .some((range) => position >= range.start && position < range.end),
      }
    })
    const unknownMapping = !cacheGroups.length || cacheGroups.some((group) => group.mappingStatus !== "observed")
    const partiallyComputed = cacheGroups.some((group) => group.kvState === "computed") && !cacheGroups.every((group) => group.kvState === "computed")
    const kvUnknown = unknown || unknownMapping || partiallyComputed
    return {
      position,
      kind: position < promptCount ? "prompt" : "output",
      tokenId: position < promptCount ? request.promptTokenIds?.[position] ?? idsByPosition.get(position) ?? null
        : acceptedIds.get(outputPosition) ?? idsByPosition.get(position) ?? null,
      tokenPiece: piecesByPosition.get(position) ?? null,
      kvState: kvUnknown ? "unknown" : cacheGroups.every((group) => group.kvState === "computed") ? "computed"
        : locations.length ? "reserved" : "not_computed",
      computedEver: unknown || unknownMapping ? null : cacheGroups.every((group) => group.computedEver === true),
      kvReason: unknown ? "A trace gap or missing inventory prevents a current token validity claim."
        : unknownMapping ? "Token validity is unavailable in one or more required cache groups."
          : partiallyComputed ? "Only some cache groups have observed contents for this token; inspect the per-group facts." : null,
      cacheGroups,
      locations,
    }
  })
}

function tokenMappingAvailable(group: CacheGroupInventory): boolean {
  return group.pool != null && group.blockSize != null
    && (group.tokenSlotMapping === "qualified_full_attention" || group.tokenSlotMapping === "authored_fixture" || group.tokenSlotMapping === "legacy_observed")
}

function groupOccupancyAvailability(group: CacheGroupInventory, block: { bindings: KvBinding[]; contents: KvWrittenRange[] } | null): { status: ObservationStatus; reason: string | null } {
  if (!tokenMappingAvailable(group)) return {
    status: group.tokenSlotMapping === "unavailable" ? "unavailable" : "unknown",
    reason: group.tokenSlotMappingReason ?? "Token-slot mapping or geometry was not observed for this cache group.",
  }
  if (group.tokenSlotMapping === "legacy_observed" && (!block || ![...block.bindings, ...block.contents].some((span) => span.cacheGroup === group.cacheGroup))) {
    return { status: "unknown", reason: "No explicit legacy token observation establishes this block's slot occupancy." }
  }
  return { status: "observed", reason: group.tokenSlotMappingReason }
}

function occupancyAvailability(groups: CacheGroupInventory[], block: { bindings: KvBinding[]; contents: KvWrittenRange[] } | null): { status: ObservationStatus; reason: string | null } {
  if (!groups.length) return { status: "unknown", reason: "The physical pool has no recorded token-slot mapping." }
  const unavailable = groups.map((group) => groupOccupancyAvailability(group, block)).find((mapping) => mapping.status === "unavailable")
  const unknown = groups.map((group) => groupOccupancyAvailability(group, block)).find((mapping) => mapping.status === "unknown")
  return unavailable ?? unknown ?? { status: "observed", reason: null }
}

function object(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function positiveInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null
}

function string(value: unknown): string | null {
  return typeof value === "string" && value ? value : null
}
