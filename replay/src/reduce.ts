/**
 * Deterministic reducer for recording schema 1.0.0.
 * Playback speed is a display concern and is not an input here.
 */

export class TraceError extends Error {}

export interface TraceEvent {
  schema_version: string
  run_id: string
  producer_id: string
  sequence: number
  monotonic_ns: number | string
  event_type: string
  [key: string]: unknown
}

export interface ContentRecord {
  kind: string
  request_id?: string
  text?: string
  text_delta?: string
  emission_index?: number
  token_ids?: number[]
  output_position_start?: number
  content_sequence?: number
  [key: string]: unknown
}

export interface ManifestGap {
  sequence?: number
  reason?: string
  missing_sequence_start?: number | null
  missing_sequence_end?: number | null
}

export interface RecordingManifest {
  schema_version?: string
  run_id?: string
  execution_kind?: string
  completion_status?: string
  content_retained?: boolean
  event_completeness?: {
    status?: string
    gap?: ManifestGap | null
  } | null
  lesson?: { title?: string; summary?: string; evidence?: string }
  arrivals?: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface ScheduledRequest {
  requestId: string
  rangeStart: number
  rangeLength: number
  promptTokens: number
  generationTokens: number
  numPromptTokens: number
}

export interface BlockTableChange {
  requestId: string
  change: string
  blockIds: number[][]
}

export interface AcceptedSpan {
  stepId: string
  start: number
  count: number
  tokenIds: number[] | null
}

export interface StepState {
  id: string
  sequence: number
  status: "scheduled" | "submitted" | "result_received"
  tokenBudget: number
  totalScheduledTokens: number
  requests: ScheduledRequest[]
  preemptedRequestIds: string[]
  blockTableChanges: BlockTableChange[]
  submittedSequence: number | null
  resultSequence: number | null
  resultObservation: string | null
  unscheduled: boolean
  policy: string | null
  queueBefore: SchedulerQueues | null
  queueAfter: SchedulerQueues | null
}

export interface QueueEntry {
  requestId: string
  priority: number | null
  computedTokens: number | null
  promptTokens: number | null
  status: string | null
}

export interface SchedulerQueues {
  waiting: QueueEntry[]
  running: QueueEntry[]
  skippedWaiting: QueueEntry[]
}

export interface CacheInventoryPool {
  pool: string
  blockSize: number | null
  capacity: number
}

export interface CacheGroupInventory {
  cacheGroup: number
  pool: string | null
  blockSize: number | null
  layerNames: string[]
  cacheType: string | null
  tokenSlotMapping: "qualified_full_attention" | "authored_fixture" | "unavailable" | "unknown" | "legacy_observed"
  tokenSlotMappingReason: string | null
}

export interface KvBinding {
  requestId: string
  cacheGroup: number
  tokenStart: number
  tokenEnd: number
  stepId: string
  sequence: number
}

export interface KvWrittenRange extends KvBinding {
  slotStart: number
  slotEnd: number
  tokenIds: number[] | null
  tokenPieces: string[] | null
}

export interface RequestState {
  id: string
  status: string
  adapterId: string | null
  loraIntId: number | null
  modelRouteKind: string | null
  modelRouteSequence: number | null
  promptTokenCount: number | null
  promptText: string | null
  promptTokenIds: number[] | null
  reservedCursor: number | null
  blockIds: number[]
  blockGroups: number[][]
  computedTokenRanges: Array<{ start: number; end: number }>
  computedTokenRangesByGroup: Record<string, Array<{ start: number; end: number }>>
  acceptedTokenCount: number
  acceptedSpans: AcceptedSpan[]
  acceptedTokenIds: number[] | null
  text: string
  preemptionCount: number
  computedTokensBefore: number | null
  computedTokensAfter: number | null
  preemptionReason: string | null
  matchedPrefixTokens: number | null
  sharedPrefixBoundary: number | null
  prefixBlockIds: number[]
  recomputing: boolean
  trust: "known" | "unknown"
  finishReason: string | null
}

export interface BlockState {
  key: string
  pool: string
  blockId: number
  generation: number
  refCount: number
  hasHash: boolean
  cachedAtZeroRefs: boolean
  lastOperation: string
  trust: "known" | "unknown"
  isNull: boolean
  bindings: KvBinding[]
  contents: KvWrittenRange[]
}

export interface TextEmission {
  requestId: string
  index: number
  sequence: number
  tokenStart: number | null
  tokenEnd: number | null
  text: string | null
}

export interface ReplayState {
  trust: "known" | "unknown"
  incomplete: boolean
  gap: { sequence: number; start: number | null; end: number | null; reason: string } | null
  requests: Record<string, RequestState>
  steps: StepState[]
  blocks: Record<string, BlockState>
  cacheInventory: CacheInventoryPool[] | null
  cacheGroups: CacheGroupInventory[]
  emissions: TextEmission[]
  lastSequence: number
  lastEventType: string | null
  lastEventNs: number | null
  appliedCount: number
}

const SCHEMA = "1.0.0"

const EVENT_TYPES = new Set([
  "request_received",
  "request_tokenized",
  "request_queued",
  "request_dispatched",
  "request_dispatch_completed",
  "http_request_received",
  "http_engine_admission_correlated",
  "http_response_completed",
  "request_model_route",
  "engine_prompt_observed",
  "engine_admitted",
  "capture_ready",
  "step_scheduled",
  "step_submitted",
  "step_result_received",
  "prefix_lookup",
  "block_change",
  "tokens_accepted",
  "text_emitted",
  "preemption",
  "cancellation",
  "completion",
  "trace_gap",
  "cache_inventory",
  "kv_block_binding",
  "kv_block_written",
  "cache_reset",
])

export function initialState(): ReplayState {
  return {
    trust: "known",
    incomplete: false,
    gap: null,
    requests: {},
    steps: [],
    blocks: {},
    cacheInventory: null,
    cacheGroups: [],
    emissions: [],
    lastSequence: 0,
    lastEventType: null,
    lastEventNs: null,
    appliedCount: 0,
  }
}

/** Keep integers outside the safe range as strings so elapsed time stays exact. */
export function parseRecordingJson<T>(text: string): T {
  return JSON.parse(quoteWideIntegers(text)) as T
}

export function parseJsonl<T>(text: string): T[] {
  const rows: T[] = []
  const lines = text.split(/\n/)
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim()
    if (!line) continue
    try {
      rows.push(parseRecordingJson<T>(line))
    } catch (error) {
      const message = error instanceof Error ? error.message : "invalid JSON"
      throw new TraceError(`line ${index + 1}: ${message}`)
    }
  }
  return rows
}

function quoteWideIntegers(text: string): string {
  // Match strings as whole tokens so numeric text inside a prompt stays text.
  return text.replace(/"(?:[^"\\]|\\[\s\S])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g, (token) => {
    return /^-?\d{16,}$/.test(token) ? `"${token}"` : token
  })
}

export function replay(events: TraceEvent[], manifest?: RecordingManifest): ReplayState {
  const state = events.reduce((current, event) => applyEvent(current, event, manifest), initialState())
  return finish(state, manifest)
}

export function applyEvent(
  state: ReplayState,
  event: TraceEvent,
  manifest?: RecordingManifest,
): ReplayState {
  if (state.trust === "unknown") return state
  if (event.schema_version !== SCHEMA) {
    throw new TraceError(`unsupported recording schema ${String(event.schema_version)}`)
  }
  if (!Number.isInteger(event.sequence) || event.sequence !== state.lastSequence + 1) {
    throw new TraceError(`expected sequence ${state.lastSequence + 1}, found ${String(event.sequence)}`)
  }
  if (!EVENT_TYPES.has(event.event_type)) {
    throw new TraceError(`unknown event type ${event.event_type}`)
  }
  const next = structuredClone(state)
  next.lastSequence = event.sequence
  next.lastEventType = event.event_type
  next.lastEventNs = Number(event.monotonic_ns)
  next.appliedCount += 1
  switch (event.event_type) {
    case "request_received":
      advanceIntakeStatus(ensureRequest(next, requestId(event)), "received")
      break
    case "request_tokenized": {
      const request = ensureRequest(next, requestId(event))
      request.promptTokenCount = numberValue(event.prompt_token_count)
      advanceIntakeStatus(request, "tokenized")
      break
    }
    case "request_queued":
      advanceIntakeStatus(ensureRequest(next, requestId(event)), "queued")
      break
    case "request_dispatched":
      advanceIntakeStatus(ensureRequest(next, requestId(event)), "dispatched")
      break
    case "request_dispatch_completed":
      // Completion of the client write is still an intake dispatch boundary;
      // it does not imply server receipt.
      advanceIntakeStatus(ensureRequest(next, requestId(event)), "dispatched")
      break
    case "http_request_received":
      advanceIntakeStatus(ensureRequest(next, requestId(event)), "received")
      break
    case "engine_prompt_observed": {
      const request = ensureRequest(next, requestId(event))
      request.promptTokenCount = numberValue(event.prompt_token_count)
      advanceIntakeStatus(request, "tokenized")
      break
    }
    case "engine_admitted":
    case "http_engine_admission_correlated":
      advanceIntakeStatus(ensureRequest(next, requestId(event)), "queued")
      break
    case "http_response_completed":
      applyHttpResponse(next, event)
      break
    case "request_model_route":
      // Routing is captured immediately before the engine add_request call.
      applyModelRoute(next, event)
      break
    case "capture_ready":
      // Run-level evidence. Keeping it in the applied prefix makes seeking and
      // recorded-time playback exact without creating request state.
      break
    case "step_scheduled":
      applyScheduled(next, event)
      break
    case "step_submitted":
      applySubmitted(next, event)
      break
    case "step_result_received":
      applyResult(next, event)
      break
    case "prefix_lookup":
      applyPrefix(next, event)
      break
    case "block_change":
      applyBlock(next, event)
      break
    case "cache_inventory":
      applyInventory(next, event)
      break
    case "kv_block_binding":
      applyKvBinding(next, event)
      break
    case "kv_block_written":
      applyKvWritten(next, event)
      break
    case "cache_reset":
      if (event.verified !== true) throw new TraceError("cache reset was not verified")
      for (const block of Object.values(next.blocks)) {
        block.contents = []
        block.bindings = []
        block.hasHash = false
        block.cachedAtZeroRefs = false
      }
      break
    case "tokens_accepted":
      applyAccepted(next, event)
      break
    case "text_emitted":
      applyText(next, event)
      break
    case "preemption":
      applyPreemption(next, event)
      break
    case "cancellation":
      applyTerminal(next, event, "cancelled")
      break
    case "completion":
      applyTerminal(next, event, "completed")
      break
    case "trace_gap":
      applyGap(next, event, manifest)
      break
    default:
      break
  }
  return next
}

export interface Checkpoint {
  sequence: number
  state: ReplayState
}

export function buildCheckpoints(
  events: TraceEvent[],
  interval = 32,
  manifest?: RecordingManifest,
): Checkpoint[] {
  const points: Checkpoint[] = [{ sequence: 0, state: initialState() }]
  let state = initialState()
  events.forEach((event, index) => {
    state = applyEvent(state, event, manifest)
    const boundary = (index + 1) % interval === 0 || index === events.length - 1
    if (boundary) points.push({ sequence: state.lastSequence, state })
  })
  return points
}

export interface Player {
  maxSequence: number
  steps: StepState[]
  requestOrder: string[]
  at(sequence: number): ReplayState
}

export function createPlayer(events: TraceEvent[], manifest?: RecordingManifest): Player {
  const checkpoints = buildCheckpoints(events, 32, manifest)
  const final = checkpoints[checkpoints.length - 1].state
  return {
    maxSequence: final.lastSequence,
    steps: structuredClone(final.steps),
    requestOrder: Object.keys(final.requests),
    at(sequence: number) {
      let start = checkpoints[0]
      for (const point of checkpoints) {
        if (point.sequence <= sequence) start = point
        else break
      }
      let state = structuredClone(start.state)
      for (const event of events) {
        if (event.sequence <= start.sequence) continue
        if (event.sequence > sequence) break
        state = applyEvent(state, event, manifest)
      }
      return finish(state, manifest)
    },
  }
}

export function attachContent(state: ReplayState, content: ContentRecord[]): ReplayState {
  const next = structuredClone(state)
  const rows = [...content].sort(
    (left, right) => Number(left.content_sequence ?? 0) - Number(right.content_sequence ?? 0),
  )
  for (const row of rows) {
    if (row.kind === "kv_tokens") attachKvTokens(next, row)
    const requestIdValue = typeof row.request_id === "string" ? row.request_id : null
    if (requestIdValue == null || next.requests[requestIdValue] == null) continue
    const request = next.requests[requestIdValue]
    if (row.kind === "prompt_text" && typeof row.text === "string") {
      request.promptText = row.text
    }
    if (row.kind === "prompt" && Array.isArray(row.token_ids)) {
      request.promptTokenIds = row.token_ids.map((token) => Number(token))
    }
    if (row.kind === "text") {
      const emission = next.emissions.find(
        (item) => item.requestId === requestIdValue && item.index === Number(row.emission_index ?? 0),
      )
      if (emission && typeof row.text_delta === "string") emission.text = row.text_delta
    }
    if (row.kind === "accepted_tokens" && Array.isArray(row.token_ids)) {
      const start = Number(row.output_position_start ?? 0)
      const span = request.acceptedSpans.find((item) => item.start === start)
      if (span) span.tokenIds = row.token_ids.map((token) => Number(token))
    }
  }
  for (const request of Object.values(next.requests)) {
    request.text = next.emissions
      .filter((item) => item.requestId === request.id && item.text != null)
      .sort((left, right) => left.index - right.index)
      .map((item) => item.text ?? "")
      .join("")
    const spans = [...request.acceptedSpans].sort((left, right) => left.start - right.start)
    if (spans.length === 0) {
      request.acceptedTokenIds = request.acceptedTokenCount === 0 ? [] : null
      continue
    }
    if (spans.some((span) => span.tokenIds == null)) {
      request.acceptedTokenIds = null
      continue
    }
    request.acceptedTokenIds = spans.flatMap((span) => span.tokenIds ?? [])
  }
  return next
}

function finish(state: ReplayState, manifest?: RecordingManifest): ReplayState {
  if (manifest?.event_completeness?.status === "incomplete" && !state.incomplete) {
    const next = structuredClone(state)
    next.incomplete = true
    return next
  }
  return state
}

function ensureRequest(state: ReplayState, id: string): RequestState {
  const existing = state.requests[id]
  if (existing) return existing
  const created: RequestState = {
    id,
    status: "observed",
    adapterId: null,
    loraIntId: null,
    modelRouteKind: null,
    modelRouteSequence: null,
    promptTokenCount: null,
    promptText: null,
    promptTokenIds: null,
    reservedCursor: null,
    blockIds: [],
    blockGroups: [],
    computedTokenRanges: [],
    computedTokenRangesByGroup: {},
    acceptedTokenCount: 0,
    acceptedSpans: [],
    acceptedTokenIds: null,
    text: "",
    preemptionCount: 0,
    computedTokensBefore: null,
    computedTokensAfter: null,
    preemptionReason: null,
    matchedPrefixTokens: null,
    sharedPrefixBoundary: null,
    prefixBlockIds: [],
    recomputing: false,
    trust: "known",
    finishReason: null,
  }
  state.requests[id] = created
  return created
}

function applyModelRoute(state: ReplayState, event: TraceEvent): void {
  const request = ensureRequest(state, requestId(event))
  request.adapterId = typeof event.adapter_id === "string" ? event.adapter_id : null
  request.loraIntId = numberValue(event.lora_int_id)
  request.modelRouteKind = typeof event.route_kind === "string" ? event.route_kind : null
  request.modelRouteSequence = event.sequence
}

const INTAKE_STATUS_RANK: Record<string, number> = {
  observed: 0,
  dispatched: 1,
  received: 2,
  tokenized: 3,
  queued: 4,
  scheduled: 5,
}

/**
 * Apply only the boundary that was recorded. Correlation and summary events
 * may be written after a scheduler or terminal event, so they must never move
 * a request backwards in the replayed state.
 */
function advanceIntakeStatus(request: RequestState, status: keyof typeof INTAKE_STATUS_RANK): void {
  if (request.status === "completed" || request.status === "cancelled" || request.status === "preempted") return
  const current = INTAKE_STATUS_RANK[request.status]
  const next = INTAKE_STATUS_RANK[status]
  if (current == null || next >= current) request.status = status
}

function applyScheduled(state: ReplayState, event: TraceEvent): void {
  const requests = scheduledRequests(event.requests)
  const changes = blockTableChanges(event.block_table_changes)
  const step: StepState = {
    id: stringField(event, "step_id", "step"),
    sequence: event.sequence,
    status: "scheduled",
    tokenBudget: numberValue(event.token_budget) ?? 0,
    totalScheduledTokens: numberValue(event.total_scheduled_tokens) ?? 0,
    requests,
    preemptedRequestIds: stringList(event.preempted_request_ids),
    blockTableChanges: changes,
    submittedSequence: null,
    resultSequence: null,
    resultObservation: null,
    unscheduled: false,
    policy: typeof event.policy === "string" ? event.policy : null,
    queueBefore: schedulerQueues(event.queue_before),
    queueAfter: schedulerQueues(event.queue_after),
  }
  state.steps.push(step)
  for (const item of requests) {
    const request = ensureRequest(state, item.requestId)
    if (request.status !== "completed" && request.status !== "cancelled") request.status = "scheduled"
    request.reservedCursor = item.rangeStart + item.rangeLength
    request.recomputing = false
    if (request.promptTokenCount == null) request.promptTokenCount = item.numPromptTokens
  }
  for (const change of changes) {
    const request = ensureRequest(state, change.requestId)
    const flat = change.blockIds.flat()
    request.blockIds = change.change === "append" ? unique([...request.blockIds, ...flat]) : flat
    request.blockGroups = change.change === "append"
      ? Array.from({ length: Math.max(request.blockGroups.length, change.blockIds.length) }, (_, index) =>
          [...(request.blockGroups[index] ?? []), ...(change.blockIds[index] ?? [])])
      : change.blockIds.map((group) => [...group])
  }
}

function applySubmitted(state: ReplayState, event: TraceEvent): void {
  const step = findOrCreateStep(state, event, "submitted")
  step.status = step.status === "result_received" ? step.status : "submitted"
  step.submittedSequence = event.sequence
}

function applyResult(state: ReplayState, event: TraceEvent): void {
  const step = findOrCreateStep(state, event, "result_received")
  step.status = "result_received"
  step.resultSequence = event.sequence
  step.resultObservation = typeof event.observation === "string" ? event.observation : null
}

function applyPrefix(state: ReplayState, event: TraceEvent): void {
  const request = ensureRequest(state, requestId(event))
  request.matchedPrefixTokens = numberValue(event.matched_token_count)
  request.sharedPrefixBoundary = numberValue(event.shared_prefix_boundary)
  request.prefixBlockIds = flattenGroups(event.block_ids)
}

function applyBlock(state: ReplayState, event: TraceEvent): void {
  const pool = stringField(event, "pool", "pool")
  const blockId = numberValue(event.block_id) ?? 0
  const key = `${pool}:${blockId}`
  const generation = numberValue(event.allocation_generation) ?? 0
  const refCount = numberValue(event.ref_count_after) ?? 0
  const previous = state.blocks[key]
  const ambiguousPool = Object.values(state.blocks).some((block) => block.blockId === blockId && block.pool !== pool)
  if (!ambiguousPool && (refCount === 0 || (previous && previous.generation !== generation))) {
    for (const request of Object.values(state.requests)) {
      request.blockIds = request.blockIds.filter((id) => id !== blockId)
      request.blockGroups = request.blockGroups.map((group, index) => {
        const mapping = state.cacheGroups.find((item) => item.cacheGroup === index)
        return mapping?.pool != null && mapping.pool !== pool ? group : group.filter((id) => id !== blockId)
      })
    }
  }
  state.blocks[key] = {
    key,
    pool,
    blockId,
    generation,
    refCount,
    hasHash: Boolean(event.has_hash_after),
    cachedAtZeroRefs: Boolean(event.cached_at_zero_refs),
    lastOperation: stringField(event, "operation", "change"),
    trust: "known",
    isNull: previous?.isNull ?? false,
    bindings: previous && previous.generation === generation && refCount > 0 ? previous.bindings : [],
    // Release and hash eviction change ownership/reusability, not the bytes.
    contents: previous && previous.generation === generation ? previous.contents : [],
  }
}

function applyInventory(state: ReplayState, event: TraceEvent): void {
  if (!Array.isArray(event.pools)) throw new TraceError("cache inventory has no pools")
  const pools: CacheInventoryPool[] = []
  const blocks: Record<string, BlockState> = {}
  for (const value of event.pools) {
    const row = record(value)
    const pool = typeof row.pool === "string" ? row.pool : ""
    const blockSize = row.block_size == null ? null : integerField(row, "block_size", 1)
    const capacity = integerField(row, "num_gpu_blocks", 0)
    if (!pool || pools.some((item) => item.pool === pool) || !Array.isArray(row.blocks)) {
      throw new TraceError("cache inventory has invalid or duplicate pools")
    }
    pools.push({ pool, blockSize, capacity })
    for (const item of row.blocks) {
      const source = record(item)
      const blockId = integerField(source, "block_id", 0)
      const key = `${pool}:${blockId}`
      if (blockId >= capacity || blocks[key]) throw new TraceError("cache inventory has invalid or duplicate blocks")
      blocks[key] = {
        key, pool, blockId,
        generation: integerField(source, "allocation_generation", 0),
        refCount: integerField(source, "ref_count", 0),
        hasHash: Boolean(source.has_hash),
        cachedAtZeroRefs: source.ref_count === 0 && Boolean(source.has_hash),
        lastOperation: "inventory",
        trust: "known",
        isNull: Boolean(source.is_null),
        bindings: [],
        contents: [],
      }
    }
    if (Object.values(blocks).filter((block) => block.pool === pool).length !== capacity) {
      throw new TraceError("cache inventory does not enumerate the full pool")
    }
  }
  state.cacheInventory = pools
  // The validated inventory establishes the untouched free generation-zero
  // baseline. Keep only exceptions/touched cells in checkpoint state.
  state.blocks = Object.fromEntries(Object.entries(blocks).filter(([, block]) =>
    block.isNull || block.refCount !== 0 || block.hasHash || block.generation !== 0))
  state.cacheGroups = Array.isArray(event.groups) ? event.groups.map((item) => {
    const row = record(item)
    const pool = typeof row.pool === "string" ? row.pool : null
    if (pool != null && !pools.some((value) => value.pool === pool)) throw new TraceError("cache group refers to an unknown pool")
    const mapping = row.token_slot_mapping ?? "unknown"
    if (mapping !== "qualified_full_attention" && mapping !== "authored_fixture" && mapping !== "unavailable" && mapping !== "unknown") {
      throw new TraceError("cache group has an unsupported token-slot mapping status")
    }
    return {
      cacheGroup: integerField(row, "cache_group", 0), pool,
      blockSize: row.block_size == null ? null : integerField(row, "block_size", 1),
      layerNames: stringList(row.layer_names),
      cacheType: typeof row.cache_type === "string" ? row.cache_type : null,
      tokenSlotMapping: mapping,
      tokenSlotMappingReason: typeof row.token_slot_mapping_reason === "string" ? row.token_slot_mapping_reason
        : mapping === "unknown" ? "This older inventory did not record cache-layout qualification."
          : mapping === "unavailable" ? "Per-token slot mapping is unavailable for this cache layout." : null,
    }
  }) : []
  if (new Set(state.cacheGroups.map((group) => group.cacheGroup)).size !== state.cacheGroups.length) throw new TraceError("cache inventory has duplicate cache-group identities")
}

function kvIdentity(state: ReplayState, event: TraceEvent): { block: BlockState; binding: KvBinding; blockSize: number } {
  const pool = stringField(event, "pool", "")
  const blockId = integerField(event, "block_id", 0)
  const generation = integerField(event, "allocation_generation", 0)
  const block = state.blocks[`${pool}:${blockId}`]
  const geometry = state.cacheInventory?.find((item) => item.pool === pool)
  if (!block || !geometry || block.generation !== generation || block.isNull) {
    throw new TraceError("KV observation does not match an inventoried block generation")
  }
  const tokenStart = integerField(event, "token_range_start", 0)
  const tokenEnd = integerField(event, "token_range_end", 1)
  const cacheGroup = integerField(event, "cache_group", 0)
  let group = state.cacheGroups.find((item) => item.cacheGroup === cacheGroup)
  if (group && group.pool !== pool) throw new TraceError("KV observation does not match the observed cache-group pool mapping")
  if (!group && state.cacheGroups.length) throw new TraceError("KV observation refers to an unknown inventory cache group")
  if (group?.tokenSlotMapping === "unavailable") throw new TraceError("KV observation uses a cache group without qualified token-slot mapping")
  if (!group) {
    // Older adapters could record explicit token ranges without inventory group
    // qualification. Preserve that observation without claiming a qualified layout.
    group = { cacheGroup, pool, blockSize: geometry.blockSize, layerNames: [], cacheType: null,
      tokenSlotMapping: "unknown", tokenSlotMappingReason: null }
    state.cacheGroups.push(group)
  }
  if (group.tokenSlotMapping === "unknown") {
    group.tokenSlotMapping = "legacy_observed"
    group.tokenSlotMappingReason = "Explicit legacy token observations are retained; cache-layout qualification was not recorded."
  }
  const blockSize = group?.blockSize ?? geometry.blockSize
  if (blockSize == null) throw new TraceError("KV observation has no observed block size")
  if (tokenEnd <= tokenStart || Math.floor(tokenStart / blockSize) !== Math.floor((tokenEnd - 1) / blockSize)) {
    throw new TraceError("KV observation must cover a nonempty range within one block")
  }
  const id = stringField(event, "request_id", "")
  if (!id) throw new TraceError("KV observation has no request identity")
  return {
    block, blockSize,
    binding: {
      requestId: id,
      cacheGroup,
      tokenStart, tokenEnd,
      stepId: stringField(event, "step_id", ""),
      sequence: event.sequence,
    },
  }
}

function applyKvBinding(state: ReplayState, event: TraceEvent): void {
  const { block, binding, blockSize } = kvIdentity(state, event)
  ensureRequest(state, binding.requestId)
  // A request's logical block can move to a different physical generation.
  for (const item of Object.values(state.blocks)) {
    item.bindings = item.bindings.filter((old) => old.requestId !== binding.requestId
      || old.cacheGroup !== binding.cacheGroup
      || old.tokenEnd <= binding.tokenStart || old.tokenStart >= binding.tokenEnd)
  }
  block.bindings.push(binding)
  markBoundContents(state, block, binding, blockSize)
}

function applyKvWritten(state: ReplayState, event: TraceEvent): void {
  const { block, binding, blockSize } = kvIdentity(state, event)
  const slotStart = binding.tokenStart % blockSize
  const slotEnd = slotStart + binding.tokenEnd - binding.tokenStart
  const kept: KvWrittenRange[] = []
  for (const old of block.contents) {
    if (old.cacheGroup !== binding.cacheGroup || old.slotEnd <= slotStart || old.slotStart >= slotEnd) kept.push(old)
    else {
      if (old.slotStart < slotStart) kept.push(sliceWritten(old, old.slotStart, slotStart))
      if (old.slotEnd > slotEnd) kept.push(sliceWritten(old, slotEnd, old.slotEnd))
    }
  }
  kept.push({ ...binding, slotStart, slotEnd, tokenIds: null, tokenPieces: null })
  block.contents = kept.sort((left, right) => left.slotStart - right.slotStart)
  markComputed(ensureRequest(state, binding.requestId), binding.tokenStart, binding.tokenEnd, binding.cacheGroup)
  for (const owner of block.bindings) {
    if (owner.cacheGroup === binding.cacheGroup) markBoundContents(state, block, owner, blockSize)
  }
}

function markBoundContents(state: ReplayState, block: BlockState, binding: KvBinding, blockSize: number): void {
  const start = binding.tokenStart % blockSize
  const end = start + binding.tokenEnd - binding.tokenStart
  for (const span of block.contents) {
    const overlapStart = Math.max(start, span.slotStart)
    const overlapEnd = Math.min(end, span.slotEnd)
    if (overlapEnd > overlapStart && span.cacheGroup === binding.cacheGroup) {
      markComputed(ensureRequest(state, binding.requestId), binding.tokenStart + overlapStart - start, binding.tokenStart + overlapEnd - start, binding.cacheGroup)
    }
  }
}

function markComputed(request: RequestState, start: number, end: number, cacheGroup: number): void {
  request.computedTokenRanges = mergeComputedRanges(request.computedTokenRanges, start, end)
  request.computedTokenRangesByGroup[cacheGroup] = mergeComputedRanges(request.computedTokenRangesByGroup[cacheGroup] ?? [], start, end)
}

function mergeComputedRanges(previousRanges: Array<{ start: number; end: number }>, start: number, end: number): Array<{ start: number; end: number }> {
  const ranges = [...previousRanges, { start, end }].sort((left, right) => left.start - right.start)
  const merged: typeof ranges = []
  for (const range of ranges) {
    const previous = merged.at(-1)
    if (previous && previous.end >= range.start) previous.end = Math.max(previous.end, range.end)
    else merged.push({ ...range })
  }
  return merged
}

function sliceWritten(span: KvWrittenRange, start: number, end: number): KvWrittenRange {
  const offset = start - span.slotStart
  return {
    ...span, slotStart: start, slotEnd: end,
    tokenStart: span.tokenStart + offset,
    tokenEnd: span.tokenStart + offset + end - start,
    tokenIds: span.tokenIds?.slice(offset, offset + end - start) ?? null,
    tokenPieces: span.tokenPieces?.slice(offset, offset + end - start) ?? null,
  }
}

function attachKvTokens(state: ReplayState, row: ContentRecord): void {
  const block = state.blocks[`${String(row.pool)}:${String(row.block_id)}`]
  if (!block || block.generation !== row.allocation_generation || !Array.isArray(row.token_ids)) return
  for (const span of block.contents) {
    if (span.requestId !== row.request_id || span.cacheGroup !== row.cache_group || span.stepId !== row.step_id) continue
    const start = numberValue(row.token_range_start)
    const end = numberValue(row.token_range_end)
    if (start == null || end == null || start > span.tokenStart || end < span.tokenEnd) continue
    const ids = row.token_ids.slice(span.tokenStart - start, span.tokenEnd - start)
    if (ids.length === span.tokenEnd - span.tokenStart && ids.every((id) => Number.isInteger(id) && id >= 0)) span.tokenIds = [...ids]
    if (Array.isArray(row.token_pieces)) {
      const pieces = row.token_pieces.slice(span.tokenStart - start, span.tokenEnd - start)
      if (pieces.length === span.tokenEnd - span.tokenStart && pieces.every((piece) => typeof piece === "string")) span.tokenPieces = pieces as string[]
    }
  }
}

function clearBindings(state: ReplayState, id: string): void {
  for (const block of Object.values(state.blocks)) block.bindings = block.bindings.filter((binding) => binding.requestId !== id)
}

function schedulerQueues(value: unknown): SchedulerQueues | null {
  if (value == null || typeof value !== "object") return null
  const row = record(value)
  if (row.available === false) return null
  if (![row.waiting, row.running, row.skipped_waiting].every(Array.isArray)) return null
  const entries = (items: unknown): QueueEntry[] => (items as unknown[]).map((item) => {
    const entry = record(item)
    return {
      requestId: String(entry.request_id ?? ""),
      priority: numberValue(entry.priority),
      computedTokens: numberValue(entry.num_computed_tokens),
      promptTokens: numberValue(entry.num_prompt_tokens),
      status: typeof entry.status === "string" ? entry.status : null,
    }
  })
  return { waiting: entries(row.waiting), running: entries(row.running), skippedWaiting: entries(row.skipped_waiting) }
}

function integerField(value: Record<string, unknown>, key: string, minimum: number): number {
  const result = numberValue(value[key])
  if (result == null || !Number.isSafeInteger(result) || result < minimum) throw new TraceError(`invalid ${key} in cache observation`)
  return result
}

function applyAccepted(state: ReplayState, event: TraceEvent): void {
  const request = ensureRequest(state, requestId(event))
  const count = numberValue(event.token_count) ?? 0
  const start = numberValue(event.output_position_start) ?? request.acceptedTokenCount
  request.acceptedTokenCount += count
  request.acceptedSpans.push({
    stepId: stringField(event, "step_id", ""),
    start,
    count,
    tokenIds: null,
  })
}

function applyText(state: ReplayState, event: TraceEvent): void {
  state.emissions.push({
    requestId: requestId(event),
    index: numberValue(event.emission_index) ?? 0,
    sequence: event.sequence,
    tokenStart: numberValue(event.token_range_start),
    tokenEnd: numberValue(event.token_range_end),
    text: null,
  })
}

function applyPreemption(state: ReplayState, event: TraceEvent): void {
  const request = ensureRequest(state, requestId(event))
  request.status = "preempted"
  request.blockIds = []
  request.blockGroups = []
  clearBindings(state, request.id)
  request.preemptionCount = numberValue(event.preemption_count) ?? request.preemptionCount + 1
  request.computedTokensBefore = numberValue(event.computed_tokens_before)
  request.computedTokensAfter = numberValue(event.computed_tokens_after)
  request.reservedCursor = request.computedTokensAfter
  request.recomputing = true
  request.preemptionReason = typeof event.reason === "string" ? event.reason : null
}

function applyTerminal(state: ReplayState, event: TraceEvent, status: "completed" | "cancelled"): void {
  const request = ensureRequest(state, requestId(event))
  request.status = status
  request.blockIds = []
  request.blockGroups = []
  clearBindings(state, request.id)
  request.recomputing = false
  request.finishReason = typeof event.reason === "string" ? event.reason : stringField(event, "status", status)
}

function applyHttpResponse(state: ReplayState, event: TraceEvent): void {
  const request = ensureRequest(state, requestId(event))
  const statusCode = numberValue(event.status_code)
  const finishReason = typeof event.finish_reason === "string" && event.finish_reason
    ? event.finish_reason
    : null

  // The instrumented HTTP server writes this boundary only after the vLLM
  // request output has supplied a terminal finish reason. A successful response
  // therefore provides a terminal fallback when vLLM's scheduler result omits
  // the separate completion event (observed for early EOS / `stop` outputs).
  // A bare response boundary or an error response is not completion evidence.
  if (finishReason == null || statusCode == null || statusCode < 200 || statusCode >= 300) return
  request.status = finishReason === "cancelled" || finishReason === "abort" ? "cancelled" : "completed"
  request.blockIds = []
  request.blockGroups = []
  clearBindings(state, request.id)
  request.recomputing = false
  request.finishReason = finishReason
}

function applyGap(state: ReplayState, event: TraceEvent, manifest?: RecordingManifest): void {
  const manifestGap = manifest?.event_completeness?.gap
  state.trust = "unknown"
  state.incomplete = true
  state.gap = {
    sequence: event.sequence,
    start: numberValue(event.missing_sequence_start) ?? numberValue(manifestGap?.missing_sequence_start),
    end: numberValue(event.missing_sequence_end) ?? numberValue(manifestGap?.missing_sequence_end),
    reason: typeof event.reason === "string" ? event.reason : "trace_gap",
  }
  for (const request of Object.values(state.requests)) request.trust = "unknown"
  for (const block of Object.values(state.blocks)) block.trust = "unknown"
}

function findOrCreateStep(state: ReplayState, event: TraceEvent, status: StepState["status"]): StepState {
  const id = stringField(event, "step_id", "step")
  const existing = state.steps.find((step) => step.id === id)
  if (existing) return existing
  const created: StepState = {
    id,
    sequence: event.sequence,
    status,
    tokenBudget: 0,
    totalScheduledTokens: 0,
    requests: [],
    preemptedRequestIds: [],
    blockTableChanges: [],
    submittedSequence: null,
    resultSequence: null,
    resultObservation: null,
    unscheduled: true,
    policy: null,
    queueBefore: null,
    queueAfter: null,
  }
  state.steps.push(created)
  return created
}

function scheduledRequests(value: unknown): ScheduledRequest[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => {
    const row = record(item)
    return {
      requestId: String(row.request_id ?? ""),
      rangeStart: numberValue(row.range_start) ?? 0,
      rangeLength: numberValue(row.range_length) ?? 0,
      promptTokens: numberValue(row.prompt_tokens) ?? 0,
      generationTokens: numberValue(row.generation_tokens) ?? 0,
      numPromptTokens: numberValue(row.num_prompt_tokens) ?? 0,
    }
  })
}

function blockTableChanges(value: unknown): BlockTableChange[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => {
    const row = record(item)
    return {
      requestId: String(row.request_id ?? ""),
      change: String(row.change ?? "append"),
      blockIds: nestedIds(row.block_ids),
    }
  })
}

function nestedIds(value: unknown): number[][] {
  if (!Array.isArray(value)) return []
  return value.map((group) => (Array.isArray(group) ? group.map((id) => Number(id)) : []))
}

function flattenGroups(value: unknown): number[] {
  return nestedIds(value).flat()
}

function requestId(event: TraceEvent): string {
  return stringField(event, "request_id", "request")
}

function stringField(event: TraceEvent, key: string, fallback: string): string {
  const value = event[key]
  return typeof value === "string" && value ? value : fallback
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : []
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  return null
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {}
}

function unique(values: number[]): number[] {
  return [...new Set(values)]
}
