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
])

export function initialState(): ReplayState {
  return {
    trust: "known",
    incomplete: false,
    gap: null,
    requests: {},
    steps: [],
    blocks: {},
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
  }
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
