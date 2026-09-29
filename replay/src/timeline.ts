/**
 * Pure indexes for sample lanes. The reducer remains the source of truth
 * for an event prefix; these helpers do not invent receipt events.
 */

import type { RecordingManifest, ReplayState, TraceEvent } from "./reduce.ts"

export type TimeBasis = "controller_origin" | "since_first_event" | "event_triggered" | "unavailable"

export type Phase =
  | "planned"
  | "awaiting"
  | "observed"
  | "dispatched"
  | "received"
  | "tokenized"
  | "queued"
  | "prefill"
  | "decode"
  | "mixed"
  | "preempted"
  | "completed"
  | "cancelled"
  | "not_submitted"
  | "unknown"

export interface PlannedSample {
  id: string
  label: string
  batchId: string
  batchLabel: string
  arrivalOffsetMs: number | null
  arrivalPolicy: string | null
  maxOutputTokens: number | null
  order: number
}

export interface PlannedBatch {
  id: string
  label: string
  arrivalOffsetMs: number | null
  sampleIds: string[]
}

export interface Roster {
  samples: PlannedSample[]
  batches: PlannedBatch[]
  groupingKnown: boolean
  timeBasis: TimeBasis
  originNs: number | string | null
  timeBasisLabel: string
}

export interface IndexedParticipant {
  requestId: string
  rangeStart: number
  rangeLength: number
  promptTokens: number
  generationTokens: number
}

export interface IndexedStep {
  id: string
  sequence: number
  tokenBudget: number
  totalScheduledTokens: number
  participants: IndexedParticipant[]
}

export interface Milestone {
  requestId: string
  sequence: number
  monotonicNs: number | string
  kind: string
  label: string
}

export interface TimePoint {
  sequence: number
  monotonicNs: number | string
  elapsedMs: number | null
}

export interface TimelineIndex {
  roster: Roster
  steps: IndexedStep[]
  milestonesByRequest: Record<string, Milestone[]>
  timePoints: TimePoint[]
  maxSequence: number
  firstEventNs: number | string | null
}

export interface ScheduledSegment {
  stepId: string
  sequence: number
  kind: "prefill" | "decode" | "mixed"
  rangeStart: number
  rangeEnd: number
  promptTokens: number
  generationTokens: number
}

export interface SampleProjection {
  sample: PlannedSample
  phase: Phase
  phaseLabel: string
  acceptedTokenCount: number | null
  scheduled: ScheduledSegment[]
  observedReceiptNs: number | string | null
  controllerAddReturnedNs: number | string | null
  promptText: string | null
  promptSource: "recording" | "submitted-plan" | "unavailable"
  outputText: string
  milestones: Milestone[]
  appliedStepIds: string[]
}

export interface BlockFact {
  key: string
  pool: string
  blockId: number
  generation: number
  refCount: number
  referencedBy: string[]
  shared: boolean
  note: string
}

const PHASE_LABELS: Record<Phase, string> = {
  planned: "Planned",
  awaiting: "Awaiting observed admission",
  observed: "Observed boundary",
  dispatched: "Dispatched",
  received: "Received",
  tokenized: "Prompt observed",
  queued: "Queued",
  prefill: "Prefill scheduled",
  decode: "Decode scheduled",
  mixed: "Mixed prompt + generation",
  preempted: "Preempted",
  completed: "Completed",
  cancelled: "Cancelled",
  not_submitted: "Not submitted",
  unknown: "Unknown observed state",
}

export function phaseLabel(phase: Phase): string {
  return PHASE_LABELS[phase]
}

export function buildTimeline(events: TraceEvent[], manifest?: RecordingManifest): TimelineIndex {
  const firstEventNs = events.length > 0 ? clockValue(events[0].monotonic_ns) : null
  const roster = buildRoster(events, manifest, firstEventNs)
  const steps: IndexedStep[] = []
  const milestonesByRequest: Record<string, Milestone[]> = {}
  const timePoints: TimePoint[] = []
  for (const event of events) {
    const monotonicNs = clockValue(event.monotonic_ns) ?? Number.NaN
    timePoints.push({
      sequence: event.sequence,
      monotonicNs,
      elapsedMs: elapsedFrom(monotonicNs, roster.originNs),
    })
    if (event.event_type === "step_scheduled") {
      steps.push(indexStep(event))
    }
    for (const milestone of milestonesFor(event)) {
      const list = milestonesByRequest[milestone.requestId] ?? []
      list.push(milestone)
      milestonesByRequest[milestone.requestId] = list
    }
  }
  return {
    roster,
    steps,
    milestonesByRequest,
    timePoints,
    maxSequence: events.length > 0 ? events[events.length - 1].sequence : 0,
    firstEventNs,
  }
}

export function sequenceAtElapsed(index: TimelineIndex, elapsedMs: number): number {
  let sequence = 0
  for (const point of index.timePoints) {
    if (point.elapsedMs == null) continue
    if (point.elapsedMs <= elapsedMs) sequence = point.sequence
    else break
  }
  return sequence
}

export function elapsedMsAt(index: TimelineIndex, sequence: number): number | null {
  let elapsed: number | null = null
  for (const point of index.timePoints) {
    if (point.sequence <= sequence) elapsed = point.elapsedMs
    else break
  }
  return elapsed
}

export function nextStepSequence(steps: Array<{ sequence: number }>, sequence: number): number | null {
  for (const step of steps) {
    if (step.sequence > sequence) return step.sequence
  }
  return null
}

export function previousStepSequence(steps: Array<{ sequence: number }>, sequence: number): number | null {
  let found: number | null = null
  for (const step of steps) {
    if (step.sequence >= sequence) break
    found = step.sequence
  }
  return found
}

export function nextStepForSample(index: TimelineIndex, sampleId: string, sequence: number): number | null {
  for (const step of index.steps) {
    if (step.sequence > sequence && step.participants.some((item) => item.requestId === sampleId)) {
      return step.sequence
    }
  }
  return null
}

export function previousStepForSample(index: TimelineIndex, sampleId: string, sequence: number): number | null {
  let found: number | null = null
  for (const step of index.steps) {
    if (step.sequence >= sequence) break
    if (step.participants.some((item) => item.requestId === sampleId)) found = step.sequence
  }
  return found
}

export function projectSamples(
  index: TimelineIndex,
  state: ReplayState,
  sequence: number,
  options?: {
    completionStatus?: string | null
    contentPrompts?: Record<string, string>
    submittedPrompts?: Record<string, string> | null
    arrivals?: Array<Record<string, unknown>>
  },
): SampleProjection[] {
  const completion = options?.completionStatus ?? null
  const contentPrompts = options?.contentPrompts ?? {}
  const submittedPrompts = options?.submittedPrompts ?? null
  const arrivals = options?.arrivals ?? []
  const elapsed = elapsedMsAt(index, sequence)
  const atEnd = sequence >= index.maxSequence
  return index.roster.samples.map((sample) => {
    const request = state.requests[sample.id]
    const phase = phaseFor(sample, state, request, elapsed, atEnd, completion)
    const scheduled = scheduledSegments(state, sample.id)
    const milestones = (index.milestonesByRequest[sample.id] ?? []).filter((item) => item.sequence <= sequence)
    const receipt = milestones.find((item) => item.kind === "http_request_received")
      ?? milestones.find((item) => item.kind === "request_received")
    const arrival = arrivals.find((item) => item.request_id === sample.id)
    const addReturned = request ? clockValue(arrival?.controller_add_returned_monotonic_ns) : null
    let promptText: string | null = null
    let promptSource: SampleProjection["promptSource"] = "unavailable"
    if (request?.promptText) {
      promptText = request.promptText
      promptSource = "recording"
    } else if (request && typeof contentPrompts[sample.id] === "string") {
      promptText = contentPrompts[sample.id]
      promptSource = "recording"
    } else if (submittedPrompts && typeof submittedPrompts[sample.id] === "string") {
      promptText = submittedPrompts[sample.id]
      promptSource = "submitted-plan"
    }
    return {
      sample,
      phase,
      phaseLabel: phaseLabel(phase),
      acceptedTokenCount: request ? request.acceptedTokenCount : null,
      scheduled,
      observedReceiptNs: receipt ? receipt.monotonicNs : null,
      controllerAddReturnedNs: addReturned,
      promptText,
      promptSource,
      outputText: request?.text ?? "",
      milestones,
      appliedStepIds: scheduled.map((item) => item.stepId),
    }
  })
}

export function describeBlocks(state: ReplayState): BlockFact[] {
  return Object.values(state.blocks)
    .sort((left, right) => left.pool.localeCompare(right.pool) || left.blockId - right.blockId)
    .map((block) => {
      const referencedBy = Object.values(state.requests)
        .filter((request) => request.blockIds.includes(block.blockId))
        .map((request) => request.id)
      const shared = block.refCount > 1 || referencedBy.length > 1
      const note = shared
        ? "This block is shared or listed by more than one request. Pool, block id, and allocation generation identify it. The block number alone is not exclusive ownership."
        : "Identity is the pool, block id, and allocation generation."
      return {
        key: `${block.pool}:${block.blockId}:${block.generation}`,
        pool: block.pool,
        blockId: block.blockId,
        generation: block.generation,
        refCount: block.refCount,
        referencedBy,
        shared,
        note,
      }
    })
}

export function promptsFromContent(content: Array<{ kind?: string; request_id?: string; text?: string }>): Record<string, string> {
  const prompts: Record<string, string> = {}
  for (const row of content) {
    if (row.kind === "prompt_text" && typeof row.request_id === "string" && typeof row.text === "string") {
      prompts[row.request_id] = row.text
    }
  }
  return prompts
}

function phaseFor(
  sample: PlannedSample,
  state: ReplayState,
  request: ReplayState["requests"][string] | undefined,
  elapsed: number | null,
  atEnd: boolean,
  completion: string | null,
): Phase {
  if (state.trust === "unknown" || state.gap) return "unknown"
  if (!request) {
    if (atEnd && (completion === "cancelled" || completion === "failed")) return "not_submitted"
    if (sample.arrivalOffsetMs != null && elapsed != null && elapsed >= sample.arrivalOffsetMs) return "awaiting"
    return "planned"
  }
  if (request.trust === "unknown") return "unknown"
  if (request.status === "completed") return "completed"
  if (request.status === "cancelled") return "cancelled"
  if (request.status === "preempted" || request.recomputing) return "preempted"
  const latest = [...state.steps].reverse().find((step) => step.requests.some((item) => item.requestId === sample.id))
  if (latest) {
    const row = latest.requests.find((item) => item.requestId === sample.id)
    if (row) {
      if (row.promptTokens > 0 && row.generationTokens > 0) return "mixed"
      if (row.generationTokens > 0 && row.promptTokens === 0) return "decode"
      if (row.promptTokens > 0) return "prefill"
    }
  }
  if (request.status === "queued") return "queued"
  if (request.status === "tokenized") return "tokenized"
  if (request.status === "received") return "received"
  if (request.status === "dispatched") return "dispatched"
  return "observed"
}

function scheduledSegments(state: ReplayState, requestId: string): ScheduledSegment[] {
  const segments: ScheduledSegment[] = []
  for (const step of state.steps) {
    const row = step.requests.find((item) => item.requestId === requestId)
    if (!row) continue
    const kind = row.promptTokens > 0 && row.generationTokens > 0 ? "mixed" : row.generationTokens > 0 ? "decode" : "prefill"
    segments.push({
      stepId: step.id,
      sequence: step.sequence,
      kind,
      rangeStart: row.rangeStart,
      rangeEnd: row.rangeStart + Math.max(row.rangeLength - 1, 0),
      promptTokens: row.promptTokens,
      generationTokens: row.generationTokens,
    })
  }
  return segments
}

function buildRoster(events: TraceEvent[], manifest: RecordingManifest | undefined, firstEventNs: number | string | null): Roster {
  const clock = readClock(manifest, firstEventNs)
  const plan = readPlan(manifest)
  const arrivals = Array.isArray(manifest?.arrivals) ? manifest.arrivals : []
  const arrivalById = new Map<string, Record<string, unknown>>()
  for (const arrival of arrivals) {
    if (arrival && typeof arrival === "object" && typeof (arrival as Record<string, unknown>).request_id === "string") {
      arrivalById.set(String((arrival as Record<string, unknown>).request_id), arrival as Record<string, unknown>)
    }
  }
  const samples: PlannedSample[] = []
  const batches: PlannedBatch[] = []
  const seen = new Set<string>()
  if (plan) {
    plan.batches.forEach((batch) => {
      const sampleIds: string[] = []
      batch.samples.forEach((sample) => {
        const arrival = arrivalById.get(sample.id)
        samples.push({
          id: sample.id,
          label: sample.label,
          batchId: batch.id,
          batchLabel: batch.label,
          arrivalOffsetMs: batch.arrivalOffsetMs,
          arrivalPolicy: typeof arrival?.requested_offset_policy === "string" ? arrival.requested_offset_policy : "controller_offset",
          maxOutputTokens: sample.maxOutputTokens,
          order: samples.length,
        })
        sampleIds.push(sample.id)
        seen.add(sample.id)
      })
      batches.push({
        id: batch.id,
        label: batch.label,
        arrivalOffsetMs: batch.arrivalOffsetMs,
        sampleIds,
      })
    })
  }
  const discovered = discoverIds(events, arrivals)
  for (const id of discovered) {
    if (seen.has(id)) continue
    const arrival = arrivalById.get(id)
    const policy = typeof arrival?.requested_offset_policy === "string" ? arrival.requested_offset_policy : null
    const offset = offsetFromArrival(arrival)
    samples.push({
      id,
      label: id,
      batchId: `ungrouped:${id}`,
      batchLabel: "Ungrouped",
      arrivalOffsetMs: offset,
      arrivalPolicy: policy,
      maxOutputTokens: null,
      order: samples.length,
    })
    batches.push({
      id: `ungrouped:${id}`,
      label: "Ungrouped",
      arrivalOffsetMs: offset,
      sampleIds: [id],
    })
    seen.add(id)
  }
  return {
    samples,
    batches,
    groupingKnown: Boolean(plan && !plan.synthesized),
    timeBasis: clock.basis,
    originNs: clock.originNs,
    timeBasisLabel: clock.label,
  }
}

function discoverIds(events: TraceEvent[], arrivals: unknown[]): string[] {
  const ids: string[] = []
  const seen = new Set<string>()
  const add = (value: unknown) => {
    if (typeof value !== "string" || !value || seen.has(value)) return
    seen.add(value)
    ids.push(value)
  }
  for (const arrival of arrivals) {
    if (arrival && typeof arrival === "object") add((arrival as Record<string, unknown>).request_id)
  }
  for (const event of events) {
    add(event.request_id)
    if (event.event_type === "step_scheduled" && Array.isArray(event.requests)) {
      for (const request of event.requests) {
        if (request && typeof request === "object") add((request as Record<string, unknown>).request_id)
      }
    }
  }
  return ids
}

function offsetFromArrival(arrival: Record<string, unknown> | undefined): number | null {
  if (!arrival) return null
  if (arrival.requested_offset_policy !== "controller_offset") return null
  const ns = arrival.requested_offset_ns
  if (typeof ns !== "number" || !Number.isFinite(ns)) return null
  return Math.round(ns / 1_000_000)
}

interface PlanBatch {
  id: string
  label: string
  arrivalOffsetMs: number | null
  samples: Array<{ id: string; label: string; maxOutputTokens: number | null }>
}

function readPlan(manifest?: RecordingManifest): { synthesized: boolean; batches: PlanBatch[] } | null {
  const raw = manifest?.workload_plan
  if (!raw || typeof raw !== "object") return null
  const plan = raw as Record<string, unknown>
  if (!Array.isArray(plan.batches)) return null
  const batches: PlanBatch[] = []
  for (const batch of plan.batches) {
    if (!batch || typeof batch !== "object") continue
    const row = batch as Record<string, unknown>
    const id = typeof row.id === "string" ? row.id : ""
    const label = typeof row.label === "string" && row.label.trim() ? row.label : id
    const offset = typeof row.arrival_offset_ms === "number" && Number.isFinite(row.arrival_offset_ms) ? row.arrival_offset_ms : null
    const samples = Array.isArray(row.samples)
      ? row.samples.flatMap((sample) => {
          if (!sample || typeof sample !== "object") return []
          const item = sample as Record<string, unknown>
          if (typeof item.id !== "string" || !item.id) return []
          return [
            {
              id: item.id,
              label: typeof item.label === "string" && item.label.trim() ? item.label : item.id,
              maxOutputTokens: typeof item.max_output_tokens === "number" ? item.max_output_tokens : null,
            },
          ]
        })
      : []
    if (!id || samples.length === 0) continue
    batches.push({ id, label, arrivalOffsetMs: offset, samples })
  }
  if (batches.length === 0) return null
  return { synthesized: plan.synthesized === true, batches }
}

function readClock(
  manifest: RecordingManifest | undefined,
  firstEventNs: number | string | null,
): { basis: TimeBasis; originNs: number | string | null; label: string } {
  const origin = manifest?.controller_origin
  if (origin && typeof origin === "object") {
    const started = clockValue((origin as Record<string, unknown>).controller_started_monotonic_ns)
    if (started != null) {
      return {
        basis: "controller_origin",
        originNs: started,
        label: "Controller execution time, after preparation",
      }
    }
  }
  const arrivals = Array.isArray(manifest?.arrivals) ? manifest.arrivals : []
  const triggered = arrivals.some((arrival) => {
    if (!arrival || typeof arrival !== "object") return false
    const policy = (arrival as Record<string, unknown>).requested_offset_policy
    return typeof policy === "string" && policy !== "controller_offset" && policy !== "immediate"
  })
  if (triggered) {
    return {
      basis: "event_triggered",
      originNs: firstEventNs,
      label: "Event-triggered arrivals. Observed time is since the first recorded event.",
    }
  }
  if (firstEventNs == null) {
    return { basis: "unavailable", originNs: null, label: "No recorded clock" }
  }
  return {
    basis: "since_first_event",
    originNs: firstEventNs,
    label: "Since first recorded event",
  }
}

function indexStep(event: TraceEvent): IndexedStep {
  const requests = Array.isArray(event.requests) ? event.requests : []
  return {
    id: typeof event.step_id === "string" ? event.step_id : "step",
    sequence: event.sequence,
    tokenBudget: typeof event.token_budget === "number" ? event.token_budget : 0,
    totalScheduledTokens: typeof event.total_scheduled_tokens === "number" ? event.total_scheduled_tokens : 0,
    participants: requests.map((item) => {
      const row = item && typeof item === "object" ? (item as Record<string, unknown>) : {}
      return {
        requestId: String(row.request_id ?? ""),
        rangeStart: typeof row.range_start === "number" ? row.range_start : 0,
        rangeLength: typeof row.range_length === "number" ? row.range_length : 0,
        promptTokens: typeof row.prompt_tokens === "number" ? row.prompt_tokens : 0,
        generationTokens: typeof row.generation_tokens === "number" ? row.generation_tokens : 0,
      }
    }),
  }
}

function milestonesFor(event: TraceEvent): Milestone[] {
  const requestId = typeof event.request_id === "string" ? event.request_id : ""
  const base = {
    sequence: event.sequence,
    monotonicNs: milestoneClock(event),
  }
  if (requestId && event.event_type === "request_dispatched") {
    return [{ ...base, requestId, kind: event.event_type, label: "Dispatch started" }]
  }
  if (requestId && event.event_type === "request_dispatch_completed") {
    const direct = typeof event.dispatch_transport === "string" && event.dispatch_transport !== "instrumented_loopback_http"
    return [{ ...base, requestId, kind: event.event_type, label: direct ? "Direct dispatch completed" : "Request body sent" }]
  }
  if (requestId && event.event_type === "http_request_received") {
    return [{ ...base, requestId, kind: event.event_type, label: "HTTP request received" }]
  }
  if (requestId && event.event_type === "engine_prompt_observed") {
    return [{ ...base, requestId, kind: event.event_type, label: "Engine prompt observed" }]
  }
  if (requestId && event.event_type === "engine_admitted") {
    return [{ ...base, requestId, kind: event.event_type, label: "Engine admitted" }]
  }
  if (requestId && event.event_type === "http_engine_admission_correlated") {
    return [{ ...base, requestId, kind: event.event_type, label: "HTTP and engine admission correlated" }]
  }
  if (requestId && event.event_type === "request_model_route") {
    const label = event.route_kind === "lora_adapter" ? "LoRA adapter selected" : "Base model selected"
    return [{ ...base, requestId, kind: event.event_type, label }]
  }
  if (requestId && event.event_type === "http_response_completed") {
    return [{ ...base, requestId, kind: event.event_type, label: "HTTP response completed" }]
  }
  if (requestId && event.event_type === "request_received") {
    return [{ ...base, requestId, kind: event.event_type, label: "Legacy receipt marker" }]
  }
  if (requestId && event.event_type === "request_tokenized") {
    return [{ ...base, requestId, kind: event.event_type, label: "Legacy tokenization marker" }]
  }
  if (requestId && event.event_type === "request_queued") {
    return [{ ...base, requestId, kind: event.event_type, label: "Queued" }]
  }
  if (requestId && event.event_type === "tokens_accepted") {
    return [{ ...base, requestId, kind: event.event_type, label: "Output accepted" }]
  }
  if (requestId && event.event_type === "text_emitted") {
    return [{ ...base, requestId, kind: event.event_type, label: "Text emitted" }]
  }
  if (requestId && event.event_type === "preemption") {
    return [{ ...base, requestId, kind: event.event_type, label: "Preempted" }]
  }
  if (requestId && (event.event_type === "completion" || event.event_type === "cancellation")) {
    return [{
      ...base,
      requestId,
      kind: event.event_type,
      label: event.event_type === "completion" ? "Completed" : "Cancelled",
    }]
  }
  if (event.event_type === "step_scheduled" && Array.isArray(event.requests)) {
    return event.requests.flatMap((item) => {
      if (!item || typeof item !== "object") return []
      const id = (item as Record<string, unknown>).request_id
      if (typeof id !== "string" || !id) return []
      return [{ ...base, requestId: id, kind: "step_scheduled", label: "Scheduled" }]
    })
  }
  return []
}

function milestoneClock(event: TraceEvent): number | string {
  const boundaryField =
    event.event_type === "request_dispatched"
      ? event.dispatch_started_monotonic_ns
      : event.event_type === "request_dispatch_completed"
        ? event.dispatch_completed_monotonic_ns
      : event.event_type === "http_request_received"
        ? event.http_received_monotonic_ns
        : event.event_type === "http_engine_admission_correlated"
          ? event.engine_admitted_monotonic_ns
          : event.event_type === "http_response_completed"
            ? event.http_responded_monotonic_ns
            : null
  return clockValue(boundaryField) ?? clockValue(event.monotonic_ns) ?? Number.NaN
}

function elapsedFrom(monotonicNs: number | string, originNs: number | string | null): number | null {
  const monotonic = parseNs(monotonicNs)
  const origin = parseNs(originNs)
  if (monotonic == null || origin == null) return null
  const delta = monotonic - origin
  if (delta > BigInt(Number.MAX_SAFE_INTEGER) || delta < BigInt(Number.MIN_SAFE_INTEGER)) return null
  return Math.round(Number(delta) / 1_000_000)
}

function clockValue(value: unknown): number | string | null {
  const parsed = parseNs(value)
  if (parsed == null) return null
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER) || parsed < BigInt(Number.MIN_SAFE_INTEGER)) return parsed.toString()
  return Number(parsed)
}

function parseNs(value: unknown): bigint | null {
  if (typeof value === "bigint") return value
  if (typeof value === "string" && /^-?\d+$/.test(value)) return BigInt(value)
  if (typeof value === "number" && Number.isInteger(value)) return BigInt(value)
  return null
}
