import assert from "node:assert/strict"
import { test } from "node:test"

import { attachContent, createPlayer, parseJsonl, type ContentRecord, type TraceEvent } from "./reduce.ts"
import {
  buildTimeline,
  describeBlocks,
  nextStepForSample,
  projectSamples,
  sequenceAtElapsed,
} from "./timeline.ts"

function event(sequence: number, eventType: string, fields: Record<string, unknown> = {}, ns = sequence * 1_000_000): TraceEvent {
  return {
    schema_version: "1.0.0",
    run_id: "test",
    producer_id: "vllm-030-adapter",
    sequence,
    monotonic_ns: ns,
    event_type: eventType,
    ...fields,
  }
}

function step(
  sequence: number,
  stepId: string,
  requests: Array<Record<string, unknown>>,
  ns?: number,
): TraceEvent {
  return event(sequence, "step_scheduled", {
    step_id: stepId,
    token_budget: 32,
    total_scheduled_tokens: requests.reduce((sum, item) => sum + Number(item.range_length ?? 0), 0),
    requests,
  }, ns)
}

const plan = {
  version: 1,
  synthesized: false,
  batches: [
    {
      id: "batch-a",
      label: "A",
      arrival_offset_ms: 0,
      samples: [
        { id: "sample-1", label: "input_1", max_output_tokens: 8, order: 0 },
        { id: "sample-2", label: "input_2", max_output_tokens: 8, order: 1 },
      ],
    },
    {
      id: "batch-b",
      label: "B",
      arrival_offset_ms: 100,
      samples: [{ id: "sample-4", label: "input_4", max_output_tokens: 8, order: 2 }],
    },
  ],
}

test("planned rows stay visible before any receipt event", () => {
  const index = buildTimeline([], {
    workload_plan: plan,
    controller_origin: {
      controller_started_monotonic_ns: 1_000_000_000,
      provenance: "injected_monotonic",
    },
    completion_status: "completed",
  })
  const player = createPlayer([])
  const view = projectSamples(index, player.at(0), 0, { completionStatus: "completed" })
  assert.deepEqual(view.map((item) => item.sample.label), ["input_1", "input_2", "input_4"])
  assert.deepEqual(view.map((item) => item.phase), ["planned", "planned", "planned"])
  assert.equal(view[2].acceptedTokenCount, null)
  assert.equal(index.roster.groupingKnown, true)
  assert.equal(index.roster.timeBasis, "controller_origin")
})

test("HTTP intake milestones retain their named boundaries and exact observed clocks", () => {
  const events = [
    event(1, "capture_ready", { cache_reset_verified: true }, 500),
    event(2, "request_dispatched", {
      request_id: "sample-1",
      dispatch_started_monotonic_ns: 1_000,
    }, 10_000),
    event(3, "request_dispatch_completed", {
      request_id: "sample-1",
      dispatch_completed_monotonic_ns: 1_500,
    }, 15_000),
    event(4, "http_request_received", {
      request_id: "sample-1",
      http_received_monotonic_ns: 2_000,
    }, 20_000),
    event(5, "request_model_route", {
      request_id: "sample-1",
      adapter_id: "adapter-a",
      lora_int_id: 1,
      route_kind: "lora_adapter",
    }, 25_000),
    event(6, "engine_prompt_observed", {
      request_id: "sample-1",
      prompt_token_count: 7,
    }, 30_000),
    event(7, "engine_admitted", {
      request_id: "sample-1",
      status: "scheduler_added",
    }, 40_000),
    event(8, "http_engine_admission_correlated", {
      request_id: "sample-1",
      engine_admitted_monotonic_ns: 4_100,
    }, 50_000),
    event(9, "http_response_completed", {
      request_id: "sample-1",
      http_responded_monotonic_ns: 5_000,
      status_code: 200,
    }, 60_000),
    event(10, "completion", { request_id: "sample-1", status: "stop" }, 70_000),
  ]
  const manifest = {
    workload_plan: plan,
    controller_origin: { controller_started_monotonic_ns: 0 },
    completion_status: "completed",
  }
  const index = buildTimeline(events, manifest)
  const player = createPlayer(events, manifest)
  const phaseAt = (sequence: number) => projectSamples(
    index,
    player.at(sequence),
    sequence,
    { completionStatus: "completed" },
  ).find((item) => item.sample.id === "sample-1")

  assert.equal(phaseAt(2)?.phase, "dispatched")
  assert.equal(phaseAt(3)?.phase, "dispatched")
  assert.equal(phaseAt(4)?.phase, "received")
  assert.equal(phaseAt(6)?.phase, "tokenized")
  assert.equal(phaseAt(7)?.phase, "queued")
  assert.equal(phaseAt(9)?.phase, "queued")
  assert.equal(phaseAt(10)?.phase, "completed")

  const final = phaseAt(10)
  assert.equal(final?.observedReceiptNs, 2_000)
  assert.deepEqual(
    final?.milestones.map((item) => [item.kind, item.label, item.monotonicNs]),
    [
      ["request_dispatched", "Dispatch started", 1_000],
      ["request_dispatch_completed", "Request body sent", 1_500],
      ["http_request_received", "HTTP request received", 2_000],
      ["request_model_route", "LoRA adapter selected", 25_000],
      ["engine_prompt_observed", "Engine prompt observed", 30_000],
      ["engine_admitted", "Engine admitted", 40_000],
      ["http_engine_admission_correlated", "HTTP and engine admission correlated", 4_100],
      ["http_response_completed", "HTTP response completed", 5_000],
      ["completion", "Completed", 70_000],
    ],
  )
})

test("prefix playback hides future output and restores it only at that event", () => {
  const events = [
    event(1, "request_received", { request_id: "sample-4" }, 1_000_100_000),
    event(2, "request_queued", { request_id: "sample-4" }, 1_000_101_000),
    step(3, "step-prefill", [{
      request_id: "sample-4",
      range_start: 0,
      range_length: 4,
      prompt_tokens: 4,
      generation_tokens: 0,
      num_prompt_tokens: 4,
    }], 1_000_110_000),
    step(4, "step-decode", [{
      request_id: "sample-4",
      range_start: 4,
      range_length: 1,
      prompt_tokens: 0,
      generation_tokens: 1,
      num_prompt_tokens: 4,
    }], 1_000_140_000),
    event(5, "tokens_accepted", {
      request_id: "sample-4",
      step_id: "step-decode",
      token_count: 1,
      output_position_start: 0,
    }, 1_000_141_000),
    event(6, "text_emitted", {
      request_id: "sample-4",
      emission_index: 0,
      token_range_start: 0,
      token_range_end: 1,
    }, 1_000_142_000),
    event(7, "completion", { request_id: "sample-4", status: "stop" }, 1_000_150_000),
  ]
  const content: ContentRecord[] = [
    { kind: "prompt_text", request_id: "sample-4", text: "Name a fruit." },
    { kind: "text", request_id: "sample-4", emission_index: 0, text_delta: " apple" },
  ]
  const manifest = {
    workload_plan: plan,
    controller_origin: { controller_started_monotonic_ns: 1_000_000_000 },
    arrivals: [{
      request_id: "sample-4",
      requested_offset_ns: 100_000_000,
      requested_offset_policy: "controller_offset",
      controller_add_returned_monotonic_ns: 1_000_100_500,
    }],
    completion_status: "completed",
  }
  const index = buildTimeline(events, manifest)
  const player = createPlayer(events, manifest)
  const early = projectSamples(index, attachContent(player.at(2), content), 2, {
    completionStatus: "completed",
    arrivals: manifest.arrivals,
  })
  const input4 = early.find((item) => item.sample.id === "sample-4")
  assert.ok(input4)
  assert.equal(input4.phase, "queued")
  assert.equal(input4.outputText, "")
  assert.equal(input4.acceptedTokenCount, 0)
  assert.equal(input4.controllerAddReturnedNs, 1_000_100_500)
  assert.equal(input4.scheduled.length, 0)
  const beforeReceipt = projectSamples(index, attachContent(player.at(0), content), 0, {
    completionStatus: "completed",
    arrivals: manifest.arrivals,
    contentPrompts: { "sample-4": "Name a fruit." },
  }).find((item) => item.sample.id === "sample-4")
  assert.equal(beforeReceipt?.controllerAddReturnedNs, null)
  assert.equal(beforeReceipt?.outputText, "")
  assert.equal(beforeReceipt?.promptText, null)
  const scheduled = projectSamples(index, attachContent(player.at(4), content), 4, {
    completionStatus: "completed",
    arrivals: manifest.arrivals,
  }).find((item) => item.sample.id === "sample-4")
  assert.equal(scheduled?.phase, "decode")
  assert.equal(scheduled?.outputText, "")
  assert.equal(scheduled?.acceptedTokenCount, 0)
  const done = projectSamples(index, attachContent(player.at(7), content), 7, {
    completionStatus: "completed",
    arrivals: manifest.arrivals,
  }).find((item) => item.sample.id === "sample-4")
  assert.equal(done?.phase, "completed")
  assert.equal(done?.outputText, " apple")
  assert.equal(done?.acceptedTokenCount, 1)
  assert.equal(done?.observedReceiptNs, 1_000_100_000)
  assert.equal(done?.controllerAddReturnedNs, 1_000_100_500)
  assert.equal(done?.promptText, "Name a fruit.")
  assert.deepEqual(done?.scheduled.map((item) => item.kind), ["prefill", "decode"])
  assert.equal(nextStepForSample(index, "sample-4", 3), 4)
})

test("mixed ranges, preemption, trace gaps, and unsubmitted samples stay distinct", () => {
  const events = [
    event(1, "request_received", { request_id: "sample-1" }, 5_000),
    step(2, "step-mixed", [{
      request_id: "sample-1",
      range_start: 3,
      range_length: 2,
      prompt_tokens: 1,
      generation_tokens: 1,
      num_prompt_tokens: 4,
    }], 6_000),
    event(3, "preemption", {
      request_id: "sample-1",
      preemption_count: 1,
      computed_tokens_before: 5,
      computed_tokens_after: 0,
    }, 7_000),
    event(4, "trace_gap", { reason: "buffer_exhausted", missing_sequence_start: 5, missing_sequence_end: 6 }, 8_000),
  ]
  const manifest = {
    workload_plan: plan,
    completion_status: "cancelled",
    arrivals: [],
    event_completeness: { status: "incomplete", gap: { sequence: 4, reason: "buffer_exhausted" } },
  }
  const index = buildTimeline(events, manifest)
  const player = createPlayer(events, manifest)
  const beforeGap = projectSamples(index, player.at(3), 3, { completionStatus: "cancelled" })
  const mixed = beforeGap.find((item) => item.sample.id === "sample-1")
  assert.equal(mixed?.phase, "preempted")
  assert.equal(mixed?.scheduled[0]?.kind, "mixed")
  const waiting = beforeGap.find((item) => item.sample.id === "sample-4")
  assert.equal(waiting?.phase, "planned")
  const afterGap = projectSamples(index, player.at(4), 4, { completionStatus: "cancelled" })
  assert.ok(afterGap.every((item) => item.phase === "unknown"))
  const ended = buildTimeline([event(1, "request_received", { request_id: "sample-1" })], {
    workload_plan: plan,
    completion_status: "cancelled",
    controller_origin: { controller_started_monotonic_ns: 100 },
  })
  const endState = createPlayer([event(1, "request_received", { request_id: "sample-1" })]).at(1)
  const terminal = projectSamples(ended, endState, 1, { completionStatus: "cancelled" })
  assert.equal(terminal.find((item) => item.sample.id === "sample-4")?.phase, "not_submitted")
  assert.equal(terminal.find((item) => item.sample.id === "sample-1")?.phase, "received")
})

test("legacy recordings do not invent an origin or arrival groups", () => {
  const events = [
    event(1, "request_received", { request_id: "request-a" }, 50_000),
    event(2, "request_received", { request_id: "request-b" }, 50_000),
  ]
  const index = buildTimeline(events, {
    arrivals: [
      {
        request_id: "request-b",
        requested_offset_ns: null,
        requested_offset_policy: "after_request_a_first_accepted_token",
        controller_add_returned_monotonic_ns: 90_000,
      },
      {
        request_id: "request-a",
        requested_offset_ns: 0,
        requested_offset_policy: "immediate",
        controller_add_returned_monotonic_ns: 50_000,
      },
    ],
  })
  assert.equal(index.roster.groupingKnown, false)
  assert.equal(index.roster.timeBasis, "event_triggered")
  assert.equal(index.roster.samples[0]?.arrivalOffsetMs, null)
  assert.equal(sequenceAtElapsed(index, 0), 2)
  const sameTime = index.timePoints.filter((point) => point.elapsedMs === 0)
  assert.deepEqual(sameTime.map((point) => point.sequence), [1, 2])
})

test("recorded time keeps elapsed across timestamps outside the safe integer range", () => {
  const origin = "9007199254740993"
  const later = (9007199254740993n + 120_000_000n).toString()
  const events = [
    event(1, "request_received", { request_id: "sample-1" }),
    event(2, "request_received", { request_id: "sample-4" }),
  ]
  events[0].monotonic_ns = origin
  events[1].monotonic_ns = later
  const index = buildTimeline(events, {
    controller_origin: { controller_started_monotonic_ns: origin },
    workload_plan: plan,
  })
  assert.equal(index.roster.timeBasis, "controller_origin")
  assert.equal(index.timePoints[0]?.elapsedMs, 0)
  assert.equal(index.timePoints[1]?.elapsedMs, 120)
  assert.equal(sequenceAtElapsed(index, 119), 1)
  assert.equal(sequenceAtElapsed(index, 120), 2)
  const parsed = parseJsonl<TraceEvent>('{"schema_version":"1.0.0","run_id":"t","producer_id":"p","sequence":1,"monotonic_ns":9007199254740993,"event_type":"request_received"}\n')
  assert.equal(parsed[0]?.monotonic_ns, origin)
})

test("a missing controller origin stays anchored at the first event", () => {
  const admitted = (9007199254740993n + 100_000_000n).toString()
  const events = [event(1, "request_received", { request_id: "sample-1" })]
  events[0].monotonic_ns = admitted
  const index = buildTimeline(events, {
    arrivals: [{
      request_id: "sample-1",
      requested_offset_ns: 100_000_000,
      requested_offset_policy: "controller_offset",
      controller_add_returned_monotonic_ns: admitted,
    }],
  })
  assert.equal(index.roster.timeBasis, "since_first_event")
  assert.equal(index.timePoints[0]?.elapsedMs, 0)
})

test("shared blocks are not assigned to one request", () => {
  const events = [
    event(1, "block_change", {
      pool: "pool-0",
      block_id: 3,
      allocation_generation: 2,
      operation: "allocate",
      ref_count_after: 2,
      has_hash_after: false,
      cached_at_zero_refs: false,
    }),
  ]
  const state = createPlayer(events).at(1)
  state.requests["sample-1"] = {
    ...blank("sample-1"),
    blockIds: [3],
  }
  state.requests["sample-4"] = {
    ...blank("sample-4"),
    blockIds: [3],
  }
  const [block] = describeBlocks(state)
  assert.equal(block?.key, "pool-0:3:2")
  assert.equal(block?.shared, true)
  assert.deepEqual(block?.referencedBy.sort(), ["sample-1", "sample-4"])
})

test("terminal and preempted requests stop owning reused block numbers", () => {
  for (const terminal of ["completion", "cancellation", "preemption", "http_response_completed"]) {
    const events = [
      event(1, "block_change", { pool: "pool-0", block_id: 3, allocation_generation: 1, ref_count_after: 1 }),
      event(2, "step_scheduled", { step_id: "first", requests: [], block_table_changes: [{ request_id: "a", change: "replace", block_ids: [[3]] }] }),
      event(3, terminal, { request_id: "a", status_code: 200, finish_reason: "stop" }),
      event(4, "block_change", { pool: "pool-0", block_id: 3, allocation_generation: 2, ref_count_after: 1 }),
      event(5, "step_scheduled", { step_id: "second", requests: [], block_table_changes: [{ request_id: "b", change: "replace", block_ids: [[3]] }] }),
    ]
    const state = createPlayer(events).at(5)
    assert.deepEqual(state.requests.a.blockIds, [], terminal)
    const [block] = describeBlocks(state)
    assert.deepEqual(block.referencedBy, ["b"], terminal)
    assert.equal(block.shared, false, terminal)
  }
})

test("block tables do not claim ownership of a free block or an ambiguous pool", () => {
  const events = [
    event(1, "block_change", { pool: "pool-0", block_id: 3, allocation_generation: 1, ref_count_after: 0 }),
    event(2, "step_scheduled", { step_id: "first", requests: [], block_table_changes: [{ request_id: "a", change: "replace", block_ids: [[3]] }] }),
    event(3, "block_change", { pool: "pool-1", block_id: 3, allocation_generation: 1, ref_count_after: 1 }),
  ]
  const state = createPlayer(events).at(3)
  assert.deepEqual(describeBlocks(state).map((block) => block.referencedBy), [[], []])
})

test("active requests shedding a block do not retain ownership of its next allocation", () => {
  const events = [
    event(1, "block_change", { pool: "pool-0", block_id: 3, allocation_generation: 1, ref_count_after: 1 }),
    event(2, "step_scheduled", { step_id: "first", requests: [], block_table_changes: [{ request_id: "a", change: "replace", block_ids: [[3]] }] }),
    event(3, "block_change", { pool: "pool-0", block_id: 3, allocation_generation: 1, ref_count_after: 0 }),
    event(4, "block_change", { pool: "pool-0", block_id: 3, allocation_generation: 2, ref_count_after: 1 }),
    event(5, "step_scheduled", { step_id: "second", requests: [], block_table_changes: [{ request_id: "b", change: "replace", block_ids: [[3]] }] }),
  ]
  const state = createPlayer(events).at(5)
  assert.deepEqual(state.requests.a.blockIds, [])
  assert.deepEqual(describeBlocks(state)[0].referencedBy, ["b"])
  assert.equal(describeBlocks(state)[0].shared, false)
})

function blank(id: string) {
  return {
    id,
    status: "queued",
    promptTokenCount: null,
    promptText: null,
    promptTokenIds: null,
    reservedCursor: null,
    blockIds: [] as number[],
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
    trust: "known" as const,
    finishReason: null,
  }
}
