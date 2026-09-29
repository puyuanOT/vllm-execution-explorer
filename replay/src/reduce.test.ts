import assert from "node:assert/strict"
import { test } from "node:test"

import { explainBlock, explainRequest } from "./explain.ts"
import {
  attachContent,
  createPlayer,
  replay,
  type ContentRecord,
  type TraceEvent,
} from "./reduce.ts"

function event(sequence: number, eventType: string, fields: Record<string, unknown> = {}): TraceEvent {
  return {
    schema_version: "1.0.0",
    run_id: "test",
    producer_id: "vllm-030-adapter",
    sequence,
    monotonic_ns: sequence * 1000,
    event_type: eventType,
    ...fields,
  }
}

function sample(): TraceEvent[] {
  return [
    event(1, "request_received", { request_id: "a" }),
    event(2, "request_tokenized", { request_id: "a", prompt_token_count: 4 }),
    event(3, "request_queued", { request_id: "a", status: "queued" }),
    event(4, "step_scheduled", {
      step_id: "step-1",
      token_budget: 32,
      total_scheduled_tokens: 4,
      requests: [
        {
          request_id: "a",
          range_start: 0,
          range_length: 4,
          prompt_tokens: 4,
          generation_tokens: 0,
          num_prompt_tokens: 4,
        },
      ],
      preempted_request_ids: [],
      block_table_changes: [{ request_id: "a", change: "replace", block_ids: [[3]] }],
    }),
    event(5, "block_change", {
      pool: "pool-0",
      block_id: 3,
      allocation_generation: 1,
      operation: "allocate",
      ref_count_before: 0,
      ref_count_after: 1,
      has_hash_before: false,
      has_hash_after: false,
      cached_at_zero_refs: false,
    }),
    event(6, "tokens_accepted", {
      request_id: "a",
      step_id: "step-1",
      output_position_start: 0,
      token_count: 1,
    }),
    event(7, "text_emitted", {
      request_id: "a",
      emission_index: 0,
      token_range_start: 0,
      token_range_end: 1,
    }),
    event(8, "preemption", {
      request_id: "a",
      computed_tokens_before: 4,
      computed_tokens_after: 0,
      preemption_count: 1,
    }),
    event(9, "block_change", {
      pool: "pool-0",
      block_id: 3,
      allocation_generation: 1,
      operation: "release",
      ref_count_before: 1,
      ref_count_after: 0,
      has_hash_before: true,
      has_hash_after: true,
      cached_at_zero_refs: true,
    }),
  ]
}

test("replay is deterministic and seeks match the event prefix", () => {
  const events = sample()
  const speed = 4
  assert.deepEqual(replay(events), replay(events))
  assert.equal(speed, 4)
  const player = createPlayer(events)
  for (const item of events) {
    assert.deepEqual(
      player.at(item.sequence),
      replay(events.filter((candidate) => candidate.sequence <= item.sequence)),
    )
  }
  assert.equal(player.at(0).appliedCount, 0)
  assert.equal(player.at(4).requests.a.reservedCursor, 4)
  assert.equal(player.at(8).requests.a.recomputing, true)
  assert.equal(player.at(8).requests.a.reservedCursor, 0)
  assert.equal(player.at(8).requests.a.preemptionReason, null)
})

test("HTTP capture boundaries apply recorded intake facts without regressing later state", () => {
  const events = [
    event(1, "capture_ready", { cache_reset_verified: true }),
    event(2, "request_dispatched", {
      request_id: "http-a",
      dispatch_started_monotonic_ns: 1_100,
      dispatch_transport: "instrumented_loopback_http",
    }),
    event(3, "http_request_received", {
      request_id: "http-a",
      http_received_monotonic_ns: 1_200,
    }),
    event(4, "engine_prompt_observed", {
      request_id: "http-a",
      prompt_token_count: 12,
      observation_boundary: "scheduler.add_request",
    }),
    event(5, "engine_admitted", {
      request_id: "http-a",
      status: "scheduler_added",
    }),
    event(6, "step_scheduled", {
      step_id: "step-http",
      token_budget: 32,
      total_scheduled_tokens: 12,
      requests: [
        {
          request_id: "http-a",
          range_start: 0,
          range_length: 12,
          prompt_tokens: 12,
          generation_tokens: 0,
          num_prompt_tokens: 12,
        },
      ],
    }),
    event(7, "completion", { request_id: "http-a", status: "stop" }),
    // Correlation and dispatch summary records can be serialized after the
    // engine event they describe. They must not move terminal state backward.
    event(8, "http_engine_admission_correlated", {
      request_id: "http-a",
      engine_admitted_monotonic_ns: 1_300,
    }),
    event(9, "request_dispatch_completed", {
      request_id: "http-a",
      dispatch_completed_monotonic_ns: 1_150,
    }),
    event(10, "http_response_completed", {
      request_id: "http-a",
      status_code: 200,
      http_responded_monotonic_ns: 2_000,
    }),
  ]
  const player = createPlayer(events)
  assert.equal(Object.keys(player.at(1).requests).length, 0)
  assert.equal(player.at(2).requests["http-a"].status, "dispatched")
  assert.equal(player.at(3).requests["http-a"].status, "received")
  assert.equal(player.at(4).requests["http-a"].status, "tokenized")
  assert.equal(player.at(4).requests["http-a"].promptTokenCount, 12)
  assert.equal(player.at(5).requests["http-a"].status, "queued")
  assert.equal(player.at(6).requests["http-a"].status, "scheduled")
  assert.equal(player.at(10).requests["http-a"].status, "completed")
  assert.equal(player.at(10).appliedCount, 10)
})

test("a successful HTTP response with an engine finish reason closes an early-EOS request", () => {
  const state = replay([
    event(1, "engine_prompt_observed", { request_id: "short", prompt_token_count: 6 }),
    event(2, "step_scheduled", {
      step_id: "step-short",
      token_budget: 32,
      total_scheduled_tokens: 1,
      requests: [
        {
          request_id: "short",
          range_start: 6,
          range_length: 1,
          prompt_tokens: 0,
          generation_tokens: 1,
          num_prompt_tokens: 6,
        },
      ],
    }),
    event(3, "tokens_accepted", {
      request_id: "short",
      step_id: "step-short",
      output_position_start: 0,
      token_count: 1,
    }),
    // vLLM can surface an early EOS finish reason to RequestOutput without
    // including it in the scheduler result observed by the completion adapter.
    event(4, "http_response_completed", {
      request_id: "short",
      status_code: 200,
      finish_reason: "stop",
    }),
  ])

  assert.equal(state.requests.short.status, "completed")
  assert.equal(state.requests.short.finishReason, "stop")
  assert.equal(state.requests.short.recomputing, false)
})

test("an HTTP boundary without a successful terminal finish reason is not completion evidence", () => {
  const withoutReason = replay([
    event(1, "request_dispatched", { request_id: "pending" }),
    event(2, "http_response_completed", { request_id: "pending", status_code: 200 }),
  ])
  const failedResponse = replay([
    event(1, "request_dispatched", { request_id: "failed" }),
    event(2, "http_response_completed", {
      request_id: "failed",
      status_code: 500,
      finish_reason: "failed",
    }),
  ])

  assert.equal(withoutReason.requests.pending.status, "dispatched")
  assert.equal(withoutReason.requests.pending.finishReason, null)
  assert.equal(failedResponse.requests.failed.status, "dispatched")
  assert.equal(failedResponse.requests.failed.finishReason, null)
})

test("a trace gap prevents later HTTP evidence from being treated as observed", () => {
  const state = replay(
    [
      event(1, "request_dispatched", { request_id: "a" }),
      event(2, "trace_gap", { reason: "buffer_exhausted", missing_sequence_start: 3 }),
      event(3, "http_request_received", { request_id: "a", http_received_monotonic_ns: 4_000 }),
      event(4, "http_response_completed", { request_id: "unseen", status_code: 200 }),
    ],
    { event_completeness: { status: "incomplete", gap: { missing_sequence_start: 3, missing_sequence_end: 4 } } },
  )
  assert.equal(state.trust, "unknown")
  assert.equal(state.requests.a.status, "dispatched")
  assert.equal(state.requests.unseen, undefined)
  assert.equal(state.lastSequence, 2)
  assert.equal(state.appliedCount, 2)
})

test("a trace gap freezes later events", () => {
  const events = [
    event(1, "block_change", {
      pool: "pool-0",
      block_id: 3,
      allocation_generation: 1,
      operation: "allocate",
      ref_count_after: 1,
      has_hash_after: false,
      cached_at_zero_refs: false,
    }),
    event(2, "trace_gap", { reason: "buffer_exhausted", missing_sequence_start: 3 }),
    event(3, "block_change", {
      pool: "pool-0",
      block_id: 3,
      allocation_generation: 2,
      operation: "allocate",
      ref_count_after: 1,
      has_hash_after: false,
      cached_at_zero_refs: false,
    }),
  ]
  const state = replay(events, {
    event_completeness: { status: "incomplete", gap: { missing_sequence_start: 3, missing_sequence_end: 3 } },
  })
  assert.equal(state.trust, "unknown")
  assert.equal(state.blocks["pool-0:3"].generation, 1)
  assert.equal(state.gap?.end, 3)
  assert.equal(state.appliedCount, 2)
})

test("content stays out of the structural state until it is attached", () => {
  const events = sample().slice(0, 7)
  const structural = replay(events)
  assert.equal(structural.requests.a.text, "")
  assert.equal(structural.requests.a.acceptedSpans[0].tokenIds, null)
  const content: ContentRecord[] = [
    { kind: "prompt_text", request_id: "a", text: "hello", content_sequence: 1 },
    { kind: "accepted_tokens", request_id: "a", output_position_start: 0, token_ids: [11], content_sequence: 2 },
    { kind: "text", request_id: "a", emission_index: 0, text_delta: "Hi", content_sequence: 3 },
  ]
  const early = attachContent(createPlayer(events).at(6), content)
  assert.equal(early.requests.a.promptText, "hello")
  assert.deepEqual(early.requests.a.acceptedTokenIds, [11])
  assert.equal(early.requests.a.text, "")
  const withText = attachContent(createPlayer(events).at(7), content)
  assert.equal(withText.requests.a.text, "Hi")
})

test("unsupported schema and a broken sequence are rejected", () => {
  const broken = event(1, "request_received", { request_id: "a" })
  broken.schema_version = "9.0.0"
  assert.throws(() => replay([broken]), /unsupported recording schema/)
  assert.throws(() => replay([event(2, "request_received", { request_id: "a" })]), /expected sequence 1/)
  assert.throws(() => replay([event(1, "kernel_span")]), /unknown event type/)
})

test("request model routing is a supported observed boundary", () => {
  const state = replay([
    event(1, "request_received", { request_id: "r1" }),
    event(2, "request_model_route", {
      request_id: "r1",
      adapter_id: "adapter-a",
      lora_int_id: 7,
      route_kind: "lora_adapter",
    }),
  ])

  assert.equal(state.lastEventType, "request_model_route")
  assert.equal(state.requests.r1.status, "received")
  assert.equal(state.requests.r1.adapterId, "adapter-a")
  assert.equal(state.requests.r1.loraIntId, 7)
  assert.equal(state.requests.r1.modelRouteKind, "lora_adapter")
  assert.equal(state.requests.r1.modelRouteSequence, 2)
})

test("explanations keep observed facts apart from policy", () => {
  const events = sample()
  const state = replay(events)
  const request = explainRequest(state, "a")
  assert.match(request.observed.join("\n"), /No decision reason was recorded/)
  assert.doesNotMatch(request.observed.join("\n"), /cache was full|memory pressure/)
  assert.match(request.explanatory.join("\n"), /did not include a decision reason/)
  assert.match(request.explanatory.join("\n"), /Candidate probabilities were not captured/)
  const block = explainBlock(state, "pool-0:3")
  assert.match(block.observed.join("\n"), /cached/)
  assert.match(block.explanatory.join("\n"), /new allocation generation/)
})
