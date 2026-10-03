import assert from "node:assert/strict"
import { test } from "node:test"
import { attachContent, createPlayer, replay, type ContentRecord, type TraceEvent } from "./reduce.ts"
import { projectArchitecture, projectKvCache, projectRequestTokens, projectScheduler } from "./learning.ts"

function event(sequence: number, event_type: string, fields: Record<string, unknown> = {}): TraceEvent {
  return { schema_version: "1.0.0", run_id: "learning", producer_id: "adapter", sequence, monotonic_ns: sequence * 100, event_type, ...fields }
}

function inventory(sequence = 1, capacity = 8): TraceEvent {
  return event(sequence, "cache_inventory", {
    pools: [{ pool: "pool-0", block_size: 8, num_gpu_blocks: capacity, inventory_complete: true,
      blocks: Array.from({ length: capacity }, (_, block_id) => ({ block_id, allocation_generation: 0, ref_count: block_id === 0 ? 1 : 0, has_hash: false, is_null: block_id === 0 })) }],
    groups: [{ cache_group: 0, pool: "pool-0", block_size: 8, layer_names: ["model.layers.0.self_attn"], cache_type: "FullAttentionSpec",
      token_slot_mapping: "authored_fixture", token_slot_mapping_reason: "Explicit test fixture slot mapper." }],
  })
}

function block(sequence: number, fields: Record<string, unknown> = {}): TraceEvent {
  return event(sequence, "block_change", { pool: "pool-0", block_id: 1, allocation_generation: 1, operation: "allocate", ref_count_after: 1, has_hash_after: false, cached_at_zero_refs: false, ...fields })
}

function kv(sequence: number, type: "kv_block_binding" | "kv_block_written", fields: Record<string, unknown> = {}): TraceEvent {
  return event(sequence, type, { step_id: "step-1", request_id: "a", pool: "pool-0", block_id: 1, allocation_generation: 1, cache_group: 0, token_range_start: 0, token_range_end: 8, ...fields })
}

function sample(): TraceEvent[] {
  return [inventory(), event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }), block(3),
    event(4, "step_scheduled", { step_id: "step-1", token_budget: 8, total_scheduled_tokens: 3,
      requests: [{ request_id: "a", range_start: 0, range_length: 3, prompt_tokens: 3, generation_tokens: 0, num_prompt_tokens: 8 }],
      block_table_changes: [{ request_id: "a", change: "replace", block_ids: [[1]] }] }),
    kv(5, "kv_block_binding"), event(6, "step_submitted", { step_id: "step-1" }),
    event(7, "step_result_received", { step_id: "step-1", observation: "future.result" }),
    kv(8, "kv_block_written", { token_range_end: 3 }),
  ]
}

const content: ContentRecord[] = [{ kind: "prompt", request_id: "a", token_ids: [10, 11, 12, 13, 14, 15, 16, 17] },
  { kind: "kv_tokens", step_id: "step-1", request_id: "a", pool: "pool-0", block_id: 1, allocation_generation: 1,
    cache_group: 0, token_range_start: 0, token_range_end: 3, token_ids: [10, 11, 12], token_pieces: ["a", "b", "c"] }]

test("full inventory is authoritative, includes null cells, and leaves checkpoint state sparse", () => {
  const state = replay([inventory(1, 35_000)])
  assert.equal(Object.keys(state.blocks).length, 1)
  const pool = projectKvCache(state).pools[0]
  assert.equal(pool.capacity, 35_000)
  assert.equal(pool.blocks.length, 35_000)
  assert.equal(pool.blocks[0].isNull, true)
  assert.equal(pool.blocks[34_999].status, "free")
  assert.equal(pool.blocks[34_999].generation, 0)
  assert.deepEqual(pool.groups[0].layerNames, ["model.layers.0.self_attn"])
  const incomplete = inventory()
  ;(incomplete.pools as Array<{ blocks: unknown[] }>)[0].blocks.pop()
  assert.throws(() => replay([incomplete]), /full pool/)
})

test("scheduled reservations and host results do not establish computed KV; partial writes do", () => {
  const player = createPlayer(sample())
  assert.equal(projectKvCache(player.at(5), null, content).pools[0].blocks[1].validTokenCount, 0)
  assert.equal(projectRequestTokens(player.at(7), "a", content)[0].kvState, "reserved")
  const cache = projectKvCache(player.at(8), null, content)
  assert.equal(cache.pools[0].blocks[1].validTokenCount, 3)
  assert.deepEqual(cache.pools[0].blocks[1].contents[0].tokenIds, [10, 11, 12])
  const tokens = projectRequestTokens(player.at(8), "a", content)
  assert.deepEqual(tokens.map((token) => token.kvState), ["computed", "computed", "computed", "reserved", "reserved", "reserved", "reserved", "reserved"])
  assert.equal(tokens[2].tokenPiece, "c")
  assert.equal(tokens[3].computedEver, false)
  assert.equal(player.at(8).blocks["pool-0:1"].contents[0].tokenIds, null)
})

test("the final sampled output is accepted content without computed K/V", () => {
  const events = sample().slice(0, 7)
  events.push(kv(8, "kv_block_written"), event(9, "tokens_accepted", { request_id: "a", step_id: "step-1", output_position_start: 0, token_count: 1 }),
    event(10, "completion", { request_id: "a", reason: "length" }))
  const rows = [...content, { kind: "accepted_tokens", request_id: "a", output_position_start: 0, token_ids: [99] }]
  const tokens = projectRequestTokens(replay(events), "a", rows)
  assert.equal(tokens.length, 9)
  assert.equal(tokens[8].kind, "output")
  assert.equal(tokens[8].tokenId, 99)
  assert.equal(tokens[8].kvState, "not_computed")
  assert.equal(tokens[8].computedEver, false)
})

test("released prefix bytes persist through reuse and hash eviction, then disappear on reassignment", () => {
  const events = sample()
  events.push(block(9, { operation: "cache", has_hash_after: true }),
    block(10, { operation: "release", ref_count_after: 0, has_hash_after: true, cached_at_zero_refs: true }),
    event(11, "engine_prompt_observed", { request_id: "b", prompt_token_count: 3 }),
    block(12, { operation: "reuse", has_hash_after: true }),
    kv(13, "kv_block_binding", { step_id: "step-2", request_id: "b", token_range_end: 3 }),
    block(14, { operation: "release", ref_count_after: 0, has_hash_after: true, cached_at_zero_refs: true }),
    block(15, { operation: "evict", ref_count_after: 0, has_hash_after: false }),
    block(16, { allocation_generation: 2 }))
  const player = createPlayer(events)
  assert.equal(projectKvCache(player.at(10)).pools[0].blocks[1].status, "cached")
  assert.equal(projectKvCache(player.at(10)).pools[0].blocks[1].validTokenCount, 3)
  const reused = projectRequestTokens(player.at(13), "b")
  assert.equal(reused[1].kvState, "computed")
  assert.equal(reused[1].computedEver, true)
  assert.equal(reused[1].locations[0].sourceRequestId, "a")
  assert.equal(projectKvCache(player.at(15)).pools[0].blocks[1].status, "free")
  assert.equal(projectKvCache(player.at(15)).pools[0].blocks[1].validTokenCount, 3)
  assert.equal(projectKvCache(player.at(16)).pools[0].blocks[1].validTokenCount, 0)
  assert.equal(projectRequestTokens(player.at(16), "a")[0].computedEver, true)
  assert.equal(projectRequestTokens(player.at(16), "a")[0].kvState, "not_computed")
})

test("pool/group/generation identity keeps independent writes and equal block IDs separate", () => {
  const initial = inventory()
  const pools = initial.pools as Array<Record<string, unknown>>
  pools.push({ ...structuredClone(pools[0]), pool: "pool-1" })
  ;(initial.groups as unknown[]).push({ cache_group: 1, pool: "pool-0", block_size: 4, layer_names: ["model.layers.1.self_attn"] },
    { cache_group: 2, pool: "pool-1", block_size: 8, layer_names: ["model.layers.2.self_attn"] })
  const events = [initial, event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }), block(3),
    block(4, { pool: "pool-1", allocation_generation: 4 }), kv(5, "kv_block_written", { token_range_end: 2 }),
    kv(6, "kv_block_written", { cache_group: 1, token_range_start: 4, token_range_end: 7 }),
    kv(7, "kv_block_written", { pool: "pool-1", cache_group: 2, allocation_generation: 4, token_range_end: 1 })]
  const state = replay(events)
  const pool0 = projectKvCache(state).pools[0].blocks[1]
  assert.deepEqual(pool0.validTokenCountsByGroup, { "0": 2, "1": 3 })
  assert.equal(pool0.contents.length, 2)
  assert.equal(projectKvCache(state).pools[1].blocks[1].generation, 4)
  assert.throws(() => replay([...events, kv(8, "kv_block_written", { allocation_generation: 99 })]), /generation/)
})

test("queue snapshots preserve observed order and priority without inventing selection reasons", () => {
  const queues = { waiting: [{ request_id: "slow", priority: 3, num_computed_tokens: 0, num_prompt_tokens: 8, status: "WAITING" },
    { request_id: "urgent", priority: -1, num_computed_tokens: 0, num_prompt_tokens: 2, status: "WAITING" }], running: [], skipped_waiting: [] }
  const state = replay([event(1, "step_scheduled", { step_id: "s", policy: "fcfs", queue_before: queues,
    queue_after: { waiting: queues.waiting.slice(1), running: [queues.waiting[0]], skipped_waiting: [] } })])
  const scheduler = projectScheduler(state)
  assert.equal(scheduler.status, "observed")
  assert.equal(scheduler.policy, "fcfs")
  assert.deepEqual(scheduler.before?.waiting.map((request) => request.requestId), ["slow", "urgent"])
  assert.equal(scheduler.after?.running[0].priority, 3)
  scheduler.before!.waiting.reverse()
  assert.equal(projectScheduler(state).before?.waiting[0].requestId, "slow")
  const unavailable = replay([event(1, "step_scheduled", { step_id: "s", queue_before: { ...queues, available: false }, queue_after: { ...queues, available: false } })])
  assert.equal(projectScheduler(unavailable).status, "unavailable")
})

test("heterogeneous group sizes and unmapped groups stay explicit instead of inventing pool geometry", () => {
  const initial = inventory()
  ;(initial.pools as Array<Record<string, unknown>>)[0].block_size = null
  ;(initial.groups as unknown[]).push({ cache_group: 1, pool: null, block_size: null, layer_names: ["unsupported"] })
  const state = replay([initial, event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }), block(3), kv(4, "kv_block_written", { token_range_end: 2 })])
  const projected = projectKvCache(state)
  assert.equal(projected.pools[0].blockSize, null)
  assert.equal(projected.groups[1].pool, null)
  assert.match(projected.reason ?? "", /no qualified token-slot mapping/)
  assert.equal(projected.pools[0].blocks[1].validTokenCount, 2)
  const token = projectRequestTokens(state, "a")[0]
  assert.equal(token.kvState, "unknown")
  assert.equal(token.cacheGroups[0].kvState, "computed")
  assert.equal(token.cacheGroups[1].mappingStatus, "unknown")
})

test("unsupported state and circular layouts keep full physical inventory without invented token slots", () => {
  for (const cacheType of ["MambaSpec", "SlidingWindowSpec", "FullAttentionSpec"]) {
    const initial = inventory()
    Object.assign((initial.groups as Array<Record<string, unknown>>)[0], {
      cache_type: cacheType, token_slot_mapping: "unavailable", token_slot_mapping_reason: "The exact loaded layout is not qualified for token slots.",
    })
    const events = [initial, event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }),
      block(3, { has_hash_after: true }), event(4, "step_scheduled", { step_id: "step-1", requests: [
        { request_id: "a", range_start: 0, range_length: 3, prompt_tokens: 3, generation_tokens: 0, num_prompt_tokens: 8 },
      ], block_table_changes: [{ request_id: "a", change: "replace", block_ids: [[1]] }] }),
      event(5, "step_result_received", { step_id: "step-1" })]
    const state = replay(events)
    const pool = projectKvCache(state).pools[0]
    assert.equal(pool.blocks.length, 8)
    assert.equal(pool.blocks[0].isNull, true)
    assert.equal(pool.blocks[1].status, "allocated")
    assert.equal(pool.blocks[1].hasHash, true)
    assert.equal(pool.blocks[7].status, "free")
    assert.equal(pool.tokenOccupancyStatus, "unavailable")
    for (const physical of [pool.blocks[1], pool.blocks[7]]) {
      assert.equal(physical.tokenOccupancyStatus, "unavailable")
      assert.equal(physical.validTokenCount, null)
      assert.deepEqual(physical.validTokenCountsByGroup, { "0": null })
      assert.match(physical.tokenOccupancyReason!, /not qualified/)
    }
    const token = projectRequestTokens(state, "a")[0]
    assert.equal(token.kvState, "unknown")
    assert.equal(token.computedEver, null)
    assert.deepEqual(token.locations, [])
    assert.equal(token.cacheGroups[0].mappingStatus, "unavailable")
    assert.throws(() => replay([...events, kv(6, "kv_block_binding")]), /without qualified token-slot mapping/)
    assert.throws(() => replay([...events, kv(6, "kv_block_written")]), /without qualified token-slot mapping/)
  }
})

test("mixed groups preserve qualified contents while aggregate token validity remains unavailable", () => {
  const initial = inventory()
  ;(initial.groups as unknown[]).push({ cache_group: 1, pool: "pool-0", block_size: 8, layer_names: ["model.layers.1"],
    cache_type: "MambaSpec", token_slot_mapping: "unavailable", token_slot_mapping_reason: "State cache has no qualified per-token slots." })
  const events = [initial, event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }), block(3),
    kv(4, "kv_block_binding"), kv(5, "kv_block_written", { token_range_end: 2 })]
  const state = replay(events)
  const physical = projectKvCache(state).pools[0].blocks[1]
  assert.equal(physical.status, "allocated")
  assert.equal(physical.validTokenCount, null)
  assert.deepEqual(physical.validTokenCountsByGroup, { "0": 2, "1": null })
  assert.equal(physical.contents.length, 1)
  const token = projectRequestTokens(state, "a")[0]
  assert.equal(token.kvState, "unknown")
  assert.equal(token.computedEver, null)
  assert.equal(token.locations[0].valid, true)
  assert.equal(token.cacheGroups[0].kvState, "computed")
  assert.equal(token.cacheGroups[0].computedEver, true)
  assert.equal(token.cacheGroups[1].kvState, "unknown")
  assert.equal(token.cacheGroups[1].computedEver, null)
})

test("legacy inventory never qualifies by type name and only explicit token observations remain usable", () => {
  const initial = inventory()
  const group = (initial.groups as Array<Record<string, unknown>>)[0]
  delete group.token_slot_mapping
  delete group.token_slot_mapping_reason
  const events = [initial, event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }), block(3),
    kv(4, "kv_block_binding"), kv(5, "kv_block_written", { token_range_end: 2 })]
  const player = createPlayer(events)
  const untouched = projectKvCache(player.at(3)).pools[0].blocks[1]
  assert.equal(untouched.tokenOccupancyStatus, "unknown")
  assert.equal(untouched.validTokenCount, null)
  assert.equal(projectRequestTokens(player.at(3), "a")[0].kvState, "unknown")
  const observed = projectKvCache(player.at(5))
  assert.equal(observed.groups[0].tokenSlotMapping, "legacy_observed")
  assert.match(observed.groups[0].tokenSlotMappingReason!, /qualification was not recorded/)
  assert.equal(observed.pools[0].blocks[1].validTokenCount, 2)
  assert.equal(observed.pools[0].blocks[7].validTokenCount, null)
  assert.equal(projectRequestTokens(player.at(5), "a")[0].kvState, "computed")
  assert.equal(projectKvCache(player.at(3)).groups[0].tokenSlotMapping, "unknown")
  group.cache_type = "MambaSpec"
  assert.equal(projectKvCache(replay([initial])).groups[0].tokenSlotMapping, "unknown")
  assert.equal(projectKvCache(replay(events)).groups[0].tokenSlotMapping, "legacy_observed")
})

test("computed proof is tracked separately by cache group rather than promoted from one group", () => {
  const initial = inventory()
  ;(initial.groups as unknown[]).push({ cache_group: 1, pool: "pool-0", block_size: 8, layer_names: ["model.layers.1"],
    cache_type: "FullAttentionSpec", token_slot_mapping: "qualified_full_attention" })
  const events = [initial, event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 8 }), block(3),
    kv(4, "kv_block_binding"), kv(5, "kv_block_binding", { cache_group: 1 }),
    kv(6, "kv_block_written", { token_range_end: 2 }), kv(7, "kv_block_written", { cache_group: 1, token_range_end: 1 })]
  const player = createPlayer(events)
  const partial = projectRequestTokens(player.at(6), "a")[0]
  assert.equal(partial.kvState, "unknown")
  assert.match(partial.kvReason!, /Only some cache groups/)
  assert.equal(partial.computedEver, false)
  assert.deepEqual(partial.cacheGroups.map((group) => group.kvState), ["computed", "reserved"])
  const tokens = projectRequestTokens(player.at(7), "a")
  assert.equal(tokens[0].kvState, "computed")
  assert.equal(tokens[0].computedEver, true)
  assert.equal(tokens[1].kvState, "unknown")
  assert.equal(tokens[1].computedEver, false)
  assert.equal(projectKvCache(player.at(7)).pools[0].blocks[1].validTokenCountsByGroup[1], 1)
})

test("unavailable mapping cannot be bypassed through a false pool or unlisted group identity", () => {
  const initial = inventory()
  const pools = initial.pools as Array<Record<string, unknown>>
  pools.push({ ...structuredClone(pools[0]), pool: "pool-1" })
  Object.assign((initial.groups as Array<Record<string, unknown>>)[0], { token_slot_mapping: "unavailable" })
  const events = [initial, block(2), block(3, { pool: "pool-1" })]
  assert.throws(() => replay([...events, kv(4, "kv_block_written", { pool: "pool-1" })]), /cache-group pool mapping/)
  assert.throws(() => replay([...events, kv(4, "kv_block_written", { cache_group: 99 })]), /unknown inventory cache group/)
  ;(initial.groups as unknown[]).push(structuredClone((initial.groups as unknown[])[0]))
  assert.throws(() => replay([initial]), /duplicate cache-group/)
})

test("overlapping writes replace only the affected group slots and seek-safe content slices", () => {
  const events = [...sample(), kv(9, "kv_block_written", { step_id: "step-2", token_range_start: 1, token_range_end: 2 })]
  const replacement: ContentRecord = { kind: "kv_tokens", step_id: "step-2", request_id: "a", pool: "pool-0", block_id: 1,
    allocation_generation: 1, cache_group: 0, token_range_start: 1, token_range_end: 2, token_ids: [77], token_pieces: ["replacement"] }
  const projected = projectKvCache(replay(events), null, [...content, replacement]).pools[0].blocks[1]
  assert.deepEqual(projected.contents.map((span) => span.tokenIds), [[10], [77], [12]])
  assert.deepEqual(projected.contents.map((span) => [span.slotStart, span.slotEnd]), [[0, 1], [1, 2], [2, 3]])
  const onlyKvContent = content.filter((row) => row.kind === "kv_tokens")
  assert.equal(projectRequestTokens(replay(sample()), "a", onlyKvContent)[0].tokenId, 10)
})

test("backward seeking and optional content retain only writes already observed", () => {
  const events = [...sample(), kv(9, "kv_block_written", { step_id: "step-2", token_range_start: 3, token_range_end: 5 })]
  const future: ContentRecord = { kind: "kv_tokens", step_id: "step-2", request_id: "a", pool: "pool-0", block_id: 1,
    allocation_generation: 1, cache_group: 0, token_range_start: 3, token_range_end: 5, token_ids: [13, 14] }
  const player = createPlayer(events)
  for (const sequence of [9, 8, 5, 9, 0, 7, 8]) {
    const expected = attachContent(replay(events.filter((item) => item.sequence <= sequence)), [...content, future])
    assert.deepEqual(attachContent(player.at(sequence), [...content, future]), expected)
  }
  assert.equal(projectKvCache(player.at(8), null, [...content, future]).pools[0].blocks[1].contents.length, 1)
  assert.equal(projectKvCache(player.at(8)).pools[0].blocks[1].contents[0].tokenIds, null)
  assert.equal(projectRequestTokens(player.at(8), "a")[0].tokenId, null)
})

test("old recordings, unsupported architectures, gaps and verified resets retain honest uncertainty", () => {
  const old = replay([event(1, "engine_prompt_observed", { request_id: "a", prompt_token_count: 2 })])
  assert.equal(projectKvCache(old, { cache_geometry: { num_gpu_blocks: 100 } }).status, "unavailable")
  assert.equal(projectScheduler(old).status, "unavailable")
  assert.equal(projectRequestTokens(old, "a")[0].kvState, "unknown")
  assert.equal(projectArchitecture({ model: { id: "Qwen/Qwen3-4B-Instruct-2507" } }).supported, false)
  assert.equal(projectArchitecture({ architecture: { supported: false, reason: "unsupported attention layout" } }).reason, "unsupported attention layout")
  const architecture = projectArchitecture({ architecture: { supported: true, model_type: "qwen3", source: "loaded_model_config",
    num_layers: 36, hidden_size: 2560, num_attention_heads: 32, num_key_value_heads: 8, head_dim: 128, intermediate_size: 9728 } })
  assert.equal(architecture.layerCount, 36)
  assert.equal(architecture.kvHeads, 8)
  const gapped = replay([...sample(), event(9, "trace_gap", { reason: "buffer_exhausted" }), block(10, { allocation_generation: 2 })])
  assert.equal(projectKvCache(gapped).status, "unknown")
  assert.equal(projectKvCache(gapped).pools[0].blocks[7].status, "unknown")
  assert.equal(projectRequestTokens(gapped, "a")[0].kvState, "unknown")
  const reset = replay([...sample(), event(9, "cache_reset", { verified: true })])
  assert.equal(projectKvCache(reset).pools[0].blocks[1].validTokenCount, 0)
  assert.throws(() => replay([...sample(), event(9, "cache_reset", { verified: false })]), /not verified/)
})
