import assert from "node:assert/strict"
import { test } from "node:test"
import { projectKvConfiguration, projectKvTensorPreview, projectLayerTokenMatrix } from "./kv.ts"
import { parseProfileAttachment, projectMeasuredProfile } from "./profiling.ts"
import { replay, type RecordingManifest, type TraceEvent } from "./reduce.ts"

const clock = { domain: "torch.profiler.chrome_trace", unit: "microseconds", origin: "trace_local" }
const manifest: RecordingManifest = {
  run_id: "cache-test", execution_kind: "gpu", engine_settings: { max_model_len: 64, kv_cache_dtype: "auto", enable_prefix_caching: true },
  architecture: { supported: true, model_type: "qwen3", num_layers: 2, hidden_size: 8, num_attention_heads: 2, num_key_value_heads: 1, head_dim: 4, intermediate_size: 16 },
  kv_cache: {
    contract_version: 1, status: "observed", unique_storage_bytes: 2048, observation_boundary: "loaded_attention_cache_tensors",
    storages: [{ storage_id: "shared", device: "cuda:0", dtype: "torch.float16", byte_count: 2048 }],
    layers: [0, 1].map((layer_index) => ({ layer_index, layer_name: `model.layers.${layer_index}.self_attn`, pool: "pool-0", cache_group: 0,
      device: "cuda:0", dtype: "torch.float16", shape: [4, 1, 4, 8], strides: [32, 32, 8, 1], storage_id: "shared", storage_offset: layer_index * 128,
      logical_bytes: 1024, backend: "FLASH_ATTN", layout: "kernel_block,kv_head,token_slot,concatenated_K_V", layout_status: "qualified",
      manager_block_size: 4, kernel_block_size: 4, kernel_blocks_per_manager_block: 1, num_kernel_blocks: 4, num_kv_heads: 1, head_dim: 4 })),
  },
  profiling: { contract_version: 1, mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock, status: "completed", attachment_status: "available" },
}

function event(sequence: number, event_type: string, fields: Record<string, unknown> = {}): TraceEvent {
  return { schema_version: "1.0.0", run_id: "cache-test", producer_id: "adapter", sequence, monotonic_ns: sequence * 100, event_type, ...fields }
}

function state() {
  const kv = { step_id: "step-0", request_id: "a", pool: "pool-0", cache_group: 0, block_id: 1, allocation_generation: 1, token_range_start: 0, token_range_end: 2 }
  return replay([
    event(1, "cache_inventory", { pools: [{ pool: "pool-0", block_size: 4, num_gpu_blocks: 4, inventory_complete: true,
      blocks: [0, 1, 2, 3].map((block_id) => ({ block_id, allocation_generation: 0, ref_count: block_id === 0 ? 1 : 0, is_null: block_id === 0, has_hash: false })) }],
      groups: [{ cache_group: 0, pool: "pool-0", block_size: 4, layer_names: ["model.layers.0.self_attn", "model.layers.1.self_attn"], cache_type: "FullAttentionSpec", token_slot_mapping: "qualified_full_attention" }] }),
    event(2, "engine_prompt_observed", { request_id: "a", prompt_token_count: 4 }),
    event(3, "block_change", { pool: "pool-0", block_id: 1, allocation_generation: 1, operation: "allocate", ref_count_after: 1, has_hash_after: true }),
    event(4, "kv_block_binding", kv), event(5, "kv_block_written", kv),
    event(6, "engine_prompt_observed", { request_id: "b", prompt_token_count: 4 }),
    event(7, "kv_block_binding", { ...kv, step_id: "step-1", request_id: "b", token_range_end: 4 }),
    event(8, "step_scheduled", { step_id: "step-1", total_scheduled_tokens: 2,
      requests: [{ request_id: "b", range_start: 2, range_length: 2, prompt_tokens: 2, generation_tokens: 0, num_prompt_tokens: 4 }] }),
    event(9, "step_result_received", { step_id: "step-1", observation: "future.result" }),
  ], manifest)
}

function source() {
  const participants = [2, 3].map((token_position, row) => ({ row, request_id: "b", token_position, pool: "pool-0", cache_group: 0,
    block_id: 1, allocation_generation: 1, slot: token_position, kernel_block_id: 1, kernel_slot: token_position }))
  return {
    contract_version: 1, run_id: "cache-test", mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock,
    activities: [0, 1].flatMap((layer_index) => [
      { activity_id: `cpu-${layer_index}`, kind: "cpu_scope", name: "cache update", device: null, stream: null, start: 90 + layer_index * 100, duration: 5,
        scope: { step_id: "step-1", layer_index, component: "attention", correlation: "explicit_scope", operation: "kv_cache_write", operation_id: `write-${layer_index}` } },
      { activity_id: `gpu-${layer_index}`, kind: "cuda_kernel", name: "reshape_and_cache", device: 0, stream: 7, start: 100 + layer_index * 100, duration: 10,
        scope: { step_id: "step-1", layer_index, component: "attention", correlation: "runtime_flow", operation: "kv_cache_write", operation_id: `write-${layer_index}` } },
    ]),
    steps: [{ step_id: "step-1", mapping_status: "observed", mapping_observation_boundary: "GPUModelRunner._prepare_inputs.returned", unpadded_tokens: 2, padded_tokens: 2,
      packed_rows: participants.map((participant) => ({ row: participant.row, request_id: participant.request_id, token_position: participant.token_position, is_padding: false })) }],
    cache_writes: [0, 1].map((layer_index) => ({ operation_id: `write-${layer_index}`, step_id: "step-1", layer_index, layer_name: `model.layers.${layer_index}.self_attn`, cache_group: 0,
      pool: "pool-0", device: "cuda:0", status: "observed", participants: structuredClone(participants), gpu_activity_ids: [`gpu-${layer_index}`], observation_boundary: "AttentionImpl.do_kv_cache_update.returned" })),
    tensor_snapshots: [{ snapshot_id: "snapshot-0", operation_id: "write-0", step_id: "step-1", request_id: "b", token_position: 2, layer_index: 0,
      cache_group: 0, pool: "pool-0", block_id: 1, allocation_generation: 1, slot: 2, device: "cuda:0", kv_head: 0, kind: "K", dtype: "torch.float16",
      shape: [2], dimension_start: 1, values: [1.25, -1.25], byte_count: 4, observation_boundary: "bounded_snapshot_after_cache_write" }],
    tensor_capture: { enabled: true, status: "completed", max_bytes: 4, retained_bytes: 4, selection: { request_id: "b", token_positions: [2], layer_indices: [0], head_indices: [0], dimension_start: 1, dimension_count: 2 }, observation_boundary: "bounded_snapshot_after_cache_write" },
  }
}

test("configuration counts shared storage once and separates context limits from physical pool occupancy", () => {
  const result = projectKvConfiguration(state(), manifest)
  assert.equal(result.uniqueStorageBytes, 2048)
  assert.equal(result.layers.length, 2)
  assert.equal(result.pools[0].storageBytes, 2048)
  assert.equal(result.pools[0].capacity, 4)
  assert.equal(result.pools[0].usableCapacity, 3)
  assert.equal(result.pools[0].tokenCapacity, 12)
  assert.equal(result.maxModelLength, 64)
  assert.deepEqual(result.pools[0].devices, ["cuda:0"])
  assert.deepEqual(result.pools[0].dtypes, ["torch.float16"])
  assert.deepEqual([result.pools[0].referencedBlocks, result.pools[0].cachedBlocks, result.pools[0].freeBlocks], [1, 0, 2])
  const legacy = projectKvConfiguration(state(), { engine_settings: { kv_cache_dtype: "auto" } })
  assert.equal(legacy.status, "unavailable")
  assert.equal(legacy.pools[0].device, null)
  assert.deepEqual(legacy.pools[0].dtypes, [])
  assert.equal(legacy.uniqueStorageBytes, null)
  assert.equal(legacy.requestedDtype, "auto")
  const mixed = structuredClone(manifest)
  const metadata = mixed.kv_cache as { storages: Array<{ dtype: string }>; layers: Array<Record<string, unknown>> }
  metadata.storages[0].dtype = "mixed"
  metadata.layers[1].dtype = "float8_e4m3fn"
  metadata.layers[1].quantization = { cache_dtype: "fp8", mode: "per_tensor", scales: {
    K: { status: "observed", shape: [], dtype: "float32", device: "cuda:0" },
    V: { status: "unavailable", shape: [], dtype: null, device: null },
  } }
  const quantized = projectKvConfiguration(state(), mixed)
  assert.equal(quantized.uniqueStorageBytes, 2048)
  assert.equal(quantized.layers[1].quantization?.mode, "per_tensor")
  assert.equal(quantized.layers[1].quantization?.scales.V.status, "unavailable")
})

test("layer matrix follows batch GPU cache writes, not host result or CPU return, with seek-safe endpoints", () => {
  const attachment = parseProfileAttachment(JSON.stringify(source()), manifest)
  const project = (timeUs?: number) => projectLayerTokenMatrix(state(), manifest, attachment, { stepId: "step-1", timeUs })
  const currentStates = (timeUs?: number) => project(timeUs).cells.filter((cell) => cell.position === 2).map((cell) => cell.state)
  assert.deepEqual(currentStates(), ["scheduled", "scheduled"])
  assert.deepEqual(currentStates(95), ["scheduled", "scheduled"])
  assert.deepEqual(currentStates(105), ["operation_active", "scheduled"])
  assert.deepEqual(currentStates(110), ["completed", "scheduled"])
  assert.deepEqual(currentStates(205), ["completed", "operation_active"])
  assert.deepEqual(currentStates(210), ["completed", "completed"])
  assert.deepEqual(currentStates(105), ["operation_active", "scheduled"])
  assert.equal(project(105).cells.filter((cell) => cell.position < 2).every((cell) => cell.state === "reused_existing_kv"), true)
  assert.equal(project(105).cells.filter((cell) => cell.position === 3)[0].state, "operation_active")
  const reusedSelection = projectLayerTokenMatrix(state(), manifest, attachment, { stepId: "step-1", requestId: "b", tokenPosition: 0, timeUs: 105 })
  assert.equal(reusedSelection.cells.every((cell) => cell.state === "reused_existing_kv"), true)
  assert.equal(Object.hasOwn(project(105).cells[0], "durationUs"), false)
  const outside = projectLayerTokenMatrix(state(), manifest, attachment, { stepId: "step-1", requestId: "a", tokenPosition: 0, timeUs: 105 })
  assert.equal(outside.cells.every((cell) => cell.state === "not_participating"), true)
  assert.equal(projectLayerTokenMatrix(state(), manifest, attachment, { maxTokens: 2 }).truncated, true)
})

test("legacy, unsupported, missing and synthetic layer evidence stay explicit", () => {
  const legacySource = source() as Record<string, unknown>
  delete legacySource.cache_writes
  delete legacySource.tensor_snapshots
  delete legacySource.tensor_capture
  const legacy = parseProfileAttachment(JSON.stringify(legacySource), manifest)
  assert.deepEqual(legacy.cacheWrites, [])
  assert.equal(projectLayerTokenMatrix(state(), manifest, legacy, { timeUs: 500 }).cells.filter((cell) => cell.position === 2).every((cell) => cell.state === "unknown"), true)
  assert.match(projectKvTensorPreview(manifest, legacy).reason!, /predates/)
  const partialSource = source()
  Object.assign(partialSource.cache_writes[1], { status: "unavailable", reason: "GPU correlation missing.", gpu_activity_ids: [] })
  const partial = parseProfileAttachment(JSON.stringify(partialSource), manifest)
  const result = projectLayerTokenMatrix(state(), manifest, partial, { timeUs: 500 })
  assert.deepEqual(result.cells.filter((cell) => cell.position === 2).map((cell) => cell.state), ["completed", "unknown"])
  const fixture = { ...manifest, execution_kind: "fixture" }
  const synthetic = projectLayerTokenMatrix(state(), fixture, partial)
  assert.equal(synthetic.evidenceKind, "fixture")
  assert.match(synthetic.reason!, /Synthetic/)
  const unsupported = { ...manifest, architecture: {} }
  delete unsupported.kv_cache
  assert.equal(projectLayerTokenMatrix(state(), unsupported).status, "unavailable")
})

test("legacy attention module names link existing prefix KV only to declared layers", () => {
  const legacy = structuredClone(manifest)
  delete legacy.kv_cache
  const current = state()
  current.cacheGroups[0].layerNames = ["model.layers.0.self_attn.attn"]
  const matrix = projectLayerTokenMatrix(current, legacy, null, { stepId: "step-1" })
  assert.equal(matrix.cells.filter((cell) => cell.position < 2 && cell.layerIndex === 0).every((cell) => cell.state === "reused_existing_kv"), true)
  assert.equal(matrix.cells.filter((cell) => cell.position < 2 && cell.layerIndex === 1).every((cell) => cell.state === "unknown"), true)
  assert.equal(matrix.tokens.filter((token) => token.scheduled).length, 2)
  assert.equal(matrix.tokens.filter((token) => !token.scheduled).length, 2)
})

test("tensor previews retain exact allocation identity and content-export omission", () => {
  const attachment = parseProfileAttachment(JSON.stringify(source()), manifest)
  const preview = projectKvTensorPreview(manifest, attachment, { requestId: "b", tokenPosition: 2, layerIndex: 0, kvHead: 0, kind: "K", pool: "pool-0", blockId: 1, generation: 1, slot: 2 })
  assert.deepEqual(preview.snapshots[0].values, [1.25, -1.25])
  assert.equal(preview.snapshots[0].dtype, "torch.float16")
  preview.snapshots[0].values[0] = 999
  assert.equal(attachment.tensorSnapshots[0].values[0], 1.25)
  assert.equal(projectKvTensorPreview(manifest, attachment, { generation: 2 }).status, "unavailable")
  const shared = projectKvTensorPreview(manifest, attachment, { requestId: "other-holder", tokenPosition: 99, stepId: "later-step", pool: "pool-0", blockId: 1, generation: 1, slot: 2 })
  assert.equal(shared.resolvedByAllocation, true)
  assert.equal(shared.requestedRequestId, "other-holder")
  assert.equal(shared.snapshots[0].requestId, "b")
  assert.equal(shared.snapshots[0].tokenPosition, 2)
  assert.equal(shared.snapshots[0].stepId, "step-1")
  assert.equal(projectKvTensorPreview(manifest, attachment, { pool: "pool-0", blockId: 1, generation: 2, slot: 2 }).snapshots.length, 0)
  assert.equal(projectKvTensorPreview(manifest, attachment, { pool: "pool-0", cacheGroup: 1, blockId: 1, generation: 1, slot: 2 }).snapshots.length, 0)
  assert.match(projectKvTensorPreview(manifest, attachment, { kind: "V" }).reason!, /not captured/)
  const structural = { ...manifest, profiling: { ...(manifest.profiling as object), attachment_status: "omitted_by_structural_export" } }
  assert.deepEqual(projectKvTensorPreview(structural, attachment).snapshots, [])
  assert.deepEqual(projectMeasuredProfile(structural, attachment).tensorSnapshots, [])
  const synthetic = projectKvTensorPreview({ ...manifest, execution_kind: "fixture" }, attachment)
  assert.match(synthetic.reason!, /not read from a GPU/)
})

test("gaps freeze dynamic occupancy and linked state without erasing static configuration", () => {
  const unknown = state()
  unknown.trust = "unknown"
  for (const block of Object.values(unknown.blocks)) block.trust = "unknown"
  const configuration = projectKvConfiguration(unknown, manifest)
  assert.equal(configuration.uniqueStorageBytes, 2048)
  assert.equal(configuration.pools[0].freeBlocks, null)
  assert.equal(configuration.pools[0].referencedBlocks, null)
  assert.match(configuration.reason!, /trace gap/)
  const matrix = projectLayerTokenMatrix(unknown, manifest, parseProfileAttachment(JSON.stringify(source()), manifest), { timeUs: 500 })
  assert.equal(matrix.status, "unknown")
  assert.equal(matrix.cells.every((cell) => cell.state === "unknown"), true)
})

test("malformed cache attribution, snapshots and unique storage accounting are rejected", () => {
  for (const mutate of [
    (value: ReturnType<typeof source>) => { value.cache_writes[0].gpu_activity_ids = ["cpu-0"] },
    (value: ReturnType<typeof source>) => { value.cache_writes[0].gpu_activity_ids = ["gpu-1"] },
    (value: ReturnType<typeof source>) => { value.cache_writes[0].participants[0].request_id = "ghost" },
    (value: ReturnType<typeof source>) => { value.tensor_snapshots[0].allocation_generation = 2 },
    (value: ReturnType<typeof source>) => { value.tensor_snapshots[0].shape = [3] },
    (value: ReturnType<typeof source>) => { value.tensor_capture.max_bytes = 3 },
    (value: ReturnType<typeof source>) => { value.tensor_capture.retained_bytes = 3 },
  ]) {
    const value = source()
    mutate(value)
    assert.throws(() => parseProfileAttachment(JSON.stringify(value), manifest))
  }
  const bad = structuredClone(manifest)
  ;(bad.kv_cache as Record<string, unknown>).unique_storage_bytes = 4096
  assert.throws(() => projectKvConfiguration(state(), bad), /physical storage once/)
})
