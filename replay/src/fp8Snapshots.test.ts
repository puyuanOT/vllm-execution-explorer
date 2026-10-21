import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { decodeFp8Byte, parseProfileAttachment } from "./profiling.ts"

// Authored contract case. It does not claim that these values were captured on a GPU.
const clock = { domain: "torch.profiler.chrome_trace", unit: "microseconds", origin: "trace_local" }
const manifest = { run_id: "fp8-fixture", execution_kind: "fixture", profiling: { contract_version: 1, mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock } }
const boundary = "same_cuda_stream_after_native_cache_update_before_reuse"
function source() {
  const participant = { row: 0, request_id: "request-1", token_position: 0, pool: "pool-0", cache_group: 0, block_id: 1, allocation_generation: 1, slot: 0, kernel_block_id: 1, kernel_slot: 0 }
  return {
    contract_version: 1, run_id: manifest.run_id, mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock,
    activities: [],
    steps: [{ step_id: "step-1", mapping_status: "observed", unpadded_tokens: 1, padded_tokens: 1, mapping_observation_boundary: "authored", packed_rows: [{ row: 0, request_id: "request-1", token_position: 0, is_padding: false }] }],
    cache_writes: [{ operation_id: "write-1", step_id: "step-1", layer_index: 0, layer_name: "model.layers.0.attn", pool: "pool-0", cache_group: 0, device: "cuda:0", status: "observed", participants: [participant], gpu_activity_ids: [], observation_boundary: "authored" }],
    tensor_snapshots: [{ snapshot_id: "snapshot-1", operation_id: "write-1", step_id: "step-1", request_id: "request-1", token_position: 0, layer_index: 0, pool: "pool-0", cache_group: 0, block_id: 1, allocation_generation: 1, slot: 0, device: "cuda:0", kv_head: 0, kind: "K", dtype: "uint8", shape: [4], dimension_start: 0, values: [1, 2, -1, -2], byte_count: 8, observation_boundary: boundary,
      quantization: { format: "fp8_e4m3", raw_bits: [56, 64, 184, 192], scale: { value: 2, dtype: "float32", shape: [1], mode: "per_tensor", head_index: null as number | null, observation_boundary: boundary }, reconstructed_values: [2, 4, -2, -4], reconstruction: "decoded_fp8_times_recorded_scale" } }],
    tensor_capture: { enabled: true, status: "completed", max_bytes: 16, retained_bytes: 8, selection: { request_id: "request-1", token_positions: [0], layer_indices: [0], head_indices: [0], dimension_start: 0, dimension_count: 4 }, observation_boundary: boundary },
  }
}

test("the production FP8 decoder matches all 512 native PyTorch CPU reinterpretations", () => {
  const reference = JSON.parse(readFileSync(new URL("../../docs/qualification/demo-facets/fp8-torch-cpu-reference.json", import.meta.url), "utf8"))
  for (const [key, encoding] of [["e4m3fn", "fp8_e4m3"], ["e5m2", "fp8_e5m2"]] as const) {
    for (let byte = 0; byte < 256; byte++) {
      const value = decodeFp8Byte(byte, encoding), expected = reference[key][byte]
      if (expected === "nan") assert.ok(Number.isNaN(value))
      else assert.equal(value, expected === "inf" ? Infinity : expected === "-inf" ? -Infinity : expected)
    }
  }
})

test("FP8 snapshots retain raw bytes, decoded storage, positive update-time scales and reconstruction", () => {
  const snapshot = parseProfileAttachment(JSON.stringify(source()), manifest).tensorSnapshots[0]
  assert.equal(snapshot.dtype, "uint8")
  assert.deepEqual(snapshot.values, [1, 2, -1, -2])
  assert.deepEqual(snapshot.quantization?.rawBits, [56, 64, 184, 192])
  assert.deepEqual(snapshot.quantization?.reconstructedValues, [2, 4, -2, -4])
  assert.equal(snapshot.quantization?.scale.value, 2)
  const perHead = source()
  Object.assign(perHead.tensor_snapshots[0].quantization.scale, { shape: [8], mode: "per_head", head_index: 0 })
  assert.equal(parseProfileAttachment(JSON.stringify(perHead), manifest).tensorSnapshots[0].quantization?.scale.mode, "per_head")
})

test("corrupted encodings, mismatched scales, accounting and reconstructions cannot become valid slices", () => {
  const changes: Array<(value: ReturnType<typeof source>) => void> = [
    (value) => { value.tensor_snapshots[0].quantization.format = "fp8_e5m2" },
    (value) => { value.tensor_snapshots[0].quantization.raw_bits[0] = 256 },
    (value) => { value.tensor_snapshots[0].values[0] = 56 },
    (value) => { value.tensor_snapshots[0].quantization.scale.value = 0 },
    (value) => { value.tensor_snapshots[0].quantization.scale.value = 0.1 },
    (value) => { value.tensor_snapshots[0].quantization.scale.head_index = 1 },
    (value) => { value.tensor_snapshots[0].quantization.scale.shape = [8] },
    (value) => { value.tensor_snapshots[0].quantization.scale.observation_boundary = "later" },
    (value) => { value.tensor_snapshots[0].quantization.reconstructed_values[0] = 56 },
    (value) => { value.tensor_snapshots[0].byte_count = 4 },
  ]
  for (const change of changes) {
    const value = source()
    change(value)
    assert.throws(() => parseProfileAttachment(JSON.stringify(value), manifest))
  }
})
