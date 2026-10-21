import test from "node:test"
import assert from "node:assert/strict"
import { createPlayer, type TraceEvent } from "./reduce.ts"
import { projectMemory, sampledDeviceMaximum, storageBreakdown } from "./memory.ts"

const origin = 9_007_199_254_740_993n
function event(sequence: number, event_type: string, fields: Record<string, unknown> = {}): TraceEvent {
  return { schema_version: "1.0.0", run_id: "memory", producer_id: "engine", sequence, monotonic_ns: String(origin + BigInt(sequence) * 100_000_000n), event_type, ...fields }
}
function sample(sequence: number, used: number, status = "observed", more: Record<string, unknown> = {}) {
  return event(sequence, "gpu_memory_sample", { sample_started_monotonic_ns: String(origin + BigInt(sequence) * 100_000_000n - 5n), sample_completed_monotonic_ns: String(origin + BigInt(sequence) * 100_000_000n),
    memory_clock_domain: "time.monotonic_ns", status, phase: "inference", source: "nvml", engine_pid: 42,
    devices: status === "observed" ? [{ gpu_uuid: "GPU-a", name: "H100", used_bytes: used, free_bytes: 1000 - used, total_bytes: 1000, process_status: "observed", process_used_bytes: used - 10 }] : [], ...more })
}

test("memory events replay as metadata without changing requests or cache", () => {
  const events = [sample(1, 100), event(2, "gpu_memory_phase", { phase: "cleanup" }), event(3, "gpu_memory_inventory", { storages: [] })]
  const state = createPlayer(events).at(3)
  assert.equal(state.lastSequence, 3)
  assert.deepEqual(state.requests, {})
  assert.deepEqual(state.blocks, {})
})

test("host-clock projection keeps future measurements out of the current sample", () => {
  const events = [sample(1, 100), sample(2, 250), sample(3, 200)]
  const view = projectMemory(events, { gpu_memory: { sample_interval_ms: 100 } }, 2)
  assert.equal(view.current?.devices[0].used, 250)
  assert.equal(view.current?.timeNs, origin + 200_000_000n)
  assert.equal(view.samples.length, 3)
  assert.equal(view.ageMs, 0)
  assert.equal(sampledDeviceMaximum(view.samples, "GPU-a"), 250)
})

test("timeline starts no later than the first query, whose event is emitted after measurement", () => {
  const events = [sample(1, 100)]
  const view = projectMemory(events, null, 1)
  assert.equal(view.originNs, view.current?.startNs)
  assert.ok(view.current!.timeNs >= view.originNs!)
})

test("failed and missing samples show gaps and retain age of last successful measurement", () => {
  const events = [sample(1, 100), sample(2, 0, "unavailable", { reason: "PermissionError" }), event(3, "capture_ready"), event(4, "capture_ready")]
  const view = projectMemory(events, null, 4)
  assert.equal(view.current?.devices[0].used, 100)
  assert.equal(view.latestAttempt?.reason, "PermissionError")
  assert.equal(view.ageMs, 300)
  assert.equal(view.gap, true)
  assert.equal(projectMemory([event(1, "capture_ready")], null, 1).reason, "GPU memory was not recorded for this run. A new capture is required.")
})

test("unmatched clock domains never align with host events", () => {
  const view = projectMemory([sample(1, 100, "observed", { memory_clock_domain: "native_profiler" })], null, 1)
  assert.equal(view.current, null)
  assert.equal(view.samples.length, 0)
})

test("trace gaps prevent future memory measurements and maxima from becoming observed", () => {
  const events = [sample(1, 100), event(2, "trace_gap", { reason: "buffer_exhausted" }), sample(3, 900)]
  const view = projectMemory(events, null, 3)
  assert.equal(view.current?.devices[0].used, 100)
  assert.equal(view.samples.length, 1)
  assert.equal(sampledDeviceMaximum(view.samples, "GPU-a"), 100)
  assert.equal(view.gap, true)
})

test("shared allocations count once and cleanup invalidates loaded attribution", () => {
  const inventory = event(2, "gpu_memory_inventory", { memory_clock_domain: "time.monotonic_ns", status: "observed", storages: [
    { storage_id: "one", device: "cuda:0", byte_count: 80, categories: ["weights"], owners: ["tied_a", "tied_b"], dtypes: ["bfloat16"] },
    { storage_id: "two", device: "cuda:0", byte_count: 40, categories: ["attention_kv", "recurrent_state"], owners: ["cache"], dtypes: ["bfloat16"] },
    { storage_id: "two", device: "cuda:0", byte_count: 40, categories: ["attention_kv"], owners: ["alias"], dtypes: ["bfloat16"] },
    { storage_id: "cpu", device: "cpu", byte_count: 160, categories: ["weights"], owners: ["embeddings"], dtypes: ["bfloat16"] },
  ] })
  const events = [sample(1, 200), inventory, event(3, "gpu_memory_phase", { phase: "cleanup" })]
  const view = projectMemory(events, null, 2)
  assert.deepEqual(storageBreakdown(view.inventory, "cuda:0"), [{ category: "weights", bytes: 80, storageCount: 1 }, { category: "shared", bytes: 40, storageCount: 1 }])
  assert.equal(storageBreakdown(view.inventory, "cpu")[0].bytes, 160)
  assert.equal(projectMemory(events, null, 3).inventory, null)
})
