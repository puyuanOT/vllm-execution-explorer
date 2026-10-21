import assert from "node:assert/strict"
import { test } from "node:test"
import type { MemoryProjection, MemorySample } from "../../../replay/src/memory.ts"
import { summarizeDeviceMemory } from "../execution/memorySummary.ts"

const GiB = 1024 ** 3
function sample(sequence: number, timeNs: bigint, used: number, uuid = "GPU-A", status = "observed"): MemorySample {
  return { sequence, timeNs, startNs: timeNs, status, phase: "inference", source: "NVML", reason: status === "observed" ? null : "query failed", pid: 10, framework: null,
    devices: [{ uuid, name: uuid, total: 80 * GiB, used, free: 80 * GiB - used, processUsed: null, processStatus: "unavailable" }] }
}
function projection(samples: MemorySample[], timeNs: bigint | null = 100n): MemoryProjection {
  return { samples, current: samples[0] ?? null, latestAttempt: samples[0] ?? null, inventory: null, timeNs, originNs: 0n, ageMs: 0, inventoryAgeMs: null, gap: false, intervalMs: 100, phase: "inference", reason: null }
}

test("recorded memory summaries retain the complete run and only include the selected GPU's successful samples", () => {
  const memory = projection([sample(1, 10n, 4 * GiB), sample(2, 50n, 78 * GiB, "GPU-B"), sample(3, 60n, 79 * GiB, "GPU-A", "unavailable"), { ...sample(4, 100n, 10 * GiB), phase: "cleanup" }])
  const early = summarizeDeviceMemory({ ...memory, timeNs: 10n }, "GPU-A", 1, false)
  const late = summarizeDeviceMemory(memory, "GPU-A", 4, false)
  assert.equal(early.maximum, 10 * GiB)
  assert.deepEqual(early, late)
  assert.deepEqual(early.phases, [{ phase: "inference", samples: 1, maximum: 4 * GiB }, { phase: "cleanup", samples: 1, maximum: 10 * GiB }])
})

test("live peaks and phase counts exclude future sequence and future completed timestamps", () => {
  const memory = projection([sample(1, 10n, 2 * GiB), sample(2, 40n, 4 * GiB), sample(3, 90n, 40 * GiB), sample(4, 45n, 70 * GiB)], 50n)
  const live = summarizeDeviceMemory(memory, "GPU-A", 3, true)
  assert.equal(live.maximum, 4 * GiB)
  assert.deepEqual(live.phases, [{ phase: "inference", samples: 2, maximum: 4 * GiB }])
})

test("absent samples and an unknown live playhead remain unavailable while measured zero stays zero", () => {
  const unavailable = { maximum: null, phases: [] }
  assert.deepEqual(summarizeDeviceMemory(projection([]), "GPU-A", 0, false), unavailable)
  assert.deepEqual(summarizeDeviceMemory(projection([sample(1, 0n, GiB)], null), "GPU-A", 1, true), unavailable)
  assert.deepEqual(summarizeDeviceMemory(projection([sample(1, 0n, 0)], 0n), "GPU-A", 1, false), { maximum: 0, phases: [{ phase: "inference", samples: 1, maximum: 0 }] })
})
