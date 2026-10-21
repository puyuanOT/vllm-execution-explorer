import assert from "node:assert/strict"
import { readRecordingFixture } from "../../test/recordingFixture.ts"
import test from "node:test"
import { projectKvTensorPreview } from "../../../replay/src/kv.ts"
import { parseProfileAttachment } from "../../../replay/src/profiling.ts"
import { numericalCells, numericalFrames } from "../execution/numericalHeatmap.ts"

const recording = new URL("../../public/recordings/gpu-kv-layer-diagnostics/", import.meta.url)
const manifest = JSON.parse(readRecordingFixture(new URL("manifest.json", recording)))
const profile = parseProfileAttachment(readRecordingFixture(new URL("profiling/activities.json", recording)), manifest)
const selection = { requestId: "prefix-prime", tokenPosition: 3, layerIndex: 0, kind: "K" as const }
const observed = projectKvTensorPreview(manifest, profile, selection)

test("real sparse BF16 captures populate only their recorded head/dimension coordinates", () => {
  const frames = numericalFrames(observed.snapshots)
  assert.equal(frames.length, 1)
  const { cells, conflictCount } = numericalCells(frames[0], 8, 128)
  assert.equal(conflictCount, 0)
  assert.equal(cells.size, 16)
  assert.deepEqual([...cells.values()].map((cell) => cell.value), observed.snapshots[0].values)
  assert.ok([...cells.values()].every((cell) => cell.head === 0 && cell.dimension < 16))
  assert.equal(cells.get(16), undefined, "uncaptured dimensions must not become zeros")
  assert.equal(cells.get(128), undefined, "uncaptured heads must not borrow head0 values")
  assert.equal(projectKvTensorPreview(manifest, profile, { ...selection, kvHead: 1 }).snapshots.length, 0)
  assert.equal(numericalCells(numericalFrames(projectKvTensorPreview(manifest, profile, { ...selection, layerIndex: 1 }).snapshots)[0], 8, 128).cells.size, 0)
})

test("shared-prefix selection uses production allocation resolution and preserves the original writer", () => {
  const reader = projectKvTensorPreview(manifest, profile, {
    ...selection, requestId: "prefix-reader", pool: "pool-0", cacheGroup: 0, blockId: 1, generation: 1, slot: 3,
  })
  assert.equal(reader.resolvedByAllocation, true)
  const { cells } = numericalCells(numericalFrames(reader.snapshots)[0], 8, 128)
  assert.equal(cells.size, 16)
  assert.ok([...cells.values()].every((cell) => cell.snapshot.requestId === "prefix-prime" && cell.snapshot.tokenPosition === 3))
  const overwritten = projectKvTensorPreview(manifest, profile, { ...selection, pool: "pool-0", cacheGroup: 0, blockId: 1, generation: 2, slot: 3 })
  assert.equal(numericalCells(numericalFrames(overwritten.snapshots)[0], 8, 128).cells.size, 0)
  const omitted = projectKvTensorPreview({ ...manifest, profiling: { ...manifest.profiling, attachment_status: "omitted_by_structural_export" } }, profile, selection)
  assert.equal(numericalCells(numericalFrames(omitted.snapshots)[0], 8, 128).cells.size, 0)
})

test("a grid never silently combines different operations or allocation identities", () => {
  const source = observed.snapshots[0]
  const differentOperation = { ...source, snapshotId: "different-operation", operationId: "another-operation", kvHead: 1 }
  const differentGeneration = { ...source, snapshotId: "different-generation", allocationGeneration: 2, kvHead: 2 }
  const frames = numericalFrames([source, differentOperation, differentGeneration])
  assert.equal(frames.length, 3)
  assert.ok(frames.every((frame) => frame.snapshots.length === 1))
  const conflicting = { ...source, snapshotId: "conflicting-slice", values: source.values.map((value, index) => index === 0 ? value + 1 : value) }
  const { cells, conflictCount } = numericalCells(numericalFrames([source, conflicting])[0], 8, 128)
  assert.equal(conflictCount, 1)
  assert.equal(cells.size, 15)
  assert.equal(cells.get(0), undefined, "conflicting retained values must remain unknown")
})

test("FP8 display representations preserve raw encodings separately from decoded and reconstructed KV", () => {
  const source = observed.snapshots[0]
  const authored = { ...source, dtype: "uint8", values: [1, 2], shape: [2], byteCount: 6,
    quantization: { format: "fp8_e4m3" as const, rawBits: [56, 64], scale: { value: 2, dtype: "float32" as const, shape: [1], mode: "per_tensor" as const, headIndex: null, observationBoundary: source.observationBoundary }, reconstructedValues: [2, 4], reconstruction: "decoded_fp8_times_recorded_scale" as const } }
  const frame = numericalFrames([authored])[0]
  assert.deepEqual([...numericalCells(frame, 8, 128, "stored").cells.values()].map((cell) => cell.value), [1, 2])
  assert.deepEqual([...numericalCells(frame, 8, 128, "raw").cells.values()].map((cell) => cell.value), [56, 64])
  assert.deepEqual([...numericalCells(frame, 8, 128, "reconstructed").cells.values()].map((cell) => cell.value), [2, 4])
  assert.equal(numericalCells(numericalFrames([source])[0], 8, 128, "raw").cells.size, 0)
})
