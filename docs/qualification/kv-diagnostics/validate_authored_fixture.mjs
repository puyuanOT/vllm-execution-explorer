import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { createHash } from "node:crypto"
import { fileURLToPath } from "node:url"
import { attachContent, parseJsonl, replay } from "../../../replay/src/reduce.ts"
import { parseProfileAttachment, readProfileArtifact } from "../../../replay/src/profiling.ts"
import { projectKvConfiguration, projectKvTensorPreview, projectLayerTokenMatrix } from "../../../replay/src/kv.ts"
import { scenarioFromRecordedPlan } from "../../../web/src/scenario/serialize.ts"
import { promptsFromContent } from "../../../replay/src/timeline.ts"
import { tensorRerunSettings } from "../../../web/src/scenario/tensorRerun.ts"
import { defaultSettings } from "../../../web/src/scenario/example.ts"

const directory = new URL("./authored-recording/", import.meta.url)
const read = (name) => readFileSync(new URL(name, directory), "utf8")
const manifest = JSON.parse(read("manifest.json"))
assert.equal(manifest.execution_kind, "fixture")
assert.equal(manifest.provenance.real_gpu, false)
assert.match(manifest.fixture_lesson.claim, /Entirely synthetic/)
const events = parseJsonl(read("events.jsonl"))
const content = parseJsonl(read("content.jsonl"))
const raw = read("profiling/activities.json")
const artifact = readProfileArtifact(manifest)
assert.equal(artifact.sha256, createHash("sha256").update(raw).digest("hex"))
assert.equal(artifact.sizeBytes, Buffer.byteLength(raw))
const profile = parseProfileAttachment(raw, manifest)
const stateAt = (sequence) => attachContent(replay(events.filter((event) => event.sequence <= sequence), manifest), content)
const writer = stateAt(manifest.qualification_bookmarks.writer_computed)
const reader = stateAt(manifest.qualification_bookmarks.reader_scheduled)
const configuration = projectKvConfiguration(reader, manifest, content)
assert.equal(configuration.evidenceKind, "fixture")
assert.equal(configuration.uniqueStorageBytes, 2048)
assert.equal(configuration.layers.length, 2)
assert.deepEqual(configuration.pools[0].devices, ["cuda:0"])
assert.equal(configuration.pools[0].capacity, 2)
assert.equal(configuration.pools[0].referencedBlocks, 2)
assert.equal(configuration.maxModelLength, 64)
assert.equal(configuration.layers[0].quantization.mode, "NONE")
const matrix = (state, stepId, timeUs) => projectLayerTokenMatrix(state, manifest, profile, { stepId, timeUs, content })
assert.deepEqual(matrix(writer, "step-1", 130).cells.filter((cell) => cell.position === 0).map((cell) => cell.state), ["operation_active", "scheduled"])
assert.deepEqual(matrix(writer, "step-1", 170).cells.filter((cell) => cell.position === 0).map((cell) => cell.state), ["completed", "operation_active"])
const reuse = matrix(reader, "step-2", 430)
assert.equal(reuse.evidenceKind, "fixture")
assert.equal(reuse.cells.filter((cell) => cell.requestId === "prefix-reader" && cell.position < 4).every((cell) => cell.state === "reused_existing_kv"), true)
assert.deepEqual(reuse.cells.filter((cell) => cell.requestId === "prefix-reader" && cell.position === 4).map((cell) => cell.state), ["operation_active", "scheduled"])
const numeric = projectKvTensorPreview(manifest, profile, { requestId: "prefix-writer", tokenPosition: 1, layerIndex: 0, kvHead: 0 })
assert.equal(numeric.evidenceKind, "fixture")
assert.equal(numeric.snapshots.length, 2)
assert.deepEqual(numeric.snapshots.map((snapshot) => snapshot.values.length), [16, 16])
assert.equal(numeric.capture.status, "partial")
assert.equal(numeric.capture.retainedBytes, 256)
assert.equal(projectKvTensorPreview(manifest, profile, { requestId: "prefix-writer", layerIndex: 1 }).status, "unavailable")
const copied = scenarioFromRecordedPlan(manifest.workload_plan, promptsFromContent(content))
assert.equal(copied.batches.length, 2)
assert.equal(copied.batches[1].samples[0].prompt, "The shared prefix ends with reuse")
const rerun = tensorRerunSettings(defaultSettings(), manifest, { requestId: "prefix-writer", tokenPosition: 1, layerIndex: 1, kvHead: 0, startStep: 1 })
assert.equal(rerun.engine, "vllm")
assert.equal(rerun.model, "Qwen/Qwen3-4B-Instruct-2507")
assert.equal(rerun.dtype, "float16")
assert.equal(rerun.tensorRequestId, "prefix-writer")
assert.equal(rerun.tensorTokenPositions, "1")
assert.equal(rerun.tensorLayerIndices, "1")
assert.equal(rerun.profilingTriggerRequestId, "prefix-writer")
const final = stateAt(events.at(-1).sequence)
assert.equal(final.requests["prefix-reader"].acceptedTokenCount, 2)
console.log(JSON.stringify({ kind: "authored_fixture_validation", recording: fileURLToPath(directory), events: events.length,
  operations: profile.cacheWrites.length, snapshots: profile.tensorSnapshots.length,
  outcomes: ["alias-safe capacity metadata", "independent-clock active/completed layer states", "prefix reuse", "bounded numerical slices", "missing slice targeted rerun", "prompt/configuration restoration"],
  real_gpu: false }, null, 2))
