import assert from "node:assert/strict"
import { test } from "node:test"
import type { RecordingManifest } from "./reduce.ts"
import { createProfileIntervalIndex, parseProfileAttachment, profileActivitiesAt, projectMeasuredProfile, readProfileArtifact } from "./profiling.ts"

const clock = { domain: "torch.profiler.chrome_trace", unit: "microseconds", origin: "trace_local" }
const manifest: RecordingManifest = { run_id: "profile-test", profiling: {
  contract_version: 1, mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock,
  status: "completed", completeness: "bounded", qualification: "unverified", attachment_status: "available",
  normalized_artifact: { path: "profiling/activities.json", sha256: "a".repeat(64), size_bytes: 100 },
  runtime_versions: { vllm: "0.30.0", torch: "2.10.0" },
} }

function activity(id: string, start: number, duration: number, fields: Record<string, unknown> = {}) {
  return { activity_id: id, kind: "cuda_kernel", name: "fused_attention", device: 0, stream: 7, start, duration,
    scope: { step_id: "step-1", layer_index: 2, component: "attention", correlation: "runtime_flow" }, ...fields }
}

function source() {
  return { contract_version: 1, run_id: "profile-test", mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock,
    activities: [activity("a", 1_000_000.25, 10), activity("b", 1_000_005.25, 10, { stream: 8 }),
      activity("cpu", 999_900, 25, { kind: "cpu_scope", device: null, stream: null,
        scope: { step_id: "step-1", layer_index: 2, component: "attention", correlation: "explicit_scope" } }),
      activity("unknown", 1_000_030, 2, { scope: { step_id: null, layer_index: null, component: null, correlation: "unmapped" } })],
    steps: [{ step_id: "step-1", mapping_status: "observed", unpadded_tokens: 3, padded_tokens: 4,
      mapping_observation_boundary: "GPUModelRunner._prepare_inputs.returned",
      packed_rows: [
        { row: 0, request_id: "b", token_position: 9, is_padding: false },
        { row: 1, request_id: "a", token_position: 2, is_padding: false },
        { row: 2, request_id: "a", token_position: 3, is_padding: false },
        { row: 3, request_id: null, token_position: null, is_padding: true },
      ] }],
  }
}

test("measured scope uses native profiler time and packed post-reorder rows, with no token/head duration", () => {
  const attachment = parseProfileAttachment(JSON.stringify(source()), manifest)
  const result = projectMeasuredProfile(manifest, attachment, { stepId: "step-1", layerIndex: 2, component: "attention", timeUs: 1_000_006 })
  assert.equal(result.status, "observed")
  assert.deepEqual(result.clock, clock)
  assert.equal(result.qualification, "unverified")
  assert.deepEqual(result.selectedStep?.packedRows.map((row) => [row.row, row.requestId, row.tokenPosition]), [[0, "b", 9], [1, "a", 2], [2, "a", 3], [3, null, null]])
  assert.deepEqual(result.activeActivities.map((activity) => activity.id), ["a", "b"])
  const layer = result.layers[0]
  assert.equal(layer.cpuScopeSpanUs, 25)
  assert.equal(layer.devices[0].spanUs, 15)
  assert.equal(layer.devices[0].busyUs, 15)
  assert.equal(layer.devices[0].summedActivityUs, 20)
  assert.equal(result.unmappedActivityCount, 1)
  assert.equal(Object.hasOwn(result.selectedStep!.packedRows[0], "durationUs"), false)
  assert.equal(Object.hasOwn(layer, "tokenDurationUs"), false)
  result.selectedStep!.packedRows[0].requestId = "modified"
  result.activities[0].scope.layerIndex = 999
  assert.equal(attachment.steps[0].packedRows[0].requestId, "b")
  assert.equal(attachment.activities[0].scope.layerIndex, 2)
})

test("profiler selection is independent of host replay and cannot map activity names to a layer", () => {
  const value = source()
  value.activities.push(activity("looks-like-layer", 1_000_012, 6, {
    name: "model.layers.2.self_attn", scope: { step_id: null, layer_index: null, component: null, correlation: "unmapped" },
  }))
  const attachment = parseProfileAttachment(JSON.stringify(value), manifest)
  const first = projectMeasuredProfile(manifest, attachment, { timeUs: 1_000_000.25 })
  const later = projectMeasuredProfile(manifest, attachment, { timeUs: 1_000_030 })
  const back = projectMeasuredProfile(manifest, attachment, { timeUs: 1_000_000.25 })
  assert.deepEqual(first, back)
  assert.deepEqual(later.activeActivities.map((activity) => activity.id), ["unknown"])
  assert.equal(first.layers.length, 1)
  assert.equal(first.unmappedActivityCount, 2)
  assert.equal(first.mappingStatus, "unavailable")
  assert.match(first.mappingReason!, /Select a captured batch/)
  assert.equal(projectMeasuredProfile(manifest, attachment, { stepId: "missing" }).mappingStatus, "unavailable")
})

test("unknown packed mapping and missing attachments remain explicit, including structural export", () => {
  const value = source()
  value.steps = [{ step_id: "step-1", mapping_status: "unavailable", unpadded_tokens: 3, padded_tokens: 4,
    mapping_observation_boundary: "unobserved", packed_rows: [], reason: "Input preparation was not observed." } as unknown as typeof value.steps[number]]
  const attachment = parseProfileAttachment(JSON.stringify(value), manifest)
  const projected = projectMeasuredProfile(manifest, attachment, { stepId: "step-1" })
  assert.equal(projected.mappingStatus, "unavailable")
  assert.match(projected.mappingReason!, /not observed/)
  assert.deepEqual(projected.selectedStep?.packedRows, [])
  assert.match(projectMeasuredProfile(manifest).reason!, /unavailable/)
  const structural = structuredClone(manifest)
  ;(structural.profiling as Record<string, unknown>).attachment_status = "omitted_by_structural_export"
  assert.match(projectMeasuredProfile(structural, attachment).reason!, /structural export/)
  assert.equal(projectMeasuredProfile({}).status, "unavailable")
  assert.equal(projectMeasuredProfile({ profiling: { mode: "off", status: "disabled" } }).status, "unavailable")
})

test("partial capture retains measured intervals with bounded qualification and no host conversion", () => {
  const partial = structuredClone(manifest)
  Object.assign(partial.profiling as object, { status: "partial", completeness: "unknown" })
  const projection = projectMeasuredProfile(partial, parseProfileAttachment(JSON.stringify(source()), partial), { timeUs: 42 })
  assert.equal(projection.status, "observed")
  assert.match(projection.reason!, /did not complete/)
  assert.equal(projection.completeness, "unknown")
  assert.equal(projection.activeActivities.length, 0)
  assert.equal(projection.activities[0].startUs, 1_000_000.25)
  assert.deepEqual(projection.extent, { startUs: 999_900, endUs: 1_000_032 })
})

test("device summaries do not merge devices or treat CPU launch spans as GPU busy time", () => {
  const value = source()
  value.activities.push(activity("device-1", 1_000_050, 3, { device: 1, stream: "9" }),
    activity("unknown-device", 1_000_070, 4, { device: null, stream: null }),
    activity("later", 1_000_020.25, 2))
  const result = projectMeasuredProfile(manifest, parseProfileAttachment(JSON.stringify(value), manifest), { stepId: "step-1" })
  const devices = result.layers[0].devices
  assert.deepEqual(devices.map((device) => device.device), [0, 1, null])
  assert.equal(devices[0].busyUs, 17)
  assert.equal(devices[0].spanUs, 22)
  assert.equal(devices[1].busyUs, 3)
  assert.equal(devices[2].busyUs, null)
  assert.equal(devices[2].summedActivityUs, 4)
})

test("observed real rows survive execution failure before padding is observed", () => {
  const value = source()
  const step = value.steps[0] as Omit<typeof value.steps[number], "padded_tokens"> & { padded_tokens: number | null }
  step.padded_tokens = null
  step.packed_rows.pop()
  const partial = structuredClone(manifest)
  Object.assign(partial.profiling as object, { status: "partial" })
  const projected = projectMeasuredProfile(partial, parseProfileAttachment(JSON.stringify(value), partial), { stepId: "step-1" })
  assert.equal(projected.mappingStatus, "observed")
  assert.equal(projected.selectedStep?.paddedTokens, null)
  assert.equal(projected.selectedStep?.paddingObservationBoundary, null)
  assert.equal(projected.selectedStep?.paddingStatus, "unavailable")
  assert.match(projected.selectedStep!.paddingReason!, /not observed/)
  assert.equal(projected.selectedStep?.packedRows.length, 3)
  assert.equal(projected.selectedStep?.packedRows.some((row) => row.isPadding), false)
})

test("synthetic attachments retain explicit fixture provenance rather than claim GPU measurement", () => {
  const fixture = { ...manifest, execution_kind: "fixture" }
  const result = projectMeasuredProfile(fixture, parseProfileAttachment(JSON.stringify(source()), fixture))
  assert.equal(result.evidenceKind, "fixture")
  assert.match(result.reason!, /not GPU measurements/)
})

test("contradictory padding observation claims are rejected", () => {
  const value = source()
  Object.assign(value.steps[0], { padding_status: "unavailable" })
  assert.throws(() => parseProfileAttachment(JSON.stringify(value), manifest), /Unavailable padding/)
  Object.assign(value.steps[0], { padding_status: "observed", padded_tokens: null })
  value.steps[0].packed_rows.pop()
  assert.throws(() => parseProfileAttachment(JSON.stringify(value), manifest), /Observed padding/)
})

test("attachment identity, modes and clock reject incompatible or foreign data", () => {
  for (const change of [{ run_id: "foreign" }, { contract_version: 2 }, { execution_mode: "cuda_graph" },
    { clock: { ...clock, unit: "nanoseconds" } }, { clock: { ...clock, origin: "host_monotonic" } }]) {
    assert.throws(() => parseProfileAttachment(JSON.stringify({ ...source(), ...change }), manifest))
  }
  assert.throws(() => parseProfileAttachment("invalid", manifest), /valid JSON/)
  assert.throws(() => parseProfileAttachment(JSON.stringify(source()), {}), /manifest.profiling/)
})

test("bad correlation, malformed rows and duplicate identities fail without inventing attribution", () => {
  const badActivities = [
    { duration: -1 }, { start: Infinity }, { scope: { step_id: "step-1", layer_index: 2, component: null, correlation: "unmapped" } },
    { scope: { step_id: "step-1", layer_index: 2, component: null, correlation: "explicit_scope" } },
  ]
  for (const fields of badActivities) {
    const value = source()
    Object.assign(value.activities[0], fields)
    assert.throws(() => parseProfileAttachment(JSON.stringify(value), manifest))
  }
  const duplicate = source()
  duplicate.activities.push({ ...duplicate.activities[0] })
  assert.throws(() => parseProfileAttachment(JSON.stringify(duplicate), manifest), /Duplicate activity/)
  for (const mutate of [
    (value: ReturnType<typeof source>) => { value.steps[0].packed_rows[0].request_id = null },
    (value: ReturnType<typeof source>) => { value.steps[0].packed_rows[3].request_id = "a" },
    (value: ReturnType<typeof source>) => { value.steps[0].packed_rows[0].row = 5 },
    (value: ReturnType<typeof source>) => { value.steps[0].padded_tokens = 5 },
    (value: ReturnType<typeof source>) => { value.steps[0].mapping_observation_boundary = "" },
  ]) {
    const value = source()
    mutate(value)
    assert.throws(() => parseProfileAttachment(JSON.stringify(value), manifest))
  }
})

test("loader artifact references are explicit relative paths with integrity metadata", () => {
  assert.deepEqual(readProfileArtifact(manifest), { path: "profiling/activities.json", sha256: "a".repeat(64), sizeBytes: 100 })
  assert.equal(readProfileArtifact({}), null)
  for (const path of ["../secret", "/absolute", "https://host/profile", "profiling\\activities.json", "profiling//a", "profiling/a?x", "profiling/./a", "profiling/%2e%2e/a"]) {
    const value = structuredClone(manifest)
    const metadata = value.profiling as { normalized_artifact: { path: string } }
    metadata.normalized_artifact.path = path
    assert.throws(() => readProfileArtifact(value), /safe relative/)
  }
})

test("native interval index preserves asynchronous overlaps, endpoints, zero events and backward seeks", () => {
  const value = source()
  value.activities.push(activity("long-cpu", 999_000, 3_000, { kind: "cpu_scope", device: null, stream: null,
    scope: { step_id: "step-1", layer_index: null, component: null, correlation: "explicit_scope" } }),
    activity("point", 1_000_015.25, 0), activity("negative-point", -5.25, 0), activity("zero-point", 0, 0))
  const attachment = parseProfileAttachment(JSON.stringify(value), manifest)
  const index = createProfileIntervalIndex(attachment.activities)
  assert.equal(index.activityCount, attachment.activities.length)
  for (const time of [1_000_006, 1_000_030, 999_900, 1_000_015.25, 1_000_015.2500001, 1_000_010.25, 1_002_000, 1_000_006, -5.25, -5.250001, 0, Number.MIN_VALUE]) {
    assert.deepEqual(index.at(time).map((activity) => activity.id), profileActivitiesAt(attachment.activities, time).map((activity) => activity.id))
  }
  assert.deepEqual(index.at(1_000_006).map((activity) => activity.id), ["a", "b", "long-cpu"])
  assert.equal(index.at(1_000_015.25).some((activity) => activity.id === "b"), false)
  assert.equal(index.at(1_000_015.25).some((activity) => activity.id === "point"), true)
  assert.throws(() => index.at(Infinity), /finite/)
  assert.deepEqual(createProfileIntervalIndex([]).at(0), [])
})

test("indexed sparse large traces match a scan at deterministically varied native playheads", () => {
  const template = parseProfileAttachment(JSON.stringify(source()), manifest).activities[0]
  const activities = Array.from({ length: 35_000 }, (_, index) => ({ ...template, id: `large-${index}`, startUs: index * 100,
    durationUs: index % 701 === 0 ? 1_500 : 4, endUs: index * 100 + (index % 701 === 0 ? 1_500 : 4) }))
  // Reverse original order: index output must retain it even though starts are sorted internally.
  activities.reverse()
  const index = createProfileIntervalIndex(activities)
  for (let seed = 1, count = 0; count < 60; count += 1) {
    seed = (Math.imul(seed, 1_664_525) + 1_013_904_223) >>> 0
    const time = seed % 3_500_100
    assert.deepEqual(index.at(time).map((activity) => activity.id), profileActivitiesAt(activities, time).map((activity) => activity.id))
  }
  assert.equal(index.at(34_999 * 100 + 2)[0].id, "large-34999")
  assert.deepEqual(index.at(4_000_000), [])
})
