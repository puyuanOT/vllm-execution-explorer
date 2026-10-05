import assert from "node:assert/strict"
import { test } from "node:test"
import { defaultSettings, staggeredExample } from "./example.ts"
import { learningExample, LEARNING_EXAMPLES, matchingLearningExample } from "./learningExamples.ts"
import { reduceScenario } from "./reducer.ts"
import { toRunBody, scenarioFromRecordedPlan } from "./serialize.ts"
import { validateScenario, validateSettings } from "./validate.ts"
import { resolveWorkloadFields } from "./workloadFields.ts"
import { playbackSpeeds } from "../playback.ts"

test("learning examples remain editable valid live drafts and preserve host/model choices", () => {
  const original = { ...defaultSettings(), model: "custom/model", sshHost: "gpu.example", runnerUrl: "http://127.0.0.1:8771" }
  for (const { id } of LEARNING_EXAMPLES) {
    const example = learningExample(id, original)
    assert.deepEqual(validateScenario(example.scenario, null, example.settings), [], id)
    assert.equal(example.settings.model, original.model)
    assert.equal(example.settings.sshHost, original.sshHost)
    assert.equal(example.settings.runnerUrl, original.runnerUrl)
    const requests = toRunBody(example.scenario, example.settings).requests
    assert.equal(new Set(requests.map((request) => request.request_id)).size, requests.length)
    assert.ok(requests.every((request) => request.arrival_offset_ms >= 0))
  }
  assert.equal(original.kvCacheMemoryBytes, "")
})

test("a loaded learning example is recognized only while its workload and overrides are unchanged", () => {
  const settings = { ...defaultSettings(), model: "custom/model" }
  assert.equal(matchingLearningExample(staggeredExample(), defaultSettings()), "staggered")
  assert.equal(matchingLearningExample({ ...staggeredExample(), id: "scenario-reset-copy" }, settings), "staggered")
  for (const { id } of LEARNING_EXAMPLES) {
    const example = learningExample(id, settings)
    assert.equal(matchingLearningExample(example.scenario, example.settings), id)
  }

  const prefix = learningExample("prefix", settings)
  assert.equal(matchingLearningExample(prefix.scenario, { ...prefix.settings, prefixCaching: false }), null)
  assert.equal(matchingLearningExample(prefix.scenario, { ...prefix.settings, model: "other/model" }), "prefix")

  const edited = reduceScenario(staggeredExample(), { type: "set-sample-prompt", sampleId: "sample-1", prompt: "Name a secondary color." })
  assert.equal(matchingLearningExample(edited, defaultSettings()), null)
  const moved = reduceScenario(staggeredExample(), { type: "commit-batch-time", batchId: "batch-c", arrivalOffsetMs: 140 })
  assert.equal(matchingLearningExample(moved, defaultSettings()), null)
  assert.equal(matchingLearningExample(reduceScenario(staggeredExample(), { type: "delete-batch", batchId: "batch-c" }), defaultSettings()), null)

  const chunked = learningExample("chunked", settings)
  const longer = { ...chunked.scenario, batches: chunked.scenario.batches.map((batch) => ({ ...batch, samples: batch.samples.map((sample, index) => index === 0 ? { ...sample, syntheticInputTokens: 640 } : sample) })) }
  assert.equal(matchingLearningExample(longer, chunked.settings), null)
  const resolved = { ...chunked.scenario, batches: chunked.scenario.batches.map((batch) => ({ ...batch, samples: batch.samples.map((sample) => ({ ...sample, resolvedInputTokens: 512 })) })) }
  assert.equal(matchingLearningExample(resolved, chunked.settings), "chunked")
})

test("automatic revisions and explicit tokenizer overrides serialize without stale pins", () => {
  const auto = toRunBody(staggeredExample(), defaultSettings())
  assert.equal(auto.model_revision, null)
  assert.equal(auto.tokenizer_revision, null)
  assert.equal(auto.model_configuration.tokenizer.locator, auto.model)
  assert.deepEqual(validateSettings(defaultSettings()), [])
  const override = toRunBody(staggeredExample(), { ...defaultSettings(), modelRevision: "release", tokenizer: "other/tokenizer", tokenizerRevision: "pinned" })
  assert.equal(override.model_configuration.source.revision, "release")
  assert.deepEqual(override.model_configuration.tokenizer, { locator: "other/tokenizer", revision: "pinned" })
})

test("raw priorities cannot silently fall back and are retained by recording edits", () => {
  const scenario = staggeredExample()
  for (const raw of ["", "1.5", "1e2", "2147483648", "-2147483648"]) {
    const result = resolveWorkloadFields(scenario, "ms", (id) => id === "sample-priority-sample-1" ? raw : undefined)
    assert.ok("issue" in result, raw)
    assert.equal(result.issue.fieldId, "sample-priority-sample-1")
  }
  const valid = resolveWorkloadFields(scenario, "ms", (id) => id === "sample-priority-sample-1" ? "-4" : undefined)
  assert.ok("scenario" in valid)
  assert.ok(validateScenario(valid.scenario, null, defaultSettings()).some((issue) => issue.fieldId === "sample-priority-sample-1"))
  const settings = { ...defaultSettings(), schedulingPolicy: "priority" as const }
  assert.deepEqual(validateScenario(valid.scenario, null, settings), [])
  const body = toRunBody(valid.scenario, settings)
  assert.equal(body.requests[0]?.priority, -4)
  assert.equal(scenarioFromRecordedPlan(body.workload_plan, {})?.batches[0]?.samples[0]?.priority, -4)
})

test("invalid runtime combinations fail before submitting a capture", () => {
  assert.ok(validateSettings({ ...defaultSettings(), tokenBudget: "2" }).some((issue) => issue.fieldId === "setting-token-budget"))
  assert.ok(validateSettings({ ...defaultSettings(), chunkedPrefill: false }).some((issue) => issue.fieldId === "setting-token-budget"))
  assert.ok(validateSettings({ ...defaultSettings(), profilingMode: "eager_diagnostic", engine: "fixture" }).some((issue) => issue.fieldId === "setting-profiling-mode"))
  for (const value of ["0", "17", "2.5"]) {
    assert.ok(validateSettings({ ...defaultSettings(), profilingMode: "eager_diagnostic", profilingMaxSteps: value }).some((issue) => issue.fieldId === "setting-profiling-steps"))
  }
  assert.ok(validateSettings({ ...defaultSettings(), kvCacheMemoryBytes: "9007199254740992" }).some((issue) => issue.fieldId === "setting-kv-memory"))
})

test("recorded clock supports much slower speeds without changing other clocks", () => {
  assert.equal(playbackSpeeds("recorded")[0], 0.001)
  assert.ok(playbackSpeeds("recorded").includes(0.0025))
  assert.equal(playbackSpeeds("events")[0], 0.25)
  assert.equal(playbackSpeeds("teaching")[0], 0.25)
})
