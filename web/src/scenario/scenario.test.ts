import assert from "node:assert/strict"
import { test } from "node:test"

import { defaultSettings, emptyScenario, staggeredExample } from "./example.ts"
import { reduceScenario } from "./reducer.ts"
import { flattenRequests, scenarioFromRecordedPlan, toRunBody } from "./serialize.ts"
import { normalizeScenario } from "./storage.ts"
import { axisSpanMs, millisecondsFromAxis, parseSecondsToMilliseconds } from "./time.ts"
import { validateScenario } from "./validate.ts"
import type { ScenarioDraft } from "./types.ts"

test("the example flattens to the 0/100/120 offsets without merging equal times", () => {
  const requests = flattenRequests(staggeredExample())
  assert.deepEqual(requests.map((item) => item.arrival_offset_ms), [0, 0, 100, 100, 100, 120])
  assert.deepEqual(requests.map((item) => item.request_id), [
    "sample-1",
    "sample-2",
    "sample-3",
    "sample-4",
    "sample-5",
    "sample-6",
  ])
  assert.equal(JSON.stringify(flattenRequests(staggeredExample())).includes("Name a primary color."), true)
})

test("authoring the acceptance sequence keeps identities when labels change", () => {
  let scenario = emptyScenario()
  scenario = reduceScenario(scenario, { type: "add-batch" })
  scenario = reduceScenario(scenario, { type: "add-batch" })
  scenario = reduceScenario(scenario, { type: "add-batch" })
  const [first, second, third] = scenario.batches
  assert.ok(first && second && third)
  assert.deepEqual(scenario.batches.map((batch) => batch.arrivalOffsetMs), [0, 100, 200])
  scenario = reduceScenario(scenario, { type: "commit-batch-time", batchId: third.id, arrivalOffsetMs: 120 })
  assert.deepEqual(scenario.batches.map((batch) => batch.arrivalOffsetMs), [0, 100, 120])
  scenario = addSamples(scenario, first.id, 2)
  scenario = addSamples(scenario, second.id, 3)
  scenario = addSamples(scenario, third.id, 1)
  const labels = ["input_1", "input_2", "input_3", "input_4", "input_5", "input_6"]
  const prompts = ["Color", "Capital", "Count", "Fruit", "Weekday", "Season"]
  let index = 0
  for (const batch of scenario.batches) {
    for (const sample of batch.samples) {
      const label = labels[index] ?? "input"
      const prompt = prompts[index] ?? "Prompt"
      scenario = reduceScenario(scenario, { type: "set-sample-label", sampleId: sample.id, label })
      scenario = reduceScenario(scenario, { type: "set-sample-prompt", sampleId: sample.id, prompt })
      scenario = reduceScenario(scenario, { type: "set-sample-output", sampleId: sample.id, maxOutputTokens: 8 })
      index += 1
    }
  }
  const ids = scenario.batches.flatMap((batch) => batch.samples.map((sample) => sample.id))
  const renamed = reduceScenario(scenario, { type: "set-sample-label", sampleId: ids[3] ?? "", label: "fruit sample" })
  assert.equal(renamed.batches[1]?.samples[1]?.id, ids[3])
  assert.equal(renamed.batches[1]?.samples[1]?.label, "fruit sample")
  assert.deepEqual(flattenRequests(renamed).map((item) => item.arrival_offset_ms), [0, 0, 100, 100, 100, 120])
  assert.deepEqual(flattenRequests(renamed).map((item) => item.request_id), ids)
})

test("duplicate, move, and delete keep other sample identities", () => {
  let scenario = staggeredExample()
  const moved = scenario.batches[1]?.samples[0]
  const destination = scenario.batches[0]
  assert.ok(moved && destination)
  scenario = reduceScenario(scenario, { type: "move-sample", sampleId: moved.id, toBatchId: destination.id })
  assert.equal(scenario.batches[0]?.samples.at(-1)?.id, moved.id)
  assert.equal(scenario.batches[1]?.samples.length, 2)
  const before = scenario.batches[0]?.samples[0]
  assert.ok(before)
  scenario = reduceScenario(scenario, { type: "duplicate-sample", batchId: destination.id, sampleId: before.id })
  const copy = scenario.batches[0]?.samples[1]
  assert.ok(copy)
  assert.notEqual(copy.id, before.id)
  assert.equal(copy.prompt, before.prompt)
  const snapshot = scenario
  scenario = reduceScenario(scenario, { type: "delete-sample", batchId: destination.id, sampleId: copy.id })
  assert.equal(scenario.batches[0]?.samples.some((sample) => sample.id === copy.id), false)
  assert.equal(snapshot.batches[0]?.samples.some((sample) => sample.id === copy.id), true)
})

test("equal arrival times stay in authoring order and are not merged", () => {
  let scenario = staggeredExample()
  const third = scenario.batches[2]
  assert.ok(third)
  scenario = reduceScenario(scenario, { type: "commit-batch-time", batchId: third.id, arrivalOffsetMs: 100 })
  assert.equal(scenario.batches.length, 3)
  assert.deepEqual(scenario.batches.map((batch) => batch.id), ["batch-a", "batch-b", "batch-c"])
  assert.deepEqual(scenario.batches.map((batch) => batch.arrivalOffsetMs), [0, 100, 100])
})

test("seconds convert exactly and reject sub-millisecond values", () => {
  assert.equal(parseSecondsToMilliseconds("0.1"), 100)
  assert.equal(parseSecondsToMilliseconds("0.12"), 120)
  assert.equal(parseSecondsToMilliseconds("0"), 0)
  assert.equal(parseSecondsToMilliseconds("0.1001"), null)
  assert.equal(parseSecondsToMilliseconds("0.1005"), null)
  assert.equal(parseSecondsToMilliseconds("-1"), null)
})

test("dragging an arrival marker lands on integer milliseconds", () => {
  assert.equal(axisSpanMs([0, 100, 120], 120_000), 200)
  assert.equal(millisecondsFromAxis(0, 0, 200, 200, 120_000), 0)
  assert.equal(millisecondsFromAxis(100, 0, 200, 200, 120_000), 100)
  assert.equal(millisecondsFromAxis(120, 0, 200, 200, 120_000), 120)
  assert.equal(millisecondsFromAxis(400, 0, 200, 200, 120_000), 200)
  const scenario = reduceScenario(staggeredExample(), { type: "commit-batch-time", batchId: "batch-c", arrivalOffsetMs: 120 })
  assert.equal(scenario.batches.filter((batch) => batch.arrivalOffsetMs === 120).length, 1)
})

test("saved v1 drafts from before input modes are normalized", () => {
  const legacy = {
    version: 1,
    id: "legacy",
    name: "Legacy draft",
    nextOrder: 1,
    batches: [{
      id: "batch-legacy",
      label: "A",
      arrivalOffsetMs: 0,
      createdOrder: 0,
      samples: [{
        id: "sample-legacy",
        label: "input_1",
        prompt: "Hello",
        maxOutputTokens: 8,
      }],
    }],
  } as unknown as ScenarioDraft
  const normalized = normalizeScenario(legacy)
  assert.equal(normalized.batches[0]?.defaultAdapterId, null)
  assert.equal(normalized.batches[0]?.samples[0]?.inputMode, "prompt")
  assert.equal(normalized.batches[0]?.samples[0]?.syntheticInputTokens, 128)
  assert.equal(normalized.batches[0]?.samples[0]?.resolvedInputTokens, null)
  assert.equal(normalized.batches[0]?.samples[0]?.adapterId, null)
})

test("run serialization emits the nested capture contract", () => {
  const body = toRunBody(staggeredExample(), defaultSettings())
  assert.equal(body.model_configuration.source.kind, "huggingface")
  assert.equal(body.vllm_configuration.max_num_scheduled_tokens, 2048)
  assert.equal(body.vllm_configuration.fresh_prefix_cache, true)
  assert.deepEqual(body.sampling, { temperature: 0, top_p: 1, seed: 0, stop: [] })
  assert.equal(body.requests[0]?.target_input_tokens, null)
  assert.equal(body.requests[0]?.resolved_input_tokens, null)
})

test("runner capabilities block an unavailable saved quantization choice", () => {
  const settings = { ...defaultSettings(), quantization: "bitsandbytes" as const }
  const issues = validateScenario(staggeredExample(), null, settings, {
    bitsandbytes: {
      status: "blocked",
      selectable: false,
      message: "Install vllm-bnb-plugin on this execution host.",
    },
  })
  assert.deepEqual(
    issues.filter((issue) => issue.fieldId === "model-quantization"),
    [{ fieldId: "model-quantization", message: "Install vllm-bnb-plugin on this execution host." }],
  )
})

test("recorded base routes remain explicit when copied into an adapter draft", () => {
  const copied = scenarioFromRecordedPlan({
    batches: [{
      id: "batch-a",
      label: "A",
      arrival_offset_ms: 0,
      default_adapter_id: "adapter-a",
      samples: [{
        id: "sample-a",
        label: "input_1",
        input_mode: "prompt",
        max_output_tokens: 4,
        adapter_id: "base",
      }],
    }],
  }, { "sample-a": "Hello" })
  assert.ok(copied)
  assert.equal(copied.batches[0]?.defaultAdapterId, "adapter-a")
  assert.equal(copied.batches[0]?.samples[0]?.adapterId, "__base__")
  assert.equal(flattenRequests(copied)[0]?.adapter_id, null)
})

function addSamples(scenario: ScenarioDraft, batchId: string, count: number): ScenarioDraft {
  let next = scenario
  for (let index = 0; index < count; index += 1) {
    next = reduceScenario(next, { type: "add-sample", batchId })
  }
  return next
}
