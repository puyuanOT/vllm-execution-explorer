import assert from "node:assert/strict"
import { test } from "node:test"

import { defaultSettings, staggeredExample } from "./example.ts"
import { reduceScenario } from "./reducer.ts"
import { toRunBody } from "./serialize.ts"
import { resolveWorkloadFields } from "./workloadFields.ts"

test("invalid raw output values block a saved valid draft", () => {
  const scenario = staggeredExample()
  for (const raw of ["", "0", "-1", "1.5", "1e3", "9007199254740992"]) {
    const result = resolveWorkloadFields(scenario, "ms", (id) => id === "sample-output-sample-1" ? raw : undefined)
    assert.ok("issue" in result, `accepted invalid output ${raw}`)
    assert.equal(result.issue.fieldId, "sample-output-sample-1")
    assert.equal(scenario.batches[0]?.samples[0]?.maxOutputTokens, 8)
  }
})

test("invalid raw offsets block the previous valid time in either display unit", () => {
  const scenario = staggeredExample()
  for (const unit of ["ms", "s"] as const) {
    for (const raw of ["", "abc", "-1", "0.0001", "9007199254740992"]) {
      const result = resolveWorkloadFields(scenario, unit, (id) => id === "batch-time-batch-a" ? raw : undefined)
      assert.ok("issue" in result, `accepted invalid ${unit} offset ${raw}`)
      assert.equal(result.issue.fieldId, "batch-time-batch-a")
      assert.equal(scenario.batches[0]?.arrivalOffsetMs, 0)
    }
  }
})

test("submission uses latest valid raw values and preserves identities and order", () => {
  const scenario = reduceScenario(staggeredExample(), { type: "set-sample-mode", sampleId: "sample-2", inputMode: "synthetic" })
  const raw: Record<string, string> = {
    "batch-time-batch-b": "0.12",
    "sample-output-sample-1": "12",
    "sample-synthetic-sample-2": "256",
  }
  const result = resolveWorkloadFields(scenario, "s", (id) => raw[id])
  assert.ok("scenario" in result)
  assert.deepEqual(result.scenario.batches.map((batch) => batch.id), scenario.batches.map((batch) => batch.id))
  assert.deepEqual(result.scenario.batches.map((batch) => batch.createdOrder), scenario.batches.map((batch) => batch.createdOrder))
  assert.deepEqual(result.scenario.batches.flatMap((batch) => batch.samples.map((sample) => sample.id)), scenario.batches.flatMap((batch) => batch.samples.map((sample) => sample.id)))
  assert.equal(result.scenario.id, scenario.id)
  assert.equal(result.scenario.nextOrder, scenario.nextOrder)
  assert.equal(result.scenario.batches[1]?.arrivalOffsetMs, 120)
  const body = toRunBody(result.scenario, defaultSettings())
  assert.equal(body.requests[0]?.max_output_tokens, 12)
  assert.equal(body.requests[1]?.synthetic_input_tokens, 256)
  assert.equal(body.requests[2]?.arrival_offset_ms, 120)
  assert.equal(scenario.batches[1]?.arrivalOffsetMs, 100)
  assert.equal(scenario.batches[0]?.samples[0]?.maxOutputTokens, 8)
  assert.equal(scenario.batches[0]?.samples[1]?.syntheticInputTokens, 128)
})

test("synthetic lengths are validated only for the active request mode", () => {
  const scenario = staggeredExample()
  const readValue = (id: string) => id === "sample-synthetic-sample-1" ? "" : undefined
  const prompt = resolveWorkloadFields(scenario, "ms", readValue)
  assert.ok("scenario" in prompt)
  const synthetic = reduceScenario(scenario, { type: "set-sample-mode", sampleId: "sample-1", inputMode: "synthetic" })
  const result = resolveWorkloadFields(synthetic, "ms", readValue)
  assert.ok("issue" in result)
  assert.equal(result.issue.fieldId, "sample-synthetic-sample-1")
})

test("unchanged fields retain their draft values without losing valid zero offsets", () => {
  const scenario = staggeredExample()
  const result = resolveWorkloadFields(scenario, "s", () => undefined)
  assert.ok("scenario" in result)
  assert.deepEqual(result.scenario, scenario)
  assert.notEqual(result.scenario, scenario)
  assert.notEqual(result.scenario.batches[0]?.samples[0], scenario.batches[0]?.samples[0])
})
