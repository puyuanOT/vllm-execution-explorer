import assert from "node:assert/strict"
import { test } from "node:test"
import { readFileSync } from "node:fs"
import { buildTimeline } from "../../../replay/src/timeline.ts"
import type { TraceEvent } from "../../../replay/src/reduce.ts"
import { firstIterationSequence, iterationEndpoints, nextIteration, previousIteration, recordedTimeRange } from "../iterationPlayback.ts"

test("real iteration endpoints retain observer order and include result-associated ownership/output", () => {
  const events = readFileSync(new URL("../../public/recordings/gpu-kv-layer-diagnostics/events.jsonl", import.meta.url), "utf8").trim().split("\n").map(line => JSON.parse(line) as TraceEvent)
  const manifest = JSON.parse(readFileSync(new URL("../../public/recordings/gpu-kv-layer-diagnostics/manifest.json", import.meta.url), "utf8"))
  const index = buildTimeline(events, manifest)
  const endpoints = iterationEndpoints(events, index.steps)
  assert.equal(endpoints.length, 13)
  assert.equal(endpoints[0], 30)
  assert.equal(firstIterationSequence(events), endpoints[0])
  assert.equal(endpoints[5], 140)
  assert.equal(nextIteration(endpoints, 30, index.maxSequence), endpoints[1])
  assert.equal(previousIteration(endpoints, 30), 0)
  assert.equal(nextIteration(endpoints, endpoints.at(-1)!, index.maxSequence), 399)
  const range = recordedTimeRange(index.timePoints)!
  assert.ok(range.startMs < -127)
  assert.ok(range.endMs > 6000)
  assert.ok(index.timePoints.every(point => point.elapsedMs == null || point.elapsedMs >= range.startMs && point.elapsedMs <= range.endMs))
  const before = index.timePoints.find(point => point.sequence === 170)!.elapsedMs!
  const after = index.timePoints.find(point => point.sequence === 171)!.elapsedMs!
  assert.equal(before, 6029.621817)
  assert.equal(after, 6029.702897)
  assert.ok(Math.abs((after - before) * 1000 - 81.08) < 0.000001)
})

test("large recorded axes retain fractional preparation bounds without an argument-limit failure", () => {
  const points = Array.from({length:200000}, (_, index) => ({ sequence:index + 1, elapsedMs:index / 1000 })) as ReturnType<typeof buildTimeline>["timePoints"]
  points[0].elapsedMs = -0.000123
  assert.deepEqual(recordedTimeRange(points), {startMs:-0.000123,endMs:199.999})
})

test("an incomplete step stops at its last observed boundary and untimed data has no invented time", () => {
  const events = [{sequence:1,event_type:"step_scheduled",step_id:"s"},{sequence:2,event_type:"step_submitted",step_id:"s"}] as TraceEvent[]
  const index = buildTimeline(events)
  assert.deepEqual(iterationEndpoints(events, index.steps), [2])
  assert.equal(recordedTimeRange(index.timePoints), null)
})

test("demo entry uses the first iteration rather than preparation or a later iteration", () => {
  const events = [
    { sequence: 1, event_type: "gpu_memory_sample" },
    { sequence: 5, event_type: "step_scheduled", step_id: "step-1" },
    { sequence: 8, event_type: "step_scheduled", step_id: "step-2" },
    { sequence: 9, event_type: "step_completed", step_id: "step-1" },
    { sequence: 12, event_type: "step_completed", step_id: "step-2" },
  ] as TraceEvent[]
  assert.equal(firstIterationSequence(events), 9)
  assert.equal(firstIterationSequence(events.slice(0, 1)), 0)
  assert.equal(firstIterationSequence([{ sequence: 2, event_type: "step_scheduled" }, { sequence: 3, event_type: "other" }] as TraceEvent[]), 2)
})
