import assert from "node:assert/strict"
import { test } from "node:test"
import { profileObservationIndex } from "./profileIndex.ts"
import type { ProfileActivity, ProfileKvObservations } from "./profiling.ts"

function activity(id: string, stepId: string | null, startUs: number, kind: ProfileActivity["kind"] = "cuda_kernel"): ProfileActivity {
  return { id, kind, name: id, startUs, endUs: startUs + 5, durationUs: 5, device: kind === "cpu_scope" ? null : 0,
    stream: kind === "cpu_scope" ? null : 1, processId: 1, threadId: 1, relatedActivityIds: [],
    scope: { stepId, layerIndex: stepId == null ? null : 0, component: null, correlation: stepId == null ? "unmapped" : "runtime_flow",
      correlationSource: null, operation: null, operationId: null } }
}

function observations(startUs: number): ProfileKvObservations {
  return { activities: [activity("same-id", "step-1", startUs), activity("cpu", "step-1", startUs, "cpu_scope"), activity("unmapped", null, startUs + 10)],
    steps: [], cacheWrites: [], tensorSnapshots: [], tensorCapture: null }
}

test("immutable recording lookup preserves source order and identities without rebuilding for playhead changes", () => {
  const source = observations(100)
  const first = profileObservationIndex(source)!
  assert.equal(profileObservationIndex(source), first)
  assert.equal(first.activitiesById.get("same-id"), source.activities[0])
  assert.deepEqual(first.gpuActivitiesByStep.get("step-1"), [source.activities[0]])
  assert.deepEqual(first.gpuActivitiesByStep.get(null), [source.activities[2]])
  assert.equal(first.activitiesById.get("cpu"), source.activities[1])
  assert.equal(profileObservationIndex(null), null)
})

test("new recording and replaced collections invalidate lookup even when recorded IDs collide", () => {
  const source = observations(100)
  const original = profileObservationIndex(source)!
  const replacement = observations(1000)
  const next = profileObservationIndex(replacement)!
  assert.notEqual(next, original)
  assert.equal(next.activitiesById.get("same-id")!.startUs, 1000)
  assert.equal(profileObservationIndex(source)!.activitiesById.get("same-id")!.startUs, 100)
  source.activities = replacement.activities
  const replaced = profileObservationIndex(source)!
  assert.notEqual(replaced, original)
  assert.equal(replaced.activitiesById.get("same-id")!.startUs, 1000)
  assert.equal(replaced.gpuActivitiesByStep.get("step-1")![0], replacement.activities[0])
})
