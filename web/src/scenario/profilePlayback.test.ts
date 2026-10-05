import assert from "node:assert/strict"
import { test } from "node:test"
import type { ProfileActivity } from "../../../replay/src/profiling.ts"
import { profileBoundaryTimes } from "../profilePlayback.ts"

test("batch navigation skips embeddings and runtime calls and deduplicates delayed sampling scopes", () => {
  function activity(name: string, stepId: string | null, startUs: number): ProfileActivity {
    return { id: name + startUs, name, kind: "cpu_scope", startUs, durationUs: 5, endUs: startUs + 5, device: null, stream: null, processId: 1, threadId: 1, relatedActivityIds: [], scope: { stepId, layerIndex: null, component: null, correlation: stepId ? "explicit_scope" : "unmapped", correlationSource: null } }
  }
  const activities = [activity("explorer.step/step-2", "step-2", 100), activity("embedding", "step-1", 12), activity("explorer.step/step-1", "step-1", 10), activity("runtime launch", "step-1", 15), activity("explorer.step/step-1", "step-1", 50), activity("unmapped", null, 1)]
  assert.deepEqual(profileBoundaryTimes(activities, "batch"), [10, 100])
  assert.deepEqual(profileBoundaryTimes(activities, "activity"), [1, 10, 12, 15, 50, 100])
})
