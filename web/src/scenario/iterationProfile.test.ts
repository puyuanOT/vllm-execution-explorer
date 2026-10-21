import assert from "node:assert/strict"
import { readRecordingFixture } from "../../test/recordingFixture.ts"
import { test } from "node:test"
import { parseProfileAttachment } from "../../../replay/src/profiling.ts"
import { createPlayer, parseJsonl, type RecordingManifest, type TraceEvent } from "../../../replay/src/reduce.ts"
import { activeIterationIntervals, iterationProfile, nativeWindowAt, participantPickerValue, participantSelection } from "../execution/iterationProfile.ts"

const source = new URL("../../public/recordings/gpu-kv-layer-diagnostics/", import.meta.url)
const manifest = JSON.parse(readRecordingFixture(new URL("manifest.json", source))) as RecordingManifest
const observations = parseProfileAttachment(readRecordingFixture(new URL("profiling/activities.json", source)), manifest)
const player = createPlayer(parseJsonl(readRecordingFixture(new URL("events.jsonl", source))) as TraceEvent[], manifest)

test("real iteration inspection links 468 native writes by recorded IDs and preserves mixed request attribution", () => {
  let count = 0
  for (const step of player.at(player.maxSequence).steps) {
    const scope = iterationProfile(observations, step.id)
    assert.equal(scope.writes.length, 36)
    assert.equal(scope.intervals.length, 36)
    for (const { write, activity } of scope.intervals) {
      assert.equal(activity.kind, "cuda_kernel")
      assert.equal(activity.scope.operationId, write.operationId)
      assert.ok(scope.extent && scope.extent.startUs <= activity.startUs && scope.extent.endUs >= activity.endUs)
      assert.equal(write.participants.length, step.totalScheduledTokens)
      for (const request of step.requests) assert.equal(write.participants.filter((row) => row.requestId === request.requestId).length, request.rangeLength)
      count++
    }
  }
  assert.equal(count, 468)
  const mixed = player.at(player.maxSequence).steps.find(step => step.requests.some(request => request.promptTokens > 0) && step.requests.some(request => request.generationTokens > 0))!
  assert.deepEqual(mixed.requests.map(request => request.rangeLength), [1, 63])
  assert.deepEqual(iterationProfile(observations, "not-recorded").intervals, [])
  assert.deepEqual(iterationProfile(null, "step-1").intervals, [])
})

test("native cursor uses half-open measured intervals and zoom resolves microsecond kernels within their native extent", () => {
  const scope = iterationProfile(observations, "step-1")
  const first = scope.intervals[0]
  assert.deepEqual(activeIterationIntervals(scope.intervals, first.activity.startUs).map(interval => interval.write.layerIndex), [0])
  assert.ok(!activeIterationIntervals(scope.intervals, first.activity.endUs).some(interval => interval.write.layerIndex === 0))
  assert.ok(scope.extent!.endUs - scope.extent!.startUs > first.activity.durationUs * 1000)
  const zoom = nativeWindowAt(scope.extent!, (first.activity.startUs + first.activity.endUs) / 2, first.activity.durationUs * 4)
  assert.ok(zoom.endUs - zoom.startUs < 12)
  assert.ok(zoom.startUs <= first.activity.startUs && zoom.endUs >= first.activity.endUs)
  const clipped = nativeWindowAt(scope.extent!, scope.extent!.startUs - 100, 5)
  assert.equal(clipped.startUs, scope.extent!.startUs)
  assert.equal(clipped.cursor, scope.extent!.startUs)
  const unlinked = structuredClone(observations)
  unlinked.activities.find(activity => activity.id === first.activity.id)!.scope.operationId = "different-operation"
  assert.equal(iterationProfile(unlinked, "step-1").intervals.length, 35)
})

test("participant selection validates current physical generation/slot and resets the picker after external block selection", () => {
  const write = iterationProfile(observations, "step-4").writes[18]
  const participant = write.participants[0]
  assert.equal(participant.tokenPosition, 173)
  const reserved = player.at(88)
  const linked = participantSelection(reserved, write, participant)
  assert.equal(linked.physical, true)
  assert.deepEqual(linked.selection, { kind: "token", requestId: participant.requestId, position: 173, kvLayerIndex: 18, kvBlockKey: "pool-0:11" })
  assert.equal(participantPickerValue(write, linked.selection), "0")
  assert.equal(participantPickerValue(write, { kind: "block", key: "pool-0:3" }), "")
  assert.equal(participantPickerValue(write, linked.selection), "0", "the same native option can be selected again after the controlled blank value")
  const replaced = structuredClone(reserved)
  replaced.blocks["pool-0:11"].generation++
  const historical = participantSelection(replaced, write, participant)
  assert.equal(historical.physical, false)
  assert.deepEqual(historical.selection, { kind: "token", requestId: participant.requestId, position: 173 })
  assert.equal(participantSelection(reserved, write, { ...participant, slot: 0 }).physical, false)
  assert.equal(participantSelection(player.at(20), iterationProfile(observations, "step-1").writes[0], iterationProfile(observations, "step-1").writes[0].participants[0]).physical, false)
})

test("iteration lookup reuses observed records but returns independent membership arrays", () => {
  const first = iterationProfile(observations, "step-1")
  const nativeActivity = first.intervals[0].activity
  const nativeWrite = first.writes[0]
  first.writes.pop()
  first.intervals.pop()
  const repeated = iterationProfile(observations, "step-1")
  assert.equal(repeated.writes.length, 36)
  assert.equal(repeated.intervals.length, 36)
  assert.equal(repeated.writes[0], nativeWrite)
  assert.equal(repeated.intervals[0].activity, nativeActivity)
  const changed = structuredClone(observations)
  const originalStart = nativeActivity.startUs
  changed.activities.find(activity => activity.id === nativeActivity.id)!.startUs += 1000
  changed.activities.find(activity => activity.id === nativeActivity.id)!.endUs += 1000
  const next = iterationProfile(changed, "step-1")
  assert.equal(next.intervals.find(interval => interval.activity.id === nativeActivity.id)!.activity.startUs, originalStart + 1000)
  assert.equal(iterationProfile(observations, "step-1").intervals[0].activity.startUs, originalStart)
})
