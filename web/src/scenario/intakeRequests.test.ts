import assert from "node:assert/strict"
import { readRecordingFixture } from "../../test/recordingFixture.ts"
import test from "node:test"
import { createPlayer, parseJsonl } from "../../../replay/src/reduce.ts"
import { observedRequestIds } from "../explore/model.ts"
import { modelIterationSummary } from "../architecture/model.ts"

test("Intake identities follow recorded arrivals and backward seeks independently of iteration membership", () => {
  const source = new URL("../../public/recordings/gpu-h100-bf16/", import.meta.url)
  const manifest = JSON.parse(readRecordingFixture(new URL("manifest.json", source)))
  const player = createPlayer(parseJsonl(readRecordingFixture(new URL("events.jsonl", source))), manifest)
  const order = Array.from({ length: 6 }, (_, i) => `request-${i + 1}`)
  for (const [sequence, expected] of [
    [173, []], [174, [1]], [189, [1]], [190, [1]], [363, [1]],
    [558, [1, 3]], [559, [1, 2, 3]], [590, [1, 2, 3, 4, 5, 6]],
    [174, [1]], [173, []],
  ] as const) {
    assert.deepEqual(observedRequestIds(player.at(sequence), order), expected.map((i) => `request-${i}`), `EVT ${sequence}`)
  }
  assert.deepEqual(modelIterationSummary(player.at(174)).participants, [])
  assert.deepEqual(modelIterationSummary(player.at(558)).participants.map((r) => r.requestId), ["request-1"])
  // Missing roster metadata must not hide a request that Intake has already observed.
  assert.deepEqual(observedRequestIds(player.at(558), []), ["request-1", "request-3"])
  // Native GPU membership remains independent of the host Intake playhead.
  const native = modelIterationSummary(player.at(174), { stepId: "step-9", activeLayers: [], packedRows: [
    { row: 0, requestId: "request-3", tokenPosition: 0, isPadding: false },
  ] })
  assert.deepEqual(native.participants.map((r) => r.requestId), ["request-3"])
  assert.deepEqual(observedRequestIds(player.at(174), order), ["request-1"])
})
