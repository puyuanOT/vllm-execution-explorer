import assert from "node:assert/strict"
import test from "node:test"

import { createPlayer, type TraceEvent } from "../../../replay/src/reduce.ts"
import { buildTimeline } from "../../../replay/src/timeline.ts"
import { arrivalAxis } from "../configure/arrivalAxis.ts"
import { buildScope, scopeChannel, sequencePosition } from "../execution/scope.ts"
import { sequencerScrollLeft } from "../execution/sequencerScroll.ts"
import { recordingEvidence } from "../execution/evidence.ts"
import { nextSequence, replayableEvents } from "../playback.ts"
import { formatTime, parseTime } from "./time.ts"

test("short arrival plans have valid whole-millisecond ticks in either display unit", () => {
  for (const latest of [0, 1, 8, 10, 100, 120, 120_000]) {
    const { span, ticks } = arrivalAxis(latest)
    assert.ok(span > latest)
    for (const tick of ticks) {
      assert.ok(Number.isSafeInteger(tick))
      for (const unit of ["ms", "s"] as const) {
        assert.equal(parseTime(formatTime(tick, unit), unit), tick)
      }
    }
  }
})

test("requested GPU execution needs observed provenance and hardware before claiming GPU evidence", () => {
  for (const manifest of [
    { execution_kind: "gpu", completion_status: "preparing" },
    { execution_kind: "gpu", completion_status: "failed", provenance: { real_gpu: false }, hardware: { gpu: null } },
    { execution_kind: "gpu", provenance: { real_gpu: false }, hardware: { gpu: "NVIDIA A100" } },
    { execution_kind: "gpu", hardware: { gpu: "NVIDIA A100" } },
    { execution_kind: "gpu", provenance: { real_gpu: true } },
  ]) {
    assert.equal(recordingEvidence(manifest, false).kind, "unverified")
    assert.doesNotMatch(recordingEvidence(manifest, false).text, /Observed GPU/)
  }
  const observed = { execution_kind: "gpu", provenance: { real_gpu: true }, hardware: { gpu: "NVIDIA A100" } }
  assert.equal(recordingEvidence(observed, false).kind, "gpu")
  // A failed/cancelled capture may still retain earlier actual GPU observations.
  assert.equal(recordingEvidence({ ...observed, completion_status: "failed" }, false).kind, "gpu")
  assert.equal(recordingEvidence({ ...observed, demo_fixture: true }, false).kind, "synthetic")
  assert.equal(recordingEvidence(observed, true).kind, "draft")
})

test("a trace gap excludes the raw tail from scope, scheduler markers and playback", () => {
  const event = (sequence: number, event_type: string, extra = {}): TraceEvent => ({ schema_version: "1.0.0", run_id: "gap", producer_id: "engine", sequence, monotonic_ns: sequence * 1_000_000, event_type, ...extra })
  const events = [
    event(1, "request_received", { request_id: "r1" }),
    event(2, "trace_gap", { reason: "lost events" }),
    event(3, "completion", { request_id: "r1" }),
    event(4, "step_scheduled", { step_id: "untrusted", scheduled: [], total_scheduled_tokens: 0, token_budget: 64 }),
  ]
  const player = createPlayer(events)
  assert.equal(player.maxSequence, 2)
  assert.equal(player.at(4).trust, "unknown")
  const playable = replayableEvents(events, player.maxSequence)
  const timeline = buildTimeline(playable)
  const scope = buildScope({
    ticks: playable.flatMap((event) => { const channel = scopeChannel(event.event_type); return channel ? [{ position: sequencePosition(event.sequence, player.maxSequence), channel }] : [] }),
    markers: timeline.steps.map((step, index) => ({ position: sequencePosition(step.sequence, player.maxSequence), label: `S${index + 1}` })),
    playhead: 1, gap: 1, columns: 100, currentPosition: 1,
  })
  assert.deepEqual(scope.bins.scheduler, [])
  assert.deepEqual(scope.bins.output, [])
  assert.deepEqual(scope.markers, [])
  assert.equal(scope.current, null)
  assert.equal(scope.gapColumn, 99)
  assert.equal(nextSequence(playable, 2), 2)
  assert.equal(timeline.steps.length, 0)
})

test("forward and backward sequencer scrolling keeps the whole current column beside sticky labels", () => {
  for (const viewport of [294, 334, 600, 1000]) {
    for (const columnWidth of [34, 64]) {
      let scroll = 0
      for (const index of [0, 1, 2, 7, 20, 7, 1, 0]) {
        scroll = sequencerScrollLeft(index, columnWidth, 220, viewport, scroll)
        const left = 220 + index * columnWidth - scroll
        const right = left + columnWidth
        assert.ok(left >= 220, `left=${left}, viewport=${viewport}, index=${index}`)
        assert.ok(right <= viewport, `right=${right}, viewport=${viewport}, index=${index}`)
      }
    }
  }
})
