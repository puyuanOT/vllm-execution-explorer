import assert from "node:assert/strict"
import test from "node:test"

import { buildScope, scopeChannel, sequencePosition } from "../execution/scope.ts"
import { telemetryThrough, type TelemetryPoint } from "../execution/telemetry.ts"

test("scope channels follow the serving subsystem that recorded each event", () => {
  assert.equal(scopeChannel("engine_admitted"), "intake")
  assert.equal(scopeChannel("step_scheduled"), "scheduler")
  assert.equal(scopeChannel("kv_block_binding"), "kv")
  assert.equal(scopeChannel("step_result_received"), "model")
  assert.equal(scopeChannel("completion"), "output")
  assert.equal(scopeChannel("trace_gap"), null)
})

test("the scope never draws ticks or step markers past the playhead", () => {
  const ticks = [1, 2, 3, 4, 5].map((sequence) => ({ position: sequencePosition(sequence, 5), channel: "kv" as const }))
  const markers = [{ position: sequencePosition(2, 5), label: "S1" }, { position: sequencePosition(5, 5), label: "S2" }]
  const scope = buildScope({ ticks, markers, playhead: sequencePosition(3, 5), columns: 101, currentPosition: sequencePosition(3, 5) })
  assert.deepEqual(scope.bins.kv, [0, 25, 50])
  assert.deepEqual(scope.markers.map((marker) => marker.label), ["S1"])
  assert.equal(scope.playheadColumn, 50)
  assert.deepEqual(scope.current, { channel: "kv", column: 50 })
  assert.equal(scope.gapColumn, null)
})

test("a trace gap keeps its own column and collapsed positions stay in range", () => {
  const scope = buildScope({ ticks: [{ position: 1, channel: "output" }, { position: Number.NaN, channel: "intake" }], markers: [], playhead: 1, gap: 0.5, columns: 11 })
  assert.deepEqual(scope.bins.intake, [0])
  assert.deepEqual(scope.bins.output, [10])
  assert.equal(scope.gapColumn, 5)
  assert.equal(sequencePosition(1, 1), 1)
  assert.equal(sequencePosition(0, 1), 0)
})

test("telemetry sparklines end at the playhead and drop later samples", () => {
  const point = (sequence: number, accepted: number): TelemetryPoint => ({ sequence, accepted, requests: 0, steps: 0, referenced: 0, capacity: null, results: 0, submissions: 0 })
  const series = [point(0, 0), point(10, 2), point(20, 5), point(30, 9)]
  const visible = telemetryThrough(series, point(14, 3))
  assert.deepEqual(visible.map((item) => [item.sequence, item.accepted]), [[0, 0], [10, 2], [14, 3]])
})
