import assert from "node:assert/strict"
import test from "node:test"

import type { RunSnapshot } from "../api.ts"
import { expectedRunFailure, shouldLoadExpectedRecording } from "./runCompletion.ts"

function snapshot(update: Partial<RunSnapshot>): RunSnapshot {
  return {
    run_id: "run-fast",
    status: "capturing",
    intake_open: false,
    text: {},
    message: "",
    ...update,
  }
}

test("only a matching completed run triggers the recording fallback", () => {
  assert.equal(shouldLoadExpectedRecording("run-fast", snapshot({ status: "completed", finalized: true }), false), true)
  assert.equal(shouldLoadExpectedRecording("run-fast", snapshot({ status: "completed", finalized: false }), false), false)
  assert.equal(shouldLoadExpectedRecording("run-fast", snapshot({ status: "failed", finalized: true }), false), false)
  assert.equal(shouldLoadExpectedRecording("run-fast", snapshot({ status: "capturing", finalized: true }), false), false)
})

test("a matching failed run preserves the controller error for Configure", () => {
  assert.equal(
    expectedRunFailure("run-fast", snapshot({ status: "failed", finalized: true, message: "FP8 kernel failed" }), false),
    "FP8 kernel failed",
  )
  assert.equal(
    expectedRunFailure("run-fast", snapshot({ status: "cancelled", finalized: true, message: "" }), false),
    "Capture was cancelled.",
  )
  assert.equal(
    expectedRunFailure("run-fast", snapshot({ status: "completed", finalized: false, message: "Manifest finalization failed" }), false),
    "Manifest finalization failed",
  )
  assert.equal(expectedRunFailure("run-other", snapshot({ status: "failed", message: "hidden" }), false), "")
})

test("the fallback ignores active, mismatched, and manually superseded runs", () => {
  assert.equal(shouldLoadExpectedRecording("run-fast", snapshot({ status: "capturing", finalized: false }), false), false)
  assert.equal(shouldLoadExpectedRecording("run-other", snapshot({ status: "completed", finalized: true }), false), false)
  assert.equal(shouldLoadExpectedRecording("run-fast", snapshot({ status: "completed", finalized: true }), true), false)
  assert.equal(shouldLoadExpectedRecording("", snapshot({ status: "completed", finalized: true }), false), false)
})
