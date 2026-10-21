import assert from "node:assert/strict"
import test from "node:test"

import type { RunSnapshot } from "../api.ts"
import { expectedRunFailure, ownedLiveRun } from "./runCompletion.ts"

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

test("only the UI-owned run can download live events or finalize a native profile", () => {
  const active = snapshot({})
  const completed = snapshot({ status: "completed", finalized: true })
  assert.equal(ownedLiveRun("run-fast", active, false), active)
  assert.equal(ownedLiveRun("run-fast", completed, false), completed)
  assert.equal(ownedLiveRun("", completed, false), null)
  assert.equal(ownedLiveRun("other-run", completed, false), null)
  assert.equal(ownedLiveRun("run-fast", completed, true), null)
  assert.equal(ownedLiveRun("run-fast", null, false), null)
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
