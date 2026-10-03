import assert from "node:assert/strict"
import { test } from "node:test"

import { targetConnectionStages } from "./targetConnection.ts"

test("SSH reachability stays active until the host has responded", () => {
  const stages = targetConnectionStages({ status: "connecting", phase: "checking_ssh", message: "Connecting over SSH" })
  assert.deepEqual(stages?.map((stage) => stage.status), ["active", "pending", "pending"])
  assert.equal(stages?.[0]?.detail, "Connecting over SSH")
})

test("runtime copying and preparation show SSH complete without claiming the controller ready", () => {
  for (const phase of ["copying_project", "preparing_runtime"]) {
    const stages = targetConnectionStages({ status: "connecting", phase, message: "Preparing runtime" })
    assert.deepEqual(stages?.map((stage) => stage.status), ["complete", "active", "pending"])
    assert.equal(stages?.[1]?.detail, "Preparing runtime")
  }
})

test("tunnel phases wait for final controller verification", () => {
  for (const phase of ["opening_tunnel", "checking_controller"]) {
    const stages = targetConnectionStages({ status: "connecting", phase, message: "Verifying controller" })
    assert.deepEqual(stages?.map((stage) => stage.status), ["complete", "complete", "active"])
  }
})

test("prior ready or failed responses and unknown phases cannot replace current progress", () => {
  for (const status of ["idle", "ready", "failed"]) {
    assert.equal(targetConnectionStages({ status, phase: "checking_ssh", message: "Previous attempt" }), null)
  }
  assert.equal(targetConnectionStages({ status: "connecting", phase: "unknown", message: "Unrecognized phase" }), null)
})
