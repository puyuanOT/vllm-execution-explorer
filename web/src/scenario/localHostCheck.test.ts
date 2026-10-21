import assert from "node:assert/strict"
import test from "node:test"

import { localHostCheckPresentation } from "./localHostCheck.ts"

const cudaMessage = "This Linux machine runs the explorer and replay. A live engine needs a CUDA GPU on this machine, or an SSH GPU host."

test("a demo fixture host check does not report the missing CUDA runtime as complete", () => {
  const result = localHostCheckPresentation("fixture", "Runner is reachable.", { ok: false, message: cudaMessage })
  assert.equal(result.readiness.label, "Ready")
  assert.equal(result.readiness.tone, "ready")
  assert.match(result.readiness.message, /synthetic evidence/)
  assert.match(result.readiness.message, /does not require CUDA/)
  assert.doesNotMatch(result.readiness.message, /needs a CUDA GPU/)
  const runtime = result.stages.find((stage) => stage.id === "runtime")
  assert.equal(runtime?.status, "complete")
  assert.match(runtime?.detail ?? "", /does not certify CUDA/)
  assert.doesNotMatch(runtime?.detail ?? "", /needs a CUDA GPU/)
})

test("a live local engine keeps a failed CUDA check visible", () => {
  const result = localHostCheckPresentation("vllm", "Runner is reachable.", { ok: false, message: cudaMessage })
  assert.equal(result.readiness.label, "Needs GPU host")
  assert.equal(result.readiness.tone, "warning")
  assert.equal(result.readiness.message, cudaMessage)
  assert.equal(result.stages.find((stage) => stage.id === "runtime")?.status, "error")
  assert.equal(result.stages.find((stage) => stage.id === "runtime")?.detail, cudaMessage)
})

test("a ready CUDA runtime is reported as ready", () => {
  const message = "This Linux machine can host the vLLM 0.30.0 engine."
  const result = localHostCheckPresentation("vllm", "Runner is reachable.", { ok: true, message })
  assert.equal(result.readiness.label, "Ready")
  assert.equal(result.readiness.tone, "ready")
  assert.equal(result.readiness.message, message)
  assert.equal(result.stages.find((stage) => stage.id === "runtime")?.status, "complete")
})

test("a missing preflight reports that the controller was not reached", () => {
  for (const engine of ["fixture", "vllm"] as const) {
    const result = localHostCheckPresentation(engine, "The runner is unreachable.", null)
    assert.equal(result.readiness.label, "Unavailable")
    assert.equal(result.readiness.tone, "error")
    assert.equal(result.stages.find((stage) => stage.id === "connect")?.status, "error")
    assert.equal(result.stages.find((stage) => stage.id === "connect")?.detail, "The runner is unreachable.")
    assert.equal(result.stages.find((stage) => stage.id === "runtime")?.status, "error")
    assert.equal(result.stages.find((stage) => stage.id === "ready")?.status, "error")
  }
})

test("each host check returns its own stage objects", () => {
  const first = localHostCheckPresentation("fixture", "Runner is reachable.", { ok: false, message: cudaMessage })
  const connect = first.stages.find((stage) => stage.id === "connect")
  assert.ok(connect)
  connect.status = "error"
  connect.detail = "changed"
  const second = localHostCheckPresentation("vllm", "Runner is reachable.", { ok: true, message: "ready" })
  assert.equal(second.stages.find((stage) => stage.id === "connect")?.status, "complete")
  assert.equal(second.stages.find((stage) => stage.id === "connect")?.detail, "Local controller")
})
