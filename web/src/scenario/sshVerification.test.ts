import assert from "node:assert/strict"
import test from "node:test"

import { defaultSettings } from "./example.ts"
import {
  createSshVerification,
  remoteRuntimeError,
  sshVerificationError,
} from "./sshVerification.ts"

const identity = { name: "test-key.pem" } as File

function sshSettings() {
  return {
    ...defaultSettings(),
    hostKind: "ssh" as const,
    sshUser: " ubuntu ",
    sshHost: " 203.0.113.10 ",
    sshPort: "22",
    runnerUrl: "http://127.0.0.1:8766",
  }
}

test("SSH verification is bound to the checked host, setup mode, key, and endpoint", () => {
  const settings = sshSettings()
  const verification = createSshVerification(settings, identity, settings.runnerUrl)
  assert.equal(sshVerificationError(settings, identity, verification), "")
  assert.match(sshVerificationError({ ...settings, sshHost: "other-host" }, identity, verification), /changed/)
  assert.match(sshVerificationError({ ...settings, automaticSetup: false }, identity, verification), /changed/)
  assert.match(sshVerificationError(settings, { name: "other.pem" } as File, verification), /changed/)
  assert.match(sshVerificationError({ ...settings, runnerUrl: "" }, identity, verification), /changed/)
})

test("SSH capture requires both a key and a completed host check", () => {
  const settings = sshSettings()
  assert.match(sshVerificationError(settings, null, null), /private key/)
  assert.match(sshVerificationError(settings, identity, null), /Check this SSH host/)
  assert.equal(sshVerificationError({ ...settings, hostKind: "local" }, null, null), "")
})

test("remote readiness requires a visible GPU and vLLM 0.30.x", () => {
  assert.match(remoteRuntimeError(null, "0.30.0"), /NVIDIA GPU/)
  assert.match(remoteRuntimeError("NVIDIA A10", null), /no vLLM install/)
  assert.match(remoteRuntimeError("NVIDIA A10", "0.29.0"), /found 0.29.0/)
  assert.equal(remoteRuntimeError("NVIDIA A10", "0.30.1"), "")
})
