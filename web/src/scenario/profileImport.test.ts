import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { test } from "node:test"
import { loadFiles } from "../api.ts"

const clock = { domain: "torch.profiler.chrome_trace", unit: "microseconds", origin: "trace_local" }
const profile = { contract_version: 1, run_id: "test-only", mode: "eager_diagnostic", execution_mode: "eager", backend: "torch.profiler", clock, activities: [], steps: [] }
const text = JSON.stringify(profile)
const artifact = { path: "profiles/normalized.json", sha256: createHash("sha256").update(text).digest("hex"), size_bytes: Buffer.byteLength(text) }
const manifest = { run_id: "test-only", execution_kind: "fixture", profiling: { ...profile, normalized_artifact: artifact } }
function files(attachment?: string, metadata: unknown = manifest): FileList {
  return [new File([JSON.stringify(metadata)], "manifest.json"), new File([""], "events.jsonl"), ...(attachment == null ? [] : [new File([attachment], "normalized.json")])] as unknown as FileList
}

test("profile imports verify bytes and recording identity while retaining host replay on failure", async () => {
  const valid = await loadFiles(files(text))
  assert.equal(valid.profile?.runId, "test-only")
  assert.equal(valid.profileError, "")
  const missing = await loadFiles(files())
  assert.equal(missing.profile, null)
  assert.match(missing.profileError ?? "", /Select the declared/)
  assert.equal(missing.manifest.run_id, "test-only")
  const wrongDigest = await loadFiles(files(text.replace("test-only", "wrong-run")))
  assert.equal(wrongDigest.profile, null)
  assert.match(wrongDigest.profileError ?? "", /digest/)
  const wrongSize = await loadFiles(files(text + " "))
  assert.match(wrongSize.profileError ?? "", /size/)
})

test("profile import rejects unsafe paths and oversized attachments before reading them", async () => {
  for (const path of ["../normalized.json", "https://example.com/profile.json", "/profile.json"]) {
    const result = await loadFiles(files(text, { ...manifest, profiling: { ...manifest.profiling, normalized_artifact: { ...artifact, path } } }))
    assert.match(result.profileError ?? "", /safe relative/)
  }
  const large = await loadFiles(files(undefined, { ...manifest, profiling: { ...manifest.profiling, normalized_artifact: { ...artifact, size_bytes: 64 * 1024 * 1024 + 1 } } }))
  assert.match(large.profileError ?? "", /64 MiB/)
})
