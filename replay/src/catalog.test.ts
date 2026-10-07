import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { test } from "node:test"
import { fileURLToPath } from "node:url"

import { explainRequest } from "./explain.ts"
import { projectArchitecture, projectKvCache, projectRequestTokens } from "./learning.ts"
import { projectKvConfiguration, projectKvTensorPreview } from "./kv.ts"
import { parseProfileAttachment, readProfileArtifact } from "./profiling.ts"
import {
  attachContent,
  createPlayer,
  parseJsonl,
  type ContentRecord,
  type RecordingManifest,
  type TraceEvent,
} from "./reduce.ts"

const catalog = join(dirname(fileURLToPath(import.meta.url)), "../../web/public/recordings")

test("demo catalog contains real GPU lessons; retained regression fixtures preserve their provenance", () => {
  const index = JSON.parse(readFileSync(join(catalog, "index.json"), "utf8")) as {
    recordings: Array<{ id: string; evidence: string }>
  }
  const ids = new Set(index.recordings.map((item) => item.id))
  assert.equal(index.recordings[0].id, "gpu-kv-layer-diagnostics")
  assert.ok(index.recordings.every((item) => item.evidence === "gpu"))
  for (const required of ["gpu-kv-layer-diagnostics", "gpu-model-batching", "gpu-prefix-reuse"]) {
    assert.ok(ids.has(required), required)
  }
  const fixtures = [
    "model-kv-walkthrough",
    "continuous-batching",
    "single-request",
    "chunked-prefill",
    "prefix-reuse",
    "prefix-miss",
    "preemption",
    "cancellation",
    "incomplete-capture",
  ]
  for (const item of [...index.recordings, ...fixtures.map((id) => ({ id, evidence: "fixture" }))]) {
    const manifest = JSON.parse(readFileSync(join(catalog, item.id, "manifest.json"), "utf8")) as RecordingManifest
    const events = parseJsonl<TraceEvent>(readFileSync(join(catalog, item.id, "events.jsonl"), "utf8"))
    const content = parseJsonl<ContentRecord>(readFileSync(join(catalog, item.id, "content.jsonl"), "utf8"))
    assert.equal(manifest.execution_kind, item.evidence)
    const player = createPlayer(events, manifest)
    const state = attachContent(player.at(player.maxSequence), content)
    if (item.id === "incomplete-capture") {
      assert.equal(state.trust, "unknown")
      assert.equal(state.incomplete, true)
    } else {
      assert.equal(state.trust, "known")
    }
    if (item.evidence === "gpu") {
      assert.equal((manifest.provenance as { real_gpu?: boolean }).real_gpu, true)
      assert.equal((manifest.engine as { version?: string }).version, "0.30.0")
      assert.equal(projectKvCache(state, manifest, content).evidenceKind, "gpu")
      const configuration = projectKvConfiguration(state, manifest)
      assert.equal(configuration.pools[0].layerIndices.length, 36)
      const artifact = readProfileArtifact(manifest)
      if (artifact) {
        const attachment = parseProfileAttachment(readFileSync(join(catalog, item.id, artifact.path), "utf8"), manifest)
        if (item.id === "gpu-kv-layer-diagnostics") {
          assert.equal(configuration.status, "observed")
          assert.ok(configuration.layers.every((layer) => layer.layoutStatus === "qualified" && layer.device === "cuda:0" && layer.dtype === "bfloat16"))
          assert.equal(configuration.uniqueStorageBytes, configuration.storages.reduce((total, storage) => total + storage.byteCount, 0))
          assert.ok(attachment.cacheWrites.length >= 36)
          assert.ok(attachment.cacheWrites.every((write) => write.status === "observed" && write.gpuActivityIds.length > 0))
          const tensors = projectKvTensorPreview(manifest, attachment)
          assert.equal(tensors.status, "observed")
          assert.equal(tensors.snapshots.length, 24)
          assert.deepEqual([...new Set(tensors.snapshots.map((slice) => slice.layerIndex))].sort((a, b) => a - b), [0, 18, 35])
          assert.ok(tensors.snapshots.every((slice) => slice.requestId === "prefix-prime" && slice.kvHead === 0 && slice.values.length === 16 && slice.values.every(Number.isFinite)))
          assert.ok(player.steps.some((step) => step.requests.some((request) => request.promptTokens > 0) && step.requests.some((request) => request.generationTokens > 0)))
          assert.ok((state.requests["prefix-reader"].matchedPrefixTokens ?? 0) > 0)
          for (const requestId of Object.keys(state.requests)) {
            const finalSample = projectRequestTokens(state, requestId, content).at(-1)
            assert.equal(finalSample?.kvState, "not_computed", requestId)
          }
        } else {
          // These archives predate per-layer cache scopes and numerical snapshots.
          assert.equal(configuration.status, "unavailable")
          assert.equal(attachment.cacheWrites.length, 0)
          assert.equal(projectKvTensorPreview(manifest, attachment).status, "unavailable")
        }
      }
      if (item.id === "gpu-model-batching") assert.ok(player.steps.some((step) => step.requests.some((request) => request.promptTokens > 0) && step.requests.some((request) => request.generationTokens > 0)))
      if (item.id === "gpu-prefix-reuse") assert.ok(Object.values(state.requests).some((request) => (request.matchedPrefixTokens ?? 0) > 0))
    }
    if (item.id === "model-kv-walkthrough") {
      const architecture = projectArchitecture(manifest)
      assert.equal(architecture.supported, true)
      assert.equal(architecture.modelType, "qwen3")
      assert.equal(architecture.source, "pinned_model_config_reference")
      assert.equal(architecture.layerCount, 36)
      assert.equal(architecture.queryHeads, 32)
      assert.equal(architecture.kvHeads, 8)
      const cache = projectKvCache(state, manifest, content)
      assert.equal(cache.status, "observed")
      assert.equal(cache.pools.length, 1)
      assert.equal(cache.pools[0].capacity, 8)
      assert.deepEqual(cache.pools[0].blocks.map((block) => block.blockId), [0, 1, 2, 3, 4, 5, 6, 7])
      assert.equal(cache.groups.length, 1)
      assert.equal(cache.groups[0].tokenSlotMapping, "authored_fixture")
      assert.equal(Object.keys(state.requests).length, 6)
      for (const requestId of Object.keys(state.requests)) {
        const tokens = projectRequestTokens(state, requestId, content)
        const finalSample = tokens.at(-1)
        assert.ok(finalSample, requestId)
        assert.equal(finalSample.kind, "output", requestId)
        assert.equal(finalSample.kvState, "not_computed", requestId)
        assert.equal(finalSample.computedEver, false, requestId)
      }
    }
    if (item.id === "continuous-batching") {
      assert.ok(player.steps.some((step) => step.requests.length >= 2))
    }
    if (item.id === "prefix-reuse") {
      assert.equal(state.requests.second.matchedPrefixTokens, 4)
      const explanation = explainRequest(state, "second")
      assert.match(explanation.observed.join("\n"), /matched 4 tokens/)
      assert.match(explanation.explanatory.join("\n"), /does not skip ordinary decoding/)
    }
    if (item.id === "preemption") {
      const preempt = events.find((entry) => entry.event_type === "preemption")
      assert.ok(preempt)
      const mid = player.at(preempt.sequence)
      assert.equal(mid.requests.crowded.recomputing, true)
      assert.equal(mid.requests.crowded.preemptionReason, null)
      assert.ok(Object.values(state.blocks).some((block) => block.blockId === 3 && block.generation === 2))
    }
    if (item.id === "single-request") {
      assert.match(state.requests["request-a"].text, /●/)
    }
    if (item.id === "chunked-prefill") {
      assert.match(state.requests.long.text, /Done/)
      assert.ok(player.steps.some((step) => step.requests.length === 2))
    }
  }
})
