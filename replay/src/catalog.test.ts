import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { test } from "node:test"
import { fileURLToPath } from "node:url"

import { explainRequest } from "./explain.ts"
import { projectArchitecture, projectKvCache, projectRequestTokens } from "./learning.ts"
import {
  attachContent,
  createPlayer,
  parseJsonl,
  type ContentRecord,
  type RecordingManifest,
  type TraceEvent,
} from "./reduce.ts"

const catalog = join(dirname(fileURLToPath(import.meta.url)), "../../web/public/recordings")

test("bundled fixture lessons replay without inventing a GPU trace", () => {
  const index = JSON.parse(readFileSync(join(catalog, "index.json"), "utf8")) as {
    recordings: Array<{ id: string; evidence: string }>
  }
  const ids = new Set(index.recordings.map((item) => item.id))
  assert.equal(index.recordings[0].id, "model-kv-walkthrough")
  for (const required of [
    "model-kv-walkthrough",
    "continuous-batching",
    "single-request",
    "chunked-prefill",
    "prefix-reuse",
    "prefix-miss",
    "preemption",
    "cancellation",
    "incomplete-capture",
  ]) {
    assert.ok(ids.has(required), required)
  }
  for (const item of index.recordings) {
    assert.equal(item.evidence, "fixture")
    const manifest = JSON.parse(readFileSync(join(catalog, item.id, "manifest.json"), "utf8")) as RecordingManifest
    const events = parseJsonl<TraceEvent>(readFileSync(join(catalog, item.id, "events.jsonl"), "utf8"))
    const content = parseJsonl<ContentRecord>(readFileSync(join(catalog, item.id, "content.jsonl"), "utf8"))
    assert.equal(manifest.execution_kind, "fixture")
    const player = createPlayer(events, manifest)
    const state = attachContent(player.at(player.maxSequence), content)
    if (item.id === "incomplete-capture") {
      assert.equal(state.trust, "unknown")
      assert.equal(state.incomplete, true)
    } else {
      assert.equal(state.trust, "known")
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
