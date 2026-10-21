import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { projectKvCache, projectRequestTokens } from "../../../replay/src/learning.ts"
import { createPlayer, type RecordingManifest, type TraceEvent } from "../../../replay/src/reduce.ts"
import { projectKvConfiguration } from "../../../replay/src/kv.ts"
import { learningBookmarks } from "../execution/learningBookmarks.ts"
import { namedKvDimensions } from "../execution/kvDimensions.ts"
import { tokenKvLabel, tokenSlotOwnership } from "../execution/tokenSemantics.ts"

const recording = new URL("../../public/recordings/gpu-kv-layer-diagnostics/", import.meta.url)
const events: TraceEvent[] = readFileSync(new URL("events.jsonl", recording), "utf8").trim().split("\n").map((line) => JSON.parse(line))
const manifest: RecordingManifest = JSON.parse(readFileSync(new URL("manifest.json", recording), "utf8"))

test("learning entry points resolve actual prefill, reserved decode, release, reuse and mixed states", () => {
  const bookmarks = learningBookmarks(events)
  assert.deepEqual(bookmarks.map((bookmark) => bookmark.id), ["prefill", "decode-input", "release", "prefix-reuse", "mixed-batch"])
  const player = createPlayer(events, manifest)
  for (const bookmark of bookmarks) {
    assert.ok(events.some((event) => event.sequence === bookmark.sequence))
    const state = player.at(bookmark.sequence)
    if (bookmark.id === "decode-input") {
      assert.equal(bookmark.selection.kind, "token")
      if (bookmark.selection.kind !== "token") continue
      const token = projectRequestTokens(state, bookmark.selection.requestId).find((item) => item.position === bookmark.selection.position)
      assert.equal(token?.kvState, "reserved")
      assert.equal(token?.computedEver, false)
    } else if (bookmark.id === "release") {
      assert.equal(bookmark.selection.kind, "block")
      if (bookmark.selection.kind !== "block") continue
      const block = state.blocks[bookmark.selection.key]
      assert.equal(block.refCount, 0)
      assert.equal(block.hasHash, true)
      assert.equal(block.bindings.length, 0)
      assert.ok(block.contents.length > 0)
    } else if (bookmark.id === "prefix-reuse") {
      assert.equal(bookmark.selection.kind, "token")
      if (bookmark.selection.kind !== "token") continue
      const token = projectRequestTokens(state, bookmark.selection.requestId).find((item) => item.position === bookmark.selection.position)
      assert.equal(token?.kvState, "computed")
      assert.ok(token?.locations.some((location) => location.valid && location.sourceRequestId !== bookmark.selection.requestId))
      const block = state.blocks[bookmark.selection.kvBlockKey!]
      assert.ok(block.bindings.some((binding) => binding.requestId === bookmark.selection.requestId))
    } else {
      assert.equal(bookmark.selection.kind, "step")
      if (bookmark.selection.kind !== "step") continue
      const step = state.steps.find((item) => item.id === bookmark.selection.id)
      assert.ok(step?.requests.some((request) => request.promptTokens > 0))
      if (bookmark.id === "mixed-batch") assert.ok(step?.requests.some((request) => request.generationTokens > 0))
    }
  }
})

test("entry points follow event boundaries when unrelated observations are inserted", () => {
  const insertedCount = 7
  const inserted: TraceEvent[] = Array.from({ length: insertedCount }, (_, index) => ({ ...events[0], event_type: "capture_ready", sequence: index + 1, monotonic_ns: index }))
  const shifted = events.map((event) => ({ ...event, sequence: event.sequence + insertedCount,
    ...(typeof event.causal_event_sequence === "number" ? { causal_event_sequence: event.causal_event_sequence + insertedCount } : {}),
  }))
  const original = learningBookmarks(events)
  assert.deepEqual(learningBookmarks([...inserted, ...shifted]), original.map((bookmark) => ({ ...bookmark, sequence: bookmark.sequence + insertedCount })))
})

test("trace gaps exclude apparent prefix reuse and mixed iterations from the untrusted tail", () => {
  const firstMatch = events.findIndex((event) => event.event_type === "prefix_lookup" && Number(event.matched_token_count) > 0)
  const interrupted = [...events.slice(0, firstMatch), { ...events[firstMatch], event_type: "trace_gap", reason: "test missing observations" }, ...events.slice(firstMatch + 1)]
  const bookmarks = learningBookmarks(interrupted)
  assert.ok(bookmarks.length > 0)
  assert.ok(bookmarks.every((bookmark) => bookmark.sequence < events[firstMatch].sequence))
  assert.ok(bookmarks.every((bookmark) => bookmark.id !== "prefix-reuse" && bookmark.id !== "mixed-batch"))
})

test("a prefix match without observed valid physical binding remains a request-level observation", () => {
  const matchIndex = events.findIndex((event) => event.event_type === "prefix_lookup" && Number(event.matched_token_count) > 0)
  const bookmark = learningBookmarks(events.slice(0, matchIndex + 1)).find((item) => item.id === "prefix-reuse")
  assert.equal(bookmark?.sequence, events[matchIndex].sequence)
  assert.deepEqual(bookmark?.selection, { kind: "request", id: events[matchIndex].request_id })
  assert.equal(bookmark?.title, "A prefix match is observed")
  assert.equal(learningBookmarks([]).length, 0)
  const noMatch = learningBookmarks(events.slice(0, matchIndex))
  assert.ok(noMatch.every((item) => item.id !== "prefix-reuse"))
})

test("a release without a retained prefix hash never claims a reusable-prefix bookmark", () => {
  const event: TraceEvent = { schema_version: "1.0.0", run_id: "release", producer_id: "engine", sequence: 1, monotonic_ns: 1,
    event_type: "block_change", operation: "release", pool: "p", block_id: 1, ref_count_after: 0, has_hash_after: false, cached_at_zero_refs: false }
  assert.ok(learningBookmarks([event]).every((bookmark) => bookmark.id !== "release"))
})

test("named KV axes require a qualified shape consistent with observed K and V dimensions", () => {
  const state = createPlayer(events, manifest).at(events.at(-1)!.sequence)
  const layer = projectKvConfiguration(state, manifest).layers[0]
  const description = namedKvDimensions(layer)
  assert.equal(description, `${layer.numKernelBlocks} kernel blocks × ${layer.numKvHeads} KV heads × ${layer.kernelBlockSize} token slots × (${layer.headDim} key + ${layer.headDim} value features)`)
  assert.equal(namedKvDimensions({ ...layer, layoutStatus: "unavailable" }), null)
  assert.equal(namedKvDimensions({ ...layer, layout: "unsupported" }), null)
  assert.equal(namedKvDimensions({ ...layer, shape: [...layer.shape.slice(0, 3), layer.headDim!] }), null)
  assert.equal(namedKvDimensions({ ...layer, headDim: null }), null)
})

test("a shared prefix has a new holder and preserves its original writer; release removes only the holder", () => {
  const bookmark = learningBookmarks(events).find((item) => item.id === "prefix-reuse")!
  assert.equal(bookmark.selection.kind, "token")
  if (bookmark.selection.kind !== "token") return
  const player = createPlayer(events, manifest)
  const reused = player.at(bookmark.sequence)
  const token = projectRequestTokens(reused, bookmark.selection.requestId).find((item) => item.position === bookmark.selection.position)!
  const ownership = tokenSlotOwnership(token, projectKvCache(reused, manifest))
  assert.equal(ownership.known, true)
  assert.deepEqual(ownership.holders, [bookmark.selection.requestId])
  assert.ok(ownership.writers.length > 0)
  assert.ok(ownership.writers.every((writer) => writer !== bookmark.selection.requestId))
  const releasedEvent = events.find((event) => event.sequence > bookmark.sequence && event.event_type === "block_change"
    && `${event.pool}:${event.block_id}` === bookmark.selection.kvBlockKey && event.operation === "release")!
  const released = player.at(releasedEvent.sequence)
  const retained = projectRequestTokens(released, bookmark.selection.requestId).find((item) => item.position === bookmark.selection.position)!
  // The holder's mapping is released. The original writer can still address retained contents.
  const originalToken = projectRequestTokens(released, ownership.writers[0]).find((item) => item.position === bookmark.selection.position)!
  assert.equal(retained.computedEver, true)
  assert.equal(retained.kvState, "not_computed")
  assert.equal(tokenKvLabel(retained), "No retained KV mapping")
  const retainedOwnership = tokenSlotOwnership(originalToken, projectKvCache(released, manifest))
  assert.equal(retainedOwnership.known, true)
  assert.deepEqual(retainedOwnership.holders, [])
  assert.deepEqual(retainedOwnership.writers, ownership.writers)
  assert.equal(originalToken.kvState, "computed")
  assert.equal(tokenSlotOwnership({ ...token, locations: token.locations.map((location) => ({ ...location, generation: location.generation + 1 })) }, projectKvCache(reused, manifest)).known, false)
})

test("unknown token history never claims the token was never computed", () => {
  assert.equal(tokenKvLabel({ kvState: "not_computed", computedEver: null }), "No observed KV mapping")
  assert.equal(tokenKvLabel({ kvState: "not_computed", computedEver: false }), "Not yet computed")
  assert.equal(tokenKvLabel({ kvState: "unknown", computedEver: null }), "Unknown")
})
