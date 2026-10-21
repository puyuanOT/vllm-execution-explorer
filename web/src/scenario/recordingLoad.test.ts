import assert from "node:assert/strict"
import { test } from "node:test"
import { createRecordingLoader, type RecordingLoadCallbacks } from "./recordingLoad.ts"

test("replacement releases old render callbacks before reading and prevents overlapping large loads", async () => {
  const events: string[] = []
  const callbacks = (prefix: string): RecordingLoadCallbacks<string> => ({
    prepare() { events.push(`${prefix}:prepare`) }, reading() { events.push(`${prefix}:reading`) },
    accept(value) { events.push(`${prefix}:accept:${value}`) }, reject() { events.push(`${prefix}:reject`) }, finish() { events.push(`${prefix}:finish`) },
  })
  const ref = { current: callbacks("old") }
  let paint = 0
  const loader = createRecordingLoader(ref, async () => {
    events.push(`paint:${++paint}`)
    ref.current = callbacks(`empty-${paint}`)
  })
  const opening = loader.open(async () => { events.push("read"); return "new" }, "recording")
  assert.equal(loader.busy, true)
  assert.equal(await loader.open(async () => { throw new Error("second read must not start") }, "other"), false)
  assert.equal(await opening, true)
  assert.deepEqual(events, ["old:prepare", "paint:1", "empty-1:reading", "paint:2", "read", "empty-2:accept:new", "empty-2:finish"])
  assert.equal(loader.busy, false)
})

test("load failures release busy state for retry and cancellation cannot publish an old result", async () => {
  const accepted: string[] = []
  const rejected: unknown[] = []
  let finished = 0
  const ref = { current: { prepare() {}, reading() {}, accept(value: string) { accepted.push(value) },
    reject(error: unknown) { rejected.push(error) }, finish() { finished += 1 } } }
  const loader = createRecordingLoader(ref, async () => {})
  const failure = new Error("missing recording")
  assert.equal(await loader.open(async () => { throw failure }, "missing"), false)
  assert.deepEqual(rejected, [failure])
  assert.equal(loader.busy, false)
  let resolve: (value: string) => void = () => {}
  const pending = loader.open(() => new Promise<string>(ready => { resolve = ready }), "pending")
  await Promise.resolve()
  await Promise.resolve()
  loader.cancel()
  resolve("stale")
  assert.equal(await pending, false)
  assert.deepEqual(accepted, [])
  assert.equal(finished, 1)
  assert.equal(await loader.open(async () => "retry", "retry"), true)
  assert.deepEqual(accepted, ["retry"])
})
