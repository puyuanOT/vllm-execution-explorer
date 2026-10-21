import assert from "node:assert/strict"
import { test } from "node:test"
import { loadRunnerRecording } from "../api.ts"

test("retiring an owned capture aborts its download before content or native profile can be read", async (context) => {
  const previous = globalThis.fetch
  context.after(() => { globalThis.fetch = previous })
  const controller = new AbortController()
  const requested: string[] = []
  let reachedEvents: () => void = () => {}
  const eventsStarted = new Promise<void>(resolve => { reachedEvents = resolve })
  globalThis.fetch = async (input, options) => {
    const url = String(input)
    requested.push(url)
    assert.equal(options?.signal, controller.signal)
    if (url.endsWith("/manifest")) return new Response("{}")
    reachedEvents()
    return new Promise<Response>((_, reject) => controller.signal.addEventListener("abort", () => reject(controller.signal.reason), { once: true }))
  }
  const loading = loadRunnerRecording("http://controller", "owned-run", controller.signal)
  await eventsStarted
  controller.abort()
  await assert.rejects(loading, { name: "AbortError" })
  assert.deepEqual(requested.map(url => url.split("/").at(-1)), ["manifest", "events"])
})

test("a cancelled profile read is not converted into a loaded recording with a profile error", async (context) => {
  const previous = globalThis.fetch
  context.after(() => { globalThis.fetch = previous })
  const controller = new AbortController()
  const requested: string[] = []
  globalThis.fetch = async input => {
    const url = String(input)
    requested.push(url)
    if (url.endsWith("/manifest")) return new Response(JSON.stringify({ profiling: { normalized_artifact: {
      path: "profiling/activities.json", sha256: "0".repeat(64), size_bytes: 8 } } }))
    if (url.endsWith("/profile")) { controller.abort(); return new Response("not-json") }
    return new Response("")
  }
  await assert.rejects(loadRunnerRecording("http://controller", "owned-run", controller.signal), { name: "AbortError" })
  assert.deepEqual(requested.map(url => url.split("/").at(-1)), ["manifest", "events", "content", "profile"])
})
