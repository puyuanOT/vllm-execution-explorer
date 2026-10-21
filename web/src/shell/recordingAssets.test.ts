import assert from "node:assert/strict"
import test from "node:test"
import { gzipSync } from "node:zlib"
import { fetchRecordingText, type RecordingAssetMap } from "../recordingAssets.ts"

test("unmapped recordings retain the normal text fetch path", async () => {
  const calls: string[] = []
  const text = await fetchRecordingText("/recordings/demo/manifest.json", {
    assets: {},
    fetcher: async (input) => {
      calls.push(String(input))
      return new Response('{"name":"原始 recording"}\n')
    },
  })
  assert.equal(text, '{"name":"原始 recording"}\n')
  assert.deepEqual(calls, ["/recordings/demo/manifest.json"])
})

test("ordered binary parts reconstruct exact text across gzip and UTF-8 boundaries", async () => {
  const original = '{"tokens":["Hello", "世界", "🦊", "\\n"]}\n'.repeat(4000)
  const gzip = gzipSync(original)
  const parts = [gzip.subarray(0, 1), gzip.subarray(1, 7), gzip.subarray(7, 19), gzip.subarray(19)]
  const urls = parts.map((_, index) => `/assets/recording.part${index}.bin`)
  const calls: string[] = []
  const assets: RecordingAssetMap = { "/recordings/demo/events.jsonl": { encoding: "gzip", parts: urls } }
  const text = await fetchRecordingText("/recordings/demo/events.jsonl", {
    assets,
    fetcher: async (input) => {
      const url = String(input)
      calls.push(url)
      const part = parts[urls.indexOf(url)]
      return new Response(new ReadableStream({
        start(controller) {
          for (let offset = 0; offset < part.length; offset += 3) controller.enqueue(part.subarray(offset, offset + 3))
          controller.close()
        },
      }))
    },
  })
  assert.equal(text, original)
  assert.deepEqual(calls, urls)
})

test("missing parts fail with the affected part URL and HTTP status", async () => {
  const gzip = gzipSync("The complete recording must be present.")
  const assets: RecordingAssetMap = { "/recordings/demo/events.jsonl": { encoding: "gzip", parts: ["/first.bin", "/missing.bin"] } }
  await assert.rejects(fetchRecordingText("/recordings/demo/events.jsonl", {
    assets,
    fetcher: async (input) => String(input) === "/first.bin" ? new Response(gzip.subarray(0, 10)) : new Response("Missing", { status: 404 }),
  }), /\/missing\.bin \(HTTP 404\)/)
})

test("invalid or truncated gzip never returns a partial recording", async () => {
  const gzip = gzipSync("Some data can decode before its truncated trailer is detected.")
  for (const bytes of [new Uint8Array([1, 2, 3, 4]), gzip.subarray(0, gzip.length - 4)]) {
    await assert.rejects(fetchRecordingText("/recordings/demo/events.jsonl", {
      assets: { "/recordings/demo/events.jsonl": { encoding: "gzip", parts: ["/bad.bin"] } },
      fetcher: async () => new Response(bytes),
    }), /Could not decode compressed recording \/recordings\/demo\/events.jsonl:/)
  }
})

test("normal HTTP failures and invalid empty transport entries remain errors", async () => {
  await assert.rejects(fetchRecordingText("/recordings/missing.json", {
    assets: {},
    fetcher: async () => new Response("Failure", { status: 503 }),
  }), /\/recordings\/missing\.json \(HTTP 503\)/)
  await assert.rejects(fetchRecordingText("/recordings/empty.json", {
    assets: { "/recordings/empty.json": { encoding: "gzip", parts: [] } },
    fetcher: async () => { throw new Error("Empty transport must not fetch anything") },
  }), /has no asset parts/)
})

test("an aborted recording load does not start fetching", async () => {
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(fetchRecordingText("/recordings/demo/events.jsonl", {
    assets: {},
    signal: controller.signal,
    fetcher: async () => { throw new Error("Aborted loads must not fetch anything") },
  }), { name: "AbortError" })
})
