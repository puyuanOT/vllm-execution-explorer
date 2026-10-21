import { recordingAssetMap } from "./recordingAssetMap.ts"
import { recordingAssetBaseUrl } from "./recordingAssetSource.ts"

export type RecordingAssetMap = Record<string, { encoding: "gzip"; parts: string[] }>

interface RecordingAssetOptions {
  assets?: RecordingAssetMap
  baseUrl?: string
  preferLocal?: boolean
  fetcher?: typeof fetch
  signal?: AbortSignal
}

/** Hosted assets use lossless transport compression; callers still receive the original recording bytes as UTF-8 text. */
export async function fetchRecordingText(url: string, options: RecordingAssetOptions = {}): Promise<string> {
  const assets = options.assets ?? recordingAssetMap
  const fetcher = options.fetcher ?? fetch
  const signal = options.signal
  signal?.throwIfAborted()
  const asset = assets[url]
  if (!asset) {
    const response = await fetcher(url, { signal })
    if (!response.ok) throw new Error(`Could not read ${url} (HTTP ${response.status}).`)
    return response.text()
  }
  if (!asset.parts.length) throw new Error(`Compressed recording ${url} has no asset parts.`)
  if (options.preferLocal) {
    const local = await fetcher(url, { signal })
    // Vite's SPA fallback serves index.html with status 200 for missing recording files.
    if (local.ok && !local.headers.get("content-type")?.includes("text/html")) return local.text()
    await local.body?.cancel()
    if (!local.ok && local.status !== 404) throw new Error(`Could not read ${url} (HTTP ${local.status}).`)
  }
  const baseUrl = (options.baseUrl ?? recordingAssetBaseUrl).replace(/\/+$/, "")

  let partIndex = 0
  let partReader: ReadableStreamDefaultReader<Uint8Array<ArrayBuffer>> | undefined
  const compressed = new ReadableStream<BufferSource>({
    async pull(controller) {
      try {
        while (true) {
          signal?.throwIfAborted()
          if (!partReader) {
            if (partIndex === asset.parts.length) {
              controller.close()
              return
            }
            const partUrl = `${baseUrl}${asset.parts[partIndex]}`
            const response = await fetcher(partUrl, { signal })
            if (!response.ok) throw new Error(`Could not read recording part ${partUrl} (HTTP ${response.status}).`)
            if (!response.body) throw new Error(`Recording part ${partUrl} has no response body.`)
            partReader = response.body.getReader()
          }
          const next = await partReader.read()
          if (next.done) {
            partReader.releaseLock()
            partReader = undefined
            partIndex += 1
          } else {
            controller.enqueue(next.value)
            return
          }
        }
      } catch (error) {
        controller.error(error)
      }
    },
    async cancel(reason) {
      await partReader?.cancel(reason)
    },
  })

  // Stream both decompression and UTF-8 decoding to avoid retaining a second complete decoded byte buffer.
  const reader = compressed.pipeThrough(new DecompressionStream("gzip")).pipeThrough(new TextDecoderStream()).getReader()
  let text = ""
  try {
    while (true) {
      const next = await reader.read()
      if (next.done) return text
      text += next.value
    }
  } catch (error) {
    signal?.throwIfAborted()
    const detail = error instanceof Error ? error.message : "Invalid gzip data."
    throw new Error(`Could not decode compressed recording ${url}: ${detail}`, { cause: error })
  } finally {
    reader.releaseLock()
  }
}
