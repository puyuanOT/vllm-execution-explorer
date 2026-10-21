import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { gunzipSync } from "node:zlib"
import { recordingAssetMap } from "../src/recordingAssetMap.ts"

const publicRoot = new URL("../public/", import.meta.url)

/** Read the same captured fixture from either canonical files or the hosted lossless transport. */
export function readRecordingFixtureBytes(url: URL | string): Buffer {
  if (existsSync(url)) return readFileSync(url)
  const filename = typeof url === "string" ? path.resolve(url) : fileURLToPath(url)
  const relative = path.relative(fileURLToPath(publicRoot), filename)
  const asset = recordingAssetMap[`/${relative.split(path.sep).join("/")}`]
  if (!asset) return readFileSync(url)
  const compressed = Buffer.concat(asset.parts.map((part) => readFileSync(new URL(part.slice(1), publicRoot))))
  return gunzipSync(compressed)
}

export function readRecordingFixture(url: URL | string): string {
  return readRecordingFixtureBytes(url).toString("utf8")
}
