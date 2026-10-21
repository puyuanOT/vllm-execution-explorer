#!/usr/bin/env node
/**
 * Prepare lossless recording assets for a static host with a per-file size cap.
 *
 * Initial preparation:
 *   node scripts/prepare-recordings.mjs --source /path/to/public/recordings
 * Verify the checked-in assets (also the default; no source folder required):
 *   node scripts/prepare-recordings.mjs --verify
 *
 * The native profiling/raw archives are intentionally excluded. The runtime's
 * normalized profiles, recording manifests, events, and content retain their
 * exact original bytes. Gzip is an explicit application transport; .bin avoids
 * HTTP servers transparently decompressing an already encoded response.
 */
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { copyFile, mkdir, mkdtemp, open, readFile, readdir, realpath, rename, rm, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { Readable, Transform, Writable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { createGunzip, createGzip } from 'node:zlib'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const target = path.join(root, 'web/public/recordings')
const mapFile = path.join(root, 'web/src/recordingAssetMap.ts')
const reportName = 'transport-report.json'
const plainLimit = 1024 * 1024
const partLimit = 16 * 1024 * 1024

function hashTap() {
  const hash = createHash('sha256')
  let size = 0
  return {
    stream: new Transform({ transform(chunk, _encoding, callback) {
      size += chunk.length
      hash.update(chunk)
      callback(null, chunk)
    } }),
    result: () => ({ size, sha256: hash.digest('hex') }),
  }
}

async function digest(file) {
  const tap = hashTap()
  await pipeline(createReadStream(file), tap.stream, new Writable({ write(_chunk, _encoding, callback) { callback() } }))
  return tap.result()
}

async function* walk(folder, prefix = '') {
  const entries = await readdir(folder, { withFileTypes: true })
  entries.sort((a, b) => a.name.localeCompare(b.name, 'en'))
  for (const entry of entries) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name
    const absolute = path.join(folder, entry.name)
    if (entry.isDirectory()) yield* walk(absolute, relative)
    else if (entry.isFile()) yield { absolute, relative, size: (await stat(absolute)).size }
    else throw new Error(`Refusing non-file recording entry: ${relative}`)
  }
}

function publicPath(relative) {
  return `/recordings/${relative}`
}

function localAsset(folder, url) {
  if (!url.startsWith('/recordings/')) throw new Error(`Invalid recording URL: ${url}`)
  const relative = url.slice('/recordings/'.length)
  const absolute = path.resolve(folder, relative)
  if (!absolute.startsWith(`${path.resolve(folder)}${path.sep}`)) throw new Error(`Recording URL escapes asset folder: ${url}`)
  return absolute
}

// Normalize the non-semantic gzip header, including its OS byte, so preparation
// is deterministic across platforms. Node's gzip payload is otherwise stable.
function stableGzipHeader() {
  let prefix = Buffer.alloc(0)
  let emitted = false
  return new Transform({
    transform(chunk, _encoding, callback) {
      if (emitted) return callback(null, chunk)
      prefix = Buffer.concat([prefix, chunk])
      if (prefix.length < 10) return callback()
      prefix.fill(0, 4, 8)
      prefix[9] = 255
      emitted = true
      callback(null, prefix)
    },
    flush(callback) {
      callback(emitted ? undefined : new Error('Incomplete gzip header'))
    },
  })
}

function chunkWriter(folder, relative) {
  const bundle = relative.split('/')[0]
  const stem = path.basename(relative).replace(/[^a-zA-Z0-9_-]/g, '-')
  const key = createHash('sha256').update(relative).digest('hex').slice(0, 16)
  const parts = []
  let handle = null
  let current = null
  let currentHash = null
  async function closePart() {
    if (!handle) return
    await handle.close()
    handle = null
    current.sha256 = currentHash.digest('hex')
    current = null
    currentHash = null
  }
  const stream = new Writable({
    write(chunk, _encoding, callback) {
      ;(async () => {
        let cursor = 0
        while (cursor < chunk.length) {
          if (!handle) {
            const suffix = String(parts.length + 1).padStart(3, '0')
            const name = `${bundle}/transport/${stem}-${key}.part-${suffix}.bin`
            await mkdir(path.dirname(path.join(folder, name)), { recursive: true })
            handle = await open(path.join(folder, name), 'wx')
            current = { url: publicPath(name), size: 0, sha256: '' }
            currentHash = createHash('sha256')
            parts.push(current)
          }
          const length = Math.min(chunk.length - cursor, partLimit - current.size)
          const piece = chunk.subarray(cursor, cursor + length)
          let written = 0
          while (written < piece.length) {
            const result = await handle.write(piece, written, piece.length - written)
            if (!result.bytesWritten) throw new Error(`Unable to write recording part: ${current.url}`)
            written += result.bytesWritten
          }
          currentHash.update(piece)
          current.size += length
          cursor += length
          if (current.size === partLimit) await closePart()
        }
      })().then(() => callback(), callback)
    },
    final(callback) { closePart().then(() => callback(), callback) },
    destroy(error, callback) { closePart().then(() => callback(error), callback) },
  })
  return { stream, parts }
}

function mapSource(assets) {
  const map = Object.fromEntries(assets.filter(asset => asset.encoding === 'gzip').map(asset => [asset.url, {
    encoding: 'gzip', parts: asset.parts.map(part => part.url),
  }]))
  return `// Generated by scripts/prepare-recordings.mjs. Original recording bytes are unchanged.\nexport const recordingAssetMap: Record<string, { encoding: 'gzip'; parts: string[] }> = ${JSON.stringify(map, null, 2)}\n`
}

async function verifyAssets(folder, report) {
  let checked = 0
  for (const asset of report.assets) {
    for (const part of asset.parts) {
      if (part.size > partLimit) throw new Error(`Oversized deployment asset: ${part.url}`)
      const result = await digest(localAsset(folder, part.url))
      if (result.size !== part.size || result.sha256 !== part.sha256) throw new Error(`Transport checksum mismatch: ${part.url}`)
    }
    const source = Readable.from((async function* () {
      for (const part of asset.parts) yield* createReadStream(localAsset(folder, part.url))
    })())
    const tap = hashTap()
    const sink = new Writable({ write(_chunk, _encoding, callback) { callback() } })
    if (asset.encoding === 'gzip') await pipeline(source, createGunzip(), tap.stream, sink)
    else await pipeline(source, tap.stream, sink)
    const reconstructed = tap.result()
    if (reconstructed.size !== asset.originalSize || reconstructed.sha256 !== asset.originalSha256) {
      throw new Error(`Reconstructed recording differs from original: ${asset.url}`)
    }
    checked++
  }
  return checked
}

function summary(report, checked) {
  return {
    verifiedRuntimeFiles: checked,
    sourceBytes: report.summary.sourceBytes,
    runtimeOriginalBytes: report.summary.runtimeOriginalBytes,
    compressedTransportBytes: report.summary.compressedTransportBytes,
    deploymentAssetBytes: report.summary.deploymentAssetBytes,
    gzipFiles: report.summary.gzipFiles,
    gzipChunks: report.summary.gzipChunks,
    maxAssetBytes: report.summary.maxAssetBytes,
    omittedRawFiles: report.summary.omittedRawFiles,
    omittedRawBytes: report.summary.omittedRawBytes,
  }
}

async function prepare(sourceArgument) {
  const source = await realpath(path.resolve(sourceArgument))
  const output = path.resolve(target)
  if (source === output || source.startsWith(`${output}${path.sep}`) || output.startsWith(`${source}${path.sep}`)) {
    throw new Error('Source and destination recording folders must be separate, non-nested directories')
  }
  await mkdir(path.dirname(target), { recursive: true })
  const staging = await mkdtemp(path.join(path.dirname(target), '.recordings-prepare-'))
  const assets = []
  let omittedRawFiles = 0
  let omittedRawBytes = 0
  let sourceBytes = 0
  try {
    for await (const file of walk(source)) {
      sourceBytes += file.size
      const segments = file.relative.split('/')
      if (segments.includes('raw')) {
        omittedRawFiles++
        omittedRawBytes += file.size
        continue
      }
      if (segments.includes('transport') || file.relative === reportName) throw new Error('Source already contains transport output; use the canonical recording folder')
      const original = hashTap()
      if (file.size > plainLimit && file.relative !== 'index.json') {
        const outputParts = chunkWriter(staging, file.relative)
        await pipeline(createReadStream(file.absolute), original.stream, createGzip({ level: 6 }), stableGzipHeader(), outputParts.stream)
        const originalResult = original.result()
        assets.push({ url: publicPath(file.relative), encoding: 'gzip', originalSize: originalResult.size, originalSha256: originalResult.sha256, parts: outputParts.parts })
      } else {
        if (file.size > partLimit) throw new Error('Catalog index exceeds the static host file size limit')
        const destination = path.join(staging, file.relative)
        await mkdir(path.dirname(destination), { recursive: true })
        await copyFile(file.absolute, destination)
        const copied = await digest(destination)
        const sourceDigest = await digest(file.absolute)
        if (copied.sha256 !== sourceDigest.sha256 || copied.size !== sourceDigest.size) throw new Error(`Copied recording differs from source: ${file.relative}`)
        assets.push({ url: publicPath(file.relative), encoding: 'identity', originalSize: copied.size, originalSha256: copied.sha256, parts: [{ url: publicPath(file.relative), size: copied.size, sha256: copied.sha256 }] })
      }
    }
    if (!assets.some(asset => asset.url === '/recordings/index.json')) throw new Error('Source does not contain index.json')
    const allParts = assets.flatMap(asset => asset.parts)
    const gzipAssets = assets.filter(asset => asset.encoding === 'gzip')
    const report = {
      version: 1,
      preparation: { gzipLevel: 6, gzipHeaderMtime: 0, gzipHeaderOS: 255, plainLimitBytes: plainLimit, maxPartBytes: partLimit, rawArchivesIncluded: false },
      summary: {
        sourceBytes,
        runtimeOriginalBytes: assets.reduce((sum, asset) => sum + asset.originalSize, 0),
        compressedTransportBytes: gzipAssets.flatMap(asset => asset.parts).reduce((sum, part) => sum + part.size, 0),
        deploymentAssetBytes: allParts.reduce((sum, part) => sum + part.size, 0),
        gzipFiles: gzipAssets.length,
        gzipChunks: gzipAssets.reduce((sum, asset) => sum + asset.parts.length, 0),
        maxAssetBytes: Math.max(0, ...allParts.map(part => part.size)),
        omittedRawFiles,
        omittedRawBytes,
      },
      assets,
    }
    const checked = await verifyAssets(staging, report)
    await writeFile(path.join(staging, reportName), `${JSON.stringify(report, null, 2)}\n`)
    const mapTemporary = `${mapFile}.tmp`
    await writeFile(mapTemporary, mapSource(assets))
    await rm(target, { recursive: true, force: true })
    await rename(staging, target)
    await rename(mapTemporary, mapFile)
    console.log(JSON.stringify(summary(report, checked), null, 2))
  } catch (error) {
    await rm(staging, { recursive: true, force: true })
    throw error
  }
}

const arguments_ = process.argv.slice(2)
if (arguments_.length === 2 && arguments_[0] === '--source') {
  await prepare(arguments_[1])
} else if (!arguments_.length || (arguments_.length === 1 && arguments_[0] === '--verify')) {
  const report = JSON.parse(await readFile(path.join(target, reportName), 'utf8'))
  const checked = await verifyAssets(target, report)
  if ((await readFile(mapFile, 'utf8')) !== mapSource(report.assets)) throw new Error('Generated recording asset map does not match checked-in assets')
  console.log(JSON.stringify(summary(report, checked), null, 2))
} else {
  throw new Error('Usage: node scripts/prepare-recordings.mjs [--source <canonical-recordings-directory> | --verify]')
}
