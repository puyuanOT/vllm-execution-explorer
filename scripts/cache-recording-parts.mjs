/** Download immutable, verified recording parts and restore canonical files for offline playback and tests. */
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm } from 'node:fs/promises';
import path from 'node:path';
import { Readable, Transform, Writable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { createGunzip } from 'node:zlib';
import { recordingAssetBaseUrl } from '../web/src/recordingAssetSource.ts';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const report = JSON.parse(await readFile(path.join(root, 'web/public/recordings/transport-report.json'), 'utf8'));
const run = promisify(execFile);

function destination(url) {
  if (!url.startsWith('/recordings/') || url.split('/').includes('..')) throw new Error('Invalid recording path.');
  return path.join(root, 'web/public', url.slice(1));
}

function hashTap() {
  const hash = createHash('sha256');
  let size = 0;
  return {
    stream: new Transform({ transform(chunk, _encoding, callback) {
      hash.update(chunk); size += chunk.length; callback(null, chunk);
    } }),
    matches: expected => size === expected.size && hash.digest('hex') === expected.sha256,
  };
}

async function verified(file, expected) {
  const tap = hashTap();
  try {
    await pipeline(createReadStream(file), tap.stream, new Writable({ write(_chunk, _encoding, callback) { callback(); } }));
    return tap.matches(expected);
  } catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

async function cachePart(part) {
  const file = destination(part.url);
  if (await verified(file, part)) return;
  await mkdir(path.dirname(file), {recursive:true});
  const temporary = file + '.partial';
  try {
    // curl uses the OS certificate trust store, including the user's installed public CAs.
    await run('curl', ['--fail', '--location', '--silent', '--show-error', '--retry', '2', '--connect-timeout', '15', '--max-time', '120', '--output', temporary, recordingAssetBaseUrl + part.url]);
    if (!await verified(temporary, part)) throw new Error(`Integrity mismatch for ${part.url}`);
    await rename(temporary, file);
  } finally { await rm(temporary, {force:true}); }
  console.log(`Downloaded and verified ${part.url}`);
}

async function* compressedBytes(parts) {
  for (const part of parts) yield* createReadStream(destination(part.url));
}

const parts = report.assets.filter(asset => asset.encoding === 'gzip').flatMap(asset => asset.parts);
for (const part of parts) await cachePart(part);

let restored = 0;
for (const asset of report.assets) {
  const file = destination(asset.url);
  const expected = {size:asset.originalSize, sha256:asset.originalSha256};
  if (await verified(file, expected)) continue;
  if (asset.encoding === 'identity') {
    await cachePart({url:asset.url, ...expected});
  } else {
    await mkdir(path.dirname(file), {recursive:true});
    const temporary = file + '.partial';
    const tap = hashTap();
    try {
      await pipeline(Readable.from(compressedBytes(asset.parts)), createGunzip(), tap.stream, createWriteStream(temporary));
      if (!tap.matches(expected)) throw new Error(`Decoded recording integrity mismatch for ${asset.url}`);
      await rename(temporary, file);
    } finally { await rm(temporary, {force:true}); }
  }
  restored += 1;
}
console.log(`Ready for offline playback and tests: ${report.assets.length} verified canonical files (${restored} restored), ${parts.length} verified cached parts.`);
