import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { recordingAssetBaseUrl } from '../web/src/recordingAssetSource.ts';

const report = JSON.parse(await readFile('web/public/recordings/transport-report.json', 'utf8'));
const parts = report.assets.filter(asset => asset.encoding === 'gzip').flatMap(asset => asset.parts);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const part of parts) {
  if (!part.url.startsWith('/recordings/') || part.url.split('/').includes('..')) throw new Error('Invalid recording part path.');
  const destination = path.join('web/public', part.url.slice(1));
  try {
    const cached = await readFile(destination);
    if (cached.length === part.size && digest(cached) === part.sha256) continue;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const response = await fetch(recordingAssetBaseUrl + part.url, {signal:AbortSignal.timeout(60000)});
  if (!response.ok) throw new Error(`Could not cache ${part.url}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length !== part.size || digest(bytes) !== part.sha256) throw new Error(`Integrity mismatch for ${part.url}`);
  await mkdir(path.dirname(destination), {recursive:true});
  const temporary = destination + '.partial';
  try { await writeFile(temporary, bytes); await rename(temporary, destination); }
  finally { await rm(temporary, {force:true}); }
  console.log(`Verified ${part.url}`);
}
console.log(`Ready: ${parts.length} verified recording parts (local test cache only).`);
