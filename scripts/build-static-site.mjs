import { execFileSync } from 'node:child_process';
import { cp, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';

// Public hosting keeps the local capture workspace visible, with controller actions disabled.
execFileSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'build', '--prefix', 'web'], {
  stdio: 'inherit', env: { ...process.env, VITE_HOSTED_DEMO: 'true' },
});
const report = JSON.parse(await readFile('web/public/recordings/transport-report.json', 'utf8'));
const remoteFiles = new Set(report.assets.filter(asset => asset.encoding === 'gzip').map(asset => asset.url.slice(1)));
await rm('dist', { recursive: true, force: true });
await cp('web/dist', 'dist', {
  recursive: true,
  filter(source) {
    const relative = path.relative('web/dist', source).split(path.sep).join('/');
    return !source.endsWith('.bin') && !source.endsWith('.partial')
      && !remoteFiles.has(relative) && !relative.split('/').includes('raw');
  },
});
let revision;
try { revision = JSON.parse(await readFile('.openai/application-source.json', 'utf8')); }
catch (error) {
  if (error.code !== 'ENOENT') throw error;
  revision = { repository: 'https://github.com/puyuanOT/vllm-execution-explorer', commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim() };
}
await writeFile('dist/application-revision.json', JSON.stringify(revision, null, 2) + '\n');
console.log('Prepared public static build in dist; recording parts use the pinned public source.');
