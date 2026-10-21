import { cp, rm, writeFile } from 'node:fs/promises';
await rm('dist', {recursive:true, force:true});
await cp('web/dist', 'dist', {recursive:true});
await writeFile('dist/_headers', '/recordings/*/transport/*\n  Content-Type: application/octet-stream\n  X-Content-Type-Options: nosniff\n');
console.log('Prepared static app in dist.');
