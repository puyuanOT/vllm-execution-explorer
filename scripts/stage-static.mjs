import { cp, rm } from 'node:fs/promises';
await rm('dist', {recursive:true, force:true});
await cp('web/dist', 'dist', {recursive:true, filter:(source)=>!source.endsWith('.bin')});
console.log('Prepared static app in dist; immutable recording parts are served from the public source commit.');
