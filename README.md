# vLLM Execution Explorer — public site

This is the static ChatGPT Sites publication of the existing vLLM explorer, copied from the reviewed localhost implementation. It preserves Explore, real recorded demos, imported recordings, the3D model/KV-cache views and the configuration workspace. No GPU model runs on this web host.

The public GitHub project is https://github.com/puyuanOT/vllm-execution-explorer . Live cluster capture uses the local Python controller; the public website never uploads SSH credentials or launches captures. Configuration is available for inspection; Check host and Prepare and capture are disabled here.

## Build

Node22: `npm ci` (installs the web dependencies), then `npm run build`.
The existing React/Vite build is copied to root `dist`; `.openai/hosting.json` declares static SPA routing.

## Recorded data

Canonical recording manifests are unchanged. Large runtime artifacts are stored as lossless gzip binary transport parts, each at most16MiB. `recordingAssetMap.ts` maps original URLs to ordered parts; the browser decodes the exact original UTF-8 bytes. Existing normalized-profile size and SHA-256 validation remains enabled. Original compressed native profiler archives are provenance referenced by the manifests; they are not served by this playback-only publication. They remain in the local capture/archive workspace.

`node scripts/prepare-recordings.mjs --source /absolute/path/to/recordings` recreates transport assets from original recordings. `node scripts/prepare-recordings.mjs --verify` verifies the checked-in assets. Original source recordings are not needed to build this site.
