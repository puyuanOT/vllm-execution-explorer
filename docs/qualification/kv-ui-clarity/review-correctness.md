# Correctness review: caption removal and token-slots Close

Reviewed the current working-tree changes against base commit `867b039bafff0ac97bb8318def8ec46f1b9a4c93` on 2026-10-08. The scope is limited to:

- `web/src/scene.tsx`: remove “Serving observations connected to captured model structure” from the supported-model overview and omit its empty text object.
- `web/src/architecture/CacheStage.tsx`: resolve an open block cartridge only for explicit block or token selection.

No actionable correctness findings.

The overview still renders its profiler-clock, trace-gap and unknown-architecture messages under their existing conditions. The removed caption creates no empty `SceneText` object.

Reviewed the Close transition through `BlockCartridge` and `App.choose`: Close selects the KV-cache component while the app deliberately retains request/token tracking. Previously, the tracked-token fallback immediately resolved another selected block and kept the cartridge visible. The new explicit selection-kind gate leaves `selected` undefined after Close. Both the cartridge and tensor-stage conditions consequently remain closed.

Checked these state cases by code inspection:

- Explicit physical-block selection resolves the requested block, including the existing key formats, and opens its cartridge.
- Explicit token selection retains the existing block-key preference, last-block preference and matching-block fallback.
- Explicit token selection with a KV layer can still enter the tensor stage when the layer mapping and block trust permit it.
- Component, request, step, layer, module, head and null selection cannot open a cartridge through a stale tracked token.
- Close preserves request/token tracking and request rim highlighting. Entering a component with an old tracked token no longer opens a drawer automatically; users can reopen it through explicit token or block selection.
- Unknown block trust and missing tensor-layer mapping retain their existing restrictions.

Independent validation: `web/node_modules/.bin/tsc --noEmit`, run from `web`, passed with exit code 0. No additional tests were added. Browser regression verification is recorded separately by the close-button investigation agent; this review does not claim browser execution.
