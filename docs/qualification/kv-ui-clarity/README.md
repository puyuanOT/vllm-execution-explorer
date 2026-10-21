# KV drawer dismissal and overview caption

Verified on 2026-10-08 through the integrated Vite UI at
`http://127.0.0.1:5173/explore`, using the saved real GPU demo
`gpu-kv-layer-diagnostics` (source `run-a413ca908152`). The browser used event
25, `step_submitted`, `step-1`, throughout the dismissal cases. No new GPU
inference was needed or performed. The live endpoint was unavailable; this
does not affect saved-recording replay.

The code is base commit `867b039` plus the two-file UI change in
`web/src/architecture/CacheStage.tsx` and `web/src/scene.tsx`.

## Original failure

Before the fix, the actual UI was navigated through **Explore KV-cache
manager → Find a physical KV block → Block 3 → Inspect block → Slot 14
(prefix-prime, position 46) → Close token slots**. The inspector changed to
the KV-cache manager, but the Block 3 token drawer remained visible. This
reproduced the user's screenshot, including the reserved positions 32–47.
The before-fix result was observed in the browser tool transcript; no
before-fix screenshot or DOM file was saved.

`App.choose` retains the tracked request/token when selecting a component.
`CacheStage` previously used that tracking fallback to reopen a block drawer
for every selection kind. The fix limits the selected drawer block to an
explicit block or token selection; the tracking/highlight path remains intact.

## Completed cases

| Case | Actual UI action and result | Evidence |
| --- | --- | --- |
| Tracked token closes | Block 3 → Slot 14, position 46 → Close; drawer disappears and component inspector appears | `tracked-close-click-dom.txt` |
| Explicit selection reopens | Inspect Block 3, then click Slot 14; drawer appears and token inspector selects position 46 | `explicit-token-reopen-dom.txt` |
| Tracking survives close | After closing and viewing Block 0, explore Model execution; matrix still follows prefix-prime, position 46 | `tracking-preserved-dom.txt` |
| Null placeholder closes | Inspect Block 0 while a token remains tracked, then Close; null drawer disappears | `null-block-dom.txt`, `null-block-close-dom.txt` |
| Ordinary block closes | Clear tracking using Whole batch, inspect Block 3, then Close; drawer disappears | `ordinary-block-open-dom.txt`, `ordinary-block-close-dom.txt` |
| Phone-width tracked token closes | At 390×844, inspect Block 3, dismiss compact inspector, select Slot 14, dismiss inspector, then Close token slots; drawer disappears | `narrow-token-open-dom.txt`, `narrow-token-close-dom.txt`, corresponding PNGs |
| Caption removed | Return to Overview; the removed sentence is absent and the model/serving overview remains visible | `overview-caption-removed-dom.txt`, `overview-caption-removed.png` |

`tracked-close-after-dom.txt` and `tracked-close-after.png` record the closed
state immediately after hot reload applied the fix to the previously failed
selection. `tracked-close-click-dom.txt` records a subsequent actual Close
click after explicitly reopening and selecting the same token.

The temporary viewport override was reset. These cases establish drawer
dismissal, reopening, tracking retention, and caption removal. They do not
establish numerical snapshot availability for position 46: that slice remains
explicitly **Not captured**, as expected for this saved demo's capture targets.

## Checks

- Web tests: 73 passed, zero failures (`web-tests.txt`).
- TypeScript check and production Vite build: passed (`web-build.txt`).
- `git diff --check`: passed.
- Existing Vite bundle-size advisory remains; it is unrelated to this change.

