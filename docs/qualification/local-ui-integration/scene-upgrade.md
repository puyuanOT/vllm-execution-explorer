# Scene visibility, identities, and model participation

Verified against the existing localhost app and its unchanged `gpu-kv-layer-diagnostics` recording on 2026-10-09. Configure, request lanes, the numerical inspector, and the original recording remain available.

## Implemented

- Main model/component edges and serving connectors use screen-space line widths (2–2.25 px) and brighter existing color tokens. Internal layer rims and the floor remain subordinate. Line geometry/material ownership includes disposal.
- Overview and KV drilldown share neutral allocation fills, raised referenced blocks, recessed available blocks, cached-prefix hatching, and crossed null blocks. Current request holders use separate substantial colored side strips; shared holders retain separate strips. Selection uses independent outer corner brackets. Original writers remain explicit token provenance.
- Token-slot reservations have dashed neutral outlines and `Reserved` text. Written contents have solid neutral interiors and `KV written` text. Color on the side identifies the holder instead of encoding the slot state.
- `Reset view` is an explicit text button. It resets the current camera and preserves the replay cursor, inspected object, and drilldown.
- Model participation is always visible independently of the selected request/token. Host mode uses the actual scheduled request breakdown; profiler mode uses only captured non-padding packed rows. Missing profiler rows do not borrow host participants. Trace gaps suppress current host participation.
- Walkthrough has a prominent illustrated moving band, an explicit `Illustrated layer flow` label, and a direct entry button. Illustrative packet loops run only in Walkthrough, pause when the document is hidden, and respect reduced motion. Profiler activity retains its measured layer/module indicators.
- Hybrid loaded architecture metadata renders thinner GDN plates versus full-attention plates and explicit GDN/Attention labels. A GDN layer/head drilldown shows captured recurrent dimensions rather than invented full-attention Q/K/V geometry. Actual new-model GPU captures are qualified separately by the parent task.
- A SceneLabel deferred-root teardown race uncovered during hot refresh was corrected by canceling pending teardown and reusing the DOM root. Subsequent hot updates and fresh navigation produced no console errors or warnings.

## Browser evidence

Owned Playwright session `scene-upgrade` was closed after verification. Browser used localhost only.

- At EVT 30, model L1 projected to `translate3d(654.3px, 136.7px, 0px)`. Pan right moved it to `(616.9px, 136.7px)`. Reset returned it exactly to `(654.3px, 136.7px)`; playhead stayed 30, breadcrumb stayed `Overview / Model execution`, and inspector stayed Model execution.
- Direct `Illustrate layer flow` entered Walkthrough and started the illustrated band while preserving the recorded iteration and participant summary. The band, label, and request breakdown were visually inspected.
- EVT 25 / block 1 showed 16 dashed reserved slots with positions but no recorded written contents. EVT 30 / block 1 showed solid `KV written` slots and original-writer provenance. Closing the cartridge removed the slot list while preserving EVT 25 and KV-cache focus.
- Desktop 1600 × 1100 and mobile 390 × 844 were inspected. The mobile participation HUD uses the free upper-left band, keeping the model and camera controls visible. Reset remains a visible button.

Screenshots: `output/playwright/scene-upgrade-overview.png`, `scene-upgrade-walkthrough.png`, `scene-upgrade-model.png`, `scene-upgrade-kv.png`, `scene-upgrade-written-slots.png`, `scene-upgrade-reserved-slots.png`, and `scene-upgrade-mobile.png`.

Targeted architecture tests: 13 passed, 0 failed (`output/playwright/scene-upgrade-tests.log`). Production TypeScript/Vite build passed (`output/playwright/scene-upgrade-build.log`); the existing large-bundle warning remains. Diff whitespace validation passed. New LoRA module branches and native GDN/Flash-Next GPU qualification require actual capture metadata; the scene does not invent those observations.
