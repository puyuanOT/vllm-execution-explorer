# Independent simplification, accessibility and rendering review

Reviewed the feature changes against `origin/codex/arrival-diamonds`, including new untracked implementation files, on 2026-10-06. This review preserves the approved techno design and targets meaningful complexity, interaction and rendering defects.

## Result

No significant issue found in the reviewed implementation. The new components keep configuration, layer participation and immutable numerical snapshots separate while sharing request/token/layer/head/K/V selection. The tests cover distinct contracts and regressions rather than merely duplicating DOM structure.

The new GPU instrumentation remains unqualified on a live vLLM GPU runtime because the authorized SSH host is unreachable. This review does not waive that draft gate or treat authored fixture values as runtime evidence.

## Inspected implementation and evidence

- Read the frontend changes in `App`, execution workspace, scene camera, model/cache stages, diagnostics configuration, layer matrix, configuration inspector and tensor inspector, plus the new replay projections and capture plumbing.
- Checked native buttons, labelled selects/inputs, shared selection state, preserved request tracking, return to whole batch, targeted rerun navigation, linked errors/helper text, and inherited visible focus styles.
- Checked evidence files for configuration, authored active/completed layers, shared-prefix writer identity, linked slice selection, targeted rerun restoration and compact inspector focus restoration. Inspected the final desktop tensor screenshot directly.
- Reviewed bounded matrix pagination, bounded 3D tensor segments, instanced physical pool rendering, demand rendering and resource ownership. The new tensor stage creates at most 32 dimension segments plus two backing meshes; it adds no per-frame animation or allocation. Declarative geometries/materials are owned by React Three Fiber; edge resources use the existing disposal path. Camera controls clean up listeners and dispose the controls instance.
- Reviewed the retained desktop scene measurements: the 113-block physical pool remains instanced, with 10 draw calls in the reported focused state and no continuously running demand loop. These are synchronous submission measurements, not GPU completion, presentation FPS or universal performance claims.

## Independent projection check

Executed the production `projectLayerTokenMatrix` with the retained A100 `gpu-model-batching` recording at scheduler step 3. Its 19,303 activities produce 128 bounded positions × 36 layers = 4,608 cells. After 10 warmups, 100 Node executions measured p50 6.77 ms, p95 7.54 ms and maximum 10.53 ms on this laptop. This checks synchronous projection cost only: it excludes React, WebGL, browser layout/presentation and GPU execution. The historical recording lacks new cache-write metadata, so this is not a live-instrumentation performance qualification.

## Limits

This is a source/evidence review, not an accessibility certification. Representative browser journeys were exercised by the implementation owner and their DOM/screenshots inspected here. Live numerical capture, native operation attribution and fresh measured demo recordings still require the reachable GPU host described in the qualification notes.

## Compact tensor-controls reassessment

The implementation owner's additional compact-screen case found that changing a layer/head/K/V slice automatically reopened the inspector and interrupted continued use of the scene controls. Reassessed the repair in `App.choose`, `CacheTensorControls` and workspace callback wiring. Automatic opening is now suppressed only for updates to the same request/token identity that retain an explicit layer slice. Choosing another token retains the existing compact inspector behavior. The explicit Inspect button updates selection and calls the inspector opener directly; on desktop the inspector is already visible.

The retained `mobile-tensor-dom.txt` and `mobile-tensor-inspector.jpg` show that the explicit inspector retains Layer 2 / KV head 1 and the original shared-prefix writer. The compact inspector uses native selects and K/V buttons, and its controls fit the 390-pixel viewport. No significant issue found in this repair.

Inspected the final regression artifacts `mobile-tensor-controls-dom.txt`, `mobile-tensor-controls.jpg` and `mobile-tensor-inspect-action-dom.txt`. Slice updates leave the scene controls available without a dialog; explicit Inspect opens the retained V snapshot with matching layer/head, and the owner verified Escape returns focus to that button. The full-page controls screenshot shows the controls wrapping within the compact layout. The measured document and viewport widths both remain 390 pixels.

Inspected `scene-profile-slices.json` across three pool↔tensor transitions. Settled geometry and draw-call counts return to 7 for the authored pool and reach 38 for the bounded tensor view, without an increasing retained geometry count across these cycles. Initial transition CPU submission reaches 5.3 ms; subsequent isolated tensor submissions are 0.3–0.5 ms. Resource counts are Three.js observations, not GPU bytes, and the times exclude React commits, browser presentation and GPU completion. The retained `browser-console-errors.json` contains no warnings or errors for the sampled main/profile sessions. No significant remaining issue found after this reassessment.

## Camera remount repair

A later shifted slice screenshot led to an actionable camera-state finding in `web/src/scene.tsx`: the workspace retains its last camera command while switching to request lanes unmounts the Canvas. `CameraRig.lastCommand` initialized to `null` on remount, causing its effect to replay the previously consumed pan/orbit command after resetting the camera. This is distinct from an ordinary viewport resize or pool↔slice transition.

The implementation owner reproduced Reset → Pan left → Request lanes → 3D, which reopened the scene shifted right. The minimal repair initializes `lastCommand` to `command?.id ?? null`, acknowledging the command already issued before mount. Subsequent command IDs still execute. This source repair avoids adding a second command state or a scene-navigation special case.

Inspected the saved before/after screenshots, `camera-remount-after-dom.txt` and `camera-regression.json`. After the same remount sequence, the selected slice anchor is centered at x=454 within the 908-pixel scene. A newly issued Pan left moves it to x=485.2, and Reset returns it to x=454. The after screenshot confirms symmetric framing. The regression demonstrates both stale-command suppression and continued use of subsequent camera commands. No significant issue remains in this reviewed scope.
