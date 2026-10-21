# Request-colored block tops — October 13

## Requested behavior and scope

The user approved request-colored top surfaces with equal split colors for shared blocks, preserving allocation-state and selection cues. This change applies to both Overview and the KV-cache pool through their shared BlockMarks component. The five changed application files are captured in [scoped.diff](../../../output/playwright/oct13-block-tops/scoped.diff); the large preexisting worktree diff is not this task's scope.

- Colored planes cover 92% of the block width/depth, leaving a thin neutral rim. Sides retain allocation-state colors/heights.
- Referenced blocks use deduplicated current request bindings in stable canonical order. Shared references get equal bands separated by a narrow neutral gap; band width is not ownership percentage or token occupancy.
- Cached zero-reference, available, null and unknown blocks receive no request color. Their existing hatch/cross/unknown mark and height remain. Selection retains external corner brackets and does not recolor the top.
- Planes remain non-pickable decorations, preserving the physical block's existing picking handler. One instanced mesh batches the planes for each bank/pool.
- Legends and block-finder help describe the new encoding. Token-slot strips continue to identify the active holder; cached blocks with no holders now explicitly say No current request references.
- During E2E a preexisting palette inconsistency was found: scene/chips used replay request insertion order while the legend used workload roster order. App now uses one canonical roster order for both its color map and scene requestOrder. In this recording Request 2 is consistently pink, Request 3 green and Request 4 orange.

No backend, recording data, playback logic, remote GPU, deployment, commit or push was changed/performed. Existing local Vite process cwd was verified as this worktree's web directory. The controller 127.0.0.1:8766 remains unavailable; this task verifies offline replay with real saved data.

## Integrated UI evaluation

Codex in-app browser only, DOM-confirmed **1920 × 1080**, actual BF16 H100 recording `gpu-h100-bf16`. UI controls and native accessibility slider setting were used; no app state injection. Source-derived cases use the current production applyEvent/projectKvCache and buildTimeline implementations: [source-cases.json](../../../output/playwright/oct13-block-tops/source-cases.json).

| Case | Outcome |
|---|---|
| Single current reference before KV written, EVT 191 | Pass. B1 top is Request 1 blue, while its inspector reports 0/16 computed and every mapped slot is Reserved. Other allocated blocks with no captured binding yet stay neutral. [Pool image](../../../output/playwright/oct13-block-tops/reserved-pool.jpg), [slot DOM](../../../output/playwright/oct13-block-tops/reserved-selected-dom.txt). |
| Cached/free retained contents, EVT 363 | Pass. After Request 1 completes, B1–B11 are neutral hatched reusable-prefix blocks; unreferenced/free B12 remains dark despite retained contents in source. B0 stays neutral with X. B1 inspector has zero refs, no current holders, recorded contents; updated footer explicitly says No current request references. [Pool](../../../output/playwright/oct13-block-tops/cached-neutral.jpg), [DOM](../../../output/playwright/oct13-block-tops/cached-selected-dom.txt). |
| Shared and single references together, EVT 1155 | Pass. B1–B10 and B59 split Request 2 pink / Request 4 orange equally; B12 is Request 3 green, B60 pink, B61 orange. Cached B11 remains neutral and hatched, B0 crossed, and free blocks dark. [Final full pool](../../../output/playwright/oct13-block-tops/pool-shared.jpg), [closer view](../../../output/playwright/oct13-block-tops/pool-detail.jpg). |
| Shared block selection/picking | Pass. Native click on the colored B1 top opens Block 1. Corner brackets mark selection without replacing pink/orange top. Inspector/cartridge list Request 2 and Request 4 as current holders, while Request 1 remains the original KV writer. Close token slots dismisses the cartridge. [Selected image](../../../output/playwright/oct13-block-tops/shared-selected.jpg), [DOM](../../../output/playwright/oct13-block-tops/shared-selected-dom.txt). |
| Same rendering in Overview | Pass. Overview shows the same single/split tops. Request chips, legend swatches and scene match after palette fix. [Overview](../../../output/playwright/oct13-block-tops/overview.jpg), [DOM and palette receipt](../../../output/playwright/oct13-block-tops/browser-receipt.json). |
| Count/state transitions | Pass. Seeking 191 → 363 → 1155 removes the single top at release and restores the larger set of single/shared tops at the later event, without stale surfaces. Two zoom-ins then one zoom-out exercise existing camera controls; the final framing retains all states. |
| Browser diagnostics | Pass. Captured warning/error log is empty in the receipt. |

Review identified early reserved/cached evidence with superseded helper wording. The reserved-pool image and both reserved/cached inspector DOM captures were refreshed against the final code; cached blocks explicitly say No current request references. The Codex screenshot tool produced 1920 × 1064 raster captures despite the DOM-confirmed 1920 × 1080 viewport; no screenshot was resized or edited.

Unknown blocks are absent from this complete recording and were not fabricated for UI testing. Their existing exclusion/pattern branch remains intact; existing tests cover the unknown allocation classification, while unknown-state rendering was not exercised in the browser. Testing used the real 227-block pool, not a claim of performance at arbitrary pool sizes. Shared examples contain two current holders; the same band division handles more, without a new cap.

## Checks

- [Web tests](../../../output/playwright/oct13-block-tops/web-tests.txt): **107/107 pass** after all source changes.
- [TypeScript and Vite production build](../../../output/playwright/oct13-block-tops/build.txt): pass, with the existing bundle-size advisory.
- `git diff --check`: pass.
- No new unit tests mirror visual dimensions; meaningful existing semantic/picking tests and actual UI interactions provide coverage.

All three final independent reviewers found no significant unresolved issues: [correctness](review-correctness.md), [integration and evidence](review-evidence.md), and [simplification](review-simplification.md). The evidence reviewer independently matched the saved cases to the production replay and added the reserved EVT 191 projection. No application changes were required by final review.
