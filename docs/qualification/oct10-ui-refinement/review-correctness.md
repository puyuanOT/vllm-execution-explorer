# Independent correctness review

Reviewed the October 10 UI scope after the integrated Codex in-app-browser cases described in `README.md`. This review used source inspection and the retained screenshots/DOM receipts; it did not open another browser or execute a new GPU workload. Earlier demo-expansion changes in the dirty worktree are outside this review.

## Resolved finding

**P2 — Back to run reset the camera when leaving a KV tensor drill.** The first reviewed implementation cleared selection, indirectly changing `tensorDrill` and triggering automatic camera fitting. This was fixed in `CameraRig`: a same-view tensor return caused by clearing selection preserves the camera, including the subsequent resize caused by removing tensor controls. Explicit Reset, entering another tensor drill, or scene navigation clears preservation and restores normal fitting.

Re-reviewed the fix and the actual Codex 1920 × 1080 tensor case. The retained [receipt](../../../output/playwright/oct10-ui/codex-tensor-return.json) records unchanged host event 200 and canvas growth from 417.72 to 546.61 px. [Before return](../../../output/playwright/oct10-ui/codex-tensor-before-back.jpg), [after return](../../../output/playwright/oct10-ui/codex-tensor-after-back.jpg), and [after explicit Reset](../../../output/playwright/oct10-ui/codex-tensor-after-reset.jpg) show that Back to run keeps the panned close camera while restoring Inspector, and Reset then fits the whole pool. The finding is resolved.

## Other checks

- Native activity is derived exclusively from current CUDA intervals, with invalid/out-of-range layers filtered. All mapped simultaneous layers are retained; host scheduling participants and manual inspection do not fill native gaps.
- The overview passes the native profile into the compact layer stack. Amber measured plates/rims and a larger cyan inspection rim remain independent, including when the inspected layer is also active.
- The UI-to-native receipt matches BF16 layer 4 at +36,510.885986 µs and hybrid layer 1 at +14,244.470947 µs to retained CUDA intervals. The hybrid generic layer activity is labeled as a Gated DeltaNet layer from captured architecture; it does not claim a specific recurrent module for a component-less interval.
- Memory chart bounds use the selected device's successful run sampled maximum with padding. Recorded scrubbing leaves those bounds and point colors stable; live chart projection excludes samples beyond either the host sequence or host timestamp. Failed samples and timestamp gaps are not interpolated.
- Back to run clears all three selection/tracking states, with the same callback in desktop and compact inspectors. Keyboard focus moves to the retained inspector heading; the compact sheet also retains its separate dialog close control.
- The GPU-only footer suppression, Reset text, GPU time label, and removal of the Illustrate layer flow button match the request. Configure remains present in the saved 1920 × 1080 browser evidence.
- Retained current test output confirms 109 web tests, 70 replay tests, and a successful production build. The existing controller fetch failure is explicitly separated from offline recording verification.

No significant correctness issue remains in the scoped change. The camera-preservation finding was resolved and verified; this review approves the current implementation.
