# Independent integration and evidence review

Reviewed October 10 UI behavior against the qualification README, retained Codex browser receipts, native recording artifacts, screenshots, and production projection paths. This review opened no browser, ran no GPU workload, and changed no application source.

## Verified

- The BF16 and hybrid screenshots show highlights in **Overview**, alongside the intact Configure/Explore navigation. The BF16 layer 4 plate is amber; the hybrid layer 1 plate is amber and labeled Gated DeltaNet. The source preserves Overview when GPU time is entered and renders all mapped active layers in the compact overview stack.
- Independently reran `parseProfileAttachment`, `projectMeasuredProfile`, and `createProfileIntervalIndex` on the retained artifacts. BF16 UI offset `36510.885986328125` µs maps exactly to native `727726910435.739` µs and activity `activity-219873`, layer index 3 / FFN. Hybrid UI offset `14244.470947265625` µs maps exactly to native `728589839215.944` µs and activity `activity-462871`, layer index 0. Both agree with `native-ui-match.json` and the actual UI labels.
- Native activity derives from CUDA intervals. Request selection and host replay do not fill native gaps; host memory continues to use its own playhead. The small clock cues in the screenshots accurately disclose this separation. Manually inspected layers use a cyan rim rather than the amber execution highlight.
- Memory receipts contain the same 59.71 GiB scale and 662 circles at hybrid event 355 and cleanup. Only muted `#71869f` and current-sample teal `#40d9ee` fills occur. The source uses the selected device's successful sampled peak, preserves actual sample timestamps, draws no connecting interpolation, and excludes future completed samples in live mode. GiB scale labels are visible in the BF16 screenshot.
- Request and keyboard receipts establish return to the default Inspector, heading focus, and unchanged host/native playheads. The callback clears object selection and both persistent tracking states. Overview camera screenshots preserve the same panned framing, with the selected KV outline clearing as expected.
- The Configure screenshot retains the existing two-batch/three-request draft, SSH host, form sections, and preparation controls. The final DOM retains Configure, Overview, Reset, and the default Inspector. Removed provenance footer behavior is covered by the request-lane receipt and conditional rendering source.
- Current retained test output reports 109/109 web and 70/70 replay tests, with TypeScript and production build success. The bundle-size warning is disclosed.
- Independently compared all 30 retained bundled recording files with the preceding demo-expansion SHA256 receipts: every file matches. Independently computed `source_identity` for runner/instrumentation: 40 files, SHA256 `56b4d03456a627b250a87e322a54cbf035643e730db8d2c27c2f3b29cff086bc`, matching the qualified captures. This supports the stated preservation of backend capture sources and bundled recording artifacts during this UI refinement.

## Evidence limits

The completed browser evidence is from Codex at a DOM-confirmed 1920 × 1080. The earlier failed Chrome screenshot is explicitly excluded from the pass claims. The empty captured console-entry list does not establish controller connectivity: the persisted `127.0.0.1:8766` failure is visible and correctly disclosed. No new remote run was needed or claimed; live future-sample filtering is helper-tested. Simultaneous multi-layer behavior is covered by a helper regression rather than an observed overlapping multi-layer interval in these selected real screenshots. The original 45-step BF16 receipt ends in a genuine native gap; the later 180-step receipt establishes the layer 4 result.

## Targeted camera regression

Another independent reviewer identified camera reframing when Back to run clears a token/tensor drill in the KV scene. Reviewed the resulting persistent camera guard in `scene.tsx` and the targeted Codex receipt/screenshots. The guard preserves camera/controls on a same-view tensor-to-pool selection clear, including the ensuing canvas height change; explicit Reset, a new tensor drill, or view navigation releases preservation. The actual BF16 regression retains host event 200 and the close panned camera after Back to run; the pool is visibly enlarged/cropped at that close position. Only explicit Reset subsequently fits the entire pool. The default Inspector and disappearance of tensor controls are visible. The 1920 × 1080 receipt records canvas height changing from 417.72 to 546.61 pixels. This resolves the identified integration issue without changing native playback or capture sources.

## Result

No additional significant correctness or evidence issue found in the final reviewed scope. The report's limits distinguish local replay verification from live-controller availability and individual token/head timing. The refreshed post-fix output reports 109/109 web tests and successful TypeScript/Vite build (2.43 seconds); `git diff --check` also passes.
