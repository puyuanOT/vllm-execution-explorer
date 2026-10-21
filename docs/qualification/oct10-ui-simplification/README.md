# October 10 playback and memory simplification

This implements the latest four decisions in the existing local Configure/Explore application. No Sites deployment, remote capture, commit, or push was performed. The worktree contains substantial prior changes; its whole Git diff is not this task's scope.

## Scope

- Q1: Iterations and GPU activity are the two primary playback choices. Advanced playback contains Raw events and Host time. The trigger displays an active advanced mode after the menu closes. Walkthrough and its obsolete artificial animation state/props are removed; the obsolete Reduce motion control for that animation is removed. Native measured highlights remain.
- Q2: No changes to GPU timeline range, gaps, clocks, navigation, or interval boundaries. CPU activity starts remain part of Next measured activity.
- Q3: The GPU memory chart is removed. The main panel displays measured used/capacity text. Memory breakdown opens the existing accessible modal with device choice, counters, sampled maximum, allocation attribution, phase samples, and CPU-storage details. Five component cards are collapsed under Component summary by default and reset to collapsed on recording changes.
- Q4: No new demo-switch control. Existing Open recordings and Configure/Explore remain.

Affected code: `web/src/App.tsx`, playback.ts, execution/{PlaybackTransport,ExecutionWorkspace,GpuMemoryPanel,ModelIterationBar,ArchitectureNavigation,LayerTokenMatrix,types,memorySummary}, architecture/{ModelStage,SceneProfileDiagnostics,model}, scene.tsx, and the focused memory-summary/architecture tests. `memoryTimeline` and its fixed-chart tests were removed with the graph. Existing native profile playback calculations, replay/backend and recording data were not edited for this task.

## Integrated browser evaluation

All cases used the Codex in-app browser at **1920 × 1080** against localhost:5173, with real bundled H100 recordings. Actions used actual controls and keyboard input; application state was not injected. Retained evidence is in [browser-receipt.json](../../../output/playwright/oct10-ui-simplification/browser-receipt.json).

| Case | Result |
|---|---|
| Initial BF16 recording | Pass: Iterations/GPU activity primary choices; no Walkthrough; Component summary collapsed; memory plot absent. |
| No sample at host event 0 | Pass: dialog explains that no successful sample exists before this playhead and offers seeking to a recorded sample. It does not invent zero usage. Escape dismisses. |
| BF16 memory at host event 200 | Pass: 9.26 GiB / 79.65 GiB; run sampled maximum 9.49 GiB. Detailed counters/attribution are visible in the modal. Canvas bounding rectangle is identical before/after opening. |
| Dialog keyboard/scrolling | Pass: initial focus is Close Memory breakdown; Tab cycles to Memory device then back to Close. End on the close button did not scroll (expected button behavior) or alter event 200; PageDown focused on the dialog scrolled to the bottom (scrollTop 326.5, clientHeight 1046, scrollHeight 1372). Escape returns focus to Memory breakdown. Close button also dismissed on repeated use. |
| Component summary | Pass: Enter expands all five cards; Scheduler card opens details; Back to run restores Inspector. Space collapses summary; host event stays 200. Switching to the hybrid recording starts collapsed. |
| Advanced playback | Pass: menu exposes the two radio choices. End selects Host time inside the menu without changing host event 200. Raw events closes the menu and remains named in its trigger; Next recorded event moves 200 to 201. Host time exposes its recorded-time axis, proportional point timestamps, and fine playback speeds. |
| GPU activity | Pass: retains full native 22,295.681 ms range. Forty-seven existing Next measured activity steps reach mapped layer 1 at +3.692 ms. Overview highlights layer 1; host replay stays at event 201. No gap or stepping changes. |
| Cleanup | Pass: Iterations → End reaches cleanup event 2610. Memory is 1.11 GiB / 79.65 GiB; framework counters are Not captured and loaded-storage inventory is explicitly invalidated. Run sampled maximum remains 9.49 GiB. |
| Hybrid 27B | Pass: existing Open recordings → Qwen3.8-27B loads the real demo. First iteration/event 355 reports 53.76 GiB / 79.65 GiB, sampled maximum 55.29 GiB, shared backing allocations and separate CPU storage. Summary starts collapsed. |
| Configure and return | Pass: existing Configure page still shows query sequence, execution host, model and vLLM sections. Explore returns to the selected recording. |
| Browser diagnostics | Pass: captured console warning/error entries were empty. Existing controller badge still says 127.0.0.1:8766 Failed to fetch; these are offline playback cases, not live-controller verification. |

The receipt's `hostModeProof.hostAxis=false` came from querying an incorrect assumed DOM label; it is not a product failure. The actual Host time DOM visibly contained individual timestamped observations, the proportional-spacing explanation and speed choices. Missing sequence in that same receipt is because Host time uses its dedicated timeline control rather than the raw Playhead input. A corrected direct check of the actual `Recorded host time axis` region passed, and Next recorded event moved hybrid event 355 to 356 at 295.762783 ms: [host-time receipt](../../../output/playwright/oct10-ui-simplification/host-time-proof.json), [full Host time DOM](../../../output/playwright/oct10-ui-simplification/host-time-dom.txt).

## Evidence and checks

- [Overview with active GPU layer](../../../output/playwright/oct10-ui-simplification/overview-gpu-activity.jpg)
- [Memory dialog](../../../output/playwright/oct10-ui-simplification/memory-dialog.jpg) and [scrolled bottom](../../../output/playwright/oct10-ui-simplification/memory-dialog-scrolled.jpg)
- [Hybrid overview, summary collapsed](../../../output/playwright/oct10-ui-simplification/overview.jpg)
- Web suite **107/107 pass**: [output](../../../output/playwright/oct10-ui-simplification/web-tests.txt). Replaced obsolete graph tests with three cases protecting device-specific recorded maxima/phase totals, live future exclusion, and unknown/zero values. Removed obsolete Walkthrough geometry expectation.
- TypeScript and production build pass: [output](../../../output/playwright/oct10-ui-simplification/build.txt). Existing Vite chunk-size warning remains.
- `git diff --check` passes.
- No fresh GPU capture; no new live or multi-device session was run. Live/multi-device summary filtering is covered at helper level, not claimed as live E2E.

Final correctness and simplification review both found a stale “Illustrated layer flow” legend in the focused model view. It was removed. Direct Component summary → Explore Model execution inspection then showed only the remaining states and no illustrated-flow legend: [regression receipt](../../../output/playwright/oct10-ui-simplification/legend-cleanup-proof.json). The production build was rerun after the one-line correction.

All three final reviewers report no significant unresolved issues: [correctness](review-correctness.md), [integration and evidence](review-evidence.md), and [simplification](review-simplification.md). The evidence reviewer independently recomputed the displayed memory values, phase counts/maxima and native profiler extent from the real recording artifacts.
