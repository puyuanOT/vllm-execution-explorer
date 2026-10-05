# Forms and accessibility final review

**Verdict: no significant issues.**

Independent, non-author review of the Cursor redesign from `25432ac` through `ca9a516` and the final uncommitted frontend repairs on `codex/control-plane-review`. This pass followed the integrated evaluation described in the qualification README. No application source was edited, no browser was operated, and no GPU execution was launched by this reviewer. This report is the only review-authored file.

## Scope and result

Reviewed ConfigurePage, ArrivalSequencer, ConsoleRail, PreflightPanel, ServerSection, VllmSection, WorkloadSection, shared controls, App configuration/validation integration, header evidence, responsive theme rules, playback/inspector interactions, and the repaired KV cartridge header. Existing model/tokenizer revision overrides, host setup/check actions, scheduling/priority validation, sampling/native options, request input modes, adapter selection, movement, duplication, deletion restoration, and time-unit conversion remain reachable. No substantiated removed-control or admission/validation regression remains.

The repairs address the demonstrated failures without changing the visual direction:

- Whole-millisecond arrival ticks round-trip through both milliseconds and seconds. The chart has a focusable, named horizontal scrolling region instead of overflowing the page.
- Dense workload rows depend on the actual editor container width. Narrow rows use three tracks, visible labels, and sufficient input space; the retained 320 px and laptop receipts show `512` and `4096` without clipping.
- Request-lane scrolling keeps the selected column outside the sticky label region in both directions. Playback and scope use the trusted replay prefix after a trace gap rather than exposing later raw events as observed state.
- The KV cartridge uses a two-row header, preserving a reachable close control on the narrow receipt. Inspector Escape and focus restoration were exercised in the retained browser evaluation.
- Both desktop and narrow draft actions now say **Undo deletion**, matching the existing deletion-snapshot operation.
- Form labels, validation focus targets, busy/disabled states, named icon controls, and the retained invalid-submit/focus/mode-change journeys are consistent with the configuration flow.

The preliminary global `sr-only` concern is withdrawn: in-memory compiled CSS confirms the responsive restoration utilities follow the global rule. Container variants are also emitted. It is not an unresolved accessibility finding.

## Finding discovered and resolved during this final pass

The initial final-review source classified every `execution_kind: "gpu"` manifest as **Observed GPU recording**. The original retained preparation failure, `run-14001494e084`, disproved that classification: `provenance.real_gpu` is false, `hardware.gpu` is null, the event file is empty, and the manifest explicitly states GPU capture did not begin. The provisional live preparation manifest also only records the requested execution kind.

This was reported as a significant evidence-label issue. The subsequent narrow repair centralizes classification in `web/src/execution/evidence.ts`: a GPU observation claim requires positive GPU provenance and hardware; missing or negative evidence yields **GPU execution unverified**. App, the header, workspace chip, and watermark use that classification. A failed or cancelled capture with actual earlier GPU observations can still retain its truthful observed classification; fixture and draft labels take precedence.

Recheck used the original archived manifests directly, without modifying the archive:

| Original capture | New classification |
| --- | --- |
| `run-14001494e084`, failed before capture, no GPU provenance/hardware | `unverified` |
| `run-5328256bf349`, completed native diagnostic capture | `gpu` |
| `run-d0ec34078edc`, completed ordinary prefix capture | `gpu` |

The new regression also covers provisional preparation, negative or absent provenance, absent hardware, successful observations, failed-after-observation recordings, fixture precedence, and draft precedence. The final production-browser receipt `failed-gpu-provenance.png` shows the retained failed run with an unverified header/watermark and disabled playback. The evaluator also rechecked the successful prefix capture's observed badge. The evidence-label issue is resolved. The narrow Undo label inconsistency found in the same pass is also resolved in `ConfigurePage.tsx`; `mobile-undo-label.png` confirms the repaired action is visible at 390 px.

## Independent verification and evidence reviewed

- Ran `node --experimental-strip-types --test src/scenario/redesignRegressions.test.ts` against the final repair: **4 passed, 0 failed**. These cover integer arrival ticks, evidence classification, trusted trace-gap playback, and forward/backward sequencer visibility.
- Executed the actual `recordingEvidence` function against all three manifest objects read directly from `a100-captures.tar.gz`; the table above reports the returned classifications.
- Independently verified the archive size/SHA-256 receipt and all 11 indexed file sizes/SHA-256 values. Original failed and successful evidence is preserved.
- Reviewed the retained README, native/prefix/rejected-budget assessments, test/build logs, and responsive/browser receipts. The final integrated evaluation reports **214 Python, 51 replay, and 68 web tests plus the production build** passing, including the additional evidence regression; these are retained evaluator results, not a claim that this reviewer reran every suite.
- Inspected the retained mobile/laptop inputs, mobile lanes/arrival chart, KV cartridge, trace-gap, actual native profile, prefix reuse, and preparation-error evidence. Actual GPU execution and complete source provenance are documented by the integrated evaluator; this reviewer did not independently execute GPU workloads.

No unresolved significant issue was found in the reviewed forms, accessibility, responsive repairs, or evidence-label integration. This is a source and retained-evidence review, not a new frame-rate benchmark, screen-reader session, or GPU qualification.
