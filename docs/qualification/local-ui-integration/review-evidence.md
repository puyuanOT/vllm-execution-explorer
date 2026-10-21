# Independent final integration and evidence review

Reviewed on 2026-10-09 against base `92fb11b00d889cb58136a987bcf09687c0381acd` in `/Users/pliu/.codex/worktrees/kv-layer-inspection/vllm-execution-explorer`. Scope includes all modified and new UI modules/helpers/tests plus the fractional host-time conversion in `replay/src/timeline.ts`. Pre-existing capture-script, earlier qualification, and six-token planning changes were excluded.

No significant outstanding integration or evidence issue was found after the timestamp repair described below. This is a source/artifact review with independent production-projection and hash comparisons; I did not drive the browser, rerun the test suites, start a GPU capture, or change application source.

## Actual application and preservation

The listener on `127.0.0.1:5173` was Node PID 57002 with its working directory in this worktree's `web` directory. Retained browser outputs identify the actual `/` and `/explore` application routes. The implementation imports existing replay and profiling projections directly; it does not hydrate a copied Sites export.

Configure/Explore, the techno shell, editable workload/configuration controls, capture/preflight setup, library/import, request lanes, original camera controls, Edit as new sequence, and targeted diagnostic reruns remain available. The broader capture preset edits Configure fields without starting a run. The final mixed-iteration case retains 64 participants, keeps host EVT238 unchanged when opening global Profile, closes the local panel, and returns to Configure through Edit as new. Global participant selection retains displayed layer 19 and identifies uncaptured numerical values correctly.

| Verified behavior | Retained evidence |
| --- | --- |
| Configure draft/preset and original library | [Configure result](../../../output/playwright/local-improvements/configure-result.txt), [Configure snapshot](../../../output/playwright/local-improvements/configure-snapshot.txt), [library](../../../output/playwright/local-improvements/library.txt) |
| Native layer zoom, keyboard cursor, participant selection | [Iteration result](../../../output/playwright/local-improvements/iteration-result.txt), [repaired desktop screenshot](../../../output/playwright/local-improvements/iteration-inspection-fixed.png) |
| External block selection followed by participant reselection; five semantic bookmarks | [Bookmarks and reselection](../../../output/playwright/local-improvements/bookmarks-and-reselection.txt) |
| Raw metadata remains expandable; mixed participants, global Profile, lanes and Edit as new | [Final integration](../../../output/playwright/local-improvements/final-integration.txt), [global participant](../../../output/playwright/local-improvements/global-participant-result.txt) |
| Invalid import and recording without correlated native intervals | [Recovery result](../../../output/playwright/local-improvements/recovery-result.txt) |
| 390 px mobile document, ordinary controls, inspector/Escape, Configure | [Mobile result](../../../output/playwright/local-improvements/mobile-result.txt), [mobile screenshot](../../../output/playwright/local-improvements/mobile-inspector.png) |
| Null block semantics, repeat HUD close/reopen, existing pan controls | [Camera and close](../../../output/playwright/local-improvements/camera-and-close.txt) |
| Exact sparse values, K/V/layer/head selection, missing coverage, clipboard, keyboard and original-writer rerun | [Numerical case manifest](../../../output/playwright/local-improvements/numerical/README.md) and its per-case artifacts |

The live-follow pause, stable run-identity reset, bookmark tracking, and unknown-history corrections were also inspected in source. Opening local inspection now disables live following as well as playback; workspace resets depend on run identity rather than changing request-order array references. These are source-reviewed behaviors, not a claim of a fresh live GPU qualification.

## Resolved timestamp finding

I found that inherited whole-millisecond rounding collapsed actual EVT170 (`6029.621817 ms`) and EVT171 (`6029.702897 ms`) into the same `6030 ms` coordinate. This made the original fine-time axis claim inaccurate even when zoomed or played at 0.001×.

The repair preserves fractional milliseconds after the existing safe BigInt delta guard, retains six decimal places in event labels, and adds zoom through 4096×. I independently imported current production `buildTimeline`: it retains all 399 observations, starts at `−127.517694 ms`, ends at `6494.679238 ms`, and preserves the `81.08 µs` separation between EVT170 and EVT171. The real-browser [fractional-time regression](../../../output/playwright/local-improvements/fractional-time-result.txt) records 55 px separation at 4096×, separate keyboard-selectable observations, and 0.001× playback advancing from EVT170 to EVT171 after the recorded 150 ms wait. The [screenshot](../../../output/playwright/local-improvements/fractional-time.png) retains the original application shell.

Earlier [rounded-axis observations](../../../output/playwright/local-improvements/recorded-time-result.txt) and [1× playback timing](../../../output/playwright/local-improvements/recorded-clock-result.txt) are historical checks before this repair. Their integer labels and integer extent are superseded by the fractional regression; the 6961 ms wall-time observation includes automation overhead and does not establish GPU execution duration. The additional 200,000-point unit case verifies safe range computation, not rendering performance for a 200,000-mark browser scene.

## Provenance and factual limits

Independent SHA-256 comparisons confirm the existing `gpu-kv-layer-diagnostics` inputs are unchanged:

| Input | SHA-256 |
| --- | --- |
| `manifest.json` | `756367012514cb31b5ad6b4ab9d7eb972bc34b64f0fe85aecbcdb68ddd1ad50a` |
| `events.jsonl` | `c076d11baf7855d5df7a30672029db193f13ca76243810da9c81f94d5067c91a` |
| `content.jsonl` | `c710bfe2137f5d4c423c5caf4762e18fd16184e75fa929a45bd4858fadebbca7` |
| `profiling/activities.json` | `e9a12f209d11962f178bd634529412175befa3aaacfcf5a0b5c27728610ae93c` |

The manifest still requests four output tokens per request. Current production parsing yields 399 observations, 13 iterations, 468 layer cache-write records and 24 numerical slices. I compared the retained [clipboard JSON](../../../output/playwright/local-improvements/numerical/02-copied-values.json) directly with parsed source `snapshot-6`: all dimensions, values and writer/location/boundary provenance agree, including K dimension 0 = `3.96875`. Numerical grouping requires matching operation, writer, layer, physical allocation, datatype and boundary; conflicting or absent coordinates remain unknown.

Current holders, original writers, reserved slots, retained computed contents and null placeholders have distinct labels. Historical numerical snapshots do not become current device-memory measurements as the host clock moves. Native intervals are shared cache-write observations correlated through recorded IDs; the implementation does not turn participant membership into individual token/head duration or full model-layer execution timing. The existing eager diagnostic profiler remains unverified for production timing fidelity.

No new GPU run, six-output recording, complete two-token/all-layer/all-head numerical capture, or live hybrid/MoE execution is established by this change. Static architecture references explicitly identify their lack of execution capture and preserve the loaded model identity. Fresh capture work remains dependent on the new cluster. No Sites publication is part of this integration, and LangSmith was explicitly waived by the user.

## Verification records and failed attempts

Inspected final retained logs show [90 web tests](../../../output/playwright/local-improvements/web-tests.txt), [58 replay tests](../../../output/playwright/local-improvements/replay-tests.txt), [234 Python tests](../../../output/playwright/local-improvements/python-tests.txt), and a successful [TypeScript/Vite production build](../../../output/playwright/local-improvements/build.txt). The existing bundle-size warning remains. New tests cover actual recording boundaries and fractional times, sparse numerical provenance/conflicts, bookmarks before trace gaps, unknown token history, and physical participant generation/slot validation.

The initial native panel's pointer interception is honestly retained in [the failed screenshot](../../../output/playwright/local-improvements/iteration-inspection.png); the ordinary-click regression and repaired screenshot demonstrate the layout fix. Hot-reload selector failures, including [the interrupted clock attempt](../../../output/playwright/local-improvements/clocks-result.txt), are not counted as successful cases. The evidence does not claim exact camera-reset pixel parity, restored mobile focus, new GPU acceptance, or complete numerical coverage.
