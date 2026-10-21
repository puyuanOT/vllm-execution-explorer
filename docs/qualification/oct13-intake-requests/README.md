# Intake identities and stable scene layout — 2026-10-13

## Requested behavior

Remove “No captured participants at this position”; identify requests in the top-left widget when Intake first shows their cubes; prevent request/iteration text from resizing the 3D view.

## Implementation

- `observedRequestIds` is shared by Intake and the widget. It includes only requests present at the host replay playhead, in the canonical color order. It retains completed observed requests, matching Intake; backward seeking removes future arrivals.
- A compact Intake legend appears before iteration details. The iteration participant list remains sourced from scheduler/native packed rows. In GPU mode the legend explicitly says “Intake · host replay”.
- The removed startup sentence is gone; trace-gap uncertainty remains.
- The widget has a constant 144 CSS-pixel height, independent of request count or layer activity. Overflow scrolls within the keyboard-focusable region. This prevents its content from resizing the canvas/camera. The initial 128-pixel implementation clipped the bottom of a normal GPU batch row; increasing the stable allocation to 144 pixels and tightening padding fixed that before final verification.

## Environment and evidence

- Existing branch/worktree: `codex/kv-layer-inspection`, `/Users/pliu/.codex/worktrees/kv-layer-inspection/vllm-execution-explorer`.
- Actual Vite process PID 57002 points to that worktree's `web` directory; browser URL `http://127.0.0.1:5173/explore`.
- Actual bundled H100 BF16 recording, without replacement or synthetic trace injection.
- Codex in-app browser only. Viewport confirmed by DOM as 1920×1080. Screenshot API raster is 1920×1064 (its existing capture behavior); images were not resized.
- Output directory: `output/playwright/oct13-intake-requests/`. `scoped.diff` compares this task against saved before-files rather than including unrelated dirty worktree changes.

## Per-case results

All 12 host cases passed: EVT0 and 173 (empty), 174 (first dispatch), 189 (before scheduling), 190 (Iteration 1), 363 (completed request), 558 and 559 (new arrivals while previous iteration remains selected), 590 (all six arrivals), 1155 (three-request iteration), and backward seeks to 174 and 173.

At EVT174, both Intake and the new legend show Request 1, while no iteration participants are fabricated. At EVT558, the legend adds Request 3 while iteration details still contain only Request 1 from the last iteration. All six identity colors match the shared production palette.

For every case, the bar bounds were `(0,103,1548,144)` and the canvas bounds were `(0,247,1548,624)`. The projected L1 label also retained identical screen coordinates, supporting stable camera framing. See `browser-cases.json`, screenshots `event-*.jpg`, and `assessment.json`.

Clicking the Request 1 Intake chip at EVT174 opened its inspector, showing “Dispatched”, zero accepted output, and no scheduled range. Evidence: `arrival-selection.txt`.

GPU-clock verification held host replay at EVT174 while selecting native batch `step-20` and advancing 12 recorded activity boundaries. The Intake legend remained Request 1; native participant rows remained Requests 3, 2 and 4. Canvas bounds remained identical, and the normal GPU widget content fit within the fixed allocation. These checkpoints had no mapped active layer; this task does not claim to requalify layer timing. Evidence: `gpu-clock.json`, `gpu-clock.jpg`.

`assess.mjs` imports the production reducer, timeline roster, shared membership helper and request color function, checks browser observations against the real trace, and asserts stable bounds and no removed sentence. Run with `node --experimental-strip-types docs/qualification/oct13-intake-requests/assess.mjs`.

## Checks and limits

- Web tests: 108/108 pass, including real-trace arrival/backward-seek regression coverage.
- TypeScript and Vite production build pass. Existing bundle-size warning remains.
- `git diff --check` passes.
- Live controller is still unreachable (`127.0.0.1:8766 · Failed to fetch`); this change is verified against actual recorded GPU data and requires no live cluster.
- Two browser harness attempts were corrected before evidence capture: AX “checkbox” did not match DOM's Pause button; diffed AX state omitted the slider. Using the DOM button and full AX state resolved both. They were automation failures, not app changes.

## Independent final reviews

Correctness, integration/evidence, and simplification reviewers all found no significant or actionable issues. The correctness reviewer independently ran the real-recording regression test. The evidence reviewer independently compared all 12 browser cases to the production replay. No code changes were requested. Final browser warning/error log query returned no entries; the preexisting controller failure remains visible as described above. Temporary viewport override was reset after testing, and the local tab was retained at EVT174 for inspection.
