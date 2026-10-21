# Iteration disclosure and illustrated KV legend — 2026-10-13

## Scope

User asked what `0 prefill · 1 decode` means and whether progressive wording is clearer, whether symbols would improve the KV key, and to collapse the tall iteration area by default.

- The detail panel now uses Base UI Collapsible, closed by default, with an explicit `Iteration details` trigger. Its compact 40px header retains arrived Intake identities to preserve the preceding user requirement. Details occupy a fixed 112px scrollable panel when open; there is no height animation. Only explicit open/close changes canvas allocation.
- Host labels now say `Prefill inputs: N · Decode inputs: N`, with `scheduled inputs this iteration` beside the iteration heading. Native GPU rows retain their separate packed-input wording. Scheduled counts do not claim exact instantaneous GPU execution.
- The overview and focused KV legends use decorative block illustrations alongside short readable labels. Geometry distinguishes raised/reference, hatched/reusable, recessed/available and null crosses. Request tops/split tops use the existing identity palette, and cyan corner brackets denote selection. Counts and trace-gap uncertainty are preserved. No 3D geometry, color rules, recording or scheduling behavior changed.

## Verified token meaning

All six requests in the bundled H100 BF16 recording have `max_output_tokens: 6`; replay confirms each accepted six outputs. Request 1 has 186 prompt tokens. Its first three iterations schedule 64, 64, and 58 prompt inputs; iteration 3 produces output token 1. Iterations 4–8 each schedule one previously generated input token and produce output tokens 2–6. Iteration 8 schedules input position 190. It is not a total-output counter.

Evidence: `output/playwright/oct13-legend-disclosure/token-evidence.json` preserves the per-request limits and relevant real events. `runner/vllm_session.py:923` forwards this limit to native `max_tokens`; `instrumentation/vllm_030/capture.py:12` classifies input ranges. Official v0.30 documentation defines max_tokens per output sequence: https://docs.vllm.ai/en/v0.30.0/api/vllm/sampling_params/ .

## Runtime verification

Actual running Vite app from `/Users/pliu/.codex/worktrees/kv-layer-inspection/vllm-execution-explorer`, existing branch `codex/kv-layer-inspection`, URL http://127.0.0.1:5173/explore. All browser interaction used Codex in-app browser at DOM-confirmed 1920×1080. Screenshot raster is 1920×1064 due the existing capture API behavior; no resizing was performed. Many unrelated worktree changes exist; task-only `scoped.diff` uses saved before-files.

The real bundled `gpu-h100-bf16` recording was exercised through UI controls without injecting app state:

| Case | Result |
|---|---|
| Collapsed EVT0,174,190,332,363,1155, then backward174 | Details absent, arrived names match Intake, canvas unchanged at 1548×727, x0/y144 |
| Expanded EVT174,190,332,1155, then backward174 | Scheduled counts match production summary; canvas unchanged at 1548×615, x0/y256 |
| Keyboard Enter/Space on trigger | Expands/collapses with correct aria-expanded; aria-controls resolves to panel when expanded |
| Native GPU step20, host EVT174 | Host Intake remains Request1; native rows contain Requests3,2,4; packed-input wording retained |
| Reload after expanding | Details default back to collapsed |
| Overview / focused KV at EVT1155 | All seven sample symbols and visible labels render; focused counts 60 referenced,1 reusable,165 available,1 null match recording |

Evidence directory: `output/playwright/oct13-legend-disclosure/` contains `host-cases.json`, `native-clock.json`, `kv-focused.txt`, screenshots `overview-collapsed.jpg`, `overview-expanded.jpg`, `kv-focused.jpg`, and assessment results. `docs/qualification/oct13-legend-disclosure/assess.mjs` imports production replay, timeline roster, membership and summary helpers, checks the stored browser observations and validates output counts.

## Checks and limits

- Web tests: 108/108 pass.
- TypeScript and Vite production build pass; existing large-bundle warning remains.
- `git diff --check` passes.
- No new live GPU capture was required. The preexisting localhost controller failure remains, while recorded playback works.
- No claim is made to requalify GPU layer timing or every responsive breakpoint; this task checks the affected interactions at the user's requested viewport.

## Final reviews and handoff

Correctness, integration/evidence, and simplification reviewers found no significant code issues. The evidence reviewer independently replayed the six-token limits and output history. A documentation reference was clarified to give the full assessment script path. Final UI inspection at EVT363 displayed Request 1 as `Accepted output 6 / 6 max` with details collapsed; see `final-request-1.txt` and `final-request-1.jpg`. Browser warning/error log query returned no entries. The temporary viewport override was reset and the local tab retained.
