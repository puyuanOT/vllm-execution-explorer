# Pool explanation and text cleanup — 2026-10-13

## Request and changes

Removed the visible `Color = current requests, not token occupancy.` sentence and the scene mouse-control instruction line from `ExecutionWorkspace.tsx`. Existing camera controls, breadcrumbs and illustrated legends remain. User's follow-up wording is applied per scheduler participant as two lines: `Tokens scheduled this iteration`, then `Prefill: N · Decode: N`. Native profiler counts retain a distinct `Captured packed inputs this iteration` label because phase classification is not provided by those rows.

This change is limited to two presentation files. `output/playwright/oct13-pool-explanation/scoped.diff` compares them to saved before-files, excluding unrelated worktree changes. No recording, replay or GPU behavior was modified.

## Explanation verified against real data

The screenshot has no event label, but its visible pool pattern matches the actual H100 BF16 recording at EVT1155 / Iteration20. `analyze-shared-blocks.mjs` imports production replay and KV projection, reads real prompt token IDs, and asserts the shared IDs and prefix matches. Run from this worktree with:

`node --experimental-strip-types docs/qualification/oct13-pool-explanation/analyze-shared-blocks.mjs`

The physical grid is ordered by block IDs (`CacheStage.tsx:93`); request block tables supply logical token order. Requests 2 and4 have tables `[1,2,3,4,5,6,7,8,9,10,59,60]` and `[1,2,3,4,5,6,7,8,9,10,59,61]`. Thus shared physical blocks1–10 and59 represent contiguous logical positions0–175, despite physical ID gaps. Bands encode two current references to the same KV block, not split token ownership.

Request 1 and Requests 2/4 share 175 prompt tokens, then differ at zero-based position175 (` Name` versus ` Explain`). This recording reuses 10 complete16-token blocks (positions0–159). Cached physical B11 retains Request 1's different positions160–175 and has zero current references. Requests 2 and4 match 177 prompt tokens, permitting 11 complete blocks; their next shared block at logical positions160–175 is physical B59. Private continuations use B60 for Request 2 and B61 for Request 4. Green B12–58 belongs to Request 3.

B59 is hash-published and reused during the same scheduling pass (1059–1071), then its write is observed after the Iteration20 result (1151). The explanation must not claim it was already computed in a previous iteration. `shared-block-audit.json` preserves the tables, bindings, contents, prompt comparisons and event order.

Primary source checks: official v0.30 prefix-caching design describes preceding-context hashes and request-to-physical-block mappings (https://docs.vllm.ai/en/v0.30.0/design/prefix_caching/). Published `FullAttentionManager.find_longest_cache_hit` explicitly scans a contiguous logical prefix; `BlockPool.get_new_blocks` draws physical entries from the free queue, rather than requiring neighboring IDs (https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/single_type_kv_cache_manager.py and corresponding block_pool.py). The exact captured tables establish this demo's behavior.

## Integrated checks

Actual Vite app in the existing `/Users/pliu/.codex/worktrees/kv-layer-inspection/vllm-execution-explorer` worktree. Codex in-app browser at DOM-confirmed 1920×1080; the capture API produces 1920×1064 rasters. No injected app state or image resizing.

- EVT332 / Iteration8: expanded details display exactly `Tokens scheduled this iteration` and `Prefill: 0 · Decode: 1`. Inspected screenshot and text (`iteration-8.jpg`, `iteration-8.txt`).
- EVT1155 focused pool: both requested visible explanations are absent; all legend symbols, counts and camera controls remain (`pool.jpg`, `browser-receipt.json`).
- Real UI block finder selected physical 59: inspector reports Request 2 and4 as current holders, Request 2 as original writer, and all 16 computed slots mapped to positions160–175. Slot15 shows `Explain` (`block-59.txt`, `block-59.jpg`).
- Native GPU batch20: rows show captured 36,27,1 inputs without fabricated prefill/decode phases (`native-wording.txt`).
- Browser warning/error log query returned no entries. The preexisting localhost controller failure remains while recorded playback works.

Web tests 108/108 pass; TypeScript/Vite build passes with the existing large-bundle warning; `git diff --check` passes. No new tests mirror these visual changes. Initial browser tab1 had expired; the current tab2 was discovered and bound before interaction.

## Final reviews

Correctness, integration/evidence, and simplification reviewers found no significant or actionable issues. The first two independently replayed the real recording to verify the shared IDs, request tables, B11's state and B59's event ordering. Final cleaned pool screenshot is `final-pool.jpg`. The local tab was retained at EVT1155 in focused KV view and the temporary viewport override reset.
