# Local application integration, 2026-10-09

The agreed inspector/playback changes are integrated into the existing application served by `http://127.0.0.1:5173/`, using the `codex/kv-layer-inspection` worktree (base `92fb11b00d889cb58136a987bcf09687c0381acd`). The listener's working directory is this checkout's `web` directory. Configure, Explore, the original techno shell, editable workloads, controller/capture setup, library/import, request lanes, and targeted reruns remain in place. No Sites deployment or new GPU capture was performed.

The browser evaluation uses Playwright CLI in an isolated browser session, with the actual Vite frontend, reachable local controller, and bundled real recordings. It does not alter the user's browser draft. Numerical and native GPU views invoke the existing production replay/profiling projections rather than a copied export from the Sites prototype. Existing GPU artifacts are unchanged.

The demo remains `run-a413ca908152`: 399 observations, 13 iterations, 468 native layer cache writes, and 24 sparse numerical slices. It still requests four output tokens. Two-token/full-head capture is an editable Configure selection for a future run; six-output recapture and broader numerical evidence require the new cluster. Historical snapshots, native profiler time, and host replay boundaries stay distinct. LangSmith was explicitly waived by the user; these UI checks invoke no new inference.

## Outcomes

| Case | Success criteria and observed outcome | Evidence |
| --- | --- | --- |
| Original navigation and editable draft | Configure/Explore, workload examples, host/model/engine configuration, preflight, library, lanes and Edit as new sequence remain. Opening the demo preserved the isolated 3-batch/6-request draft. | Browser snapshots; `configure-result.txt` |
| Optional broader capture selection | Button fills positions 0,1; all 36 layers; heads 0–7; 128 dimensions; 1 MiB limit. Fields remain editable and no capture starts. | `configure-result.txt` |
| Iteration and raw stepping | From zero, iterations end at 30,53,76,93,110,140; Previous from 140 reaches 110; Raw Next reaches 111. Result-associated release/output transitions are included without removing raw observer order. | Root CLI interaction; production-data `iterationPlayback.test.ts` |
| Timestamp axis and zoom | Final axis retains fractional milliseconds: preparation begins at −127.517694 ms and EVT399 is 6494.679238 ms. EVT170/171 are 6029.621817/6029.702897 ms, 81.08 µs apart. They are 55 px apart at 4096× and separately selectable; 0.001× playback advances between them. All 399 point observations remain. | `fractional-time-result.txt`, `fractional-time.png` |
| Recorded clock | Before the precision repair, 1× browser replay took 6961 ms including automation overhead, consistent with the then-rounded 6623 ms extent, and reached EVT399. After repair, the sub-ms slow-play regression reaches EVT171 from EVT170; the exact source extent is 6622.196932 ms. | `recorded-clock-result.txt`, `fractional-time-result.txt` |
| Native iteration inspection | Before the first step, Inspect is disabled. At host EVT76, layer 19 selection, microsecond zoom, cursor keyboard End and participant selection work without seeking the host. Zoom window is 10.752 µs. | `iteration-result.txt`, `iteration-inspection-fixed.png` |
| Native participant reselection | Select token, select physical block 2 elsewhere, then reselect the same participant: picker resets to empty and selects token again. | `bookmarks-and-reselection.txt` |
| Optional learning bookmarks | Five actual boundaries jump to 30/88/128/178/238. Reserved decode shows write unobserved; released prefix retains original writer; reused token separates reader holder from prime writer; mixed step records 1 decode plus 63 prefill positions. | `bookmarks-and-reselection.txt` |
| Bookmark tracking regression | Choose the prime decode token, jump to the reader's reused token, then choose model Layer19. Matrix keeps following reader position0 rather than the previously tracked prime token. | `bookmark-tracking-result.txt` |
| Global and local profile round trips | Local mixed iteration contains 64 participants (reader position178 and mixed positions46–108). Global Profile closes local inspection without changing host EVT238; lanes and Edit as new sequence work. Clicking a global Layer19 participant retains Layer19/block9/slot0/position128. | `final-integration.txt`, `global-participant-result.txt` |
| Unverified import provenance | Import a copy of the manifest with execution/provenance declarations omitted, retaining the original events/content/profile files. Local native inspection remains available but the scene says “source unverified”; it does not label the import as a verified GPU run. | `unverified-provenance-result.txt`, `unverified/manifest.json` |
| Static hybrid references | Expandable architecture blueprints explicitly say no execution capture and preserve the recorded model identity. | `bookmarks-and-reselection.txt` |
| Missing profile and import recovery | Invalid import reports missing manifest/events; opening the historical partial-prefix recording recovers. Global Profile is disabled; iteration inspection explicitly reports no correlated native writes rather than inventing timings. | `recovery-result.txt` |
| Mobile inspection and Configure | At 390×844, the document stays 390 px wide. Native layer zoom is clickable after closing the existing layer inspector; selecting a participant opens the token inspector, Escape closes it, and Configure remains reachable without page overflow. | `mobile-result.txt`, `mobile-inspector.png` |
| Null placeholder and slot close | Block 0 shows no holders/writers and no request token slots. Block 1's slot HUD closes, reopens, and closes a second time with ordinary mouse clicks. | `camera-and-close.txt` |
| Existing camera controls | Button pan, right-drag pan and Control-left-drag pan visibly change the canvas; panning keeps the slot HUD closed, and Reset remains clickable. | `camera-and-close.txt` |
| Numerical data, unknown coverage, copy, keyboard, rerun | Real values, K/V/layer/head links, conflict/unknown cells, clipboard provenance, original-writer prefix reuse and partial-dimension rerun pass. | [Numerical cases](../../../output/playwright/local-improvements/numerical/README.md) |

The first native-panel desktop test found canvas pointer interception over the panel controls caused by flex shrinking. The layout was corrected by retaining nonshrinking scene/panel siblings within the existing workspace scroll owner. The same layer-19 zoom and Close controls then worked using ordinary mouse interaction; no force clicks were used. `iteration-inspection.png` retains the failed layout and `iteration-inspection-fixed.png` shows the successful regression. During implementation, hot reload also reset two automation selectors; those attempts were rerun after a stable checkpoint and are not counted as successful UI cases.

Independent review found inherited whole-millisecond conversion in the production timeline. It is now removed after the existing safe BigInt delta check. The original `recorded-time-result.txt`/`recorded-time.png` show the superseded rounded behavior and are retained as failure evidence, not final precision evidence. Final fractional-time results are listed above. A 200,000-point unit case verifies range computation without exceeding JavaScript argument limits; it does not qualify rendering 200,000 marks in the browser.

Review also repaired global participant layer/state wiring, bookmark tracking, and local inspection's live-follow/reset state paths. Opening inspection disables follow-live as well as playback; reset effects use run identity instead of freshly rebuilt request-order array identity. These live-stream state fixes have independent source review; a new GPU live capture has not been run.

## Checks

- 90 web tests passed: `npm test --prefix web`.
- 58 replay tests passed: `npm test --prefix replay`.
- 234 Python tests passed using the existing main checkout's Python environment against this worktree: `/Users/pliu/Documents/vllm-execution-explorer/.venv/bin/python -m pytest -q`.
- `npm run build --prefix web` passed (TypeScript + Vite; existing bundle-size warning retained).
- `git diff --check` passed.

Logs, CLI interaction outputs and images are under [output/playwright/local-improvements](../../../output/playwright/local-improvements/). New tests protect actual-recording iteration boundaries, sparse numerical provenance/conflicts, semantic bookmark/gap boundaries, and current-versus-historical native participant mappings.

Remaining GPU work is explicitly unverified: a fresh six-output recording, complete two-token/all-layer/all-head numerical snapshots, and live hybrid/MoE runtime instrumentation. Static model-card references do not establish those execution observations.

## Independent review

All three reviewers inspected the actual additive integration and retained browser evidence. Significant findings were repaired and reassessed:

- [Correctness](review-correctness.md): linked selection, unknown provenance and live-follow/reset repairs; no significant unresolved finding.
- [Integration and evidence](review-evidence.md): fractional timestamp precision, unchanged source artifacts and realistic local browser outcomes; no significant unresolved finding.
- [Simplification](review-simplification.md): production projections and narrowly scoped helpers/tests retained.
