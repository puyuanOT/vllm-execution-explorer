# KV and layer inspection qualification

This change adds configuration-first KV inspection, explicit physical block labels and allocation-state colors, linked token/layer/head/K–V slice views, whole-batch model participation, and bounded eager diagnostic capture settings for vLLM 0.30.0. Numerical snapshots are opt-in. Missing values remain unavailable and can prepare a targeted rerun from the retained workload and inference settings.

**Fresh runtime qualification is pending.** SSH to the authorized GPU instance at `144.24.39.189` timed out repeatedly, including a final bounded connection probe on 2026-10-06 (exit255). The new CUDA cache-write hooks, effective tensor metadata, and numerical snapshot/readback path have not run against a live GPU during this change. This is a draft-PR release gate, not a passed test. LangSmith was explicitly waived by the user for this project.

## Environments and evidence

- Implementation branch: `codex/kv-layer-inspection`, based on `origin/codex/arrival-diamonds` (`df59936fda1f5b6b94070c2ca0a9526b7e15042d`, PR7).
- Local controller: production `runner.serve` at port8766 with this worktree's Python sources. The integrated fixture capture completed as `run-fcf034caeade`; it exercised the controller, recording writer and browser replay, without running vLLM or CUDA.
- Browser: current Vite application at port5181; separate scene instrumentation preview at5182. Default desktop viewport1280×720; compact inspector at390×844. Browser observations and screenshots are retained below.
- Historical A100 captures: bundled `gpu-model-batching` and `gpu-prefix-reuse` use the original event/content/profile files from `docs/qualification/control-plane-review/a100-captures.tar.gz`. Manifest provenance records the source run and archive SHA256. They demonstrate actual old-runtime scheduler/KV/packed-batch observations; neither contains the new per-layer cache-write scopes or numerical snapshots.
- New optional contract: `authored-recording/` is deliberately synthetic, including all timing, device, architecture and numerical values. The production recording writer, normalizer, replay parser and frontend rerun restoration validate it. See its README for exact boundaries and reproduction commands. It is excluded from the production demo catalog.

## Local checks

| Check | Outcome | Evidence |
| --- | --- | --- |
| Python suite | 227 passed | `python-tests.log` |
| Replay suite | 58 passed | `replay-tests.log` |
| Web scenario suite | 73 passed | `web-tests.log` |
| Python compilation | Passed | `python-compile.log` |
| TypeScript check and Vite build | Passed; existing large-bundle warning remains | `build.log` |
| Whitespace/diff check | Passed | `diff-check.log` |
| Authored recording contract validation | Passed; real_gpu=false | `authored-validation.json` |

The 358 tests cover strict optional profile parsing, correlated CUDA endpoints versus CPU returns, seek behavior, unsupported layouts, bounded selection and bytes, exact allocation generation, original writer versus current holder, structural export omissions, trace gaps, workload/config restoration, and existing scheduler/arrival behavior. Unit tests do not establish live CUDA integration.

## Browser cases and assessed outcomes

| Case and success criterion | Observed outcome | Evidence |
| --- | --- | --- |
| Configure diagnostic window, request trigger and custom numeric selection; deleted request stays explicit and undo restores it | Passed. Live CUDA capture is correctly blocked on this Mac | `configuration-dom.txt` |
| Run actual integrated controller fixture and open its recording | Passed. Capture completed; full scripted pool replayed; unavailable device/dtype stayed explicit | `live-fixture-dom.txt`, `live-fixture-cache.jpg` |
| Open default real GPU lesson and inspect the full pool | Passed. Actual113-block inventory; context limit and physical usable capacity presented separately; historical metadata omissions explicit | `mobile-inspector-dom.txt`, `mobile-inspector.jpg` |
| Inspect real mixed prefill/decode batch | Passed at step9: long request1 decode input and short request16 prefill inputs. Matrix labels scheduled versus bounded prior positions separately; selected request token drill and Whole batch reset passed | `real-model-batch-dom.txt`, `real-model-batch.jpg` |
| Use independent native profiler clock on historical GPU capture | Passed locally. Host event stays independent. Existing correlated layer activity is retained; per-layer cache-write state remains unknown because old scopes are absent | Historical profile files and replay tests |
| Import authored optional profile schema; compare CPU return and GPU write interval | Passed. At125µs layer1 write is active and layer2 scheduled; at165µs layer1 completed and layer2 active. No CPU return or per-token duration is presented as completion | `authored-layer-active-dom.txt`, `authored-layer-active.jpg`, `authored-layer-completed-dom.txt` |
| Inspect shared prefix by current holder and original writer | Passed. Reader token0 resolves writer token0, block0/generation1/slot0, float16/cuda:0 and immutable toy K/V values. Allocation counts do not grow when drilling into layer/head slices | `authored-shared-prefix-tensor-dom.txt`, `authored-shared-prefix-tensor.jpg` |
| Change slice in3D controls and inspector; select an uncaptured layer | Passed. Shared layer/head/K–V selection updates both views; missing values stay unavailable | `linked-slice-selection-dom.txt` |
| Prepare targeted rerun for missing shared-prefix slice | Passed. Original writer selected; retained workload and model/inference settings restored; diagnostics open and focused, window begins at admission. Later decode positions explicitly require adjusting the window | `targeted-rerun-dom.txt`, `targeted-rerun.jpg` |
| Compact inspector, keyboard dismissal, focus and horizontal overflow | Passed at390×844. Escape closes dialog and restores Open inspector focus; document width390 | `mobile-inspector-dom.txt`, `mobile-inspector.jpg` |
| Compact linked slice controls and explicit inspector opening | Passed after repair. Three slice controls update without a modal; explicit Inspect shows the selected V numerical example, and Escape restores that button's focus. Width stays390 | `mobile-tensor-controls-dom.txt`, `mobile-tensor-controls.jpg`, `mobile-tensor-inspect-action-dom.txt` |
| Keyboard camera pan/reset | Passed through visible controls | Browser interaction; scene tests |
| Camera after request-lanes → 3D remount | Passed after repair. Existing pan is not replayed on remount (anchor center454px); subsequent new pan moves it485.2px and Reset restores454px | `camera-regression.json`, `camera-remount-before.jpg`, `camera-remount-after.jpg`, `camera-remount-after-dom.txt` |
| Paused demand-render behavior on full physical pool | Observed113-block pool rendered in10draw calls; retained CPU render submissions commonly0.2–0.4ms after initial transitions. Paused loop is not continuous; controller polling causes isolated frames around2.5s apart | `scene-profile-desktop.json` |
| Mount/unmount selected tensor slices three times | Passed for these transitions. Two-block pool settles at7draw calls/geometries; selected tensor view settles at38. Geometry counts return to7 on each unmount. Transition CPU submission peaks5.3ms; later isolated slice frames0.3–0.5ms | `slice-mount-cycles-dom.txt`, `scene-profile-slices.json` |
| Browser console | No captured errors or warnings in either local preview | `browser-console-errors.json` |

Scene metrics measure synchronous CPU submission and R3F callback-to-render end, not GPU completion, browser presentation, end-to-end frame rate, or memory bytes. Initial transitions cost more and remain in the evidence. They are observations, not a performance guarantee.

## Repairs found during evaluation

- Tensor drill initially covered the scene center with controls and long explanatory labels. Controls now sit below the canvas; the selected layer/head/K–V view is framed when entering the drill, and explanations remain in the inspector.
- Historical recordings were initially labeled as lacking all GPU layer timing while using host playback. They now distinguish available Profile mode timing from absent cache-write evidence.
- Historical vLLM module names end with `.self_attn.attn`; prefix-KV linkage previously only recognized `.self_attn`. Both exact names now link only declared layers. A regression test preserves unknown state for undeclared layers.
- The matrix initially combined prior KV positions and new scheduled inputs into a single count. Columns now state scheduled counts separately from prior positions, with a bounded-matrix notice.
- Targeted rerun initially reused an old absolute step ordinal. It now starts at the selected original writer's admission; later decode positions require an explicit window adjustment because scheduling may change between captures.
- Independent correctness review found that a retained automatic tokenizer revision could leave an unrelated old draft commit in place. Restoring the tokenizer locator now clears that revision; regression verifies the submitted revision is null.
- Compact slice changes initially reopened the inspector on every layer/head/K–V edit. Same-token slice updates now stay in the scene. Explicit Inspect opens the dialog, and Escape restores its focus; the browser regression passed.
- The last camera pan could replay when Canvas remounted after switching request lanes/3D or refreshing during development. The new camera acknowledges the already-issued command on mount, then executes only subsequent command IDs. Before/after screenshots and a subsequent-pan/reset regression verify the fix. The earlier `authored-slice-final.jpg` retains the shifted pre-repair state; `camera-remount-after.jpg` is the final framing.

## Remaining GPU release gate

Use an accessible CUDA host with the pinned vLLM0.30.0 dense-Qwen3 eager runtime to record a small real partial-prefix/mixed-prefill-decode workload. Verify effective cache shape/dtype/device/storage alias accounting against the actual backend, actual slot mappings, CUDA runtime-flow cache-write correlation, stream-ordered snapshot content, selected tensor bounds/bytes and overhead. Inspect its raw trace, normalized attachments and exported/imported replay. Replace or supplement the historical lessons with that real new-schema recording. Do not promote this PR out of draft until this gate passes.

Three independent reviews are retained as `review-correctness.md`, `review-evidence.md` and `review-simplification.md` once complete. Review outcomes do not waive the live-GPU gate.
