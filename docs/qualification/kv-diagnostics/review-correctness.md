# Independent correctness review

Reviewed the uncommitted feature against `origin/codex/arrival-diamonds` on 2026-10-06. Scope: native cache metadata and update hooks, capture window and limits, profiler normalization, replay evidence parsing and allocation identity, targeted rerun restoration, and linked UI plumbing. No implementation files were edited by this reviewer.

## Outcome

No remaining significant correctness findings after the targeted rerun fix below. This is a local code/contract review, not live GPU qualification; the new CUDA cache-write and numerical capture path still requires the documented GPU release gate.

## Resolved finding

**P2: a targeted rerun retained an unrelated tokenizer revision when the recording selected automatic revision resolution.** `web/src/scenario/tensorRerun.ts` replaced the retained tokenizer locator but assigned `tokenizerRevision` only for a string revision. A recorded `{ locator: "Qwen/model", revision: null }` combined with the current draft's `tokenizerRevision: "old-other-commit"` returned that unrelated commit, potentially failing or changing the tokenizer during a rerun. The implementation now restores the locator and sets the revision to its retained string or `""`. The existing restoration test covers a stale unrelated revision and verifies the serialized revision becomes `null`. Rechecked the fix and regression: passed.

## Checks and evidence boundaries

- Independently ran `PYTHONPATH=instrumentation /Users/pliu/Documents/vllm-execution-explorer/.venv/bin/python -m pytest -q tests/test_kv_diagnostics.py`: **13 passed**.
- Independently ran `/Users/pliu/.local/bin/node --experimental-strip-types --test replay/src/kv.test.ts web/src/scenario/diagnosticConfiguration.test.ts`: **11 passed**.
- Checked the actual [vLLM 0.30 FlashAttention implementation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/flash_attn.py) and [attention dispatch](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/attention/attention.py). The recorded rank-four layout, native update arguments, and same-stream post-update snapshot boundary agree with the selected implementation. Source inspection and authored tensor tests do not establish that the hooks ran successfully on the inaccessible GPU host.
- The replay contract rejects mismatched operation/step/layer CUDA references and mismatched packed rows or allocation generations. Cache-write completion comes from retained correlated CUDA activity ends, not the CPU update return. Missing launches, unsupported numerical layouts, negative native slot mappings, and missing old attachments remain explicit.
- Shared-prefix numerical inspection resolves the exact pool/group/block/generation/slot and retains the original writer and capture step. Layer/head/K/V views share the physical allocation rather than increasing capacity.
- Diagnostic window selection waits for both the earliest step and optional request arrival; numerical snapshots are opt-in, bounded and cloned before reuse. The rerun announcement correctly warns that later decode positions may need a later capture window.

The historical real GPU recordings qualify only their retained host events and pre-existing layer/profile evidence. The authored diagnostic recording exercises the new contract and UI but contributes no real GPU measurements or tensor values.

## Compact tensor navigation follow-up

Reassessed the subsequent three-file navigation fix in `App.tsx`, `CacheTensorStage.tsx`, and `ExecutionWorkspace.tsx`. Changes to a layer/head/K/V slice for the same selected request and token no longer automatically reopen the compact inspector. Selecting a different request/token or another object still follows the existing inspector behavior. The explicit “Inspect token and captured values” action updates the shared slice selection and invokes the inspector-opening callback separately. The callback remains optional because the desktop inspector is already visible. No additional correctness findings from this change. Inspected `mobile-tensor-controls-dom.txt` and `mobile-tensor-controls.jpg`: the selected V slice remains in the compact scene with the controls available and no inspector dialog. Inspected `mobile-tensor-inspect-action-dom.txt`: the explicit action opens the inspector with the same V/head-0/layer-1 slice and historical synthetic V values. The parent browser evaluation additionally confirms that Escape restores focus to the Inspect button, as recorded in the qualification README. This regression passed locally; it adds no GPU qualification.

## Camera remount follow-up

Reassessed the one-line `CameraRig.lastCommand` initialization in `web/src/scene.tsx`. Initializing the ref from the existing command ID prevents a newly mounted Canvas from replaying an old pan after its layout effect centers the scene. Subsequent incremented command IDs still execute the unchanged command branch. No new correctness findings. Inspected `camera-remount-after.jpg`, `camera-remount-after-dom.txt` and `camera-regression.json`: the remounted scene center is 454 px; a new PanLeft changes it to 485.2 px; Reset restores 454 px. The local browser regression passed and concerns scene navigation only.
