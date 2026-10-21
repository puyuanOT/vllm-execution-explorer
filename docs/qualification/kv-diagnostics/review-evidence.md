# Independent integration and evidence review

Reviewed on 2026-10-06 against `origin/codex/arrival-diamonds`, including the uncommitted implementation, retained browser evidence, authored recording, and historical GPU recordings. This reviewer did not author or modify the implementation.

**Outcome: no significant unsupported claim or local integration/evidence defect found. Fresh CUDA runtime qualification remains blocked and is required before leaving draft.** The completed local evidence supports recording/replay/UI behavior; it does not establish that the new native cache-update hooks or snapshot path work on a live vLLM GPU.

## Checks performed independently

- Compared every bundled event/content/profile/raw file in `gpu-model-batching` and `gpu-prefix-reuse` byte-for-byte with its source in `docs/qualification/control-plane-review/a100-captures.tar.gz`. All matched. The archive SHA256 is `8c5b2c7c5a829f6668e30eec193cafdcbcb0089623fe6490a501b9bfd6b3fbaa`. Only the `lesson` and `lesson_source` manifest keys differ. Both lessons explicitly disclose absent new cache-write scopes and numerical snapshots.
- Inspected production call plumbing from session initialization through request-admission/window triggers, native hook installation, profiler finalization, manifest retention and replay parsing. The authored generator/validator invokes current production modules rather than a copied replay or normalization implementation.
- Ran `node --experimental-strip-types docs/qualification/kv-diagnostics/validate_authored_fixture.mjs`: passed, 36 events, six authored operations, eight authored snapshots, `real_gpu=false`.
- Ran `PYTHONPATH=instrumentation python -m pytest tests/test_kv_diagnostics.py -q` with the project virtual environment: 13 passed. These fake-tensor tests cover storage alias accounting, effective quantization metadata, clone ownership before reuse, manager/kernel addressing, bounds, skipped/invalid slots, native-scope versus snapshot-kernel correlation and delayed capture triggers. They are not real CUDA tests.
- Invoked current `runner.export.export_structural` on the authored recording in a temporary directory. Output contained only `events.jsonl` and `manifest.json`; content/profile/raw files were absent, profile attachment status was `omitted_by_structural_export`, and static KV configuration remained available. Replay tests also prevent a retained in-memory attachment from bypassing that omission.
- Inspected screenshots and DOM artifacts for authored active/completed operations, exact shared-allocation snapshots, linked slices, targeted rerun, mobile inspector and real mixed step9. Synthetic timing, placement and numbers remain visibly labeled. The real step9 matrix distinguishes 1 decode input and 16 prefill inputs from bounded prior KV positions. Its host scheduling does not claim measured cache-write completion.

## Evidence boundaries

The cache matrix uses its independent profiler clock for cache-update chronology. CPU return and host result observations are not substituted for GPU completion. Missing old scopes, incomplete correlation, skipped slots, unsupported layout addressing and trace gaps retain explicit unknown/unavailable states. Multiple token positions may share one operation interval; the implementation and evidence do not assign individual token or head durations.

Numerical previews select immutable recording-time slices using pool/group/block/generation/slot identity. The shared-prefix example preserves the original writer while selecting a current reader. Missing slices prepare capture settings; they are not reconstructed. The toy fixture restores a reference Qwen workload for a future rerun, and its README explicitly says that this will not reproduce the invented tensor numbers or toy architecture.

Scene evidence reports synchronous CPU submission/resource counts only. It preserves initial-transition costs and isolated paused frames caused by polling; it does not establish GPU duration, presentation frame rate, memory bytes or broad performance stability.

The root's complete suite/build results are summarized in the qualification README. This reviewer independently reran the authored validator and the 13 diagnostic Python tests rather than repeating the complete suite. Browser journey artifacts were inspected; this reviewer did not repeat the primary agent's browser actions.

## Unresolved release gate

SSH to the authorized GPU host `144.24.39.189` is reported to time out. Neither the unchanged historical A100 captures nor the authored fixture exercises the new runtime instrumentation. Actual effective tensor views, real CUDA runtime-flow correlation, stream-ordered slot/snapshot cloning and readback, selected values/bounds/overhead, failure retention and imported/exported new-schema replay still require a fresh qualified GPU capture. Keep the PR in draft until that evidence and its final independent review are complete.
