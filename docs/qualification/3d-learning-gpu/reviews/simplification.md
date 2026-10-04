# Final independent simplification review

Reviewer: `final_simplification_review`, an agent that did not author the implementation. Reviewed on 2026-10-03 after GPU qualification and browser verification completed.

Scope: implementation diff from `6918efa` through `8f0ca92d2ee2741723ba20e5637decb91982694b`, plus the finalized qualification documentation and retained archives. Distributed capture fingerprint: `83df6c7bc792273eabb37d22ab76a29476b6c69a9325848505deafa1201e5056` over 38 files.

## Outcome

The original P2 finding below is **closed after independent repair review and runtime reassessment** at `dcab49f6ff0e194711c031f4d72399b93f300877`. No substantiated significant issue remains. No other significant simplification or maintainability issue was found in the reviewed implementation. No implementation, qualification helper, or GPU service was changed by this reviewer.

## Finding

### Closed P2 — Clear the owned process handle before releasing admission

Location: `runner/isolation.py:195–197`, with new admission at `runner/isolation.py:128–139` and shutdown handling at `runner/isolation.py:226–234`.

`_finalize_exit` sets `state.execution_process_alive = False` inside `state.lock`, releases that lock, and then assigns `self._process = None`. `start` can admit the next capture once that gate is false and install the next process handle. If the old supervisor resumes after this admission, its unconditional assignment clears the **new** worker handle. `close()` subsequently cannot terminate that owned worker after its shutdown grace period because it reads `None` from `self._process`; it instead reaches the still-alive supervisor error path. The timeout supervisor still holds its own process reference, but that does not preserve bounded controller shutdown.

The interleaving was reproduced locally using the actual `ExperimentIsolation._finalize_exit` and `ExperimentIsolation.start` methods. A reentrant lock wrapper paused the old supervisor immediately after releasing the final state lock; a fake process factory prevented process/GPU creation, and the new supervisor was held before starting its process. This scheduling pause models a valid thread switch between lines 196 and 197. The observed state was:

```text
after admission: {'new_handle_retained': True, 'readiness_gate': True}
after previous supervisor finishes: {'new_handle_retained': False, 'handle': None, 'readiness_gate': True}
```

Repair the ownership transition atomically: clear only the old owned handle before releasing the admission gate, with a process identity guard and/or consistent locking shared with `start`. Preserve the existing lock acquisition order if using the isolation lock. Add an interleaving regression that admits the next capture at this boundary and verifies its handle survives and remains available for bounded shutdown. No broad lifecycle rewrite is needed.

The paragraphs above retain the original `8f0ca92` finding and reproduction. The repair and current verdict are recorded below.

## Independent repair reassessment

Reviewed the complete `8f0ca92..dcab49f` repair diff, which changes only `runner/isolation.py` and `tests/test_isolation.py`. The capture source now fingerprints to `c48b0fd1c8eccdc0b0d1d1d37e08b02f854a6e8bfd13531f81bf116bc524704d` over 38 files; the reviewer recomputed that value from the working tree and matched it to the deployment and successful capture metadata.

The old handle is now cleared only when `self._process is process`, before releasing the readiness gate, while holding the same lifecycle mutex that protects new admission. Lock ordering remains lifecycle mutex before state lock. This removes the demonstrated interleaving without a new lifecycle abstraction. The related independently reported concurrent reaping issue is addressed by waiting on the OS sentinel outside the mutex and serializing every `Process` operation under that mutex; sentinel waiting cannot reap the child or compete with publication of its exit code. The separate termination mutex has been removed.

The reviewer independently ran all **17 isolation tests** on `dcab49f`: passed. The reviewer also loaded the exact `8f0ca92` isolation source in memory, without changing repository files, and ran the two new regressions against it. Both failed at the intended boundary assertions: the next owned handle became `None`, and shutdown could acquire the lifecycle mutex while the supervisor was between native reaping and exit-code publication. These are meaningful behavior regressions, not assertions that merely mirror the repair.

After the final runtime cue, the reviewer read the complete local raw evidence under `recordings/qualification-3d-live/lifecycle-dcab/`, the deployment/validation receipts and the new per-case reports. The observed cases were:

| Case | Independently inspected retained outcome |
| --- | --- |
| Staggered `run-62f5572c5570` | Ordinary V2 serving, six requests with 128 accepted output tokens each and explicit length terminals; retained prompts/output prose and IDs match the case. Generated prose is truncated workload output, not qualified instructional ground truth. Worker 17247 exited with code zero. |
| Cancellation `run-b61c3a49fa0d` | Diagnostic V1 eager execution retains three returned steps, zero accepted outputs and actual `FINISHED_ABORTED` at the returned native abort boundary. The native profile remains partial; raw and normalized attachment sizes/hashes match. Worker 17502 exited with code zero. |
| Preparation failure `run-b39b7d48dcca` | A one-byte KV budget fails before admission, with zero scheduler events and no inference output. The expected failed outcome is retained, rather than counted as inference success. This case has no initialized trace source metadata; its deployed source is supported by the separate exact deployment/helper receipt. Worker 17772 exited with code zero. |
| Recovery and late intake `run-0b38dea71a00` | The next run accepts the immediate late request and the planned request after ten seconds. Raw output text is `admitted` and `scheduled`, with three and two output token IDs respectively, including the terminal IDs. Both requests have native `stop` terminals. Worker 17987 exited with code zero. |

All four post-close NVIDIA receipts report **3 MiB** and no compute applications, after finalized readiness with distinct owned worker PIDs. Effective runtime settings match ordinary versus diagnostic selection. The deployment receipt preserves the original controller as alive. The validation receipt retains the root's complete 214-test pass and exact-source CI pass; these are distinct from this reviewer's own 17-test rerun.

The narrow repair does not change instrumentation, replay, UI or qualification helpers. Their earlier `8f0ca92` evidence remains separately scoped; this reassessment does not relabel the earlier seven-case matrix as executions at `dcab49f`. No new significant maintainability or simplification issue was found, and the demonstrated owned-handle finding is closed.

## Review checks

- Inspected worker ownership, supervision, interrupted evidence recovery, shared state transactions, configuration/revision resolution, observability, instrumentation, profiling normalization, replay projection and the added 3D/navigation/playback modules.
- Checked that the process isolation and file transaction lock solve distinct observed failures; they are justified by retained memory and dead-lock regressions, rather than unnecessary abstractions.
- Checked that separate native profiling/replay paths preserve distinct clock and evidence contracts; shared geometry, slot mapping and picking helpers avoid duplicating interpretation across controls and the scene.
- Inspected the added test cases for distinct behavior, including failure during a state transaction, interrupted UTF-8, startup before recording initialization, exact cache-layout qualification, post-reorder row identity, interval overlaps and final sampled tokens. No substantial redundant test set was identified.
- Independently reran `tests/test_isolation.py`, `tests/test_profiling.py`, and `tests/test_observability.py`: **30 passed**.
- Independently reran `replay/src/learning.test.ts` and `replay/src/profiling.test.ts`: **28 passed**.
- Verified all three `SHA256SUMS` archive entries. Read all 11 final-source recording manifests and their actual event/content streams from the final archive, including the expected preparation failure. The ten inference captures have distinct owned worker PIDs and exit code zero. The failure has no observed scheduler events; cancellation retains one native cancellation and a partial three-step profile; the dedicated pressure case retains three preemptions.
- Verified both final native profiles' raw and normalized member byte sizes and SHA-256 hashes against their manifests. Read finalized browser observations and preserved screenshots/evidence references.

## Limits

This review inspected retained GPU/browser evidence; it did not launch inference or repeat browser interaction. The original interleaving reproduction establishes the controller ownership race, not its frequency in live GPU use; the new forced regressions establish its repaired boundary behavior. Runtime reassessment covers the affected lifetime paths, while broader features retain their earlier source-specific qualification. Single-GPU dense Qwen3/eager diagnostic boundaries, absent per-token/head timings, separate clocks and the existing bundle-size warning remain accurately documented. Broader model/layout support or browser GPU memory performance is not established by this review.
