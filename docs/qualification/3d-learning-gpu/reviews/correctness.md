# Independent final correctness review

Reviewed on 2026-10-03 against base `6918efa`, the complete implementation/evidence at `8f0ca92d2ee2741723ba20e5637decb91982694b`, and the final two-file lifecycle repair at `dcab49f6ff0e194711c031f4d72399b93f300877`. This is an independent reassessment of the producer, replay, UI contracts, retained evidence and repair. It does not reuse the preliminary process-isolation endorsement as final approval.

**Final outcome: no significant correctness findings remain.** The confirmed P2 below is closed by the independently checked lifecycle repair, forced-interleaving regressions and final-source runtime evidence. No additional significant correctness findings were identified in the captured scheduler/KV, native terminal, profiler, immutable revision or source-identity contracts. The full GPU matrix remains scoped to its unchanged `8f0` archive; the narrower four-case runtime qualification is explicitly scoped to `dcab`.

## Closed P2 — Give the owned worker one synchronized reaper

At `8f0`, `runner/isolation.py:145` called `process.join()` in the supervisor without the termination lock. Concurrent controller shutdown called `process.join()` and `process.is_alive()` through `runner/isolation.py:205–210`. CPython's multiprocessing process object does not synchronize these concurrent `waitpid`/return-code updates: one waiter can reap the child, while the other receives `ECHILD` and temporarily sees `returncode=None`. The second path could therefore raise “The owned execution process did not exit after bounded termination” even though the owned worker had already exited. Shutdown raised before its manager-last cleanup completed and falsely reported an unconfirmed process exit.

The existing native-spawn regression `tests/test_isolation.py:220` failed naturally in the focused 80-test run: 79 passed, one failed in 6.03 seconds, with the raised process already represented as stopped with `exitcode=-SIGTERM`. Eight isolated reruns passed, so the scheduling window is intermittent rather than a consistently slow worker.

An independent temporary native-spawn probe reproduced the cause deterministically without a GPU or a fake process. It used the existing `stuck_inside_transaction` target, then paused only the supervisor immediately after the real `os.waitpid` returned the reaped child and before `Popen.poll` stored its return code. Concurrent `close()` raised the same exception in 0.54 seconds. Releasing that legal scheduling pause confirmed exit code `-15`. No repository implementation files were changed for the probe.

The requested repair was to serialize owned-process polling/reaping operations, or let the supervisor wait on the sentinel while retaining one synchronized reaper. Increasing the short test timeout would not address the concurrent reaping window.

At `dcab`, the supervisor waits on the OS sentinel outside the lifecycle mutex; this wait never reaps the child. Native process start, polling, joining, termination and exit-code observations now serialize under the same lifecycle mutex. Shutdown releases that mutex before waiting for the supervisor. The identity-matched old handle is cleared before the state admission gate is released under that mutex, also fixing the independently reported stale-finalizer handle-publication window. State transactions do not acquire the lifecycle mutex in reverse order; confirmed/unconfirmed exit gating, process-death lock release and manager-last cleanup remain intact.

Both new regressions force their specific interleavings. The native-spawn test pauses the real `waitpid` return before CPython publishes the exit code and verifies shutdown waits for the serialized reaper, succeeds after release, finalizes evidence and cleans up the manager. The publication test admits at the old gate-release boundary and verifies the new owned handle survives and remains usable by shutdown. The existing native shutdown-held-lock and unrelated-process survival tests remain present. Independent execution of `tests/test_isolation.py` at `dcab` passed all 17 tests in 5.92 seconds. The author's separate temporary-project counterfactual check reports that both new regressions fail against `8f0`; this reviewer inspected the tests and independently executed their repaired behavior, rather than claiming to have run that counterfactual.

## Independent final-source runtime reassessment

Recomputed the final 38-file runner/instrumentation fingerprint as `c48b0fd1c8eccdc0b0d1d1d37e08b02f854a6e8bfd13531f81bf116bc524704d`, matching the deployed-source receipt and each initialized final-source recording. Only `runner/isolation.py` and `tests/test_isolation.py` changed between `8f0` and `dcab`. The deployment and validation receipts identify `dcab`, the isolated controller and the preserved original controller. The recorded full Python suite is 214 passes in 7.63 seconds, with current implementation CI passing; these are inspected root receipts, distinct from this reviewer's 17-test command.

Read all four raw recordings under `recordings/qualification-3d-live/lifecycle-dcab`, authored and appended request payloads, effective configuration, retained text/IDs, final state, memory receipts, profiler metadata and declared raw/normalized attachments. Independently reran `analyze.py` against the exact `dcab` ref, requiring native profiling and cancellation for the cancellation case. All four regenerated JSON reports are identical to the frozen `results/*-dcab.json` files:

| Case | Independently checked result |
| --- | --- |
| Staggered `run-62f5572c5570` | Pass; six original prompts preserved, six length terminals, 768 accepted IDs, effective caps of 128 and authored sampling/priority preserved. |
| Partial profile cancellation `run-b61c3a49fa0d` | Pass; 3,500-token synthetic input, actual native `FINISHED_ABORTED`, zero accepted outputs and three retained prepared/profile batches. Profile remains partial, completeness unknown and qualification unverified; raw native correlations have no errors. |
| Expected preparation failure `run-b39b7d48dcca` | Strict inference report remains failed; authored one-byte KV budget, native capacity error, zero events and zero admissions retained. No initialized source/model/pool provenance is invented. Failure cleanup is supported by the separate deployment/worker/memory receipts. |
| Recovery/live intake `run-0b38dea71a00` | Pass; appended and delayed prompts preserved, effective eight-output ceilings retained, expected one-word text preserved exactly, three and two accepted IDs respectively, and both actual native natural-STOP terminals retained. |

The four distinct owned workers (17247, 17502, 17772 and 17987) have exit code zero and exited status before finalized trace readiness. Each actual GPU-host receipt follows its worker's exit and records the same 3 MiB allocation baseline with no compute applications. The final-source cases contain nine admitted streams and 773 accepted IDs; the preparation failure is separate from those inference counts. Accepted-token sidecars match retained engine-output IDs exactly, complete emitted text matches the retained reviewed outputs, and canonical configuration snapshots and hashes agree. For all three inference/cancellation structural exports, event bytes are unchanged, content is omitted and native attachment references are cleared. The preparation failure has no structural export and is not presented as an inference pass.

These local forced regressions and actual final-source success/cancellation/failure/recovery observations close the original P2. They do not relabel the prior full matrix as newly captured at `dcab`.

## Checks and independently reproduced evidence

- Read the base-to-final implementation diff and reassessed process ownership, POSIX transaction-lock death release, atomic intake/admission, interrupted-byte preservation, native STOP/abort observation, scheduled versus returned KV writes, physical generation/refcount/prefix reuse, profiler row/correlation boundaries and the separate native/host/teaching clocks. Also read the completed qualification README and final browser-assessment scope.
- Recomputed the `8f0` 38-file runner/instrumentation fingerprint: `83df6c7bc792273eabb37d22ab76a29476b6c69a9325848505deafa1201e5056`, matching that capture identity. Verified all three original archive sizes and SHA-256 values, and every indexed member's size and SHA-256: 555 files, no missing, additional or mismatched files. The new lifecycle archive/index is assembled separately by the root after this review; the raw final-source evidence above was read directly.
- Extracted the final archive into a temporary directory and independently reran `analyze.py` against **all 11 recordings** with exact source ref `8f0ca92d2ee2741723ba20e5637decb91982694b` and the appropriate prefix, preemption, cancellation and profiling requirements. Every regenerated report was identical to its frozen `results/*-8f0.json` counterpart: ten inference/cancellation passes and the intentionally failed preparation inference report. This invokes the actual Python loader/integrity evaluator and TypeScript replay/profile projections, including backward seeking and raw native CUDA-correlation validation.
- Directly counted retained engine outputs: 32 admitted streams and 2,645 accepted IDs. Confirmed distinct exited workers with code zero for the 11 archived scenarios, complete token/text sidecars for admitted streams, two native natural-STOP terminals in the recovery case, the observed `FINISHED_ABORTED` cancellation, and three actual preemptions in the extended pressure capture. The preparation failure retains zero admitted streams and lacks initialized provenance; it remains failure cleanup evidence.
- Rechecked the final browser capture's requested `main` revisions and resolved immutable model/tokenizer pin, the complete 5,414-block inventory, and the distinction between post-reorder packed row identity and per-token/head duration. The completed documentation retains the failed pre-STOP-fix checkpoint, partial-profile status, unexercised nonzero padding, unverified native qualification label and instrumented V1 versus ordinary V2 runtime boundary.

Focused local command:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_isolation.py tests/test_learning_capture.py tests/test_profiling.py tests/test_observability.py
```

Original `8f0` result: **79 passed, one shutdown race failure in 6.03 seconds**. The failed shutdown case subsequently passed eight isolated reruns; the native scheduling probe above confirmed the underlying race. This failed result is retained as repair history.

Final `dcab` local command and result:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_isolation.py
```

**17 passed in 5.92 seconds.** `git diff --check` passed at both review snapshots. The root's retained full-suite/build/CI receipts were inspected as evidence, rather than reported as tests executed by this reviewer.

## Limits

No GPU, remote runtime or model calls were launched by this review. Live browser interaction was performed by the root and inspected here through retained state, screenshots and paired controller receipts. LangSmith is explicitly waived by the user; no persistence claim is made. The retained qualification supports the exact vLLM 0.30.0 single-A10 dense-Qwen3 layout and observed scenarios. It does not establish numerical K/V values, nonzero padding on the real GPU, per-token/head device durations, answer quality or browser GPU performance. Shutdown scheduling windows are covered by local forced native-spawn regressions; the four new GPU cases exercise final-source worker success, cancellation, preparation failure and recovery, without claiming that arbitrary thread schedules were reproduced on the GPU host.
