# Live GPU KV diagnostics and recorded demo

The user supplied a reachable replacement GPU host on 2026-10-07 and authorized live verification and replacing the synthetic demo with a real run. The production demo action now opens a saved GPU recording with its own original workload and settings. Configure remains an editable live workload; opening the demo does not replace that draft or start inference. Synthetic recordings remain regression fixtures and are absent from the demo library. LangSmith was explicitly waived for this project.

**The live GPU runtime gate passes for this observed BF16 A6000 configuration.** All five final production API cases pass their artifact assessments, and the separate native-input reference passes all ten checks with 24 exact source/cache comparisons. A separate retained-artifact audit passes 19 checks. The earlier cancellation bookkeeping defect is repaired and its fresh regression passes. The actual UI-triggered targeted rerun also passes 33 checks and presents the newly captured values. The final demo is bundled; these results do not qualify other runtimes.

## Runtime and capture boundary

The observed runtime is a single NVIDIA RTX A6000 running vLLM 0.30.0, PyTorch 2.13.0+cu130 and CUDA 13.0. The dense model and tokenizer are `Qwen/Qwen3-4B-Instruct-2507`, both pinned to `cdbee75f17c01a7cc42f958dc650907174af0554`. Model and cache tensors are BF16, with no model or cache quantization. The inspected cache-update implementation is `FlashAttentionImpl` using FlashAttention 2. This evidence does not qualify other model architectures, attention backends, cache layouts, dtypes or devices.

The saved demo uses the production HTTP controller, engine-owner thread, recording writer, native profiler, normalizer and replay parser. Its workload has three actual text requests: `prefix-prime`, then `prefix-reader` and `mixed-prefill` after a planned 6,000 ms offset. The first two share a prefix; the third provides independent prefill work. Temperature is 0, seed is 0 and each request accepts at most four output tokens. The context limit is 1,024, the sequence limit is three, the scheduling budget is 64 tokens, prefix caching and chunked prefill are enabled, and the requested KV memory budget is 256 MiB. Profiling runs in eager mode and changes execution overhead.

The actual pool contains 113 physical blocks with 16 token positions per block. Block 0 is the reserved null block, leaving 112 usable blocks and 1,792 usable token positions. The context limit is separate from this shared capacity. All 36 layer views have native shape `[113, 8, 16, 256]`, with concatenated K and V, eight KV heads and 128 dimensions per K or V vector. The views alias one backing allocation of 266,600,448 bytes (254.25 MiB). Unique storage is counted once; drilling into a layer or head does not create more cache allocations.

The selected numerical capture retains positions 0–3 of one request, zero-based layers 0, 18 and 35 (displayed as layers 1, 19 and 36), head 0, and dimensions 0–15 of both K and V. That is 24 bounded slices, 16 BF16 values each, totaling 768 logical bytes under the 1 MiB default limit. Snapshots are cloned on the same CUDA stream after the native cache update and before later reuse; readback occurs after profiling stops. The saved values represent this selected snapshot, not arbitrary live tensor browsing.

## Retained cases and assessments

`capture.py` validates case payloads without executing when `--execute` is omitted. With explicit execution on the GPU host, it invokes the production API and retains requests, launch/final state, memory observations, recording artifacts, structural exports and assessment results. `assess.py` checks the new cache/tensor contract against those artifacts. Captured source identities cover the 39 distributed runner/instrumentation Python and JSON files at recording creation; they do not imply a Git commit when `declared_revision` is null.

The final distributed source SHA256 is `46fa2ddb99ea96309a50c07580a70ddaba8ab39241d8b798185ce0ea504fc978`. It is distinct from the preceding-capture fingerprint `67d993088a5f326fb1fea9cea7f64e7043dd2355a23388505b8508925049d3d9`. Final artifacts live under `final-captures/`; the earlier `captures/` and `boundary-captures/` remain the pre-repair observations.

| Case | Success criterion | Preceding evidence and outcome | Final-source repeat |
| --- | --- | --- | --- |
| Demo | Real readable requests; partial-prefix reuse and remaining prefill; mixed prefill/decode; all selected snapshots and complete profiler correlation | `captures/demo/`: 160-token prefix hit, mixed step, 468 layer cache-update operations, 24 snapshots, 32 new-contract checks passed | Passed: `final-captures/demo/new-contract-assessment.json`, run `run-a413ca908152`, 33 checks, 13 profiled steps, 468 layer writes, 24 snapshots; mixed steps 8–10 |
| Request trigger | Profiling starts only after admission of the selected request and follows the bounded window; selected numerical slices match the target | `boundary-captures/trigger/`: `mixed-prefill` admission and steps 7–13 retained; 24 snapshots | Passed: `final-captures/trigger/new-contract-assessment.json`, run `run-821b81a1a76f`, 30 checks; admission sequence 151, steps 7–13, 24 slices |
| Byte limit | Numeric budget truncates explicitly without dropping structural observations or claiming all values were retained | `boundary-captures/partial-budget/`: two slices/64 bytes, status `partial`, explicit limit reason | Passed: `final-captures/partial-budget/new-contract-assessment.json`, run `run-d95ca06f23f9`, 30 checks; two slices/64 bytes and explicit partial reason |
| Diagnostics off | Serving capture retains structural/cache evidence without a native diagnostic attachment or numerical values | `boundary-captures/off/`: API run and structural assessment passed; helper expectation repair is retained separately | Passed: `final-captures/off/new-contract-assessment.json`, run `run-c3bb90ebd0b7`, 20 checks |
| Cancellation | Cancelled request retains accepted token content consistently with engine output, explicit terminal status and a bounded partial profile; worker exits | `boundary-captures/cancel-profile/`: initial new-contract assessment failed; accepted content and retained engine outputs disagreed | Passed after repair: `final-captures/cancel-profile/new-contract-assessment.json`, run `run-f7269d1dbe7f`, 31 checks; three profiled steps, explicit partial profile, 24 slices |

The initial cancellation failure remains in `boundary-captures/cancel-profile/before-controller-fix.json` and the accompanying recording. `assessment.json` contains the older structural assessment, which did not detect the mismatch; its success must not be confused with the failed new-contract assessment. The production fix preserves accepted outputs at cancellation. The final regression retains token ID `95456` (`Certainly`) in both accepted-token content and `engine_output_token_ids`, with one text emission covering `[0, 1]`. The comparison was retained rather than weakened. Earlier artifacts and raw traces are packaged in `previous-attempts.tar.gz`; `archive-index.json` records its SHA256 and contents.

## Independent native-input reference

`native_reference.py` performs a separate direct-engine observer run using the production session and recorder. It clones selected native K/V inputs and native slot addresses before, and outside, the production cache-write profiler scopes. After profiling stops, it compares those source values with production cache snapshots at the declared physical allocation, generation and slot. This observer run is separate from the HTTP demo and adds its own overhead.

The final `final-native-reference/native-reference-assessment.json` reports ten passing checks and 24 comparisons under run `native-reference-final-2026-10-07`, using the final distributed source fingerprint above. Selected coverage is exact, values are finite, physical addresses and participants match, and float32 conversions of all selected BF16 source/cache slices are bitwise equal; maximum numerical error is zero. Observer CPU scopes and GPU launches are outside the native-write scopes. This checks the selected source/cache relationship rather than comparing the same cache indexing path to itself. The preceding `native-reference/` capture remains in the earlier-attempt archive.

Recorded `profiling.qualification` remains `unverified`: successful profile normalization is not external qualification. Assessment files provide scoped external evidence for this specific runtime and selection; they do not silently rewrite recording metadata or certify every tensor element. A CUDA cache-update interval describes a shared batch operation. It does not measure an individual token, layer tensor transfer, KV head or dimension duration. CPU-return timestamps do not establish CUDA completion.

## Browser evidence and assessed outcomes

The browser uses the actual application and imported/saved recording attachments. The retained DOM observations and screenshots establish the following journeys. The shared-token, tensor, missing-slice, mixed-batch and native-clock observations have been repeated on the final demo recording. The actual targeted rerun completed and displayed values for the previously uncaptured slice.

| Journey and success criterion | Observed outcome | Evidence |
| --- | --- | --- |
| Open the real demo from an altered draft, return to Configure and reload | Edited request fields and editable workload remain present; recorded workload is independent | `demo-draft-before-dom.txt`, `demo-draft-preserved-dom.txt`, `demo-draft-reloaded-dom.txt` |
| Offer saved GPU lessons and preserve importing recordings | Library contains the new GPU diagnostics lesson and two historical real GPU lessons | `demo-library-dom.txt` |
| Unavailable demo files and retry | Error changed from raw JSON internals to an actionable recording-files message; retry remains available | `demo-unavailable-before-dom.txt`, `demo-unavailable-after-dom.txt` |
| Local CUDA check blocks live capture while allowing saved demo | Live capture disabled on the Mac; demo enabled | `demo-local-blocked-dom.txt` |
| Compact Configure launch bar | Demo and capture actions remain visible at 390×844 | `demo-mobile-configure-dom.txt`, `demo-mobile-configure.png` |
| Inspect actual device/cache configuration | BF16 on cuda:0, actual unique byte capacity, layer membership and reserved/usable pool state are shown | `real-cache-configuration-dom.txt`, `real-cache-configuration.png` |
| Inspect shared prefix and original writer | Reader token resolves the original writer's block/generation/slot and retained numerical slice on the final recording | `final-shared-token-dom.txt`, `final-shared-tensor-dom.txt`, `final-shared-tensor.png` |
| Select an uncaptured layer | Numerical values remain unavailable and the proposed rerun targets the original writer | `final-missing-slice-dom.txt`, `final-targeted-rerun-plan-dom.txt` |
| Inspect mixed batch membership | Final recording shows simultaneous prefill/decode; scheduled counts are separate from bounded prior KV positions | `final-model-mixed-dom.txt`, `final-model-mixed.png` |
| Native-clock cache operation completion | At 6,136,549 µs since profiler trace start, layer 1 has an active shared write for 63 `mixed-prefill` inputs and one `prefix-reader` decode input. At 6,136,551 µs, both cells show the shared operation completed; host replay stays at event 0 | `final-profile-write-active-dom.txt`, `final-profile-write-completed-dom.txt` |
| SSH preflight through the actual Configure flow | Replacement host connects, reports the A6000/vLLM 0.30.0 runtime and verifies the loopback controller | `final-ssh-preflight-dom.txt` |
| UI-triggered targeted rerun | Passed: actual run `run-cb8b01c88cd0`, 33 artifact checks; original writer `prefix-prime`, position 3, layer index 1/head 0; both K/V slices retained (64 bytes). Inspector changes from unavailable to captured values on displayed Layer 2 | `ui-targeted-rerun/new-contract-assessment.json`, `final-targeted-rerun-plan-dom.txt`, `final-ui-live-launch-dom.txt`, `final-ui-targeted-slice-dom.txt`, `final-ui-targeted-slice.png` |
| Compact tensor controls and keyboard dismissal | Passed at 390×844 with document width 390. Switching to V synchronizes scene and inspector; Escape closes the dialog and restores focus to Inspect block | `final-mobile-tensor-dom.txt`, `final-mobile-tensor.png`, `final-mobile-focus-restored-dom.txt` |
| Fresh default, recorded-time speed and console | New Explore tab opens the real demo; recorded-time playback offers 0.001×. No browser warnings or errors in the fresh tab | `final-fresh-default-dom.txt`, `final-recorded-time-dom.txt`, `final-browser-console.json` |
| Real pool ruler spacing | Columns 0, 4 and 8 remain legible on the 113-block pool; physical IDs and selection are unchanged | `final-ruler-spacing-dom.txt`, `final-demo.png` |

The UI rerun's `expected-request.json` is an expected payload reconstructed from the original demo request and the UI-selected diagnostic controls, not a captured HTTP request body. `ui-targeted-rerun/request-provenance.json` records this distinction. The independent expectation and retained inference snapshot have matching SHA256 `66fe3466e5481f0af0ba22a482c77425a61cf6363e0b0056963512d23d84cd2c`; the recorded workload remains the shared three-request plan. The admission trigger targets the original writer and records steps 1–8. Browser selection of the missing slice, restored Configure controls, actual launch, completed capture and rendered numerical values provide the integrated journey evidence.

The final demo `run-a413ca908152` is bundled under `web/public/recordings/gpu-kv-layer-diagnostics`. Its files match the assessed final recording except for the lesson annotation; recording qualification is not overridden. Synthetic regression fixtures remain available for cases that cannot be manufactured honestly in a small real demo, such as an authored trace gap.

## Reproduction and retained artifacts

`previous-attempts.tar.gz` preserves the preceding runs, including the failed cancellation recording. `final-artifacts.tar.gz` preserves all five final API recordings, the UI targeted rerun, the native-input reference, raw native traces, normalized profiles and structural exports. Both archives are SHA256-indexed in `archive-index.json`. Compact inputs and assessments remain readable alongside the archives. Extract an archive into a separate folder to recover its complete recording directories before running an assessment.

Run from the repository root with its Python environment:

```sh
python docs/qualification/kv-live-gpu/capture.py --output /tmp/kv-plan
# On the prepared CUDA host; --execute starts real inference:
python docs/qualification/kv-live-gpu/capture.py --execute --output recordings/kv-repeat --recordings recordings/runs --gpu-memory
python docs/qualification/kv-live-gpu/native_reference.py --execute --output recordings/kv-native-repeat
# Read-only checks after extracting final-artifacts.tar.gz into /tmp/kv-evidence:
python docs/qualification/kv-live-gpu/assess.py --recording /tmp/kv-evidence/final-captures/demo/recording --request /tmp/kv-evidence/final-captures/demo/request.json --tensor-status completed --require-prefix --require-mixed-step --require-prefix-snapshot
python docs/qualification/kv-live-gpu/assess.py --recording /tmp/kv-evidence/final-native-reference --native-reference
```

## Validation and scope

All five final production API cases, the separate native-input comparison, and the actual UI-triggered targeted rerun pass within the observed scope above. The final browser native-clock/shared-cache and compact keyboard journeys pass. The complete suite passes 234 Python, 58 replay and 73 web tests (365 total), Python compilation, TypeScript/Vite build and whitespace checks. Web tests and build were repeated after the ruler-spacing and unused inspector-prop cleanup; native write chronology was rechecked in the browser. The existing bundle-size warning remains.

All three fresh independent reviews found no remaining significant issue: `review-correctness.md`, `review-evidence.md` and `review-simplification.md`. The earlier local/synthetic reviews are historical and do not replace these reviews of completed live evidence. Full automated check logs and their hashes are retained in `checks/`.

These captures are qualification evidence, not production performance benchmarks. Eager profiling, cache hooks and optional numerical clones add overhead. The diagnostics-off case provides a serving-path observation, not a controlled estimate of profiler overhead: request scheduling and measured durations can vary between runs. GPU memory observations can establish release after worker exit only for the sampled processes and times; they do not establish general leak freedom. Multi-GPU, distributed clocks, other layouts/quantization modes and unselected tensor values remain outside this evidence.
