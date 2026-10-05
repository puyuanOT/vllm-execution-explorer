# GPU qualification of the 3D learning explorer

**Recorded-contract and browser qualification passed at `8f0ca92d2ee2741723ba20e5637decb91982694b`; the final lifecycle repair is qualified separately at `dcab49f6ff0e194711c031f4d72399b93f300877`.** The full visualization matrix contains ten actual GPU captures, 32 admitted request streams and 2,645 accepted output tokens. Four fresh lifecycle cases add nine admitted streams and 773 accepted tokens, including partial-profile cancellation and an expected preparation failure. Across both checkpoints, 13 inference captures contain 41 admitted streams and 3,418 accepted tokens; two separate expected preparation failures qualify cleanup and recovery, not inference. The original captures include real partial prefix reuse, chunked prefill, priority ordering, three cache preemptions with recomputation, native layer activity, natural EOS and browser capture with blank revision overrides. Complete inputs, outputs, configuration, native files and failed iterations are retained in the [evidence index](evidence-index.json) with [archive checksums](SHA256SUMS).

The user explicitly waived LangSmith for this project. Qualification therefore uses actual request payloads, initialized-engine configuration, retained token/text streams, scheduler/KV observations, native profiler files, source fingerprints and browser interaction. A successful helper result is scoped to its recorded contracts; it does not establish answer quality, browser GPU performance, or support for other engine/model layouts.

## Final lifecycle repair and repeated GPU checks

Independent final reviews found two controller races after the full matrix passed: an old finalizer could clear the next worker's handle, and concurrent `Process` waiters could reap the same child before its exit code was published. Revision `dcab49f` changes only `runner/isolation.py` and its tests. The supervisor waits on the OS exit sentinel without reaping; one lifecycle mutex serializes all process operations. Identity-matched handle clearance precedes admission-gate release in the same transaction. Both forced interleaving regressions fail against the previous implementation and pass with the repair.

The final 38-file capture fingerprint is `c48b0fd1c8eccdc0b0d1d1d37e08b02f854a6e8bfd13531f81bf116bc524704d`. The actual isolated A10 controller ran four consecutive cases with this exact source; the complete [lifecycle archive](archives/lifecycle-dcab.tar.gz) retains their requests, outputs, configuration, raw native files, worker exits, memory receipts and controller log. Earlier archives and reports remain attached to their original source bytes.

| Case and independent report | Actual observation |
| --- | --- |
| [Normal staggered capture](results/staggered-dcab.json), `run-62f5572c5570` | Six requests each accept 128 tokens and terminate at their authored length limits. Changing continuous batches and result-gated KV writes pass the production replay checks. |
| [Partial-profile cancellation](results/cancel-profile-dcab.json), `run-b61c3a49fa0d` | Native abort is retained with zero output tokens, three actual profiled batches and 1,602 CUDA intervals. Raw correlation, packed rows and partial-profile contracts pass. |
| [Expected preparation failure](results/preparation-failure-dcab.json), `run-b39b7d48dcca` | The one-byte KV budget fails before admission, with zero events. Its failed inference analysis remains failed; the separate cleanup/recovery assessment passes. |
| [Immediate recovery and late intake](results/recovery-late-intake-dcab.json), `run-0b38dea71a00` | Late intake is accepted, both one-word answers naturally reach EOS, and native stop terminals precede the successful HTTP responses. Final sampled EOS remains uncomputed in KV. |

Workers 17247, 17502, 17772 and 17987 each exit with code zero before final readiness. After every case, GPU memory returns to 3 MiB with no compute application allocation. The original controller (PID 4007) remains running. The six normal outputs are length-limited workload prose; the recovery answers are exactly “admitted” and “scheduled,” with three and two accepted IDs including EOS. This follow-up qualifies changed process lifetime behavior; it does not relabel the full model/KV/browser matrix as newly executed at `dcab49f`.

All 214 Python tests pass, including 17 isolation checks. Python compilation and whitespace checks pass; [implementation CI](https://github.com/puyuanOT/vllm-execution-explorer/actions/runs/37172912654/job/111349384526) passes. The frontend, replay and instrumentation are unchanged by this final repair. The [independent review gate](reviews/README.md) has passed: all three original final reviews are retained, both lifecycle findings are closed after affected reviewers reassessed the repair and new runtime evidence, and no significant issue remains unresolved. Exact scopes are retained in [correctness](reviews/correctness.md), [integration/evidence](reviews/integration-evidence.md) and [simplification](reviews/simplification.md).

## Runtime and source identity

The isolated controller ran on the supplied Ubuntu GPU instance through its actual loopback HTTP API. The existing controller and checkout were preserved. The observed runtime is NVIDIA A10 with 23,028 MiB, vLLM `0.30.0`, PyTorch `2.13.0+cu130` and CUDA `13.0`.

All successful GPU cases use `Qwen/Qwen3-4B-Instruct-2507`, with both model and tokenizer resolved to `cdbee75f17c01a7cc42f958dc650907174af0554`. Architecture metadata comes from the initialized model configuration: 36 dense layers, hidden size 2,560, FFN size 9,728, 32 query heads, eight KV heads and explicit head dimension 128. Dividing hidden size by query-head count would give an incorrect head dimension for this model.

Ordinary captures preserve vLLM's V2 runner. The opt-in eager diagnostic mode selects the V1 `GPUModelRunner`, disables CUDA graphs and records its actual runner choice. Its native timings qualify that instrumented diagnostic workload; they are not a production V2 benchmark. [Lifetime and case details](profile-lifetime.md) describe this boundary and the per-capture execution process.

The full model/KV/browser matrix identity remains separate from the final lifecycle delta above and the preserved bounded-length checkpoint:

| Identity | Full matrix `8f0ca92` checkpoint | Preserved `75f2ccd` checkpoint |
| --- | --- | --- |
| Declared deployed source | `8f0ca92d2ee2741723ba20e5637decb91982694b` | `75f2ccd3a13b585c576f2597449a23dcad670921` |
| Runner/instrumentation digest | `83df6c7bc792273eabb37d22ab76a29476b6c69a9325848505deafa1201e5056` | `af7f86400baecd380b7ec08de10ac26f8bf320e22108fff5b7e20d4cbff5703c` |
| Fingerprint scope | 38 runner/instrumentation `.py` and `.json` files | Same scope |

The declared Git revision alone is not proof of deployed bytes. The analyzer recomputes the fingerprint from the exact audited ref. Documentation and frontend files are outside this distributed-source fingerprint and have their own versioned review evidence. The earlier transient `9fce` deployment had uncommitted source bytes; it remains explicitly separate from committed-source qualification. The retained independent reports identify analyzer digest `3bfbc5e6ba4995c4f8c4de26344251829a0baa524a37f0d20479d5490221020b`.

## Seven consecutive cases at source `8f0ca92`

The helper retained authored payloads, actual launch/final state, manifests, complete event/content streams, raw and normalized native files where applicable, integrity reports, structural exports and GPU-memory receipts. Independent analysis uses the actual TypeScript replay projections, checks backward seeking, and inspects every retained input/output stream and its effective limits, priority, sampling and terminal observation. [The independent stream review](results/stream-review-8f0.json) also evaluates what the inputs and generated outputs mean.

| Case and retained report | Actual observation | Qualification boundary |
| --- | --- | --- |
| [Staggered](results/staggered-8f0.json), `run-a85fc19d413b` | Six LLM-domain prompts at authored offsets 0/0/100/100/100/120 ms; 256 scheduler steps reach four running and two waiting requests; all six accept 128 output tokens. | Changing continuous-batch membership and FCFS admission are observed. No prefix reuse or preemption is exercised. |
| [Partial prefix reuse](results/prefix-8f0.json), `run-7a7d6779e405` | The prime finishes before the follower. The 286-token follower reuses 272 tokens in 17 blocks and newly prefills positions 272–285. | Cached positions retain the prime request as content provenance while their live binding belongs to the follower. Reassigned suffix storage carries the follower's provenance. |
| [Chunked prefill](results/chunked-8f0.json), `run-48c392b24643` | Exact 512- and 16-token synthetic inputs share a 64-token budget. The first step schedules 16 short-prompt tokens plus 48 long-prompt tokens; mixed later steps use one decode input plus 63 prefill inputs. | The short request completes while long prefill continues. Synthetic output is a stream-preservation workload, not a teaching answer. |
| [Cache admission pressure](results/pressure-8f0.json), `run-6d1ce4f328a0` | Four 192-token synthetic prompts, 32-output limits and a 72 MiB KV budget produce 32 physical blocks, including one reserved-null block. Two requests run while two wait. | Admission/backpressure is observed. This particular case records zero preemptions and zero evictions. |
| [Native profile](results/profile-8f0.json), `run-b57709a84086` | Eight eager batches retain post-reorder CPU token rows and correlated CUDA activity for all 36 layers and attention/FFN mappings. All 4,776 normalized CUDA intervals agree with raw native timing/correlation. | The first batch has 55 actual rows; the other seven have two. Real and total row counts are equal, so nonzero padding is unexercised. |
| [Cancellation and partial profile](results/cancel-profile-8f0.json), `run-76018939e6b0` | Cancellation during a 3,500-token prefill preserves three 64-row batches, 192 computed prompt positions, native `FINISHED_ABORTED`, block releases and an HTTP cancelled response. All 1,602 CUDA intervals match raw evidence. | The profile remains partial: three captured steps of a 16-step limit. Zero output tokens are expected; sampled-final-token checks are explicitly not applicable. |
| [Priority](results/priority-8f0.json), `run-0282822a273f` | Authored priorities 5/−5/0 are retained. Observed admissions are 0,5,−5; the waiting queue is −5,0,5 and, with one running sequence, execution order is −5,0,5. | Native priority ordering is observed independently of dispatch/admission order. Priority preemption is not claimed. |

These seven cases contain 20 request streams and 1,568 accepted output tokens. Natural-language outputs concern their requested LLM topics but stop at the configured length limits; some are visibly incomplete or technically imprecise. Their prose is retained workload output, not verified explanatory content. Synthetic cases deliberately use repeated token IDs and do not assess answer quality.

After each case, its distinct owned worker exited with code zero before final readiness. GPU memory returned to the same 3 MiB baseline with no compute application allocation. This is measured on the GPU host, not inferred from a local SSH-forwarded controller or a call to `empty_cache()`.

## Additional `8f0ca92` cases

The [independent extra/UI review](results/extra-ui-stream-review-8f0.json) checks authored settings against effective engine configuration, original and appended request limits/priorities, actual retained outputs, worker exits, memory receipts and replay state.

[Extended pressure capture](results/pressure-preemption-8f0.json) `run-9ff4ff7f29ba` uses four 192-token synthetic prompts, a 72 MiB / 32-block KV pool, a context limit of 480 and 256-output limits. It records three actual preemptions with computed cursors reset to zero. Accepted output IDs remain retained across preemption; later scheduled work recomputes from position zero, including those previously accepted outputs. All four streams finish with 256 accepted outputs. The shorter seven-case pressure workload remains separately scoped to admission/backpressure.

[Expected preparation failure](results/preparation-failure-8f0.json) `run-5cd47c5ab70a` repeats the intentionally unusable one-byte KV budget. Its native capacity error, failed manifest, zero admitted requests and joined worker are retained. The ordinary GPU inference analyzer remains failed because no inference began; the independent review classifies inference, loaded geometry and pool qualification as not applicable. The empty recording has no initialized source fingerprint. A separate deployed-source receipt identifies the actual tree without inventing initialized GPU provenance.

The immediate next [recovery/live-intake capture](results/recovery-late-intake-8f0.json), `run-1cd5db48bd59`, accepts `late-immediate` through the actual intake API and later dispatches `planned-after-10s`. The complete one-word answers are “admitted” and “scheduled,” with three and two accepted IDs respectively, including final EOS ID `151645`. Both have native `completion` events with reason/status `stop` (sequences 34 and 59), followed by successful HTTP stop terminals. This actually exercises the repaired falsey `STOP=0` condition before the eight-output ceilings. Prior consumed output positions become computed K/V; the sampled final EOS remains `not_computed`. The late request completes before the delayed planned request is admitted. All three extra workers exit and return GPU memory to 3 MiB with no compute application allocation.

## Historical checkpoint and discovered failure

An extended cache-pressure workload, `run-a1284af7a0fd`, uses the same 32-block pool with 256-output limits. It records three actual preemptions and cursor resets, then completes all four output streams. Its full retained payload, results and raw recording belong to the additional `75f2ccd` evidence, rather than retroactively changing the seven-case pressure result.

The deliberately invalid one-byte KV budget, `run-94df8b2bd6c0`, fails during preparation with the native “No available memory for the cache blocks” error. No request is admitted; the failed worker exits, and GPU memory returns to 3 MiB. The recording has no events and does not pass the normal successful-capture integrity/GPU-contract checks. It is retained as expected failure/recovery evidence, not a successful inference recording. [Failure analysis](results/preparation-failure-75f.json).

The next run, `run-920e4e7bae9b`, accepts a late request through the actual cross-process intake API and later dispatches the planned 10-second request. Both naturally finish after short outputs, but their native completion events are missing because `STOP=0` was ignored. The independent analyzer correctly fails per-request terminal checks despite a superficially complete run. [Contradictory recovery analysis](results/recovery-late-intake-75f.json). Fresh recovery cases at both `8f0ca92` and `dcab49f` pass independent final-token, stream and replay checks with both native stop observations. The historical failed report is preserved unchanged.

## Browser and configuration evidence

At source `8f0ca92`, the production browser preview submits six short requests with model revision, tokenizer revision and tokenizer repository all blank. [Capture `run-1021e5882f0d`](results/ui-capture-8f0.json) completes through the actual GPU controller and opens Explore automatically after its owned worker exits. The recorded requested revisions are `main`; both resolve to the immutable canonical SHA, and effective configuration records the resolved model/tokenizer identity. These fields are optional overrides, rather than mandatory knowledge for starting a run. All six requests have eight accepted outputs and length terminals; their short plain-text prompts/continuations establish the workflow, not answer-quality correctness. Requested precision is `auto`; a separately observed resolved tensor dtype is not claimed for this UI case.

[Final browser observations](browser-assessment.md#qualified-final-source-browser-journeys) retain interaction with the newly initialized **5,414-block pool**. At event 27, after result event 26, block 1 / generation 1 has one live reference and five of 16 computed positions, mapped to sample-2's actual prompt token IDs/fragments. Block 5,413 is addressable, available, generation zero, with zero references and no computed positions. Pan/reset preserve selection. [Final pool screenshot](screenshots/final-kv-pool.png) shows both the complete grid and readable computed slots. Each initialized engine reports its own capacity; the older 4,477-block observations are not substituted for this pool.

The final native profile imports with its declared normalized attachment. Step 1 contains 55 actual post-reorder rows: profile-b positions 0–27, followed by profile-a positions 0–26. At native 16.154 ms, row 0 selects profile-b / position 0 and an actually correlated layer-0 attention interval is active. The model exposes all 36 layers, 32 query heads sharing eight KV heads, explicit dimension 128, and separate CPU/device measurements. Host replay stays at event zero while the native cursor moves. [Final native attention screenshot](screenshots/qualified-native-attention.png) retains the state at 0.001×. Token/batch participation does not establish an individual token/head duration. The native qualification label remains unverified: external independent reports do not fabricate a manifest acceptance badge.

The live-follow repair was exercised while another capture was active. The browser retained its finalized historical recording and exposed neither Follow live nor Reconnect live for the unrelated current run. Paired controller/browser state and control counts are retained in the qualified archive.

[Spacing measurements](form-spacing.json) show a 6 px control margin and 7 px rendered text-to-control gap for Repository, Precision, both revisions and SSH port. [Model fields](screenshots/configuration-revisions.png) and [SSH fields](screenshots/ssh-form-spacing.png) preserve the layout. The actual bounded SSH reachability probe is retained in [ssh-reachability.json](ssh-reachability.json); automatic bootstrap of the original remote checkout was not exercised because it would replace the existing controller.

CPU submission/resource observations in [scene-cpu-resource-observations.json](scene-cpu-resource-observations.json) are scoped to observed demand frames and Three.js counters. They do not measure GPU completion, frame presentation, FPS or GPU allocation bytes. The ordinary production build excludes the opt-in scene diagnostic logger; its existing bundle-size warning remains. The `8f0ca92` checkpoint retains 212 Python tests, 51 replay tests, 52 web tests, successful production build/compile checks and implementation CI success in `validation-8f0.json` inside the qualified archive. Final independent reviews are linked above; their original lifecycle findings and repair reassessments remain visible. Final documentation CI is recorded in the PR checks.

## Evidence boundaries

The visualization keeps three clocks distinct: host-recorded observations, an explanatory model-layer walkthrough, and trace-local native CPU/CUDA microseconds. Architecture alone does not establish a measured layer timeline. Native CUDA-to-layer mapping requires launch/external-ID correlation inside explicit step/layer scopes; CPU scope duration and kernel names cannot substitute for device activity. Unmapped activity remains reported rather than assigned an invented identity.

Every physical pool block is addressable, including the reserved-null block. Token-slot contents additionally require the qualified single-GPU, unsharded, causal full-attention group/layer layout. Cached zero-reference blocks can be available for reassignment while retaining prefix content. Physical availability, live request bindings and content provenance remain separate observations.

Scheduler reservation and packed model input are distinct from returned computation. Recorded K/V writes must follow their associated returned result, remain within its scheduled range and physical generation, and agree with retained request token IDs. A sampled output becomes computed K/V only if a later model input actually covers it. There is no numerical K/V readback, per-token or per-head device duration, or claimed alignment between host and profiler clocks. Unsupported cache/model layouts remain explicitly unavailable.

## Retained regressions

Earlier attempts are preserved so a passing final case cannot hide the failures that changed the implementation:

| Attempt | Observed failure and resulting repair |
| --- | --- |
| Initial serving / repeat capture | In-process close retained about 8 GiB of compiled model allocations; a following preparation failed for insufficient free GPU memory. |
| Public compiler reset | The attempted reset did not release those allocations. It was removed; production now owns a dedicated process for each complete capture and gates readiness on exit/join. |
| First spawned-worker capture | A trace request arrived before the child created its recording directory and received 404. The actual capture completed and returned memory to 3 MiB. The accepted current run now has an empty provisional trace until its files appear; unknown IDs remain 404. |
| Initial cancellation profile | Native abort bypassed the ordinary output-return hook, so HTTP cancellation alone was retained. A hook now records only newly returned native `FINISHED_ABORTED` requests; a later partial profile contains the actual abort event. |
| Zero-output analysis | The helper originally required a sampled final token even after prefill cancellation. It now treats observed zero-output cancellation as explicitly not applicable, with counterfactual checks preserving failure for a missing native abort. |
| Natural EOS at `75f2ccd` | The additional recovery case exposed the falsey `STOP=0` finish reason. Its failing report remains retained; fresh recovery at `8f0ca92` and `dcab49f` records both natural-stop terminals. |

[Historical raw attempts](archives/historical-attempts.tar.gz), the [pre-STOP-fix matrix](archives/matrix-75f-before-stop-fix.tar.gz), per-case reports and controller logs retain their original scope. The first actual serving analysis, [run-225baaef2a99](normal-staggered-run-225baaef2a99.json), qualifies its own `7da8f31` recording only. Synthetic fixtures and the authored host-empty native fixture remain loader/UI contract artifacts; they never qualify a GPU run.

## Reproduce the analysis

`analyze.py` uses Python's standard library and the actual TypeScript replay through Node. It reads a downloaded recording without mutation, network access, model loading or GPU calls. Its shareable report retains stream hashes/counts instead of copying prompt/output text; the complete archive remains available for human inspection. Individual token fragments are not assumed to concatenate into the original prompt or full emitted answer.

The [qualified archive](archives/qualified-8f0.tar.gz) contains `final-8f0/` for the seven-case matrix, `extra-8f0/` for pressure/failure/recovery, and `ui-capture-8f0/` for the browser recording, alongside deployment, memory, controller, browser and validation receipts. The [evidence index](evidence-index.json) provides each archive's hash/size and each member's hash/size; [SHA256SUMS](SHA256SUMS) verifies archive bytes. From the repository root:

```sh
(cd docs/qualification/3d-learning-gpu && shasum -a 256 -c SHA256SUMS)
mkdir -p /tmp/vllm-3d-evidence
tar -xzf docs/qualification/3d-learning-gpu/archives/qualified-8f0.tar.gz -C /tmp/vllm-3d-evidence
python3 docs/qualification/3d-learning-gpu/analyze.py /tmp/vllm-3d-evidence/final-8f0/prefix/recording --source-ref 8f0ca92d2ee2741723ba20e5637decb91982694b --require-prefix-reuse --output /tmp/vllm-3d-evidence/prefix-analysis.json
python3 docs/qualification/3d-learning-gpu/analyze.py /tmp/vllm-3d-evidence/final-8f0/profile/recording --source-ref 8f0ca92d2ee2741723ba20e5637decb91982694b --require-profile --output /tmp/vllm-3d-evidence/profile-analysis.json
python3 docs/qualification/3d-learning-gpu/analyze.py /tmp/vllm-3d-evidence/final-8f0/cancel-profile/recording --source-ref 8f0ca92d2ee2741723ba20e5637decb91982694b --require-profile --require-cancellation --output /tmp/vllm-3d-evidence/cancel-analysis.json
python3 docs/qualification/3d-learning-gpu/analyze.py /tmp/vllm-3d-evidence/extra-8f0/pressure-preemption/recording --source-ref 8f0ca92d2ee2741723ba20e5637decb91982694b --require-preemption --output /tmp/vllm-3d-evidence/preemption-analysis.json
```

The final lifecycle delta has its own archive and exact source ref:

```sh
tar -xzf docs/qualification/3d-learning-gpu/archives/lifecycle-dcab.tar.gz -C /tmp/vllm-3d-evidence
python3 docs/qualification/3d-learning-gpu/analyze.py /tmp/vllm-3d-evidence/lifecycle-dcab/staggered/recording --source-ref dcab49f6ff0e194711c031f4d72399b93f300877 --output /tmp/vllm-3d-evidence/lifecycle-normal-analysis.json
python3 docs/qualification/3d-learning-gpu/analyze.py /tmp/vllm-3d-evidence/lifecycle-dcab/cancel-profile/recording --source-ref dcab49f6ff0e194711c031f4d72399b93f300877 --require-profile --require-cancellation --output /tmp/vllm-3d-evidence/lifecycle-cancel-analysis.json
```

Analyze `extra-8f0/recovery-late-intake/recording` and `ui-capture-8f0/recording` with the same exact `--source-ref` and no feature-required flags. The expected preparation failure intentionally fails normal inference analysis; inspect its retained assessment and the independent classification above. To inspect historical attempts, extract their separate archives and use their actual declared source refs or the separately retained transient source bytes.
Use `--require-preemption` only for an explicitly retained case containing actual preemption. `--allow-incomplete` inspects an authoritative prefix without claiming a complete run. Exit codes are 0 for passed recorded-contract checks, 1 for contradictions/malformed data and 2 for required unavailable evidence. Without `--source-ref`, the analyzer compares current working-tree source bytes; it never silently accepts a declared commit after those bytes change.

`qualify.py` validates and writes the seven public request payloads by default. It makes actual API/inference calls only with `--execute`. For complete raw native files, run it on the isolated controller host with that host's recording root; add `--gpu-memory` there for correctly scoped read-only memory receipts. [Execution commands and process ownership](profile-lifetime.md) explain the operator workflow. Credentials, model weights and virtual environments are excluded from the retained evidence archives.
