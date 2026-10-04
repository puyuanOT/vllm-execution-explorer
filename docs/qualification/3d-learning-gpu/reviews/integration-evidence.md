# Independent final integration and evidence review

Reviewed on 2026-10-03 by a non-author reviewer. Implementation scope is
`6918efa..8f0ca92d2ee2741723ba20e5637decb91982694b`, with the completed GPU
qualification README, browser assessment and retained evidence package.
No implementation, helper, recording or remote-runtime changes were made in this
review.

**Result: no actionable integration/evidence contradiction or unresolved
significant issue found within the documented qualification scope.** This is an
independent inspection of actual retained data and fresh analysis, rather than an
endorsement of the existing report summaries.

## Independently reproduced checks

- Verified all three archive sizes and SHA-256 digests against the evidence
  index and `SHA256SUMS`. Verified every indexed member's name, byte size and
  digest: 228 historical members, 160 pre-STOP-fix members and 167 final members.
  There are no unindexed regular files or missing indexed files. Checked archive
  paths and exclusions; no private-key payload, `.pem`, model-weight,
  virtual-environment or dependency-tree asset was found.
- Extracted `qualified-8f0.tar.gz` into a separate temporary directory and reran
  `analyze.py` on all ten inference captures with the exact final source ref.
  Applied the profile, cancellation, prefix-reuse and preemption required flags
  to the relevant cases. Every fresh report passed, with 34–46 recorded-contract
  checks per capture. These runs use the actual TypeScript replay projections,
  including pool enumeration, result-gated token state and backward seeking.
- Independently recomputed the committed distributed-source fingerprint from
  all 38 runner/instrumentation `.py` and `.json` files. It is
  `83df6c7bc792273eabb37d22ab76a29476b6c69a9325848505deafa1201e5056`, matching the
  initialized capture manifests, final index and deployed-tree receipts.
- Independently counted admissions and accepted-token events across the raw
  final recordings: ten inference captures contain 32 admitted streams and
  2,645 accepted output tokens. The separate preparation failure has zero
  admissions and zero outputs. All eleven recordings retain distinct owned
  worker PIDs with exit status zero and exit/join observations. Each corresponding
  post-close GPU-host receipt reports the same GPU UUID, 3 MiB used and no compute
  applications. The runtime receipt preserves the original controller as alive.
- Read the actual authored request payloads, prompt text/IDs, generated text
  deltas/IDs, request limits/priorities, terminal events and effective engine
  settings. The ordinary and eager-diagnostic runner choices are recorded
  separately. Blank browser revision overrides resolve `main` to the canonical
  immutable model/tokenizer SHA; the UI precision claim remains requested
  `auto`, without asserting a separately observed resolved tensor dtype.

## Direct checks beyond the existing analyzer summaries

The raw prefix trace records the follower's 272-token hit in blocks 1–17 and its
first newly scheduled range of positions 272–285. The raw chunked case starts
with 16 short-prompt plus 48 long-prompt inputs and subsequently mixes one decode
input with 63 prefill inputs. The priority case's actual admission priorities are
0, 5, −5, while its first waiting queue and first execution order are −5, 0, 5.
This supports native priority ordering without equating authored/dispatch order
with scheduler order. The short pressure case contains no preemption; the
separate extended case contains three actual preemptions with computed cursors
240, 222 and 224 reset to zero. The reports preserve that distinction.

The cancellation trace retains three returned prefill results, twelve block
writes covering 192 prompt positions, native `FINISHED_ABORTED` at the returned
finish hook, and a cancelled HTTP terminal. It contains zero accepted outputs.
Its three-of-sixteen-step profile remains explicitly partial, and the sampled
final-token assessment is not applicable.

For the full native profile, independently checked all 4,776 normalized CUDA
intervals against their raw event indices, timestamps, durations, launch/native
correlation IDs and containing explicit step/layer/component scopes. Independently
computed first-batch layer-0 attention's device activity span as approximately
9,708.446 µs and interval union as 157.728 µs, matching the retained UI screenshot.
The browser's 16.154 ms cursor is relative to the native extent origin; at that
offset it intersects the layer-0 attention CUDA interval starting at approximately
16.143535 ms. The first packed batch is actually profile-b positions 0–27,
followed by profile-a positions 0–26. All 55 rows are real; nonzero padding was
not exercised. The separate cancellation profile's 1,602 CUDA intervals also
pass fresh raw-correlation analysis.

Compared the final pool screenshot with its raw source recording. The manifest
and inventory enumerate 5,414 blocks, including reserved block zero and final
block 5,413. At host event 27, the block-1 write follows result event 26 and covers
sample-2 positions 0–4. The sidecar retains IDs `[675, 264, 7513, 6722, 13]` and
fragments `Name`, ` a`, ` European`, ` capital`, `.`. That agrees with the five
computed slots and live binding visible in the screenshot. The final-block
browser state agrees with its initial available generation-zero inventory entry.
The native screenshot also retains host replay at zero, an unverified manifest
qualification label, separate CPU/device measurements and the 0.001× speed.

Inspected the retained natural-EOS correction independently of the length-limit
cases. The live-intake receipt accepts `late-immediate`, whose answer is
`admitted`; the planned ten-second request answers `scheduled`. Accepted IDs
include final EOS 151645, and native stop completions occur at sequences 34 and
59 before their HTTP stop terminals. The fresh replay analysis leaves final EOS
uncomputed. The historical `75f2ccd` recovery archive actually lacks both native
completions, and its preserved report correctly remains failed. The final report
does not erase that contradiction or use its HTTP outcomes as native evidence.

Reran normal inference analysis on the one-byte preparation-failure recording.
It remains failed, with empty-event and unavailable initialized-GPU contract
checks. Its separate failure assessment records the native capacity error,
failed finalization, worker exit and restored memory baseline. The documentation
correctly classifies inference, loaded geometry and pool qualification as not
applicable. Its separate deployed-tree receipt does not fabricate an initialized
capture fingerprint.

## Meaning and limits

The retained natural-language LLM-domain responses are on-topic but mostly
length-truncated, and some explanations are imprecise. Repeated-token workloads
produce repetitive or unrelated prose. The six short browser prompts are plain
text rather than chat-formatted turns; some continuations do not answer the
question before the eight-token ceiling. The documentation and independent
stream reviews describe those outputs as workload observations, not validated
teaching content or answer-quality evidence.

This review did not launch another GPU workload or independently repeat live
browser interactions. It checked complete raw recordings, runtime/ownership and
memory receipts, retained browser states/screenshots, reproduction commands and
fresh local analysis. Test/build/CI counts are supported by the retained
validation receipt; they were not independently rerun here.

The qualified scope remains vLLM 0.30.0, this single-GPU dense Qwen3 layout and
the explicitly selected eager V1 diagnostic path for native layer timing.
It establishes logical KV occupancy and token/batch participation, not numerical
K/V values or individual token/head latency. Host and profiler clocks remain
separate. Actual nonzero padding, unsupported cache/model layouts, automatic SSH
bootstrap, browser GPU completion/FPS and GPU allocation bytes remain outside
the claimed evidence. The scene observations are CPU submission/resource
counters, and the existing production bundle-size warning is retained. LangSmith
was explicitly waived by the user and is not an unresolved gate.
