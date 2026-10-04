# Real GPU qualification analysis

`analyze.py` reads a downloaded recording without modifying it. It uses Python's standard library and the actual TypeScript replay projections through Node. It prints a compact evidence report without prompt text, decoded output text, credentials, or tensor dumps. No remote command, model load, inference, or network access occurs.

For the ordinary canonical serving capture:

```sh
python3 docs/qualification/3d-learning-gpu/analyze.py /absolute/path/to/downloaded-recording --output /absolute/path/to/serving-analysis.json
```

For the eager diagnostic capture, require measured layer coverage as well:

```sh
python3 docs/qualification/3d-learning-gpu/analyze.py /absolute/path/to/downloaded-diagnostic-recording --require-profile --output /absolute/path/to/profile-analysis.json
```

Add `--require-prefix-reuse`, `--require-preemption`, or `--require-cancellation` for cases explicitly intended to exercise those events. `--allow-incomplete` inspects the authoritative host prefix without claiming full-run qualification. Exit codes are 0 for passed recorded-contract checks, 1 for contradictions or malformed data, and 2 for required evidence that is unavailable. An exit-zero report does not qualify UI rendering, GPU/browser performance, all supported model families, or a production profiling mode.

Use `--source-ref 7da8f31` when the deployed capture came from that preserved Git revision and local source changed afterward. The digest is recomputed from that ref's exact runner/instrumentation bytes. Without this option, the comparison uses files currently on disk; a mismatch means the source identities differ and requires a deployment receipt, rather than silently accepting a declared revision.

For a deployed source tree containing reviewed fixes before its next commit, use `--source-root current` (or the path to a separately preserved tree). This compares the actual source bytes and makes no commit-identity claim. Final qualification should preserve and use the committed deployment ref.

Every request's retained input IDs/text, accepted output IDs/text emissions, effective sampling settings, output limit, priority and terminal event are inspected. The shareable analysis contains per-stream SHA-256 hashes/counts rather than copies of prompt/output text; the full recording archive remains the source for human inspection. Individually decoded token fragments are not assumed to concatenate into the original prompt or complete output text.

The canonical comparison is Qwen/Qwen3-4B-Instruct-2507 at `cdbee75f17c01a7cc42f958dc650907174af0554` for both model and tokenizer: 36 dense layers, hidden size 2560, FFN size 9728, 32 query heads, 8 KV heads, and explicit head dimension 128. On real captures the source must be `loaded_model_config`, not the fixture's public-config reference. Preserve deployment/runtime receipts alongside the recording; the script checks source fingerprints but cannot independently observe the remote model after the fact.

The checks keep three evidence boundaries separate:

1. Complete physical inventory includes every block and its allocation generation, references, hash and reserved-null state. Token-slot occupancy additionally requires qualified single-GPU, unsharded, causal full-attention cache groups and group/layer mapping. Cached zero-reference blocks can be available for reassignment while retaining prefix contents.
2. Scheduler reservations and packed model inputs are distinct from computed K/V. Every recorded write must follow its step's returned result and stay within the scheduled range and physical generation. Content IDs must agree with the request's prompt/output positions. The final sampled output remains uncomputed unless a later scheduled input actually covers it. No numerical K/V readback is claimed.
3. Packed rows must come from the returned post-reorder `_prepare_inputs` CPU metadata. Their request/position multiset must agree with that scheduled step; their order is the runner's observed order. Padding has no request/token identity. Profiling remains in native trace-local microseconds and does not move the host replay cursor. Layer GPU coverage requires raw native launch/external-ID correlation to containing explicit step/layer scopes, not kernel names or CPU scope duration. Missing correlations remain unavailable; unmapped activity counts are retained.

The existing host-empty authored profile fixture is a loader/UI contract artifact and is expected to fail full-capture analysis. The ordinary teaching fixture can smoke-test the helper with `--allow-fixture`; its report is explicitly scoped to fixture contracts and never qualifies GPU evidence.

This analysis is independent of the user's LangSmith waiver: the source fingerprint can still be present with tracing unconfigured, and no LangSmith persistence claim is required here.

## First actual serving capture

[run-225baaef2a99 analysis](normal-staggered-run-225baaef2a99.json) passes the recorded-contract checks against deployed revision `7da8f3105819581c21ad8cc4d59a78387ba0d7cc`. The actual initialized architecture matches the canonical Qwen3 geometry. All 4,477 physical blocks are projected, including block 4,476; one null block is reserved, leaving 4,476 initially available blocks. All six requests retained their input/output streams and each accepted 64 output tokens before a length completion. Every final sampled token remains `not_computed` with `computedEver=false`, and the first scheduled input was reserved before its result.

This staggered case exercised no prefix hit, eviction, preemption or cancellation. Native profiling was disabled and is explicitly unavailable for this case. The analysis does not turn these unexercised features into successful qualification. The source snapshot predates subsequent preparation/shutdown fixes; repeating captures under the final deployed revision remains part of the acceptance work.
