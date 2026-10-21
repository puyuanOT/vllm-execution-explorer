# Independent correctness review: demo expansion

No significant unresolved correctness issue was found in the final reviewed changes and retained qualification evidence. This conclusion follows code inspection, independent artifact checks, current assessor execution, and inspection of the actual browser receipts. It is scoped to the qualified configurations and behavior below.

The review compared the working tree with baseline `92fb11b00d889cb58136a987bcf09687c0381acd`, including the existing local UI work. All eight independently assessed final-source captures match runner/instrumentation source SHA-256 `56b4d03456a627b250a87e322a54cbf035643e730db8d2c27c2f3b29cff086bc` across 40 source files. The reviewer did not launch GPU work or operate the main verification browser.

## Corrections checked during review

- A missing final-state file could previously qualify as successful finalization. The assessor now requires a present receipt for the same run, finalized/completed/nonprovisional flags, matching persisted manifest/source/process identity, and a matching owner PID that exited successfully after all recorded events. Missing receipts and mismatched identities fail.
- Native CUDA `kv_cache_dtype=fp8` can produce a qualified byte cache with recorded E4M3 encoding. The picker previously reported that known encoding as unknown because it consulted only the native alias. It now uses qualified observed `uint8` numerical encoding; an unqualified ambiguous alias remains unknown. The real alias capture independently confirms this path.
- A newly preparing live run previously said that a new capture was required before its first memory sample arrived. It now shows a waiting state. Expanded replay memory guidance now directs users to a retained successful sample when the playhead precedes it, while preserving new-capture guidance for recordings without measured memory.
- The archived `/trace` endpoint legitimately has an empty current-controller message. The assessor no longer treats presentation wording as a finalization requirement. All identity, completion, source, owner-exit and clock guards remain. A regression reads the real archived FastAPI endpoint and still rejects a wrong owner PID.
- The final wrapped toolbar overlap was repaired with natural-height flex items. The bounded layer matrix now distinguishes its shown positions from the full scheduled/native packed-row count. These repairs were checked against the final source and direct desktop/mobile or host/native-clock browser receipts.

## Independently verified captures

The current production assessor passes all 41 mandatory checks for each of the six bundled cases. Each has six real requests in three matching arrival groups, actual returned content, and six accepted output tokens per request. The five 4B cases retain identical prompts, arrival offsets, and output limits; their precision and adapter routes differ intentionally.

| Bundled case | Actual run | Model iterations | Retained CUDA activities |
| --- | --- | ---: | ---: |
| BF16 weights / BF16 KV | `run-af950e986ddc` | 44 | 24,264 |
| FP8 weights / BF16 KV | `run-b8840655c01c` | 44 | 28,440 |
| BF16 weights / FP8 KV | `run-5fb78beeb09f` | 39 | 22,785 |
| LoRA on all requests | `run-aa48e4ab22a1` | 39 | 40,048 |
| Mixed base + LoRA | `run-0e45b935707d` | 42 | 36,893 |
| Qwen3.8-27B hybrid | `run-0ea99962864c` | 20 | 57,666 |

Every bundled manifest, event/content file, normalized profile and raw native trace was compared byte-for-byte with its original capture and checked against the bundle hashes and sizes. All 210,096 retained CUDA activities across 228 iterations match the raw trace's original IDs, names, start times, durations, devices, streams and order. Runtime-flow step ownership and retained CPU correlation references were checked directly. The 27B name dictionary expands to the original native names. CPU-detail bounds do not omit GPU activities or their required correlations, and all normalized attachments remain within 64 MiB. See [independent case evidence](review-independent-cases.json), [bundle receipt](bundle-receipt.json), and [case matrix](case-matrix.json).

Two additional actual Configure-to-SSH captures also pass the current 41-check assessor: `run-308ba3e3d031` and the native FP8 alias run `run-5495998a324d`. The reviewer independently compared their 21,357 and 25,872 CUDA activities with the raw traces across 39 and 44 iterations, respectively. Each returns six actual six-token outputs. Both have matching finalized owner/source receipts and strict post-owner GPU baseline proof. See [first live assessment](live-ui/six-request/review-independent-assessment.json), [first live native comparison](live-ui/six-request/review-independent-native-check.json), [FP8 alias assessment](live-fp8-alias-ui/six-request/review-independent-assessment.json), and [alias native comparison](live-fp8-alias-ui/six-request/review-independent-native-check.json).

## Semantics and state checks

Memory counters are explicitly separate overlapping views. The collector retains physical GPU UUID/PID identity, sample query boundaries, unavailable attempts, and host-clock phases. Loaded storage is deduplicated by backing allocation, including the hybrid allocation shared by recurrent and attention state. Cleanup invalidates storage attribution instead of displaying earlier weights/caches as current allocations.

The original FP8 pre-exit retained measurements remain intact: 5,469,372,416 bytes for the weights case and 8,879,341,568 bytes for the KV case. Their pre-exit 90% decrease checks remain false. Successful matching owner exit and independently timed before/after management receipts prove a separate allocation-lifecycle outcome: same local host clock, UUID/physical total, owner PID absent, and usage returning to the pre-run baseline within the disclosed 64 MiB tolerance. This does not identify particular freed tensors or establish the cause of pre-exit retention.

Native GPU and controller clocks remain distinct. Measured activity describes shared batch/layer operations; scheduler participation and illustrative layer flow do not become individual token/head execution-time claims. Numerical frames retain operation, layer, token position, cache group, physical block, allocation generation, slot and observation-boundary identity. Raw FP8 bytes, decoded storage values and recorded-scale reconstruction remain separate representations; missing coordinates are unknown. The independent numerical qualification and real UI checks retain 1,152 exact displayed-value comparisons in [facet verification](../demo-facets/README.md).

The 27B recording preserves 48 Gated DeltaNet and 16 attention layers, with 64 FFN scopes. Recurrent cells disclose that per-token attention KV does not apply. Numerical recurrent-state contents are not invented. Its matrix displays 128 shown positions separately from all 784 scheduled positions in both host and native profile modes. These counts and labels were checked in the [host receipt](../../../output/playwright/qwen27-subset-independent-host.txt) and [native receipt](../../../output/playwright/qwen27-subset-independent-profile.txt).

Configuration and Explore remain accessible. Requested compute, weight storage and KV dtype remain separate from observed formats. Saved draft routing/persistence, pending All/Mixed adapter intent, selected adapter artifact identity, request routing and same-route prefix lineage are checked. All six exact picker combinations open their corresponding retained real artifact. Status uses neutral shape/pattern/fill and request identity uses side strips. The stable DOM iteration summary remains readable while the camera moves. Direct Edit as new, Reset view preserving the playhead, memory expansion/collapse, native iteration inspection and Request lanes pass at 1200×717 and 390×844. See [layout evidence](iteration-summary-layout.md).

The real live UI receipt follows Configure, SSH host verification, actual capture submission and Explore. It shows the corrected waiting state followed by three NVML sample observations, successful sample counts increasing from 2 to 41 to 83, matching GPU UUID/PID, and direct memory expansion/collapse. See [live interaction receipt](../../../output/playwright/live-fp8-alias-capture-memory.txt).

## Checks and limits

The reviewer independently reran 20 assessment-helper tests and 10 demo-facet tests after the final repairs. Earlier focused Python and replay checks also passed. Final retained full-suite logs show [287 Python tests](python-final-checks.txt), [70 replay tests](replay-final-checks.txt), and [103 web tests](web-final-checks.txt) passing. The [production TypeScript/Vite build](build-final-checks.txt) and diff whitespace check passed; the build retains its existing large-chunk advisory.

This review does not establish exhaustive numerical tensor capture, individual token kernel timing, adapter output quality, benchmark representativeness or every model/precision/adapter combination. Selected numerical slices are bounded; the hybrid demo captures no numerical recurrent-state values. Flash-Next and FP4 remain explicitly unqualified on this one-H100 deployment. Original failed attempts and their measurements remain available. No Sites deployment or Git publication forms part of this qualification.
