# Meaningful H100 demo recordings

This change adds the version-2 Meridian operator workload to each supported captured configuration. The six fictional engineering requests ask for incident impact, a cause assessment, rollout guardrails, a scaling investigation, capacity planning, and an on-call handover. Requests 1, 2, and 4 share exactly the same incident document; their tasks differ. This preserves a real prefix-reuse lesson while making the inputs useful to read.

The configured output caps are **64, 96, 128, 160, 192, and 256** tokens. Every bundled new capture must have actual accepted outputs within 64–256 tokens, matching recorded output IDs and completion reasons. Natural stops remain allowed. Length stops can truncate sentences; generated text is preserved exactly. These are serving-mechanics demonstrations, not response-quality or performance benchmarks. Each case has `output-quality-notes.json` describing concrete limitations of the real responses.

## Configuration and timing

The three planned arrival groups contain Requests 1; Requests 2–3; and Requests 4–6. They are client arrival groups, not a promise of three engine iterations. Captured scheduling shows many iterations, multiple concurrent requests, and prefill/decode overlap. Dense and hybrid offsets are 0/20,000/20,025 ms; LoRA offsets are 0/40,000/40,025 ms. See [the measured timing failure and correction](capture-timing.md).

The captures preserve pinned model revisions, original 2048-token context limits, original dense/hybrid scheduling and cache-byte budgets, observed precision, and actual adapter routing. Diagnostic capture extends to 1024 iterations. The canonical editable preset adapts LoRA routing/timing while preserving the selected model, host, and precision. The trained rank-8 adapter demonstrates routing and cache isolation; no adapter-quality claim is made.

Supported new cases are BF16 weights/BF16 KV, online FP8 weights/BF16 KV, BF16 weights/FP8 E4M3 KV, LoRA on all requests, mixed base/LoRA requests, and Qwen3.8-27B hybrid attention/Gated DeltaNet. Unsupported FP4 and Flash-Next combinations remain explicitly unsupported and have no fabricated capture.

## Evidence and distribution

Recorded on the user-provided H100 PCIe at `209.20.157.120`, with vLLM 0.30.0, Torch 2.13.0+cu130, and GPU UUID `GPU-aeff92e5-dfdc-5620-d255-d5ad1d0987be`. Engine source SHA256 is `db403bda651d8fd6fe8308a5889f0d465b18af2e63a1974ba5ee1aae4017107b` over 41 runner/instrumentation files. Every case is checked against that source identity.

The per-case `assessment.json` checks production integrity, pinned identity, native packed-row/scheduler equality, all iterations and layers, precision/routing, prefix lineage, memory, actual output counts, and cleanup. `expansion-assessment.json` checks arrival semantics and overlapping work. `native-tokenizer-assessment.json` compares instrumented prompt IDs with the pinned tokenizer encoding of the exact authored prompts. `native-independent-assessment.json` independently compares every raw CUDA record's ID/order/fields and every declared CPU correlation candidate with the expanded browser artifact. Unmapped CUDA is allowed; token/head duration is never inferred.

[The bundle receipt](bundle-receipt.json) lists immutable artifact bytes, SHA256 hashes, source identity, and assessment hashes. New captures use new `gpu-h100-realistic-*` IDs. Original short-capture bytes are preserved and the historical H100 entries are labeled **Archived short demo**. The new BF16 capture is the default Explore recording. No Sites deployment is part of this change.

The long captures exposed the former raw/profile size limits. Limits are now explicit and bounded: 4 GiB decompressed raw input, 4,000,000 normalized activities, and a 512 MiB browser attachment. Above a 64 MiB plain attachment threshold, unrelated CPU detail is reduced and repeated activity names/scopes/cache participants are losslessly interned. All CUDA activities and their correlations, launches, scopes, and packed participants remain; complete raw native evidence is retained. The manifest still identifies these normalized profiles as partial because unrelated CPU detail was removed. Oversize or malformed evidence fails closed.

The original reducer needed 92.1 seconds to fold the real BF16 trace. The optimized player uses privately owned mutable working states with at most 128 independent checkpoints while keeping public event application immutable. All 379 iteration endpoints plus 100 other raw prefixes matched the saved original reducer exactly. Player construction took 0.82 seconds in that comparison. Shared immutable observation indexes avoid rebuilding full activity maps on every frame. See the task-local benchmark/equivalence receipts under `output/playwright/oct13-realistic-demos/`.

## Verification

The browser verification uses the **Codex in-app browser at a 1920×1080 DOM viewport**. Screenshots retain the browser's native raster size. Test evidence includes opening actual recordings via model/precision/LoRA filters, request prompts and accepted output counts, host iteration labeling, GPU layer highlighting in the overview, backward seeks, inspector reset, and memory breakdown. The removed phrase **Host time · scheduler participants** is absent while the single **Tokens scheduled this iteration** heading remains.

The first long capture hit the old profile limits, and the initial LoRA-all attempt missed the intended completed-prime timing. Both attempts and their failure receipts are retained. A configuration-switch test also exposed a renderer crash while the old and new large profiles overlapped in memory; the corrected behavior passed BF16 → FP8 weights → FP8 KV → BF16 switching. A cold load of the largest hybrid recording also passed. These failures are not reclassified as successful captures.

The final case matrix, automated check results, browser receipts, and independent review results are retained alongside this report. LangSmith was waived by the user for this project.

## Final captured cases

| Configuration | Accepted output tokens for Requests 1–6 | Iterations | CUDA records |
|---|---|---:|---:|
| bf16 | 64, 96, 128, 160, 192, 189 | 379 | 215,378 |
| fp8-kv | 64, 96, 128, 160, 192, 183 | 410 | 247,927 |
| fp8-weights | 64, 96, 128, 160, 192, 194 | 418 | 269,472 |
| lora-all | 64, 96, 125, 160, 192, 150 | 373 | 377,277 |
| lora-mixed | 64, 96, 128, 160, 192, 152 | 418 | 356,322 |
| qwen38-27b | 64, 96, 128, 160, 192, 256 | 353 | 929,970 |

All six copied cases pass 44 production assessment checks, exact prepared-plan comparison, tokenizer-ID parity, immutable artifact hashes, and independent raw CUDA auditing. Total: **2,351 iterations and 2,396,346 CUDA records/CPU links, with zero independent mismatches**. Hybrid generation reaches its caps inside native reasoning, before a final answer; those real outputs are preserved and clearly described in the library.

The recording loader now releases the old heavy view before replacement, uses an explicit read-only profile view while preserving the isolated public projection API, and retains only a source/URL for the latest capture. Only the UI-owned live run is downloaded. One live-finalization owner handles both drained trace pages and completed-file recovery, with abortable artifact reads. A real H100 UI capture (`run-d61d7481fe0b`) completed with 64 accepted tokens, opened automatically, and reopened through Latest capture. The full six-request draft was restored afterward.

Automated final checks: **335 Python, 116 web, and 78 replay tests passed**, plus TypeScript/production build and diff-whitespace checks. Existing production bundle-size warnings remain. Per-case production-path load benchmarks took 2.44–4.64 seconds in Node; the largest hybrid run peaked at 2.33 GiB RSS with transient parser allocations. These are local load measurements, not GPU-performance comparisons. Browser switching and interaction verification is separate evidence.

All three [independent final reviews](final-reviews.md) approved without a demonstrated blocking finding. The evidence reviewer independently repeated every published raw CUDA/CPU audit and re-tokenized all 36 prompts; results matched the retained receipts. A nonblocking unused fallback helper remains as a documented follow-up.
