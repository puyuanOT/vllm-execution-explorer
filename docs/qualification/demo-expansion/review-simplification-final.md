# Final simplification review

Reviewed the incremental demo expansion in the existing localhost Configure/Explore application, beyond the previously reviewed local integration. The review covered observed demo facets and capture preparation, memory recording/projection/accounting, profiler retention and lossless name encoding, FP8 numerical representations, hybrid layer presentation, the stable iteration summary, shared cache marks, and their focused tests. No GPU execution or browser actions were performed by this reviewer.

## Resolved finding

`assess.py:finalization_evidence` required the presentation sentence `Recording finalized.` in addition to completed, finalized, non-provisional receipt flags, a matching completed manifest, source identity, and successful matching owner exit. The genuine archived trace API in `runner/serve.py` deliberately returns an empty message when that run is no longer the controller's current run. This unnecessary wording dependency rejected the otherwise valid live recording `run-308ba3e3d031`.

The authorized repair changes only the qualification assessor and its test. Message text remains in diagnostic evidence; semantic finalization and identity guards remain required. The new regression obtains the archived response through the actual production `/api/runs/{run_id}/trace` endpoint, verifies its empty message is accepted, and rejects a mismatched owner PID. Missing receipts, unfinished flags, wrong run identity, failed owner exit and malformed/mismatched manifest cases still fail. All **20 focused assessor tests passed**, and `git diff --check` passed.

The original failed assessment is preserved as `live-ui/six-request/review-independent-assessment-before-archived-message-fix.json`. The raw archived `final-state.json` and recording files were unchanged. The separate corrected `live-ui/six-request/evidence-assessment.json` passes all **41 mandatory checks** against capture source `56b4d03456a627b250a87e322a54cbf035643e730db8d2c27c2f3b29cff086bc`: 39 iterations, 21,357 CUDA activities and six accepted outputs of six tokens each.

## Final assessment

**No significant unresolved simplification finding.** The added helpers have concrete shared responsibilities and do not introduce a second product shell or a parallel capture implementation.

- `web/src/shell/demoCatalog.ts` derives facets from retained native evidence, preserves historical unknowns, and prepares the existing Configure flow. Its FP8 alias repair uses the qualified native encoding rather than treating raw `uint8` or a requested setting as sufficient evidence. One shared LoRA intent guard serves both readiness and capture validation.
- `runner/gpu_memory.py`, `replay/src/memory.ts` and `GpuMemoryPanel.tsx` separate collection, projection and presentation. Device/process/framework counters overlap; storage attribution deduplicates backing allocations. Failed samples, mismatched UUIDs/clocks, future samples, trace gaps, inventory age and cleanup invalidation are distinct guards that should remain.
- The profiler's activity-count bound, browser-byte bound and optional exact-name dictionary address different observed limits. `_link_cache_writes` replaces repeated activity scans with one index while preserving launch completeness. The encoder/parser/independent assessor checks prevent lossless encoding from obscuring incomplete evidence; they are justified rather than needless duplicated production paths.
- `BlockMarks.tsx` shares allocation-state marks and request-identity strips between overview and pool views. `ModelIterationBar.tsx` places verified host or native participation above the canvas. Recurrent state remains distinct from attention KV. Full scheduled counts and bounded visible cell counts are intentionally separate.
- The final toolbar repair is limited to three `shrink-0` additions. Direct desktop/mobile Edit as new, memory expansion/collapse, reset, native inspection and request-lane receipts demonstrate normal flow and sizing; no pointer or stacking workaround was introduced.
- The tests exercise actual artifact integrity, packed-row coverage, native activity identity, independent FP8 byte patterns, corruption, route isolation, source/receipt mismatches and accounting boundaries. The all-six catalogue checks reuse one assertion helper and retain meaningful case differences; no substantial low-value mirror test or dead path was identified.

## Evidence considered

`README.md`, `case-matrix.json`, `bundle-receipt.json` and `review-independent-cases.json` establish all six actual bundled cases against the same final capture source. The independent case receipt records byte-preserved bundles, exact native CUDA event comparison and **41 passing mandatory checks per case**. The facet/numerical browser receipts cover all six exact filters, all 1,152 displayed FP8 values, mixed routing and the 27B recurrent/attention distinction.

`iteration-summary-layout.md` and the final desktop/mobile layout receipts verify the wrapped toolbar repair and direct navigation to the original Configure page. The reported integrated verification includes 70 replay tests, 103 web tests and a successful production build; this reviewer did not rerun those unchanged suites.

The additional completed FP8-alias UI capture `run-5495998a324d` has both `live-fp8-alias-ui/six-request/evidence-assessment.json` and `review-independent-assessment.json` passing all 41 checks against the same source, with matching owner exit 0, 44 iterations and 25,872 CUDA activities. `live-memory-ui.json` retains the actual transition from waiting for the first sample to measured NVML data, UUID/PID details and memory expansion/collapse. Its requested `fp8` cache is observed as qualified `fp8_e4m3`; post-owner cleanup evidence remains separate from retained pre-exit measurements.

No further source simplification is recommended for this increment.
