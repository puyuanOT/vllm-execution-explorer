# Profiler finalization verification

The KV-write linker previously scanned every activity twice for every write. It now builds one operation index, retains native activity order, and still marks completion unavailable when any observed launch lacks correlated GPU coverage.

The native trace read is bounded at 192 MiB, independently of the 64 MiB browser attachment. This accommodates the observed 135 MiB hybrid trace with initial Gated DeltaNet compilation. Browser bounds include scheduling rows, write participants and numerical snapshots. If needed, a deterministic fallback omits only unrelated CPU detail: every normalized CUDA activity (including unmapped activities), its CPU correlations, every CUDA launch, and every explicit explorer scope remains. All scheduling steps and captured tensor/write metadata remain. The attachment records original/retained counts, zero omitted GPU activities, and whether GPU coverage was already incomplete. It is marked partial, with an explicit CPU-detail omission reason; the original native trace remains available. If mandatory evidence still exceeds the browser bound, exact repeated activity names are shared losslessly through an optional `activity_names` dictionary and integer `name_id` fields. `normalization.attachment_encoding.kind` declares `activity_name_dictionary_v1`, with exact dictionary/activity counts. The replay parser and independent native-source assessor validate references, reject ambiguous direct names, and expand original strings before projection or native comparison. IDs, ordering, timestamps, device/stream, scopes, correlations, all scheduling rows and cache-write participants remain unchanged. If the encoded mandatory evidence still exceeds the bound, no normalized attachment is claimed.

The HTTP engine-owner join allows 120 seconds for enabled diagnostic profiling, compared with 30 seconds for ordinary cleanup. This covers native stop/export and bounded attachment normalization. Finalization is an explicit state, and pending, timed-out or failed profiling raises a cleanup error rather than claiming a completed capture. The outer recorder marks cleanup errors failed while retaining returned outputs.

Validation:

- Python profiling and HTTP cancellation suites: **34 passed**. Regressions cover one activity pass for 2,000 writes, native ordering, uncovered launches, partial GPU evidence, full scheduling-step retention, CPU-only detail omission, late GPU events beyond the activity limit, combined detail bounds, oversized required evidence, finalization ordering, and cleanup status/deadlines.
- Replay catalog, profile and KV suites: **23 passed**. CPU-detail bounds preserve measured GPU intervals, row identities and cache-write correlations; the projection displays the explicit omission reason. The catalog checks all 44 real BF16 iterations against native packed rows, the three arrival groups, selected tensor snapshots and GPU memory lifecycle, while preserving older fixture expectations.
- Web TypeScript/Vite production build passed. `git diff --check` passed.

The local benchmark used the actual aborted attempt-2 native trace (74,268,875 uncompressed bytes; 235,920 native events; 191,039 normalized activities). Normalization took 0.8203 seconds. Indexing and linking 1,404 operation IDs took 0.0207 seconds. Forcing the byte bound one byte below that trace's encoded size exercised detail reduction in 0.3729 seconds: 54,102 required activities remained in 23,774,024 bytes, with 136,937 unrelated CPU activities omitted and zero CUDA activities omitted.

The benchmark's write descriptors were authored solely to exercise indexing; the aborted run did not retain real scheduling/write metadata, and this benchmark does not claim to recover its capture. Exact selected measurements are in `output/profile-finalization-benchmark.json`. No GPU execution or remote capture was performed for this check; a fresh native capture must verify the final end-to-end result.

The previously captured hybrid trace has 314,071 timed activities, including 46,724 native CUDA activities; 29,984 CUDA activities occur beyond the old first-250,000-activity cutoff. The normalizer now computes scope inheritance and correlation ambiguity from the complete native CPU stream, then selects required evidence before applying the activity output limit. Its 142,538 retained activities occupy 62,763,340 bytes before scheduling/write metadata. Every CUDA activity matches the unrestricted baseline exactly, including original ID/order, timestamps, duration, device/stream, scope, and related CPU IDs. All 16 recorded iteration scopes survive. Native normalization took 1.5661 seconds and encoding took 0.1973 seconds; the secondary byte-detail reduction can reduce this source to 100,168 activities / approximately 49.97 MB without omitting CUDA activities. Combined bounds retain original source counts rather than resetting them. The exact benchmark is `output/hybrid-profile-normalization-benchmark.json`; it uses only offline native evidence and does not reconstruct lost packed rows or numerical snapshots.

The second hybrid attempt (`run-ffa6b814452a`) contains 385,458 timed native activities and 57,746 CUDA activities across all 20 host iteration scopes. After unrelated CPU-detail omission, the activity body alone occupies 61,361,327 bytes; repeated names account for 22,663,042 bytes while its 636 unique names occupy just 45,735 bytes. Offline lossless encoding produces 39,485,274 bytes, leaving 27,623,590 bytes under 64 MiB for real packed rows, writes and other metadata. Every GPU object equals the unrestricted normalization exactly, and all 123,892 retained activity objects round-trip identically, including source IDs, native names, order, timing, scopes and CPU references. Encoding took 0.6671 seconds. The benchmark deliberately forces a 54 MiB activity-only bound to exercise the optional encoding; the production attachment bound remains 64 MiB. Exact measurements are in `output/hybrid-attempt2-name-encoding-benchmark.json`. The failed run did not retain packed-row/cache-write metadata, so this benchmark does not reconstruct or qualify that capture; a fresh complete capture is required.

## Final complete capture verification

The fresh complete captures required by the earlier offline checks are now
finished. All six [final catalogue cases](README.md) use frozen runner and
instrumentation SHA256
`56b4d03456a627b250a87e322a54cbf035643e730db8d2c27c2f3b29cff086bc` and pass
all 41 mandatory evidence checks, including actual native packed rows,
per-layer scopes, write participants and successful profiler/owner finalization.
Two independent reviewers compare all 210,096 retained CUDA activities across
228 host iterations to original raw native evidence, preserving exact
identity/order, names, timing, devices, streams, scopes and correlations.
All 30 bundled recording files are byte-identical to their original capture
artifacts; the native compressed traces remain available.

The final 27B run `run-0ea99962864c` retains all 57,666 CUDA activities across
20 iterations, with 123,732 total retained activities and 637 exact activity
names in a 53,041,681-byte attachment. Only explicitly declared unrelated
CPU detail is omitted. This qualifies the actual complete hybrid recording;
the earlier failed attempts and benchmarks remain historical evidence.

Three further actual UI captures finalize successfully against the same
frozen source. Independent raw-native comparisons verify all 21,357, 21,357
and 25,872 CUDA activities respectively. The
[final evidence review](review-evidence-final.md) records these comparisons
and their 41/41 passing assessments. Full automated checks finish with
287 Python, 70 replay and 103 web tests passing, plus TypeScript/Vite build.
