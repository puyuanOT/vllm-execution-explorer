# GPU memory accounting contract

The implementation records management API memory samples in `events.jsonl` with
the execution process's `time.monotonic_ns` clock. Native CUDA profiler activity
timestamps remain a separate clock. The host playhead selects the latest returned
memory sample; measurements are never interpolated between samples. The memory
plot starts at memory capture, while request dispatch retains its controller origin.

`runner/gpu_memory.py` starts before model construction and prefers NVML. Its
fallback uses `nvidia-smi` management queries, whose MiB values have lower precision.
The requested polling interval is 100 ms; query failures retain explicit unavailable
events. Exact native Worker boundaries distinguish model loading, available-memory
profiling, cache initialization and native warmup. Explorer warmup, inference and
cleanup have explicit phases. Framework counters are read from the exposed worker
without synchronizing CUDA or initializing a previously uninitialized CUDA context.

Device used/free/total, engine-process usage and PyTorch allocated/reserved are
overlapping accounting views. They must not be stacked or added together. The
largest device sample may miss brief spikes. PyTorch's peak counters cover the
allocator since initialization or its last external reset; native vLLM startup can
reset those counters. Worker UUIDs join management UUIDs after namespace-aware
normalization; local CUDA indices alone do not establish physical identity.

Loaded backing storages are deduplicated by device and pointer inside the worker.
Only opaque allocation IDs leave the process. Recorded categories cover model
parameters/registered buffers, exposed attention KV/recurrent caches, LoRA stacks,
and directly exposed runner tensor buffers. Categories sharing one allocation
appear as shared storage and are counted once. CPU storages have a separate host
view and do not enter GPU totals. Opaque CUDA graph pools, transient activations,
workspaces and other process allocations remain unattributed. A process remainder
is explicitly computed from separately timed samples, not a simultaneous census.
Cleanup invalidates loaded-storage attribution. KV allocation bytes describe the
entire pool; logical release does not imply a reduction in that allocated storage.

The new structural events are `gpu_memory_sample`, `gpu_memory_inventory` and
`gpu_memory_phase`, additive to schema 1.0.0. The optional manifest `gpu_memory`
record declares source, polling interval, process PID, clock and peak semantics.
Historical recordings show unavailable measurements; KV-only inventory does not
become an estimated device/process/framework memory measurement.

Focused tests exercise physical aliasing and shared ownership, CPU exclusion,
query failures, actual collector event ordering, dtype qualification, LoRA artifact
identity, CUDA visibility, UUID normalization, future playback isolation, gaps,
cleanup invalidation and clock mismatch. The integrated localhost browser was
opened through Configure, the historical real GPU demo was loaded, and memory
breakdown was expanded; missing counters remained explicit with zero console
errors. Screenshot: `output/playwright/memory-accounting/legacy-unavailable.png`.
At a 1280×720 viewport, expanding the panel exposed an existing scene flex wrapper
that could shrink below its child canvas and component cards: the cards then
covered the memory header and intercepted the collapse button. A fresh reload
reproduced it independently of development hot reload. The spatial wrapper's
intrinsic minimum height was preserved; expansion and collapse were retested with
direct Playwright clicks at the same viewport and pass. Evidence:
`output/playwright/memory-accounting/legacy-expanded-before-layout-fix.png`.
After-fix evidence: `output/playwright/memory-accounting/legacy-expanded-after-layout-fix.png`.

The first real GPU attempt exposed a bare PyTorch UUID versus a `GPU-` prefixed
NVML UUID. The join was fixed and a regression uses that actual UUID, including
MIG and ambiguous-namespace cases. That attempt must remain in qualification
evidence. The second attempt retained valid measurements and UUID joins but did
not qualify: engine-owner cleanup timed out while native profiling finalized,
leaving memory/profiling marked recording. HTTP request success and the legacy
workload PASS do not establish finalized recording evidence.

The BF16 capture `run-9bef0a1cbe2e` initially passed the generalized read-only
assessment with capture-time source `4150a12c…`. After subsequent production
profiler/LoRA repairs, the frozen capture source became `682eb1df…`; its latest
`bf16-qualified/six-request/evidence-assessment.json` now correctly fails only
source parity. It remains measurement and browser integration evidence, while
fresh captures against the final source qualify the final demo catalogue.
Its observed memory evidence is:
264 successful NVML samples cover preparation, model loading, memory profiling,
cache initialization, warmup, inference and cleanup. Worker PID 6904 joins the
H100 UUID exactly. Deduplicated loaded GPU storages comprise 8,112,045,632 bytes
of model weights/registered buffers, 535,560,192 bytes of attention KV and 536
bytes of named runtime tensors. CPU runtime tensors comprise 36 bytes separately.
Final cleanup detaches the framework and reduces measured process usage to
677,380,096 bytes; remaining CUDA context allocations are not called zero.

All 44 host iterations have native packed-row and CUDA coverage across the actual
36 model layers, with 24,264 runtime-correlated CUDA activities. Only unrelated CPU
detail was bounded; native GPU activities and their correlations remain retained.
Capture source SHA256 matched the checkout at the initial assessment. Actual weight tensors qualify BF16
and native automatic KV tensors qualify BF16 separately from compute dtype.
Request 2's prefix reuse has known same-route physical allocation lineage; Request
4's lookup also succeeds but one block's owner binding occurs after the lookup in
the same scheduler call, so that block's prior host lineage stays unavailable.
The assessment records this limitation rather than invent an earlier write.

`assess.py` supports dense and hybrid recordings, resolves each model's actual
layer count, requires separate measured recurrent state for hybrid models, checks
exact requested weight/KV formats, adapter routes/configuration hashes, source
identity and cleanup. Its tests reject changed artifacts, unqualified FP8 bytes,
reused allocation generations, cross-adapter cache origins, missing/duplicate
packed positions and unfinished memory observers.

The newly bundled `gpu-h100-bf16` was opened through Configure in the localhost
browser at 1280×720. Iteration 1 displays 9.27 GiB device used, 8.79 GiB process
used, 8.18 GiB framework allocated, 7.55 GiB weight storage, 510.75 MiB attention
KV and a separately timed 755 MiB remainder. The actual UUID and PID are visible.
The expanded panel collapses directly without interception. Keyboard End on the
host playhead reaches cleanup: 1.11 GiB device used, 646 MiB process used, worker
framework sampling ended and earlier loaded storage attribution invalidated.
Console inspection reports zero errors and warnings. The first query can precede
its emitted event, so the plot origin includes query start to prevent negative
elapsed labels. Small storage allocations remain visible in bytes/KiB rather than
rounding to zero MiB. Screenshots:
`output/playwright/memory-accounting/h100-bf16-measured-expanded.png` and
`output/playwright/memory-accounting/h100-bf16-cleanup-expanded.png`.

The retained FP8 weights attempt `run-b8840655c01c` illustrates why in-process
cleanup and owner exit are separate observations. Last recorded cleanup still
shows 5,469,372,416 process bytes and no worker framework counter, compared with
6,031,409,152 process bytes at the last inference sample. The chart retains those
actual bytes. The helper's pre-exit loaded-storage release threshold remains
false; a separate management receipt after successful PID 9277 exit shows the
same GPU at its original 4 MiB device baseline and no compute PID. The corrected
assessment qualifies this allocation lifecycle using external post-owner
evidence, rather than replacing recorded memory or attributing its pre-exit
retention to an unproven native reference owner. The original failed assessment
and separately named corrected evidence assessment remain reviewable in
`fp8-weights-attempt-1/six-request/`.

## Final integrated captures

The earlier examples above retain their original capture chronology. The final
catalogue now contains six qualified actual H100 captures, all against source
SHA256 `56b4d03456a627b250a87e322a54cbf035643e730db8d2c27c2f3b29cff086bc`.
Their measured memory samples, exposed storages, same-device identity and
separate successful owner-exit receipts pass the mandatory evidence checks.
[The final case table](README.md) names the original runs and bundle hashes;
the hybrid allocation is counted once across recurrent/attention aliases.

Three additional actual UI captures also pass all 41 evidence checks and
workload assessments. In the final generic-FP8 UI capture
`run-5495998a324d`, the live panel first waits for a successful sample, then
shows arriving NVML sample counts of 2, 41 and 83. The
[live receipt](live-fp8-alias-ui/six-request/live-memory-ui.json) records actual
UUID/PID, sample age and direct expansion/collapse. Before any successful
replay sample, the panel asks the user to seek to a measured sample rather
than suggesting an otherwise complete recording lacks capture data.
[Separate bundled BF16 End-seek verification](../../../output/playwright/bundled-bf16-replay-verified.txt)
uses run `run-af950e986ddc` / PID 8918 and reaches recorded cleanup at
1.11 GiB device usage with a 9.49 GiB sampled maximum and invalidated earlier
storage attribution. The initial automation filename incorrectly referred to
FP8; its unchanged receipt was copied to the accurate BF16 name, and it does
not qualify a replay inspection of the FP8 alias PID 12640. The separate post-owner
receipt subsequently establishes the 4 MiB baseline; no interpolation or
invented tensor-free observation joins those two measurements.

Twenty assessor regression tests pass, including archived finalized trace
responses whose transient presentation message is intentionally empty.
Finalization identity/state/source and matching successful owner exit remain
mandatory. Original failed assessments and automation attempts remain
available. Four earlier root-browser connection-refused polling errors came
from tunnel/reconnect setup; clean-console claims above apply only to the
specific isolated checks that recorded them. Final full checks pass:
287 Python, 70 replay and 103 web tests, plus TypeScript/Vite build.
