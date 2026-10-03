# vLLM 0.30 learning explorer

The intended audience is an engineer learning vLLM internals and an operator investigating performance problems. The first model drill targets dense Qwen3, with Qwen/Qwen3-4B-Instruct-2507 as the canonical qualification model. The header identifies vLLM 0.30.0; future version selection requires a corresponding instrumentation and evidence contract.

## Model and serving structure

The main scene connects request intake, scheduler, a model layer stack, physical KV pool, and output. Users can explore Model → Layer → Attention or Feed-forward → Query or KV head. Query-to-KV grouping uses captured dimensions, including Qwen3's explicit head dimension. Geometry and operator explanations describe architecture; they do not imply that numeric tensors, attention weights, or per-head timings were captured.

Each token retains request and position identity. Scheduled ranges identify prefill and decode participants in the host replay batch. Queue snapshots retain observed order and priorities; reasons for skipping a request remain unknown unless observed. A sampled output token becomes model input only on a later step, so accepted output alone does not prove computed KV.

## Three independent clocks

| Mode | Meaning | Evidence boundary |
| --- | --- | --- |
| Recorded events/time | Serving observations and recorded host intervals | No GPU kernel timing or profiler alignment inferred |
| Layer walkthrough | Logical progression of the host replay batch through model structure | Explanatory stage duration |
| Measured profile | Native CPU/CUDA profiler intervals and post-reorder packed input rows | Separate clock; eager diagnostic execution with profiler overhead |

Recorded and measured playback support speeds down to 0.001×. Native batch navigation follows explicit explorer step scopes, rather than unrelated runtime calls. Keyboard navigation uses the selected clock. Packed kernels do not establish individual token or head duration. Unmapped activities remain unmapped.

Measured diagnostics are opt-in, bounded to the first configured workload steps, and initially restricted to synchronous single-GPU dense Qwen3 eager execution. CUDA graphs are disabled in this mode. Raw native traces are retained alongside a normalized, hash-checked attachment. Cancellation or failure retains partial evidence and its status. Structural exports explicitly omit profiling attachments and token content.

## Physical KV pool

Every physical block is addressable; the view does not display a sampled subset. Instancing and grid picking keep the geometry bounded. Inventory identity includes pool, block ID, and allocation generation. A referenced block is distinct from a zero-reference cached block, which remains available for allocation. Reserved null blocks are represented separately.

Drilling into a qualified block shows token slots, current request/prefix bindings, and original content provenance. Reservations do not establish a completed write. Computed occupancy follows observed result-associated writes. Reuse keeps content provenance while selection follows the current binding. Reassignment removes the previous generation's occupant; backward seeking reconstructs the earlier state.

New captures qualify token mapping only for exact ordinary full-attention layouts. Compressed, windowed, chunked, noncausal, state, MLA, and related unsupported layouts retain physical inventory with explicit unavailable token mapping. Per-group writes never prove another group's token state. Older recordings keep only their explicit legacy token observations; untouched cells remain unknown. Trace gaps freeze observed state and prevent current occupancy claims.

## Configuration and examples

Model and tokenizer revisions are optional overrides. Preparation resolves Hugging Face refs to immutable commits. The tokenizer defaults to the same repository and resolved revision; separate overrides remain available. Execution-host paths are identified as artifacts rather than invented Hub revisions.

Continuous batching is explained as built-in serving behavior, with actual scheduler batch membership in replay. There is no fictitious enable/disable toggle. Controls include FCFS/priority scheduling, per-request priority, chunked prefill, long-prefill threshold, prefix caching, and KV memory bytes. Editable examples cover staggered arrivals, partial prefix reuse, chunked prefill, and cache pressure. Their live-runtime requirements are explicit.

## Qualification gate

Local fixture and authored profile tests establish contracts and UI behavior only. Before this redesign can be merged as GPU-qualified, run the canonical Qwen model and representative learning scenarios on the actual vLLM 0.30.0 host, inspect retained native profiler mappings and partial/failure cases, inspect every experiment's actual LangSmith request/response traces and human-readable hierarchy, then obtain three independent final reviews of that completed evidence. See [local acceptance](acceptance/3d-learning-local/README.md) for the current state.
