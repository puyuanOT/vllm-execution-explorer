# vLLM Execution Explorer: agreed redesign

Prepared September 27, 2026, from the completed design interview.

**Status:** Implemented for the v0.1 milestone. This document preserves the accepted design contract; current verification and known limits are recorded in the repository acceptance notes.

This brief supersedes conflicting choices in `UI-REDESIGN-PLAN.md`, including the combined workspace, timeline-first visualization, optional 3D, and direct-engine execution path.

## 1. Outcome and scope

Build an educational tool for engineers learning how vLLM serves real requests. The primary journey is **configure → capture → explore**. Reduce initial detail, finish the visual styling, and make the progression between those stages clear.

The first implementation supports one GPU on one existing CUDA-ready Linux host, used locally or through SSH. It launches an isolated, instrumented vLLM HTTP serving instance using an exact tested version. Dense text models, supported weight quantization, and compatible LoRA adapters are in scope. Multi-GPU, multi-host, MoE, multimodal, and transformer-layer/kernel visualization are outside the initial support envelope.

The current inspected instrumentation targets vLLM 0.30.0. Use that as the starting compatibility target; record the exact tested build and source identity. Validate normal HTTP serving and its process boundaries rather than carrying forward the existing direct-engine restrictions without disclosure.

Use the existing React/Vite, React Three Fiber/Three.js, Python controller, and deterministic replay code where appropriate. Backend changes are required: this is more than a visual restyle.

## 2. Pages and visual design

### Configure page

Keep four sections in the order supplied in the user's sketch:

1. Query Batch Sequence.
2. Visualization Server.
3. Model Configuration.
4. vLLM Configuration.

Each section has a concise summary and expandable details. Show the workload immediately, expose required fields when relevant, and collapse advanced options. Show readiness and actionable errors beside the relevant section. The primary action begins preparation and recording; merely opening the page never launches a run.

Capture progress remains part of this workflow: checking the host, transferring/downloading weights, preparing the model, capturing requests, and finalizing the recording. Preserve the submitted configuration as an immutable run snapshot. Editing a completed experiment creates a new draft and run; it does not alter a recording.

### Explore page

Give the 3D workflow the large left area and the inspector/playback controls the right area, following the user's sketch. Start with five collapsed components: **request intake, scheduler, KV-cache manager, model execution, and output**. Connections represent repeated scheduling/execution cycles and cache interactions.

Single-click selects an item and updates the right inspector. Double-click, or an accessible **Explore component** action, opens a focused view of that component's internals. Breadcrumbs and **Back to overview** restore the parent view. Navigation preserves the playhead, playing/paused state, and relevant selection. Keep stable request identities across every view.

Provide camera orbit, zoom, and reset. Camera reset affects the camera only. Request lists/search and keyboard-accessible selection provide access to every request, including when the overview aggregates activity. On narrower windows, panels adapt with contained scrolling and accessible inspector access.

### Theme and baseline

Use [LLM-Visualized](https://www.llm-visualized.com/) as the visual reference: near-black canvas, charcoal surfaces, subtle borders, compact controls, and clear technical typography. Preserve the user's page layout. Use a restrained action accent; categorical request colors convey data and are paired with labels.

Apply the `baseline-ui` design rules: Tailwind default scales or existing semantic tokens, `cn` for class logic, one accessible component-primitive family, labeled icon buttons, visible focus, inline errors, tabular numerals, appropriate text wrapping, dynamic viewport height, and fixed-element safe-area support. Respect reduced motion. Requested visualization playback and camera interaction are functional motion; UI transitions, if needed, follow the same timing and animation constraints. Keep gradients, decorative glow, and unnecessary motion out of the design.

## 3. Query batches and workload authoring

The editable default remains:

| Planned send offset | Requests |
| --- | --- |
| 0 ms | input_1, input_2 |
| 100 ms | input_3, input_4, input_5 |
| 120 ms | input_6 |

Support adding, editing, duplicating, and removing batches and requests; changing batch size and send offset; and moving requests between batches. IDs remain stable when labels or ordering change. Equal-time batches retain their authoring identities. Provide precise numeric time editing with milliseconds/seconds presentation.

Support two authoring modes:

- **Prompt text:** editable real prompts with input-token counts measured using the selected tokenizer.
- **Synthetic workload:** requests generated for specified input-token lengths, verified against the actual input submitted to the engine. Display the resolved length explicitly; do not substitute character counts or silently approximate token counts.

Each request has an output-token limit. It is a maximum: EOS or stop conditions can finish generation earlier. Workload defaults include temperature, with top-p, seed, and optional stop strings under advanced sampling. Start with greedy sampling. Record resolved settings without promising identical scheduling or output across repeated runs.

When adapters are attached, provide a default adapter per batch and an optional per-request override, including the base model. Keep those controls collapsed when they are irrelevant.

Target tens of requests, including long prompts, with every request individually inspectable. Replace the current eight-request, 4,000-character, and 128-output-token restrictions with tokenizer/context-aware validation and explicit resource/recording limits. Establish and document supported limits through GPU acceptance testing. Keep the small six-request default.

Drafts survive navigation, validation failures, and reloads where supported. Save non-secret configuration and retain recording import/reopen workflows. Recordings remain separate from editable drafts.

## 4. Host selection and automatic setup

Support a local CUDA-ready Linux execution host or an existing remote CUDA-ready Linux host reached through SSH. The operator's Mac can run the UI, controller, and replay while execution occurs remotely. Reflect actual local capabilities in the UI.

Automatic setup checks the host, GPU/driver availability, storage, and prerequisites; creates an isolated runtime; installs pinned dependencies and instrumentation; obtains model artifacts; starts the serving instance; and establishes the required connection. Show staged progress, useful logs/details, failure location, and retry/reconnect actions.

Cloud-instance creation, rental, and driver installation are outside scope. Keep SSH private keys on the operator's machine. Avoid including credentials in logs, saved browser drafts, URLs, recordings, or exported artifacts. Preserve the existing loopback/tunnel approach.

The timed workload driver runs on the GPU host. The browser controls the experiment; network latency from the operator's laptop does not define request arrival offsets. Record planned dispatch, actual dispatch, HTTP receipt, and engine admission as separately named observations. Timestamp comparisons must respect clock domains.

An arrival batch means concurrent request submissions with the same planned offset. It does not force an atomic vLLM execution batch. Show the actual combinations and splits selected by vLLM.

## 5. Model configuration

Support all three artifact sources:

| Source | Behavior |
| --- | --- |
| Operator-local weights | Use directly for local execution, or provide built-in transfer to the remote host with destination, size, progress, and retry status. |
| Execution-host weights | Validate and use the existing path on the selected host. |
| Hugging Face | Select a repository/revision and download onto the execution host. Support a token for authenticated/gated access. |

Resolve and record model/tokenizer revisions or equivalent local-artifact identity. Keep revision and other specialized fields under advanced settings where appropriate. Validate model, tokenizer, dtype, context, adapter, quantization, and hardware compatibility before launch whenever possible.

Provide model precision and validated quantization choices. Support attaching multiple compatible LoRA adapters and routing requests as described above. The number attached and the number usable simultaneously in an execution batch are distinct; reflect the tested capacity and effective settings.

Every advertised quantization/adapter capability must work end to end for a documented, tested configuration. Unsupported combinations receive a clear explanation rather than a silent fallback. The final support matrix identifies actual models, adapters, formats, and hardware exercised.

## 6. vLLM configuration and run initialization

Expose the accepted curated controls:

- Maximum model context length.
- Maximum concurrent sequences.
- Tokens per scheduler iteration.
- Chunked prefill.
- Prefix caching.
- GPU memory fraction, under advanced options.

Explain each control briefly and validate relationships using the pinned runtime's rules. GPU memory fraction is an engine memory budget; show resulting KV capacity after initialization. Record effective resolved settings rather than only what the user typed.

Every experiment begins with a **fresh prefix cache**. Model loading, compilation/warmup, a verified cache reset, and capture readiness finish before workload `t=0`. Weights may remain loaded between compatible runs, but prefix-cache contents must not carry over. Prefix reuse within the run remains fully supported. Cross-run warm-cache experiments are outside this initial workflow.

During preparation/capture, show progress and a cancellation action. Once finalization succeeds, make the recording available in Explore. A failed or cancelled capture preserves valid captured evidence and clearly identifies its completeness. Reconnection/reload must not accidentally launch duplicate experiments.

## 7. Recording fidelity and drill-down content

The recording is the source of truth for all displayed execution state. Faithfulness means replaying the observations from the actual instrumented serving run, including identities, recorded ordering/causality, token ranges, cache transitions, and output. Instrumentation can affect timing; the recording is not a claim of identical uninstrumented performance.

Keep observed facts, values derived from those facts, and general educational explanations distinguishable. Do not invent unrecorded scheduler causes, interpolate unobserved state transitions as facts, or expose future results before the playhead reaches them. Missing data remains unknown.

| Component | Focused view and inspector content |
| --- | --- |
| Request intake | Planned batch membership, actual dispatch and HTTP receipt, stable request identity, adapter, prompt/tokens, measured preprocessing and admission observations. |
| Scheduler | Waiting/running queue snapshots, actual step participants, prompt/decode token ranges, token budget used, preemptions, and adapter membership. Decision reasons appear as facts only when captured. |
| KV-cache manager | Capacity, free/referenced/cached blocks, request block tables, shared-prefix matches, and allocation/reuse/release/eviction history with pool and allocation identity. |
| Model execution | Submitted step and participants, effective model/dtype/quantization/adapters, executor submission/result observations, and accurately named host-observed intervals. |
| Output | Accepted tokens linked to steps, generated text through the playhead, emitted chunks, and recorded completion/stop/cancellation reasons. |

The 3D layout organizes serving observations. Transformer tensors, attention matrices, layer internals, kernel timings, and GPU utilization are not shown without measurements and are outside the accepted initial scope.

Required capture work includes genuine HTTP/preprocessing/output observations, request and step correlation across processes, queue and initial-cache snapshots, complete cache transitions needed by the views, and a defined merge/clock/causality contract. Avoid equating collector receipt order with a total causal order of concurrent processes.

The existing arrival/tokenization/queue markers occur after scheduler admission and must be replaced or relabeled. Existing host execution intervals must not be renamed GPU execution duration. Adapt hooks for normal serving process boundaries and overlapping work; preserve trace-gap and incomplete-recording handling.

## 8. Replay behavior

Offer both accepted modes:

- **Recorded-time playback:** intervals scale proportionally with the selected playback speed.
- **Event/step exploration:** navigate recorded events and scheduler steps while retaining their actual timestamps.

Provide play/pause, speed, seek, previous/next event, previous/next scheduler step, and restart. Playback controls affect the recording only. Scrubbing reconstructs state at the selected position, including when moving backward. Component navigation preserves position and playing/paused state. The inspector's interactive widgets explore recorded state; changing execution settings creates another experiment.

Keep provenance visible: real GPU recordings, existing demo fixtures, and incomplete recordings are distinguishable. Existing fixtures remain useful for local UI/replay checks but cannot certify GPU correctness.

## 9. Implementation gates

1. **Recording contract and HTTP capture:** establish the exact runtime, real process-boundary observations, correlation, cache initialization, and deterministic complete/partial replay. Build local contract tests. Implement the GPU path before claiming it works.
2. **Configure and capture workflow:** implement the four sections, workload modes, host setup, all artifact sources and transfer, adapter routing, validation, progress, and run lifecycle.
3. **Explore workflow:** implement the five-component 3D overview, focused scenes, inspector, camera/navigation, and both replay modes at the intended workload scale.
4. **Local UI acceptance:** exercise every locally testable function through the browser, fix failures, and run appropriate backend/replay/scenario/build checks. Use fixtures and controlled local failures to cover states unavailable without a GPU.
5. **GPU acceptance when access is supplied:** exercise the implemented workflow through the UI against the user's CUDA host. Validate actual HTTP traces, model sources, adapters, quantization, scale, initialization, failures, and recovery. Fix issues and repeat affected tests.
6. **Completion evidence:** deliver the verified feature/support matrix, tested hardware/artifact identities, supported recording limits, and test results. Full completion requires the GPU gate; successful local tests alone are an intermediate checkpoint.

UI and backend work may proceed independently where contracts are settled. No complete-looking control may silently ignore a configuration or substitute a fixture for an actual GPU run.

## 10. UI and GPU acceptance checklist

All implemented user-facing functionality must be exercised through the UI, and discovered bugs must be fixed. Automated tests supplement this requirement.

| Area | Required verification |
| --- | --- |
| Default and batch editor | Exact default 2/3/1 batches at 0/100/120 ms; add/edit/delete/duplicate/move; equal timestamps; seconds conversion; invalid/empty values; stable identity and draft persistence. |
| Prompt and synthetic modes | Measured prompt lengths; requested synthetic lengths equal actual submitted input lengths; context validation; per-request output caps; sampling defaults; representative long prompts. |
| Host and setup | Local capability feedback; SSH connect; clean runtime installation on a CUDA-ready host; progress; invalid host/key; dependency/setup failures; retry and tunnel reconnect. |
| Model sources | Local execution paths; operator-local to remote transfer; existing remote paths; public and authenticated Hugging Face access; progress; invalid/missing artifacts and incompatible configuration. |
| Model options | Each advertised dtype/quantization mode in its supported configuration; base model and multiple LoRA adapters; batch defaults/request overrides; recorded effective identities/settings. |
| vLLM settings | Each exposed control and relevant combinations affect the actual run; invalid relationships fail clearly; resolved settings and KV capacity are correct. |
| Run lifecycle | Preparation before `t=0`; verified cache reset after warmup; capture/finalization; cancellation; duplicate-run prevention; reload/reconnect; incomplete trace and failure recovery. |
| Actual execution | Host-local HTTP dispatch; planned versus measured times; cross-process request/step correlation; real scheduling and cache operations; accepted tokens/output agree with engine and HTTP observations. |
| Cache behavior | Prefix reuse within a run, correct shared-block ownership, relevant allocation/release/eviction/preemption scenarios, and fresh cache in the next experiment. |
| 3D and inspector | Initially collapsed overview; all five detail views; single-click selection; double-click and keyboard alternative; breadcrumbs/back; preserved replay state; orbit/zoom/reset; every request reachable. |
| Replay | Play/pause/speed/seek/restart; event and scheduler-step navigation; recorded-time intervals; backward seek removes future state/output; camera reset leaves time unchanged; gaps remain unknown. |
| Recordings | Reopen/import valid recordings, clear invalid-import errors, missing content, incomplete traces, provenance, and separate editable copies. |
| Visual/accessibility | Desktop layouts at 1440×900 and 1280×720, narrow 768×1024 layout, 200% zoom, keyboard/focus, contrast, reduced motion, contained scrolling, readable labels, and reachable controls. |
| Scale and responsiveness | Representative tens-of-requests and long-prompt GPU recordings, responsive selection/seeking, explicit limits, complete capture or a clearly reported capture failure. |

Maintain test evidence as implementation proceeds, including screenshots where they establish visual correctness and recording checks where they establish execution fidelity. The user will supply SSH access for the GPU acceptance stage after implementation is ready for testing.

## 11. Inspection references

- `runner/vllm_session.py`: current in-process engine construction, warmup, cache reset, and fixed settings.
- `runner/controller.py`: current timed admission between synchronous steps.
- `instrumentation/vllm_030/install.py`: existing observations and compatibility-sensitive hooks.
- `instrumentation/vllm_030/collector.py`: capture limits and incomplete-recording behavior.
- `replay/src/reduce.ts` and `replay/src/explain.ts`: deterministic replay and separation of observations/explanations.
- `runner/workload.py`: existing provisional workload restrictions to replace.
- `web/src/shell/SettingsDialog.tsx` and `web/src/execution/ExecutionWorkspace.tsx`: existing combined UI to replace.
- [vLLM 0.30.0 HTTP launcher](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/entrypoints/launchers/api_server/entry.py), [AsyncLLM](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py), [scheduler settings](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/scheduler.py), [cache settings](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/cache.py), [LoRA settings](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py), and [sampling settings](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/sampling_params.py).
