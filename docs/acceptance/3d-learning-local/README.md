# Preliminary local acceptance — 2026-10-03

This preserves the initial **local synthetic and contract checkpoint**, not GPU qualification. At this checkpoint the redesign had not run the canonical Qwen model or actual native CUDA profiler. No redesign LLM calls had been made: LangSmith credentials were unavailable on the laptop and execution host. The user subsequently waived LangSmith for this project on 2026-10-03; that credential question is no longer a blocker. No key contents are stored here.

## Final local checks

- Python: 164 tests passed.
- Replay: 51 tests passed.
- Web helpers: 52 tests passed.
- Strict frontend type check and production build passed. The existing large-bundle warning remains; no loading or frame-time improvement is claimed.
- Python compilation and whitespace checks passed.
- Read-only checks against installed vLLM 0.30.0 classes qualified exact FullAttentionSpec and uniform full-attention wrappers, and rejected actual MLA/state/RSWA/sink, compressed/windowed/chunked/noncausal specs. These created configuration objects only, without loading a model or computing GPU tensors.

## Case assessments

| Case | Success criteria | Observed result |
| --- | --- | --- |
| Fresh integrated staggered capture, run-609e5a887c09 | Actual UI/controller capture; complete inventory, ordered queues, request outcomes and valid recording | 289 events, 16 steps, six completed requests, eight accepted output tokens each; all nine integrity checks and configuration hash passed independently |
| Default offline walkthrough | Fresh preview opens an explicitly synthetic lesson with architecture and complete pool available | The verified staggered fixture is the first bundled lesson; pinned public Qwen3 geometry and all eight blocks are available without importing files or loading model weights |
| KV result boundary | Reserved slots are not computed; returned writes establish occupancy | First scheduler step at event 14 showed reservations; event 23 showed block 1 with 4/16 computed positions; slot 1 selected sample-2 position 1, ID 2 |
| Final output token | Accepted final token must not be marked computed | Actual replay keeps the final token of all six requests not_computed |
| Architecture drill | All layers and heads reachable; captured GQA mapping retained | UI Model → Attention → Query head 8; Qwen3 geometry 2560 hidden, 9728 FFN, 32 Q/8 KV, 128 head dimension; Q8 shares KV2 |
| Complete large pool | Every block addressable, including final ID | Authored expansion to 35,000 blocks imported through UI; block 34999 selected; physical availability retained. Older unqualified inventory showed unknown token contents |
| Native profile contract | Native clock independent, explicit scope correlation, unknown activities preserved | Authored six-interval attachment imported; at 10 µs, host event 0 unchanged, layer 1 attention CPU 30 µs/device busy union 12 µs, packed row a/position0; one lookalike activity remained unmapped |
| Native keyboard | Arrow/Home/End/Space control native time even with no host events | ArrowRight → 10 µs, End → 100 µs, Home → 0 µs, Space → Pause; host event 0 stayed unchanged |
| Import/reset transitions | Reopening same manifest resets clock; unsupported modes recover normal speed | Repeated manifest.json import reset to 0; measured→bundled lesson restored event mode 1×; teaching→bundled lesson played through all 36 events at 1× |
| Teaching completion | Logical final stage can finish and restart despite trailing host events | Final stage completed to Replay; Replay restarted at first scheduler batch/event 14 and explanatory stage 1 |
| Camera and compact controls | Translation, drag-safe selection, reduced motion and keyboard dismissal | Pan button moved the scene; drag kept block 34999 selected;0.001× recorded speed/reduced motion selected; 768px inspector opened and Escape dismissed it with focus restored |
| Missing profile attachment | Explain missing normalized JSON, preserve ordinary replay contract | UI reports “Select the declared normalized profile JSON together with this recording”; host-empty authored fixture remains disabled rather than inventing events |
| Backward seeking, generations, prefix reuse, gaps, unsupported groups | History and uncertainty follow observed data | Distinct replay tests pass; source/content projection checked on actual fixture. Unsupported counts are null and per-group histories stay independent |
| Trace fidelity and preparation failure | Effective per-call settings, reproducible source identity, atomic rejection | Independent targeted tests pass; SDK submission is labeled queued, not inspected; malformed inventory fails before hooks and closes prepared engine |

The [authored profile fixture](profile-contract-fixture/README.md) intentionally has empty host events. It is a profile-loader/UI contract fixture and **does not pass recording integrity**. Its intervals are never described as GPU measurements. The fresh eight-block capture is preserved under [staggered-fixture](staggered-fixture/manifest.json); its token payloads are authored example content. Reproduce the large synthetic inventory with `python docs/acceptance/3d-learning-local/expand_pool.py docs/acceptance/3d-learning-local/staggered-fixture <new-output-directory>`; import its three JSON/JSONL files. The generator refuses existing destinations and GPU inputs.

## Discovered defects and remediation

Local UI and independent reviews found canvas clipping and intrinsic-height feedback after expanding/collapsing controls, overlapping head explanation and token labels, keyboard shortcuts using the wrong clock, unsupported teaching playback stalling, fallback speeds remaining at 0.001× in event mode, repeated manifest filenames retaining profiler time, teaching completion using the host tail instead of logical progress, incomplete inventory passing the report, unsupported cache layouts receiving generic token mappings, and per-call trace settings being omitted. These were fixed and affected checks rerun. Development HMR also reset in-progress views; subsequent browser checks used the production preview.

Three independent preliminary reviewers reassessed correctness, integration/evidence, and simplification. Their significant findings were addressed. The later [actual GPU qualification](../../qualification/3d-learning-gpu/README.md) records the subsequent live captures and final review status; the synthetic evidence below remains scoped to this checkpoint.

## Qualification required after this checkpoint

1. LangSmith was explicitly waived by the user on 2026-10-03; preserve and inspect local request/output and native trace evidence instead.
2. Run actual Qwen3-4B-Instruct-2507 serving and eager diagnostic scenarios, including prefix reuse, chunked prefill, cache pressure, cancellation, and partial/error retention. Use an isolated checkout/runtime and preserve existing services.
3. Check every recording, returned token outcome, immutable model/tokenizer revision, actual loaded architecture, complete physical pool, scheduler queue snapshots, and packed post-reorder row mapping. Inspect raw native traces and correlations, and record unsupported/unknown evidence honestly.
4. Inspect each experiment’s retained inputs/outputs, effective settings, scheduler/KV observations, controller log and native profiler attachments. This replaces the waived LangSmith inspection requirement.
5. Evaluate frame time, picking, memory/resource stability on the real pool and target browser; synthetic addressability alone is not a GPU performance benchmark.
6. Obtain three independent final reviews, pass final CI and merge only after the evidence gate is met.
