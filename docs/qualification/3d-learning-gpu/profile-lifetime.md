# Live qualification cases and diagnostic lifetime

The helper uses the actual controller API and imports the current workload,
configuration, integrity-report, and structural-export implementations. Its
default invocation validates and writes seven public Qwen3 request payloads. It
makes no API or GPU calls until `--execute` is supplied. Execution belongs to
the operator running the isolated controller.

```sh
.venv/bin/python docs/qualification/3d-learning-gpu/qualify.py --output /tmp/qwen3-qualification-plan
```

Run through the isolated controller's loopback endpoint, selecting individual
cases with `--case` when diagnosing a failure. To retain native raw profiler
files, execute on the controller's machine and provide its actual recording
root through `--recordings`.
Use `--gpu-memory` on that host to retain read-only pre/post-case GPU-memory
receipts, including compute-process allocation. Those receipts explicitly name
the helper's host scope; running the helper through an SSH-forwarded endpoint
does not make local memory readings measurements of the remote host.

| Case | Configuration and observations required |
| --- | --- |
| Staggered | Six chat-formatted prompts at 0/100/120 ms; changing multi-request scheduler membership and completed outputs |
| Partial prefix reuse | Shared reference with different suffixes at 0/1500 ms; prime completes first, a nonzero prefix hit remains shorter than the follower prompt, and remaining prefill is observed |
| Chunked prefill | Exact synthetic 512/16-token inputs; context 1024, scheduling budget 64 and long-prefill cap 64; multiple capped prompt ranges |
| Cache pressure | Four exact 192-token inputs with 32-token output limits; context 256, budget 256 and explicit 72 MiB KV cache; actual inventory capacity versus prompt demand and observed preemptions reported separately |
| Priority | Three equal-offset chat prompts with priorities 5/−5/0 and one running sequence; compare authored priorities with admission observations and require an ordered queue snapshot containing multiple waiting requests |
| Native profile | Two chat-formatted prompts; explicit eager diagnostics bounded to the first eight workload steps, post-reorder token rows, and native CUDA-to-layer correlation |
| Cancellation and partial profile | Exact synthetic 3500-token input; budget 64 and profile limit 16; cancel after two observed returned workload steps, preserving incomplete native evidence |

All cases pin `Qwen/Qwen3-4B-Instruct-2507` and its tokenizer to
`cdbee75f17c01a7cc42f958dc650907174af0554`. Synthetic inputs qualify structural
token scheduling and cache behavior; their generated prose is not a quality
benchmark. The helper retains exact payloads, actual outputs, events, content,
integrity reports, profile artifacts, and per-case assessments. A pressure
configuration alone does not establish why the scheduler chose a request.
Meaningful output review, seek consistency, and actual browser interaction
remain separate qualification work. LangSmith is explicitly waived by the
user for this project; the recordings remain the inspected evidence.

vLLM 0.30.0 defaults to Model Runner V2. The initial eager diagnostic hooks
require the older `GPUModelRunner` boundaries. A diagnostic session explicitly
selects V1 before engine construction and holds that selection until engine
shutdown, then restores the previous environment value and invalidates stale
vLLM environment-cache reads. A process-wide session lease prevents a second
in-process engine from inheriting another session's override. Ordinary runs
preserve their existing runner choice, and manifests record the actual choice.

Production captures spawn a dedicated execution process for the whole recording,
HTTP request service, engine, tokenizer, and profiler lifetime. The controller
retains only small control state and output previews; event and content streams
are written directly by the worker. A POSIX file lock serializes late intake,
cancellation, and snapshots and is released automatically if the worker dies.
An admission mutex prevents an active fixture from being replaced during state
migration. The child inherits the controller's Python environment and revision.

Preparation failures and successful close still shut down the native engine,
invoke its owned finalizer, drop engine and profile references, collect cycles,
and empty unused CUDA allocations. Real repeated-capture measurements showed
that these steps, including an attempted public compiler reset, retained about
8 GiB after the normal compiled runner exited its session. Production therefore
waits for the dedicated process to exit and join before exposing final readiness
or admitting another engine. The ineffective process-wide compiler reset has
been removed; compiler disk caches are preserved.

Controller shutdown first requests cancellation and closes intake. It waits a
bounded grace period, then terminates and, if necessary, kills only its registered
execution process, joins it, finalizes retained evidence, and shuts down the state
manager last. The supervisor waits on the OS process sentinel without reaping, while one
lifecycle mutex serializes process polling, joining and signalling. Only the
identity-matched owned handle is cleared before admission is released under
that mutex. A worker that cannot be reaped keeps its handle and blocks another
capture. An unexpected exit retains the valid event/content prefix, records an
explicit trace gap, and labels the capture failed. Before repairing an interrupted
last record, the original bytes are retained in `*.interrupted` files. Interior
corruption is reported rather than rewritten as complete evidence.

Local checks use real spawned fixture workers to cover repeat capture, live
intake, cancellation, joined readiness, death inside a transaction, shutdown,
bounded termination, crash recovery, and raw-byte preservation. Those checks
establish controller semantics. Subsequent live consecutive captures and
post-close GPU memory receipts are retained in the [GPU qualification report](README.md).
