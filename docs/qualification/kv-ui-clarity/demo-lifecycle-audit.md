# Real-demo lifecycle audit

Read-only investigation on 2026-10-08 of bundled `gpu-kv-layer-diagnostics`, run `run-a413ca908152`. No new inference or browser execution is claimed. The accompanying `demo-lifecycle-projection.json` records source hashes and 16 playheads evaluated through the production reducer and projections.

## Why release can look like an unfinished three-token request

The recorded observations are not an atomic scheduler transaction. Native scheduler output handling receives and appends the already sampled fourth output, checks its output limit, and frees the finished request before returning its output. Sampling occurs in model execution before this scheduler update. The instrumentation emits block-release observations inside that native call, then emits its accepted-token and completion observations after the call returns.

Consequently, Events playback genuinely shows an intermediate state with three accepted outputs and an unfinished request while the blocks turn brown. A description that the request “finished after three outputs” would be wrong.

Reproduce the receipt with Node 22 or newer from the repository root:

```sh
node --experimental-strip-types docs/qualification/kv-ui-clarity/replay-demo-lifecycle.mjs
```

The script imports the production reducer and projections. It reads the bundled recording and writes `demo-lifecycle-projection.json`; an optional output-path argument writes to a separate file for comparison. It does not run inference or alter the recording.

| Scheduler step | Model input processed | Newly sampled output | Accepted-token event | Cache observation |
|---|---|---|---:|---|
| 3 | Remaining prompt positions 128–172 | `The`, ID 785 | 76 | Prompt block 11 contains positions 160–172 |
| 4 | Position 173, `The` | ` first`, ID 1156 | 93 | Event 92 writes slot 13 of block 11 |
| 5 | Position 174, ` first` | ` thing`, ID 3166 | 110 | Event 109 writes slot 14 of block 11 |
| 6 | Position 175, ` thing` | ` recorded`, ID 12433 | 139 | Event 127 writes slot 15 of block 11 |

All three requests in this recording have `max_output_tokens: 4`. The prime's fourth output is retained in content line 25 and manifest `engine_output_token_ids`. It is emitted as text in event 141. Its sequence position 176 never receives a KV write, because no subsequent model step runs for that completed request.

Actual production replay states:

| Event | Controller elapsed time | Prime request phase | Accepted output | Emitted text | Referenced / reusable blocks |
|---:|---:|---|---:|---|---|
| 127 | 424.023 ms | Decode | 3 / 4 | `The first thing` | 11 / 0 |
| 128 | 425.192 ms | Decode | 3 / 4 | `The first thing` | 10 / 1 |
| 138 | 425.370 ms | Decode | 3 / 4 | `The first thing` | 0 / 11 |
| 139 | 425.481 ms | Decode | 4 / 4 | `The first thing` | 0 / 11 |
| 140 | 425.501 ms | Completed, reason `length` | 4 / 4 | `The first thing` | 0 / 11 |
| 141 | 425.908 ms | Completed | 4 / 4 | `The first thing recorded` | 0 / 11 |
| 151 | 6021.130 ms | Completed; distinct reader admitted | 4 / 4 | `The first thing recorded` | 0 / 11 |
| 167 | 6029.444 ms | Completed | 4 / 4 | `The first thing recorded` | 10 / 1 |

Events 128–138 release physical blocks 11 through 1 in reverse order. Their reference counts fall to zero and their prefix hashes remain. This represents host allocator bookkeeping, not per-block GPU clearing. Events 158–167 reference blocks 1 through 10 again for a distinct request, `prefix-reader`, after event 157 records a 160-token prefix hit. Allocation generation remains 1. Physical block 11 remains cached throughout the rest of this recording; the reusable count falls from 11 to 1, not zero. At final event 399, the pool has zero referenced, 23 reusable and 89 uncached free blocks.

The first 12 physical cells are reserved null block 0 plus prime blocks 1–11. Only block 11's final three slots receive the three generated input tokens above.

## UI findings

The generated-output inspector and sequencer use the production sample projection: they retain all four accepted IDs and show `The first thing recorded` once event 141 is reached. No output text is discarded in that path.

There is a separate token-fragment limitation. `projectRequestTokens` reads accepted IDs but does not read `accepted_tokens.token_pieces`; its fragments come from KV contents. As a result, position 176 has ID 12433 and `not_computed` KV state, but its individual token fragment remains unavailable despite ` recorded` being retained in `content.jsonl`. Earlier newly sampled outputs also lack their individual fragment until their next-step KV observation. This can reinforce the impression that only three outputs exist. No fix is included in this audit.

Request labels do not collapse these requests: the default demo explicitly labels `prefix-prime`, `prefix-reader` and `mixed-prefill`. However, tracked request selection is retained across playback, reused prefix text is unchanged, and original-writer provenance remains `prefix-prime`. The block holder is recorded separately. At event 178, block 1's holder is `prefix-reader`, while its slot 3 contents remain `Read` from original writer `prefix-prime`. A user looking mainly at token text or original-writer labels can reasonably perceive continuity of one request.

## Other bundled real recordings

| Recording | Physical pool blocks | Output limit | Recorded output |
|---|---:|---:|---|
| `gpu-kv-layer-diagnostics` | 113 | 4 per request | Prime: `The first thing recorded`; reader: `A shared prefix can`; mixed: `The scheduler's first` |
| `gpu-model-batching` | 113 | 8 per request | Long: ` and 1 ( 1 1`; short: ` and 100% natural,` |
| `gpu-prefix-reuse` | 455 | 8 per request | First: ` The three things recorded are requests, scheduler`; second: ` Provide a memory hierarchy explanation.\n\nCertainly!` |

The reported exact words match the new default demo, but pool size 113 alone does not identify it. The user's precise observed playhead cannot be recovered from prose alone. The audited raw-event and projection states establish the behavior of the shipped recording without treating every reported count as independently reproduced.

## Source references

- Bundled maximum output and request identity: `web/public/recordings/gpu-kv-layer-diagnostics/manifest.json:3251`.
- Third KV input, releases, fourth acceptance, completion and final text: `web/public/recordings/gpu-kv-layer-diagnostics/events.jsonl:127` through event 142.
- Retained fourth output fragment: `web/public/recordings/gpu-kv-layer-diagnostics/content.jsonl:25`.
- Return-boundary acceptance observer: `instrumentation/vllm_030/install.py:198`; nested release observer: line 396; acceptance/completion emission: line 96.
- Output text attachment: `replay/src/reduce.ts:481`; request token fragments: `replay/src/learning.ts:191`; sample output: `replay/src/timeline.ts:274`.
- Generated-output presentation: `web/src/execution/SelectionInspector.tsx:248`; holder/original-writer legend: `web/src/architecture/CacheStage.tsx:231`.
- Pinned vLLM 0.30.0 appends each output before checking stop: [scheduler.py lines 2347–2364](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L2347-L2364).
- It frees a stopped request before constructing its output: [scheduler.py lines 2117–2157](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L2117-L2157).
- The output-token limit causes length-capped completion: [utils.py lines 101–106](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/utils.py#L101-L106).
