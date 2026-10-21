# Decode input reservation audit

This is a source and production-replay audit of the saved real GPU recording
`gpu-kv-layer-diagnostics`, source run `run-a413ca908152`. It does not report a
new browser verification or a new inference run. On the follow-up audit,
browser surfaces were unavailable: the browser inventory was empty and
creating an in-app browser tab returned unavailable. The earlier drawer
dismissal browser evidence is described separately in `README.md`.

The accompanying `decode-reservation-audit.json` retains the original relevant
events, accepted/computed token content, eight production replay projections,
36 native step-4 layer-write mappings, recording hashes, and four published
source hook-fingerprint comparisons.

## Concrete token chronology

Request `prefix-prime` has 173 prompt tokens. Its first generated token is
`The`, vocabulary ID 785, output index 0, absolute request position 173.
Positions, vocabulary IDs, physical block IDs, and slots are different axes.

| Event | Observation |
| --- | --- |
| 56 | Physical Block 11 is allocated during prefill step 3, before execution. Its 16 slots cover positions 160–175. |
| 59–70 | Step 3 schedules positions 128–172. Block 11's binding ends at 173, exclusive. |
| 71–75 | Step 3 is submitted and its result boundary observed; KV contents are recorded through position 172. |
| 76 | Step 3 accepts `The`, ID 785, at absolute position 173. Its KV is not computed yet. |
| 77 | Step 4 schedules that already accepted token as its input. No block-table change is recorded. |
| 88 | The existing Block 11 binding extends to 174, exclusive, covering Slot 13 / position 173. |
| 89 | Step 4 is submitted. |
| 91–92 | The result boundary is observed and position 173 is recorded as computed KV. |
| 93 | Step 4 accepts its next output, ` first`, ID 1156, at absolute position 174. |

The step-4 reservation does not allocate a new physical block. It assigns the
next logical token position to an unused slot in a block already held by the
request. An input token can have a known ID while its own layer K/V is still
uncomputed; the previous iteration sampled that input token.

## What the production UI projections expose

The actual `createPlayer`, `attachContent`, `projectRequestTokens`,
`projectKvCache`, and `mappedBlockSlot` implementations produced these results:

| Event | Request token 173 | Grid Slot 13 in Block 11 |
| --- | --- | --- |
| 75 | Does not exist yet | No position, ID, fragment, or binding |
| 76–77 | ID 785 known; `not_computed`; no KV location | No position, ID, fragment, or binding |
| 88–91 | ID 785 known; `reserved`; location exists with `valid:false` | Position 173 appears; ID and fragment remain absent |
| 92–93 | ID 785, fragment `The`, `computed`; location `valid:true` | Position 173, ID 785, and `The` appear |

The grid obtains `Pos 173` from the recorded binding, and ID/text from a
computed content span (`web/src/architecture/model.ts`, lines 77–87;
`web/src/architecture/CacheStage.tsx`, lines 222–226). The token inspector can
resolve an earlier accepted ID from the request (`replay/src/learning.ts`,
lines 258–262). Displaying the position at reservation therefore does not
mean the current iteration just sampled that token.

## Published vLLM source and recording boundary

Official release tag `v0.30.0` resolves to immutable SHA
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

- [Scheduler input budget and allocation](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L656-L733): determine outstanding input positions and call `allocate_slots` before model execution.
- [Physical block count](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/single_type_kv_cache_manager.py#L393-L400): allocate the difference between required blocks and existing blocks, returning without new blocks when capacity suffices.
- [Engine iteration order](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L600-L616): schedule, execute, obtain the forward result, sample when needed, and update the scheduler from output.
- [Layer cache update](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backends/flash_attn.py#L1438-L1472): scatter newly computed K/V into addresses supplied by the slot mapping.
- [Sampling after forward](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py#L4595-L4605): sample from logits and update runner state afterward.

The adapter copies bindings after `scheduler.schedule` returns
(`instrumentation/vllm_030/install.py`, lines 188–196). It stages scheduled
write content there (`inventory.py`, lines 170–221), then emits
`kv_block_written` at a result boundary (`install.py`, lines 63–83).
Accepted-token events follow the scheduler's output update (`install.py`,
lines 198–205). These are host observations, not exact CUDA write completion
timestamps. Native profile timing has its own clock.

This demo disables async scheduling (`manifest.engine_settings`;
`runner/vllm_session.py`, line 514), and the runner rejects speculative
decoding (lines 837–840). CUDA launches remain asynchronous. Other async,
speculative, connector, or parallel configurations can have different
reservation and output-order details.

The published bodies of `GPUModelRunner.execute_model`, `_prepare_inputs`,
`_update_states`, and `FlashAttentionImpl.do_kv_cache_update` match the four
hook fingerprints recorded in the demo. The JSON records each comparison.
The manifest's full upstream revision is null, so this demonstrates parity
for those functions rather than every runtime source file.

## Reproduce the projections

From the repository root, run this against the retained production modules:

```sh
node --experimental-strip-types --input-type=module <<'JS'
import { readFileSync } from 'node:fs';
import { createPlayer, attachContent, parseJsonl } from './replay/src/reduce.ts';
import { projectRequestTokens, projectKvCache } from './replay/src/learning.ts';
import { mappedBlockSlot } from './web/src/architecture/model.ts';

const p = 'web/public/recordings/gpu-kv-layer-diagnostics/';
const events = parseJsonl(readFileSync(p + 'events.jsonl', 'utf8'));
const content = parseJsonl(readFileSync(p + 'content.jsonl', 'utf8'));
const manifest = JSON.parse(readFileSync(p + 'manifest.json', 'utf8'));
const player = createPlayer(events, manifest);

for (const sequence of [75, 76, 77, 88, 89, 91, 92, 93]) {
  const state = attachContent(player.at(sequence), content);
  const block = projectKvCache(state, manifest).pools
    .flatMap(pool => pool.blocks).find(block => block.blockId === 11);
  const token = projectRequestTokens(state, 'prefix-prime')
    .find(token => token.position === 173);
  const slot = mappedBlockSlot(block, 16, 13, 'prefix-prime');
  console.log(JSON.stringify({
    sequence,
    acceptedTokenCount: state.requests['prefix-prime'].acceptedTokenCount,
    requestToken: token ? {
      tokenId: token.tokenId, tokenPiece: token.tokenPiece,
      kvState: token.kvState, locations: token.locations,
    } : null,
    gridSlot: {
      position: slot.position, tokenId: slot.tokenId,
      tokenPiece: slot.tokenPiece, hasBinding: !!slot.binding,
      hasComputedSpan: !!slot.span,
    },
  }));
}
JS
```

Compare the eight outputs with the `projections` array in the audit JSON.
