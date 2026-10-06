# Authored KV diagnostic recording — no GPU evidence

Every device name, operation interval, token ID, tensor number, storage shape and architecture dimension in this recording is **synthetic**. No vLLM engine, CUDA profiler, tokenizer, model weights, or numerical cache tensor was executed or inspected. This fixture exercises the recording/replay/UI contract; it does not qualify the new runtime instrumentation.

The production `Recording` writer generates the JSONL envelopes. The production `normalize_trace` function normalizes deliberately authored Chrome-trace-shaped input. The production replay reducer, profile parser, cache projections, and frontend rerun-setting restoration validate the resulting files.

## What it represents

- One authored `cuda:0` pool contains two physical blocks with four token slots each: eight token positions of capacity, independently of the 64-token context setting.
- A small **authored** two-layer decoder has two KV heads with head dimension16. Each layer cache has shape `[2 blocks, 2 heads, 4 slots, 32 packed K/V dimensions]`, float16 storage, and a separate 1,024-byte backing allocation. Total backing storage is 2,048 bytes. These are toy dimensions, not the dimensions of the reference Qwen3 model.
- `prefix-writer` prefills four readable fragments: `The`, ` shared`, ` prefix`, ` ends`. Its sampled period is never processed or written to KV.
- `prefix-reader` reuses those four cached positions, then prefills ` with` and ` reuse` into block1. Its next step decodes the previously sampled `!`. The final sampled ` Done` is never processed or written to KV.
- Token IDs10–15 and100–102 are authored identities. The readable fragments are supplied explicitly, not decoded by a tokenizer.
- Eight authored numerical K/V slices cover positions0–3, layer0, head0, dimensions0–15. They exactly consume a 256-byte float16 capture budget. Layer1 numerical values are deliberately omitted, exercising the partial-capture and targeted-rerun UI.

## Useful UI bookmarks

The manifest’s `qualification_bookmarks` provides structural-event sequences for writer computation, reader scheduling, reader suffix computation and reader decode computation. The structural clock is a deterministic authored clock, independent of the authored profiler clock.

| Selection | Native profiler time | Expected synthetic state |
| --- | ---: | --- |
| `step-1`, writer position0 | 130µs | layer0 cache-update operation active; layer1 scheduled |
| `step-1`, writer position0 | 170µs | layer0 completed; layer1 operation active |
| `step-2`, reader position4 | 430µs | layer0 operation active; layer1 scheduled |
| `step-2`, reader positions0–3 | any | reused previously computed prefix KV |
| writer position1, layer0, head0 | any | two retained numerical examples: K and V |
| writer position1, layer1, head0 | any | numerical values unavailable; targeted rerun can be prepared |

The layer operations share their intervals across participating token positions. The fixture makes no individual-token or individual-head latency claim. CPU update scopes return before the authored CUDA activity ends, so CPU return cannot be mistaken for GPU completion.

For browser import, use `manifest.json`, `events.jsonl`, `content.jsonl`, and `profiling/activities.json`. The raw gzip trace is optional retained evidence. The fixture is intentionally absent from the production demonstration catalog. Temporary test-serving copies must remain labeled synthetic.

## Reproduce and validate

From the repository root:

```sh
python docs/qualification/kv-diagnostics/create_authored_fixture.py
node --experimental-strip-types docs/qualification/kv-diagnostics/validate_authored_fixture.mjs
```

The validator checks hashes, the strict optional profile schema, physical capacity, device/dtype/quantization metadata, layer active/completed states, prefix reuse, bounded numeric values, missing-value behavior, and workload/configuration restoration for a targeted rerun. The passing result is retained in `../authored-validation.json`.

Rerun preparation restores the reference `Qwen/Qwen3-4B-Instruct-2507` configuration and readable workload. It does not reproduce this toy architecture or these invented tensor numbers. A real capture still requires an independently qualified CUDA vLLM host.
