# Arrival timing for the recorded workload

The three arrival groups retain their membership: Request 1, Requests 2–3, then Requests 4–6. Default second/third offsets are 20,000/20,025 ms for the dense BF16, FP8-KV, FP8-weight, and hybrid variants. The two LoRA variants use 40,000/40,025 ms.

The original 20-second LoRA-all attempt, `run-57e7d9e1b2a1`, is retained in `lora-all-attempt-1`. Request 1 completed **25.287 seconds** after its admission, while the second group was admitted at **20.124/20.137 seconds**. This violated the intended completed-prime scenario. All output lengths, native coverage, adapter identities, cache lineage, and cleanup checks passed; the attempt remains unqualified because the timing check failed.

The first iteration took **5.647 seconds** from scheduling to returned result. Initial runtime work and diagnostic profiling can add overhead, but these observations do not isolate its cause or prove JIT compilation caused the delay. The 40-second offsets provide margin while preserving the three-group teaching scenario. These recordings are diagnostic demonstrations, not performance benchmarks.

`prepare.py` applies those variant defaults when its offsets are omitted. Provide **both** `--second-ms` and `--third-ms` to explicitly override every variant, for example:

```sh
python docs/qualification/realistic-demos/prepare.py \
  --second-ms 45000 --third-ms 45025 --output /tmp/new-demo-plans
```

Text, token caps, adapter routes, model revisions, and engine settings are preserved when changing arrival timings. The canonical corpus keeps its 20-second baseline; each actual capture retains its own authored and observed arrival times.
