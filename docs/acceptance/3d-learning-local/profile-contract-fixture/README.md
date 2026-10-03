# Authored profile contract fixture

This bundle contains six authored profiler-shaped intervals, including an explicit runtime-flow correlation and an unmapped lookalike kernel. It is **synthetic, not GPU evidence**. The pinned public Qwen3 architecture reference was added for testing model rendering, without loading weights. Host events are intentionally empty.

Import `manifest.json`, `events.jsonl`, and `profiling/activities.json` together. This exercises attachment validation, the separate profiler clock, mapped layer highlights, packed-row inspection, and missing host token contents. It does not qualify actual CUDA profiling or runtime hooks.
