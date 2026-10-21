#!/usr/bin/env python3
"""Retain real API recordings for the model/KV demo and bounded diagnostics.

Without --execute this validates and saves a plan. Execute on the isolated GPU
controller host so the original native profiler artifacts can also be retained.
The payloads contain readable prompts; no synthetic token generation is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("legacy_qualification", ROOT / "docs/qualification/3d-learning-gpu/qualify.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def cases() -> dict[str, dict]:
    shared = ("Read this shared reference before answering. "
              + "The explorer records request arrivals, scheduler decisions, token positions, model layers, and cache blocks. " * 8)
    base = {
        "engine": "vllm", "model": legacy.MODEL, "model_revision": legacy.REVISION,
        "tokenizer_revision": legacy.REVISION, "dtype": "bfloat16", "quantization": "none",
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "stop": []},
        "max_model_len": 1024, "max_num_seqs": 3, "token_budget": 64,
        "prefix_caching": True, "chunked_prefill": True, "gpu_memory_utilization": 0.8,
        "kv_cache_memory_bytes": 256 * 1024 * 1024, "scheduling_policy": "fcfs",
        "long_prefill_token_threshold": 64,
        "requests": [
            legacy.request("prefix-prime", shared + "Name the first thing recorded.", output=6),
            legacy.request("prefix-reader", shared + "Explain why a shared prefix can save prefill work.", 6000, 6),
            legacy.request("mixed-prefill", "Explain this independent reference. "
                           + "A scheduler admits requests, limits the tokens in a batch, and updates cache ownership after model execution. " * 8
                           + "Name the scheduler's first action.", 6000, 6),
        ],
    }
    base["workload_plan"] = {"version": 1, "batches": [
        {"id": "prime", "label": "Prime the shared prefix", "arrival_offset_ms": 0,
         "samples": [{"id": "prefix-prime", "label": "prefix-prime", "max_output_tokens": 6}]},
        {"id": "reuse-and-prefill", "label": "Reuse prefix and prefill independent request", "arrival_offset_ms": 6000,
         "samples": [{"id": request_id, "label": request_id, "max_output_tokens": 6}
                     for request_id in ("prefix-reader", "mixed-prefill")]},
    ]}
    tensor = {"enabled": True, "preset": "custom", "request_id": "prefix-prime",
              "token_positions": [0, 1, 2, 3], "layer_indices": [0, 18, 35],
              "head_indices": [0], "dimension_start": 0, "dimension_count": 16,
              "max_bytes": 1048576}
    diagnostic = {"mode": "eager_diagnostic", "max_steps": 16, "tensor_capture": tensor}
    return {
        "demo": {**base, "profiling_configuration": diagnostic},
        "trigger": {**base, "profiling_configuration": {**diagnostic,
                    "trigger_request_id": "mixed-prefill", "max_steps": 8,
                    "tensor_capture": {**tensor, "request_id": "mixed-prefill"}}},
        "partial-budget": {**base, "profiling_configuration": {**diagnostic,
                           "tensor_capture": {**tensor, "max_bytes": 64}}},
        "off": {**base, "profiling_configuration": {"mode": "off", "max_steps": 8,
                "tensor_capture": {"enabled": False}}},
        "cancel-profile": {**base, "requests": [legacy.request("cancel-long", shared + "Explain every item in detail.", output=128)],
                           "workload_plan": {"version": 1, "batches": [{"id": "cancel", "label": "Cancel during diagnostics", "arrival_offset_ms": 0,
                                "samples": [{"id": "cancel-long", "label": "cancel-long", "max_output_tokens": 128}]}]},
                           "profiling_configuration": diagnostic | {"tensor_capture": tensor | {"request_id": "cancel-long"}}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recordings", type=Path)
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpu-memory", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {name: cases()[name] for name in args.case or ("demo", "trigger", "partial-budget", "off", "cancel-profile")}
    for name, payload in selected.items():
        legacy.RunRequest(**payload)
        requests, plan = legacy.parse_run_workload(payload)
        legacy.parse_run_configuration(payload, requests, plan)
        legacy.save(args.output / name / "request.json", payload)
    legacy.save(args.output / "plan.json", {"cases": list(selected), "execute_requested": args.execute,
                "capture_source_identity": legacy.source_identity(ROOT),
                "notes": ["Actual GPU API runs, readable prompts and immutable model/tokenizer revisions.",
                          "Separate independent native-input observer run verifies numerical values.",
                          "Diagnostic profiling and numerical clones change execution overhead; recorded times are not production benchmarks."]})
    if not args.execute:
        print(f"Validated plan saved to {args.output}; no GPU calls made.")
        return 0
    results = [legacy.execute(name, payload, args) for name, payload in selected.items()]
    legacy.save(args.output / "base-assessments.json", results)
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
