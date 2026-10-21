#!/usr/bin/env python3
"""Capture six readable requests in three arrival groups through the real API."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("qualification", ROOT / "docs/qualification/3d-learning-gpu/qualify.py")
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
assessment_spec = importlib.util.spec_from_file_location("expansion_evidence", Path(__file__).with_name("assess.py"))
assessment = importlib.util.module_from_spec(assessment_spec)
assessment_spec.loader.exec_module(assessment)
# Reuse the actual API capture/export path with the dense/hybrid evidence checks.
qualification.assess = lambda _name, directory, expected: assessment.assess(directory, expected)


OPERATOR_CORPUS = ROOT / "web/src/scenario/operatorDemo.json"


def corpus_workload(*, second_ms: int = 20000, third_ms: int = 20025,
                    adapter_routes: dict[str, str] | None = None) -> dict:
    """Use the same authored documents, tasks and output caps as Configure."""
    corpus = json.loads(OPERATOR_CORPUS.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in corpus["requests"]}
    offsets, batches = {}, []
    for index, batch in enumerate(corpus["batches"]):
        offset = [batch["arrival_offset_ms"], second_ms, third_ms][index]
        samples = []
        for id in batch["request_ids"]:
            row = by_id[id]
            offsets[id] = offset
            samples.append({"id": id, "label": row["label"], "max_output_tokens": row["max_output_tokens"],
                            "adapter_id": (adapter_routes or {}).get(id, "base")})
        batches.append({"id": batch["id"], "label": batch["label"], "arrival_offset_ms": offset, "samples": samples})
    requests = []
    for row in corpus["requests"]:
        text = f'{corpus["documents"][row["document"]]}\n\nTask\n{row["task"]}'
        requests.append({**qualification.request(row["id"], text, offsets[row["id"]], row["max_output_tokens"]),
                         "adapter_id": (adapter_routes or {}).get(row["id"], "base")})
    return {"requests": requests, "workload_plan": {"version": 1, "batches": batches},
            "demo_qualification": {"corpus_name": corpus["name"], "corpus_version": corpus["version"], **corpus["qualification"]}}


def workload(*, second_ms: int = 20000, third_ms: int = 20025) -> dict:
    return {
        "engine": "vllm", "model": qualification.MODEL, "model_revision": qualification.REVISION,
        "tokenizer_revision": qualification.REVISION, "dtype": "bfloat16", "quantization": "none", "kv_cache_dtype": "auto",
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "stop": []},
        "max_model_len": 2048, "max_num_seqs": 3, "token_budget": 64,
        "prefix_caching": True, "chunked_prefill": True, "gpu_memory_utilization": 0.8,
        "kv_cache_memory_bytes": 512 * 1024 * 1024, "scheduling_policy": "fcfs", "long_prefill_token_threshold": 64,
        **corpus_workload(second_ms=second_ms, third_ms=third_ms),
        "profiling_configuration": {"mode": "eager_diagnostic", "max_steps": 1024,
            "tensor_capture": {"enabled": True, "preset": "custom", "request_id": "request-1",
                "token_positions": [0, 1], "layer_indices": [0, 18, 35], "head_indices": [0, 1],
                "dimension_start": 0, "dimension_count": 16, "max_bytes": 1048576}},
    }


def assess_expansion(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    admitted = {event["request_id"]: event["sequence"] for event in events if event["event_type"] == "engine_admitted"}
    finished = {event["request_id"]: event["sequence"] for event in events if event["event_type"] == "completion"}
    schedule = [event for event in events if event["event_type"] == "step_scheduled"]
    groups = manifest["workload_plan"]["batches"]
    checks = {
        "six_requests_three_arrival_groups": len(admitted) == len(finished) == 6 and [len(group["samples"]) for group in groups] == [1, 2, 3],
        "prime_finished_before_reuse_group": finished.get("request-1", float("inf")) < admitted.get("request-2", -1),
        "third_group_arrives_while_second_active": admitted.get("request-4", float("inf")) < finished.get("request-3", -1),
        "observed_multi_request_iterations": any(len(event["requests"]) > 1 for event in schedule),
        "observed_prefill_decode_overlap": any(any(item["prompt_tokens"] > 0 for item in event["requests"]) and any(item["generation_tokens"] > 0 for item in event["requests"]) for event in schedule),
        "memory_samples_retained": any(event["event_type"] == "gpu_memory_sample" for event in events),
    }
    return {"ok": all(checks.values()), "checks": checks, "admitted": admitted, "finished": finished,
            "iteration_participants": [{"step": event["step_id"], "requests": event["requests"]} for event in schedule]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recordings", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--second-ms", type=int, default=20000)
    parser.add_argument("--third-ms", type=int, default=20025)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpu-memory", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text()) if args.payload else workload(second_ms=args.second_ms, third_ms=args.third_ms)
    qualification.RunRequest(**payload)
    requests, plan = qualification.parse_run_workload(payload)
    qualification.parse_run_configuration(payload, requests, plan)
    args.output.mkdir(parents=True, exist_ok=False)
    qualification.save(args.output / "six-request/request.json", payload)
    if not args.execute:
        print(f"Validated plan: {args.output}")
        return 0
    result = qualification.execute("six-request", payload, args)
    expansion = assess_expansion(args.output / "six-request/recording")
    qualification.save(args.output / "expansion-assessment.json", expansion)
    return 0 if result["ok"] and expansion["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
