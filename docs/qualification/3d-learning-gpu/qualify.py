#!/usr/bin/env python3
"""Produce or execute bounded qualification cases through the actual runner API.

Default invocation writes a plan only. GPU execution requires --execute against
an already running, isolated controller. Run from the repository's Python env.
Raw profile copying requires --recordings pointing to that controller's files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from runner.configuration import parse_run_configuration
from runner.export import export_structural
from runner.load import load_recording
from runner.observability import source_identity
from runner.report import evaluate
from runner.serve import RunRequest
from runner.workload import parse_run_workload

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CASE_NAMES = ("staggered", "prefix", "chunked", "pressure", "priority", "profile", "cancel-profile")


def chat(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def request(request_id: str, text: str, offset: int = 0, output: int = 64, *, tokens: int | None = None) -> dict:
    item = {"request_id": request_id, "prompt": chat(text) if tokens is None else "",
            "arrival_offset_ms": offset, "max_output_tokens": output, "priority": 0}
    if tokens is not None:
        item.update(input_mode="synthetic", target_input_tokens=tokens)
    return item


def cases() -> dict[str, dict]:
    base = {"engine": "vllm", "model": MODEL, "model_revision": REVISION,
            "tokenizer_revision": REVISION, "dtype": "bfloat16", "quantization": "none",
            "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "stop": []},
            "max_model_len": 4096, "max_num_seqs": 4, "token_budget": 256,
            "prefix_caching": False, "chunked_prefill": True,
            "gpu_memory_utilization": 0.8, "scheduling_policy": "fcfs",
            "long_prefill_token_threshold": 0, "profiling_configuration": {"mode": "off", "max_steps": 8}}
    prefix = "Read this shared reference before answering. " + "The explorer records requests, scheduler steps and cache blocks. " * 24
    return {
        "staggered": {**base, "requests": [request(f"staggered-{i + 1}",
            f"Write a detailed numbered explanation of {topic}, with six examples.", offset, 128)
            for i, (topic, offset) in enumerate([
                ("continuous batching", 0), ("prefill", 0), ("decode", 100),
                ("KV caching", 100), ("prefix reuse", 100), ("GPU scheduling", 120)])]},
        "prefix": {**base, "prefix_caching": True, "token_budget": 2048,
            "requests": [request("prefix-prime", prefix + "Name the three things recorded.", output=8),
                         request("prefix-reuse", prefix + "Explain why cache blocks are useful.", 1500, 8)]},
        "chunked": {**base, "max_model_len": 1024, "token_budget": 64,
            "long_prefill_token_threshold": 64,
            "requests": [request("chunked-long", "", output=8, tokens=512),
                         request("chunked-short", "", output=8, tokens=16)]},
        "pressure": {**base, "max_model_len": 256, "token_budget": 256,
            "kv_cache_memory_bytes": 72 * 1024 * 1024,
            "requests": [request(f"pressure-{i + 1}", "", output=32, tokens=192) for i in range(4)]},
        "priority": {**base, "max_num_seqs": 1, "scheduling_policy": "priority",
            "requests": [{**request(f"priority-{i + 1}", "Explain GPU scheduling in twelve detailed numbered points.", output=128), "priority": priority}
                         for i, priority in enumerate([5, -5, 0])]},
        "profile": {**base, "profiling_configuration": {"mode": "eager_diagnostic", "max_steps": 8},
            "requests": [request("profile-a", "Explain continuous batching in twelve detailed numbered points.", output=128),
                         request("profile-b", "Explain KV cache allocation in twelve detailed numbered points.", output=128)]},
        "cancel-profile": {**base, "token_budget": 64, "long_prefill_token_threshold": 64,
            "profiling_configuration": {"mode": "eager_diagnostic", "max_steps": 16},
            "requests": [request("cancel-long", "", output=512, tokens=3500)]},
    }


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def api(base: str, path: str, payload: dict | None = None, *, text: bool = False) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(base.rstrip("/") + path, data=data,
                  headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
    try:
        with urlopen(req, timeout=30) as response:
            value = response.read().decode("utf-8")
    except HTTPError as error:
        raise RuntimeError(f"{req.method} {path}: HTTP {error.code}: {error.read().decode('utf-8')}") from error
    return value if text else json.loads(value)


def assess(name: str, directory: Path, expected: dict) -> dict:
    data = load_recording(directory)
    report = evaluate(data)
    (directory / "integrity-report.txt").write_text(report.text(), encoding="utf-8")
    manifest, events = data.manifest, data.events
    checks: list[dict] = []
    def check(label: str, ok: bool, detail: object) -> None:
        checks.append({"check": label, "ok": bool(ok), "detail": detail})
    check("integrity", report.ok, [finding.code for finding in report.findings if not finding.ok])
    actual_identity = (manifest.get("observability") or {}).get("code_identity") or {}
    expected_identity = source_identity(ROOT)
    check("distributed capture source matches assessment source", actual_identity.get("source_sha256") == expected_identity["source_sha256"],
          {"actual": actual_identity, "expected": expected_identity})
    check("real GPU HTTP execution", manifest.get("execution_kind") == "gpu"
          and (manifest.get("provenance") or {}).get("real_gpu") is True
          and (manifest.get("capture") or {}).get("http_boundary_observed") is True, manifest.get("hardware"))
    check("loaded dense Qwen3 geometry", (manifest.get("architecture") or {}).get("supported") is True
          and manifest["architecture"].get("source") == "loaded_model_config"
          and manifest["architecture"].get("num_layers") == 36, manifest.get("architecture"))
    check("pinned model and tokenizer", manifest.get("model", {}).get("revision") == REVISION
          and manifest.get("tokenizer", {}).get("revision") == REVISION, {"model": manifest.get("model"), "tokenizer": manifest.get("tokenizer")})
    pools = (manifest.get("cache_geometry") or {}).get("pools", [])
    groups = (manifest.get("cache_geometry") or {}).get("groups", [])
    check("full authoritative inventory and qualified token mapping", bool(pools) and bool(groups)
          and all(pool.get("inventory_complete") is True and len(pool.get("blocks", [])) == pool.get("num_gpu_blocks") for pool in pools)
          and all(group.get("token_slot_mapping") == "qualified_full_attention" for group in groups),
          {"pool_capacities": [pool.get("num_gpu_blocks") for pool in pools], "mapping": [group.get("token_slot_mapping") for group in groups]})
    check("returned-result KV writes and readable content", any(event["event_type"] == "kv_block_written" for event in events)
          and any(row.get("kind") == "kv_tokens" and row.get("token_pieces") for row in data.content), len(data.content))
    scheduled = [event for event in events if event["event_type"] == "step_scheduled"]
    check("scheduler before/after queues", bool(scheduled) and all(event.get("queue_before", {}).get("available") is True
          and event.get("queue_after", {}).get("available") is True for event in scheduled), len(scheduled))
    prompts = {event["request_id"]: event["prompt_token_count"] for event in events if event["event_type"] == "engine_prompt_observed"}
    accepted: dict[str, int] = {}
    for event in events:
        if event["event_type"] == "tokens_accepted":
            accepted[event["request_id"]] = max(accepted.get(event["request_id"], 0), event["output_position_start"] + event["token_count"])
    completed = {event["request_id"] for event in events if event["event_type"] == "completion"}
    final_boundaries = {rid: not any(event["event_type"] == "kv_block_written" and event.get("request_id") == rid
        and event["token_range_start"] <= prompts[rid] + accepted[rid] - 1 < event["token_range_end"] for event in events)
        for rid in completed if rid in prompts and accepted.get(rid)}
    if name != "cancel-profile":
        check("all requests complete with output", manifest.get("completion_status") == "completed"
              and completed == {item["request_id"] for item in expected["requests"]} and all(accepted.get(rid, 0) > 0 for rid in completed), accepted)
        check("final sampled tokens have no computed KV", bool(final_boundaries) and all(final_boundaries.values()), final_boundaries)
    if name == "staggered":
        memberships = [{item["request_id"] for item in event["requests"]} for event in scheduled]
        check("continuous batch membership changes", any(len(batch) > 1 for batch in memberships)
              and len({tuple(sorted(batch)) for batch in memberships}) > 1, [sorted(batch) for batch in memberships[:12]])
    if name == "prefix":
        hits = [event["matched_token_count"] for event in events if event["event_type"] == "prefix_lookup" and event.get("request_id") == "prefix-reuse"]
        remainder = [item for event in scheduled for item in event["requests"] if item["request_id"] == "prefix-reuse" and item["prompt_tokens"] > 0]
        check("partial prefix hit and remaining prefill", bool(hits) and 0 < max(hits) < prompts.get("prefix-reuse", 0) and bool(remainder), {"hits": hits, "remaining_prefill": remainder})
        prime_done = next((event["sequence"] for event in events if event["event_type"] == "completion" and event["request_id"] == "prefix-prime"), None)
        follower = next((event["sequence"] for event in events if event["event_type"] == "engine_admitted" and event["request_id"] == "prefix-reuse"), None)
        check("prime completes before follower admission", prime_done is not None and follower is not None and prime_done < follower, {"prime_done": prime_done, "follower_admitted": follower})
    if name == "chunked":
        ranges = [item for event in scheduled for item in event["requests"] if item["request_id"] == "chunked-long" and item["prompt_tokens"]]
        check("multiple capped prefill chunks", len(ranges) >= 8 and all(item["prompt_tokens"] <= 64 for item in ranges), ranges)
    if name == "pressure":
        preemptions = [event for event in events if event["event_type"] == "preemption"]
        capacity = sum(pool.get("num_gpu_blocks", 0) - sum(block.get("is_null", False) for block in pool.get("blocks", [])) for pool in pools)
        demand = sum(math.ceil(item["target_input_tokens"] / 16) for item in expected["requests"])
        check("KV budget applied and prompt demand exceeds usable capacity", manifest.get("engine_settings", {}).get("kv_cache_memory_bytes") == 72 * 1024 * 1024 and demand > capacity,
              {"prompt_blocks": demand, "usable_blocks": capacity, "observed_preemptions": len(preemptions), "preempted_request_ids": sorted({event["request_id"] for event in preemptions})})
    if name == "priority":
        expected_priorities = {item["request_id"]: item["priority"] for item in expected["requests"]}
        admitted = {event["request_id"]: event.get("priority") for event in events if event["event_type"] == "engine_admitted"}
        observed_queues = [event[boundary]["waiting"] for event in scheduled for boundary in ("queue_before", "queue_after")
                           if event.get(boundary, {}).get("available") is True and len(event[boundary].get("waiting", [])) >= 2]
        check("priority policy and authored priorities applied", manifest.get("engine_settings", {}).get("scheduling_policy") == "priority"
              and admitted == expected_priorities, {"authored": expected_priorities, "admitted": admitted})
        check("multiple observed waiters ordered by priority", bool(observed_queues)
              and all([item["priority"] for item in queue] == sorted(item["priority"] for item in queue)
                      and all(item["priority"] == expected_priorities[item["request_id"]] for item in queue) for queue in observed_queues), observed_queues[:8])
    if name in {"profile", "cancel-profile"}:
        profile = manifest.get("profiling") or {}
        attachment = directory / "profiling/activities.json"
        normalized = json.loads(attachment.read_text()) if attachment.is_file() else {}
        activities, steps = normalized.get("activities", []), normalized.get("steps", [])
        correlated_layers = [activity for activity in activities if activity["kind"].startswith("cuda_")
            and activity.get("scope", {}).get("correlation") == "runtime_flow" and activity["scope"].get("layer_index") is not None]
        check("eager native CUDA layer correlation", manifest.get("engine_settings", {}).get("enforce_eager") is True
              and manifest.get("engine_settings", {}).get("model_runner") == "v1"
              and bool(correlated_layers) and profile.get("clock") == {"domain": "torch.profiler.chrome_trace", "unit": "microseconds", "origin": "trace_local"},
              {"status": profile.get("status"), "steps": len(steps), "correlated_layer_activities": len(correlated_layers)})
        check("post-reorder packed rows", bool(steps) and all(step.get("mapping_status") == "observed" and step.get("padding_status") == "observed" for step in steps),
              [{"step_id": step["step_id"], "rows": len(step["packed_rows"]), "boundary": step.get("mapping_observation_boundary")} for step in steps])
        check("raw and normalized artifact integrity", bool(profile.get("raw_artifacts")) and all(artifact_ok(directory, artifact) for artifact in profile.get("raw_artifacts", []))
              and artifact_ok(directory, profile.get("normalized_artifact")), profile.get("raw_artifacts"))
        if name == "profile":
            check("exact first eight workload steps", [step["step_id"] for step in steps] == [event["step_id"] for event in scheduled[:8]] and len(steps) == 8, [step["step_id"] for step in steps])
        else:
            check("cancelled capture retains partial profile", manifest.get("completion_status") == "cancelled" and profile.get("status") == "partial" and 0 < len(steps) < 16,
                  {"capture": manifest.get("completion_status"), "profile": profile.get("status"), "steps": len(steps)})
    exported = export_structural(directory, directory / "structural-export")
    exported_report = evaluate(load_recording(exported))
    check("structural export retains integrity and omits content", exported_report.ok and not (exported / "content.jsonl").exists(), exported_report.text())
    return {"case": name, "run_id": manifest["run_id"], "ok": all(item["ok"] for item in checks), "checks": checks,
            "observability_requirement": "LangSmith waived by the user for this project; actual recording evidence inspected here.",
            "remaining_review": "Output meaning, replay seek consistency, and browser measured-profile presentation require human/independent inspection."}


def artifact_ok(directory: Path, artifact: dict | None) -> bool:
    if not artifact or not isinstance(artifact.get("path"), str):
        return False
    path = (directory / artifact["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        return False
    data = path.read_bytes()
    return len(data) == artifact.get("size_bytes") and hashlib.sha256(data).hexdigest() == artifact.get("sha256")


def gpu_memory_receipt() -> dict:
    """Read-only receipts from this machine; never a forwarded host inference."""
    receipt = {"clock_domain": "qualification_host:time.monotonic_ns", "observed_monotonic_ns": time.monotonic_ns(),
               "status": "observed", "host_scope": "machine executing this qualification helper"}
    for name, fields in (("gpus", "uuid,memory.used,memory.free,memory.total"), ("compute_apps", "pid,used_memory")):
        query = "--query-gpu=" if name == "gpus" else "--query-compute-apps="
        try:
            result = subprocess.run(["nvidia-smi", query + fields, "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            receipt.update(status="unavailable", reason=type(error).__name__)
            return receipt
        receipt[name] = {"fields": fields.split(","), "units": "memory in MiB", "returncode": result.returncode,
                         "rows": [line.strip() for line in result.stdout.splitlines() if line.strip()]}
        if result.returncode:
            receipt["status"] = "unavailable"
    return receipt


def execute(name: str, payload: dict, args: argparse.Namespace) -> dict:
    destination = args.output / name
    if args.gpu_memory:
        save(destination / "gpu-memory-before.json", gpu_memory_receipt())
    accepted = api(args.base_url, "/api/runs", payload)
    save(destination / "launch.json", accepted)
    run_id = accepted["run_id"]
    api(args.base_url, "/api/runs/current/close-intake", {})
    deadline, sequence, completed_steps = time.monotonic() + args.timeout, 0, set()
    last_phase, cancellation = None, None
    while time.monotonic() < deadline:
        trace = api(args.base_url, f"/api/runs/{run_id}/trace?after_sequence={sequence}&limit=2000")
        sequence = trace["next_event_sequence"]
        for event in trace["events"]:
            if event["event_type"] == "step_result_received":
                completed_steps.add(event["step_id"])
        phase = (trace.get("capture") or {}).get("phase")
        if phase != last_phase:
            print(f"{name}: {run_id}: {phase}", flush=True)
            last_phase = phase
        if name == "cancel-profile" and len(completed_steps) >= 2 and cancellation is None and not trace["finalized"]:
            current = api(args.base_url, "/api/runs/current")
            if current["run_id"] != run_id:
                raise RuntimeError("Controller switched runs; refusing to cancel another capture.")
            cancellation = api(args.base_url, "/api/runs/current/cancel", {})
            save(destination / "cancellation.json", {"response": cancellation, "observed_completed_steps": sorted(completed_steps), "after_sequence": sequence})
        if trace["finalized"]:
            save(destination / "final-state.json", trace)
            if args.gpu_memory:
                save(destination / "gpu-memory-after.json", gpu_memory_receipt())
            break
        time.sleep(0.01 if name == "cancel-profile" else 0.1)
    else:
        raise TimeoutError(f"{name}: capture {run_id} did not finalize within {args.timeout}s; inspect and cancel it explicitly.")
    recording = destination / "recording"
    recording.mkdir(exist_ok=True)
    for file, endpoint in (("manifest.json", "manifest"), ("events.jsonl", "events"), ("content.jsonl", "content")):
        (recording / file).write_text(api(args.base_url, f"/api/recordings/{run_id}/{endpoint}", text=True), encoding="utf-8")
    manifest = json.loads((recording / "manifest.json").read_text())
    profile = manifest.get("profiling") or {}
    if profile.get("normalized_artifact"):
        (recording / "profiling").mkdir(exist_ok=True)
        (recording / "profiling/activities.json").write_text(api(args.base_url, f"/api/recordings/{run_id}/profile", text=True), encoding="utf-8")
    for artifact in profile.get("raw_artifacts", []):
        if args.recordings:
            source = (args.recordings / run_id / artifact["path"]).resolve()
            target = (recording / artifact["path"]).resolve()
            if Path(artifact["path"]).is_absolute() or not source.is_relative_to((args.recordings / run_id).resolve()) or not target.is_relative_to(recording.resolve()):
                raise ValueError("Unsafe raw artifact path.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    result = assess(name, recording, payload)
    save(destination / "assessment.json", result)
    print(f"{name}: {'PASS' if result['ok'] else 'FAIL'}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run real GPU cases through an existing isolated runner.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory (must not exist).")
    parser.add_argument("--recordings", type=Path, help="Runner recording root on this machine, for copying raw native traces.")
    parser.add_argument("--case", choices=CASE_NAMES, action="append", help="Selected cases; default is all seven.")
    parser.add_argument("--gpu-memory", action="store_true", help="Retain pre/post-case nvidia-smi receipts from this helper's machine; use on the actual controller host.")
    parser.add_argument("--timeout", type=int, default=1800, help="Preparation plus capture timeout per case.")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {name: cases()[name] for name in args.case or CASE_NAMES}
    for name, payload in selected.items():
        RunRequest(**payload)
        requests, plan = parse_run_workload(payload)
        parse_run_configuration(payload, requests, plan)
        save(args.output / name / "request.json", payload)
    save(args.output / "plan.json", {"model": MODEL, "revision": REVISION, "cases": list(selected),
         "capture_source_identity": source_identity(ROOT), "execute_requested": args.execute,
         "notes": ["No case is qualified until its actual observations meet the assessment checks.",
                   "Pressure preemption counts are observations; no decision cause is inferred.",
                   "Native profiler time stays distinct from host time; no token or head duration is claimed."]})
    if not args.execute:
        print(f"Validated plan written to {args.output}; no API or GPU calls made.")
        return 0
    results = [execute(name, payload, args) for name, payload in selected.items()]
    save(args.output / "results.json", results)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
