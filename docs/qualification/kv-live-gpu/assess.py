#!/usr/bin/env python3
"""Audit an unchanged API recording against its submitted live request.

This helper reads native profiler evidence and recorded structural identities;
it does not execute GPU work or modify a recording. Numerical equality with
the native input K/V tensors is a separate native_reference.py qualification.
The default checks require the supported single-GPU, dense Qwen3 BF16 path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from runner.configuration import parse_run_configuration
from runner.export import export_structural
from runner.load import load_recording
from runner.observability import source_identity
from runner.profile_trace import PROFILE_CLOCK
from runner.report import evaluate
from runner.workload import parse_run_workload

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_TRACE_BYTES = 64 * 1024 * 1024
SLICE_BOUNDARY = "same_cuda_stream_after_native_cache_update_before_reuse"


def artifact_ok(directory: Path, artifact: Any) -> bool:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        return False
    path = (directory / artifact["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        return False
    payload = path.read_bytes()
    return len(payload) == artifact.get("size_bytes") and hashlib.sha256(payload).hexdigest() == artifact.get("sha256")


def native_trace(directory: Path, artifact: dict) -> dict:
    path = directory / artifact["path"]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        payload = stream.read(MAX_TRACE_BYTES + 1)
    if len(payload) > MAX_TRACE_BYTES:
        raise ValueError("Native trace exceeds the qualified 64 MiB normalization bound.")
    return json.loads(payload)


def integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def row_identity(row: dict) -> tuple:
    return row.get("row"), row.get("request_id"), row.get("token_position"), row.get("is_padding")


def allocation_identity(row: dict) -> tuple:
    return row.get("pool"), row.get("cache_group"), row.get("block_id"), row.get("allocation_generation")


def assess(directory: Path, payload: dict, *, tensor_status: str | None = None,
           require_prefix: bool = False, require_mixed_step: bool = False,
           require_prefix_snapshot: bool = False, expected_profile_steps: int | None = None,
           allow_cancelled: bool = False, request_provenance: str = "submitted") -> dict:
    data = load_recording(directory)
    manifest, events, content = data.manifest, data.events, data.content
    requests, plan = parse_run_workload(payload)
    configuration = parse_run_configuration(payload, requests, plan)
    request_ids = {request.request_id for request in requests}
    checks: list[dict] = []

    def check(label: str, ok: bool, detail: Any) -> None:
        checks.append({"check": label, "ok": bool(ok), "detail": detail})

    integrity = evaluate(data)
    check("recording integrity", integrity.ok, [finding.detail for finding in integrity.findings if not finding.ok])
    actual_identity = (manifest.get("observability") or {}).get("code_identity") or {}
    expected_identity = source_identity(ROOT)
    check("distributed capture source matches assessment source", all(actual_identity.get(key) == expected_identity[key]
          for key in ("source_sha256", "source_file_count", "source_scope")), {"actual": actual_identity, "expected": expected_identity})
    check("real GPU HTTP execution", manifest.get("execution_kind") == "gpu"
          and (manifest.get("provenance") or {}).get("real_gpu") is True
          and (manifest.get("capture") or {}).get("http_boundary_observed") is True
          and manifest.get("engine", {}).get("version") == "0.30.0", manifest.get("hardware"))
    check("pinned model and tokenizer", manifest.get("model", {}).get("id") == MODEL
          and manifest.get("model", {}).get("revision") == REVISION
          and manifest.get("tokenizer", {}).get("id") == MODEL
          and manifest.get("tokenizer", {}).get("revision") == REVISION,
          {"model": manifest.get("model"), "tokenizer": manifest.get("tokenizer")})
    recorded_config = manifest.get("run_configuration") or {}
    check("submitted inference settings retained", all(recorded_config.get(key) == configuration[key]
          for key in ("model", "vllm", "sampling", "profiling")),
          {"expected_sha256": configuration["snapshot_sha256"], "recorded_sha256": recorded_config.get("snapshot_sha256")})
    hashed_config = {key: value for key, value in recorded_config.items() if key != "snapshot_sha256"}
    digest = hashlib.sha256(json.dumps(hashed_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    check("canonical configuration hash", digest == recorded_config.get("snapshot_sha256"), digest)
    effective = manifest.get("engine_settings") or {}
    requested_engine = configuration["vllm"]
    engine_fields = ("max_model_len", "max_num_seqs", "max_num_scheduled_tokens", "enable_prefix_caching",
                     "enable_chunked_prefill", "gpu_memory_utilization", "scheduling_policy",
                     "long_prefill_token_threshold", "kv_cache_memory_bytes")
    check("requested scheduler and cache settings applied by the loaded engine", all(effective.get(key) == requested_engine[key] for key in engine_fields)
          and effective.get("async_scheduling") is False and effective.get("continuous_batching") is True,
          {key: {"expected": requested_engine[key], "actual": effective.get(key)} for key in engine_fields})
    initial = manifest.get("initial_cache_state") or {}
    check("workload begins after verified fresh-prefix-cache reset", initial.get("kind") == "fresh"
          and initial.get("reset_verified") is True and initial.get("capture_ready_after_reset") is True
          and initial.get("warmup_completed_before_reset") is True, initial)
    architecture = manifest.get("architecture") or {}
    check("loaded dense Qwen3 geometry", architecture.get("supported") is True
          and architecture.get("source") == "loaded_model_config" and architecture.get("model_type") == "qwen3"
          and (architecture.get("num_layers"), architecture.get("num_key_value_heads"), architecture.get("head_dim")) == (36, 8, 128),
          {key: architecture.get(key) for key in ("source", "model_type", "num_layers", "num_key_value_heads", "head_dim")})
    geometry = manifest.get("cache_geometry") or {}
    pools = {pool["pool"]: pool for pool in geometry.get("pools", [])}
    groups = {group["cache_group"]: group for group in geometry.get("groups", [])}
    check("full authoritative inventory and qualified token mapping", bool(pools) and bool(groups)
          and all(pool.get("inventory_complete") is True and set(block["block_id"] for block in pool.get("blocks", [])) == set(range(pool["num_gpu_blocks"])) for pool in pools.values())
          and all(group.get("token_slot_mapping") == "qualified_full_attention" and group.get("pool") in pools for group in groups.values()),
          {"pool_capacities": {name: pool.get("num_gpu_blocks") for name, pool in pools.items()}, "groups": list(groups)})
    metadata = manifest.get("kv_cache") or {}
    layers = {layer["layer_index"]: layer for layer in metadata.get("layers", [])}
    storages = {storage["storage_id"]: storage for storage in metadata.get("storages", [])}
    metadata_errors, layout_errors = [], []
    for index, layer in layers.items():
        group, storage = groups.get(layer.get("cache_group"), {}), storages.get(layer.get("storage_id"), {})
        manager, kernel = layer.get("manager_block_size"), layer.get("kernel_block_size")
        shape, strides = layer.get("shape", []), layer.get("strides", [])
        valid = layer.get("backend") == "FlashAttentionImpl" and layer.get("device") == "cuda:0" and layer.get("dtype") == "bfloat16"
        valid = valid and group.get("pool") == layer.get("pool") and layer.get("layer_name") in group.get("layer_names", [])
        valid = valid and integer(manager, 1) and group.get("block_size") == manager
        valid = valid and len(shape) == 4 and all(integer(size, 1) for size in shape) and len(strides) == len(shape) and all(integer(stride) for stride in strides) and integer(layer.get("storage_offset"))
        valid = valid and storage.get("device") == layer.get("device") and storage.get("dtype") == "bfloat16" and integer(storage.get("byte_count"), 1)
        if valid:
            last_element = layer["storage_offset"] + sum((size - 1) * stride for size, stride in zip(shape, strides))
            valid = last_element * 2 + 2 <= storage["byte_count"] and layer.get("logical_bytes") == math.prod(shape) * 2
        scales = (layer.get("quantization") or {}).get("scales") or {}
        valid = valid and all(scales.get(kind, {}).get("status") == "observed" and isinstance(scales[kind].get("shape"), list)
                              and isinstance(scales[kind].get("dtype"), str) and scales[kind].get("device") == "cuda:0" for kind in ("K", "V"))
        if not valid:
            metadata_errors.append(index)
        qualified = layer.get("layout_status") == "qualified" and layer.get("layout") == "kernel_block,kv_head,token_slot,concatenated_K_V"
        qualified = qualified and integer(manager, 1) and integer(kernel, 1) and manager % kernel == 0
        qualified = qualified and layer.get("kernel_blocks_per_manager_block") == manager // kernel and shape == [pools.get(layer.get("pool"), {}).get("num_gpu_blocks", 0) * (manager // kernel), 8, kernel, 256]
        qualified = qualified and layer.get("num_kernel_blocks") == shape[0] and layer.get("num_kv_heads") == 8 and layer.get("head_dim") == 128
        if not qualified:
            layout_errors.append(index)
    check("all 36 loaded BF16 cuda:0 cache views are observed", metadata.get("status") == "observed"
          and metadata.get("observation_boundary") == "loaded_attention_cache_tensors"
          and len(metadata.get("layers", [])) == 36 and set(layers) == set(range(36)) and not metadata_errors,
          {"layer_count": len(layers), "invalid_layers": metadata_errors})
    if configuration["profiling"]["mode"] == "eager_diagnostic":
        check("all 36 eager FlashAttention cache addresses qualified", set(layers) == set(range(36)) and not layout_errors,
              {"layer_count": len(layers), "invalid_layers": layout_errors})
    else:
        unavailable = [layer for layer in layers.values() if layer.get("layout_status") == "unavailable"]
        check("ordinary V2 numerical addressing remains explicitly unavailable", effective.get("model_runner") == "v2"
              and len(unavailable) == 36 and all(layer.get("layout") == "unavailable" and bool(layer.get("layout_reason")) for layer in unavailable),
              {"model_runner": effective.get("model_runner"), "unavailable_layers": len(unavailable), "reasons": sorted({layer.get("layout_reason", "") for layer in unavailable})})
    check("unique backing allocations counted once", bool(storages)
          and len(storages) == len(metadata.get("storages", [])) and set(storages) == {layer.get("storage_id") for layer in layers.values()}
          and metadata.get("unique_storage_bytes") == sum(storage.get("byte_count", 0) for storage in storages.values()),
          {"allocation_count": len(storages), "unique_storage_bytes": metadata.get("unique_storage_bytes"), "logical_layer_bytes": sum(layer.get("logical_bytes", 0) for layer in layers.values())})
    scheduled = [event for event in events if event["event_type"] == "step_scheduled"]
    scheduled_by_id = {step["step_id"]: step for step in scheduled}
    check("scheduler before/after queues", bool(scheduled) and all(step.get("queue_before", {}).get("available") is True
          and step.get("queue_after", {}).get("available") is True for step in scheduled), len(scheduled))
    prompts = {event["request_id"]: event["prompt_token_count"] for event in events if event["event_type"] == "engine_prompt_observed"}
    prompt_rows = {row["request_id"]: row for row in content if row.get("kind") == "prompt"}
    prompt_text = {row["request_id"]: row["text"] for row in content if row.get("kind") == "prompt_text"}
    accepted = defaultdict(list)
    output_text = defaultdict(str)
    for row in content:
        if row.get("kind") == "accepted_tokens":
            accepted[row["request_id"]].extend(row.get("token_ids", []))
        if row.get("kind") == "text":
            output_text[row["request_id"]] += row.get("text_delta", "")
    completed = {event["request_id"] for event in events if event["event_type"] == "completion"}
    if allow_cancelled:
        cancelled = {event["request_id"] for event in events if event["event_type"] == "cancellation"}
        check("cancelled capture retains explicit terminal outcomes", manifest.get("completion_status") == "cancelled"
              and bool(cancelled) and not (cancelled & completed) and cancelled | completed == request_ids
              and (manifest.get("event_completeness") or {}).get("status") == "complete"
              and all(accepted[rid] and output_text[rid].strip() for rid in completed),
              {"capture_status": manifest.get("completion_status"), "completed": sorted(completed), "cancelled": sorted(cancelled)})
    else:
        check("all requests complete with readable output", manifest.get("completion_status") == "completed" and completed == request_ids
              and all(accepted[rid] and output_text[rid].strip() for rid in request_ids),
              {rid: {"tokens": len(accepted[rid]), "text": output_text[rid]} for rid in sorted(request_ids)})
    check("accepted token content agrees with engine outputs", all(manifest.get("engine_output_token_ids", {}).get(rid) == accepted[rid] for rid in request_ids),
          {rid: len(accepted[rid]) for rid in sorted(request_ids)})
    check("readable tokenizer prompt and KV token content", all(request.input_mode == "prompt" and prompt_text.get(request.request_id) == request.prompt
          and len(prompt_rows.get(request.request_id, {}).get("token_ids", [])) == prompts.get(request.request_id)
          and len(prompt_rows.get(request.request_id, {}).get("token_pieces", [])) == prompts.get(request.request_id) for request in requests)
          and any(row.get("kind") == "kv_tokens" and row.get("token_pieces") for row in content), len(content))
    final_boundaries = {rid: not any(event["event_type"] == "kv_block_written" and event.get("request_id") == rid
        and event["token_range_start"] <= prompts[rid] + len(accepted[rid]) - 1 < event["token_range_end"] for event in events)
        for rid in completed if rid in prompts and accepted[rid]}
    check("completed requests' final sampled tokens have no computed KV", set(final_boundaries) == (completed if allow_cancelled else request_ids)
          and all(final_boundaries.values()), final_boundaries)
    hits = [event for event in events if event["event_type"] == "prefix_lookup" and event.get("matched_token_count", 0) > 0]
    if require_prefix or require_prefix_snapshot:
        partial_hits = [hit for hit in hits if hit["matched_token_count"] < prompts.get(hit["request_id"], 0)]
        remaining = [item for step in scheduled for item in step["requests"] if any(item["request_id"] == hit["request_id"] and item["range_start"] == hit["matched_token_count"] for hit in partial_hits) and item["prompt_tokens"] > 0]
        check("real partial-prefix reuse and remaining prefill", bool(partial_hits) and bool(remaining),
              [{"request_id": hit["request_id"], "matched": hit["matched_token_count"]} for hit in partial_hits])
    if require_mixed_step:
        mixed = [step["step_id"] for step in scheduled if any(item["prompt_tokens"] for item in step["requests"]) and any(item["generation_tokens"] for item in step["requests"])]
        check("mixed prefill and decode batch", bool(mixed), mixed)
    profile = manifest.get("profiling") or {}
    expected_profile = configuration["profiling"]
    if expected_profile["mode"] == "off":
        check("ordinary execution retains no profiler or numerical artifacts", profile.get("status") == "disabled" and not profile.get("enabled")
              and not profile.get("raw_artifacts") and profile.get("normalized_artifact") is None
              and not profile.get("hook_sources") and not profile.get("runtime") and not (directory / "profiling").exists()
              and profile.get("tensor_capture", {}).get("status") == "disabled" and profile.get("tensor_capture", {}).get("retained_bytes") == 0,
              {"profile_status": profile.get("status"), "tensor_status": profile.get("tensor_capture", {}).get("status")})
    else:
        _assess_profile(directory, manifest, events, scheduled, scheduled_by_id, layers, profile, expected_profile,
                        check, tensor_status=tensor_status, expected_profile_steps=expected_profile_steps,
                        require_prefix_snapshot=require_prefix_snapshot, prefix_hits=hits, allow_cancelled=allow_cancelled)
    with TemporaryDirectory(prefix="kv-qualification-export-") as temporary:
        exported = export_structural(directory, Path(temporary) / "structural-export")
        exported_data = load_recording(exported)
        check("structural export omits numerical/content evidence and retains cache geometry", evaluate(exported_data).ok
              and not (exported / "content.jsonl").exists() and not (exported / "profiling").exists()
              and exported_data.manifest.get("kv_cache") == metadata and exported_data.manifest.get("cache_geometry") == geometry,
              {"integrity": evaluate(exported_data).ok, "retains_loaded_cache_metadata": exported_data.manifest.get("kv_cache") == metadata})
    return {"run_id": manifest["run_id"], "ok": all(item["ok"] for item in checks), "checks": checks,
            "assessment_request": {"kind": request_provenance, "claim": "Saved original submitted POST /api/runs JSON payload." if request_provenance == "submitted"
                                   else "Expected configuration derived from the retained workload and independently observed UI selections; the raw submitted HTTP body was not retained."},
            "limits": "Native input/cache numerical equality, rendered replay interaction and overhead require separate qualification evidence. Shared native CUDA scopes describe batch operations, not individual token or head duration.",
            "observability_requirement": "LangSmith waived by the user; retained source, native profiler and recording evidence are inspected here."}


def _assess_profile(directory: Path, manifest: dict, events: tuple, scheduled: list, scheduled_by_id: dict,
                    layers: dict, profile: dict, expected: dict, check: Any, *, tensor_status: str | None,
                    expected_profile_steps: int | None, require_prefix_snapshot: bool, prefix_hits: list,
                    allow_cancelled: bool) -> None:
    raw = profile.get("raw_artifacts") or []
    artifacts_ok = len(raw) == 1 and artifact_ok(directory, raw[0]) and artifact_ok(directory, profile.get("normalized_artifact"))
    check("raw and normalized profile hashes and sizes", artifacts_ok, {"raw": raw, "normalized": profile.get("normalized_artifact")})
    if not artifacts_ok:
        return
    normalized = json.loads((directory / profile["normalized_artifact"]["path"]).read_text())
    trace = native_trace(directory, raw[0])
    native = trace.get("traceEvents", [])
    activities = {activity["activity_id"]: activity for activity in normalized.get("activities", [])}
    steps = normalized.get("steps", [])
    writes = normalized.get("cache_writes", [])
    snapshots = normalized.get("tensor_snapshots", [])
    check("native diagnostic profile retains the expected completion status", profile.get("status") == ("partial" if allow_cancelled else "completed")
          and manifest.get("engine_settings", {}).get("enforce_eager") is True and manifest.get("engine_settings", {}).get("model_runner") == "v1"
          and profile.get("clock") == PROFILE_CLOCK and normalized.get("clock") == PROFILE_CLOCK
          and normalized.get("run_id") == manifest["run_id"] and normalized.get("normalization", {}).get("status") == "completed"
          and normalized.get("normalization", {}).get("gpu_activity_count", 0) > 0,
          {"status": profile.get("status"), "normalization": normalized.get("normalization"), "captured_steps": len(steps)})
    if allow_cancelled:
        check("cancellation retains a bounded partial profile with an explicit reason", 0 < len(steps) < expected["max_steps"]
              and profile.get("completeness") == "unknown" and "cancelled" in profile.get("reason", ""),
              {"steps": len(steps), "step_limit": expected["max_steps"], "completeness": profile.get("completeness"), "reason": profile.get("reason")})
    sources = profile.get("hook_sources") or {}
    source_names = ("GPUModelRunner._update_states", "GPUModelRunner._prepare_inputs", "GPUModelRunner.execute_model", "FlashAttentionImpl.do_kv_cache_update")
    check("installed diagnostic boundaries retain source fingerprints", profile.get("runtime", {}).get("vllm") == "0.30.0"
          and all(isinstance(sources.get(name), str) and len(sources[name]) == 64 and all(char in "0123456789abcdef" for char in sources[name]) for name in source_names),
          {name: sources.get(name) for name in source_names})
    trigger = expected.get("trigger_request_id")
    admission = next((event["sequence"] for event in events if event["event_type"] == "engine_admitted" and event.get("request_id") == trigger), None)
    eligible = [step for ordinal, step in enumerate(scheduled, 1) if ordinal >= expected.get("start_step", 1)
                and (trigger is None or (admission is not None and step["sequence"] > admission))]
    expected_steps = [step["step_id"] for step in eligible[:expected["max_steps"]]]
    check("exact bounded step/request-trigger capture window", [step["step_id"] for step in steps] == expected_steps
          and bool(steps) and profile.get("captured_steps") == len(steps)
          and (expected_profile_steps is None or len(steps) == expected_profile_steps),
          {"expected": expected_steps, "actual": [step["step_id"] for step in steps], "trigger_request_id": trigger, "admission_sequence": admission})
    mapping_errors = []
    mappings = {}
    for step in steps:
        real = [row for row in step.get("packed_rows", []) if row.get("is_padding") is False]
        padding = [row for row in step.get("packed_rows", []) if row.get("is_padding") is True]
        scheduled_step = scheduled_by_id.get(step["step_id"], {})
        expected_tokens = {(item["request_id"], position) for item in scheduled_step.get("requests", []) for position in range(item["range_start"], item["range_start"] + item["range_length"])}
        actual_tokens = [(row.get("request_id"), row.get("token_position")) for row in real]
        valid = step.get("mapping_status") == "observed" and step.get("padding_status") == "observed"
        valid = valid and step.get("mapping_observation_boundary") == "GPUModelRunner._prepare_inputs.returned" and step.get("padding_observation_boundary") == "GPUModelRunner._preprocess.returned"
        valid = valid and len(real) == step.get("unpadded_tokens") == scheduled_step.get("total_scheduled_tokens") and set(actual_tokens) == expected_tokens and len(actual_tokens) == len(expected_tokens)
        valid = valid and [row.get("row") for row in real] == list(range(len(real))) and integer(step.get("padded_tokens"), len(real))
        valid = valid and [row.get("row") for row in padding] == list(range(len(real), step.get("padded_tokens", 0))) and all(row.get("request_id") is None and row.get("token_position") is None for row in padding)
        if not valid:
            mapping_errors.append(step["step_id"])
        mappings[step["step_id"]] = real
    check("post-reorder packed rows match exact scheduled logical ranges", bool(steps) and not mapping_errors,
          {"invalid_steps": mapping_errors, "real_rows": sum(len(rows) for rows in mappings.values())})
    bindings = defaultdict(list)
    structural_writes = defaultdict(list)
    for event in events:
        if event["event_type"] == "kv_block_binding":
            bindings[(event["step_id"], event["request_id"], event["cache_group"])].append(event)
        if event["event_type"] == "kv_block_written":
            structural_writes[(event["step_id"], event["request_id"], event["cache_group"])].append(event)
    participant_errors, cuda_errors = [], []
    operation_ids = [write.get("operation_id") for write in writes]
    memberships = Counter((write.get("step_id"), write.get("layer_index")) for write in writes)
    for write in writes:
        layer = layers.get(write.get("layer_index"), {})
        participants = write.get("participants", [])
        valid = write.get("status") == "observed" and write.get("observation_boundary") == "AttentionImpl.do_kv_cache_update.returned"
        valid = valid and all(write.get(key) == layer.get(key) for key in ("layer_name", "pool", "cache_group", "device"))
        valid = valid and [row_identity(row) for row in participants] == [row_identity(row) for row in mappings.get(write.get("step_id"), [])]
        for row in participants:
            candidates = bindings[(write["step_id"], row.get("request_id"), write.get("cache_group"))]
            binding = next((item for item in candidates if item["token_range_start"] <= row["token_position"] < item["token_range_end"]), None)
            valid = valid and binding is not None and allocation_identity(row) == allocation_identity(binding or {})
            if binding is not None:
                slot = row["token_position"] - binding["token_range_start"]
                valid = valid and row.get("slot") == slot and row.get("kernel_block_id") == row.get("block_id") * layer["kernel_blocks_per_manager_block"] + slot // layer["kernel_block_size"] and row.get("kernel_slot") == slot % layer["kernel_block_size"]
                valid = valid and any(allocation_identity(item) == allocation_identity(row) and item["token_range_start"] <= row["token_position"] < item["token_range_end"] for item in structural_writes[(write["step_id"], row["request_id"], write["cache_group"])])
        if not valid:
            participant_errors.append(write["operation_id"])
        gpu = [activities.get(identity, {}) for identity in write.get("gpu_activity_ids", [])]
        op_activities = [activity for activity in activities.values() if activity.get("scope", {}).get("operation_id") == write["operation_id"]]
        launches = {activity["activity_id"] for activity in op_activities if not activity["kind"].startswith("cuda_") and activity["name"].startswith(("cudaLaunch", "cuLaunch"))}
        valid_cuda = bool(gpu) and bool(launches) and any(activity.get("kind") == "cuda_kernel" and "reshape_and_cache_flash" in activity.get("name", "") for activity in gpu)
        valid_cuda = valid_cuda and set(write["gpu_activity_ids"]) == {activity["activity_id"] for activity in op_activities if activity["kind"].startswith("cuda_") and activity["scope"].get("correlation") == "runtime_flow"}
        correlated = set()
        for activity in gpu:
            scope = activity.get("scope", {})
            valid_cuda = valid_cuda and activity.get("kind", "").startswith("cuda_") and scope.get("correlation") == "runtime_flow" and scope.get("operation") == "kv_cache_write"
            valid_cuda = valid_cuda and scope.get("operation_id") == write["operation_id"] and scope.get("step_id") == write["step_id"] and scope.get("layer_index") == write["layer_index"]
            valid_cuda = valid_cuda and _native_correlation_matches(activity, activities, native)
            correlated.update(activity.get("related_activity_ids", []))
        if not valid_cuda or not launches.issubset(correlated):
            cuda_errors.append(write["operation_id"])
    check("one native cache update per layer/step with exact physical participants", len(set(operation_ids)) == len(operation_ids)
          and memberships == Counter({(step["step_id"], layer): 1 for step in steps for layer in range(36)}) and not participant_errors,
          {"operations": len(writes), "participants": sum(len(write.get("participants", [])) for write in writes), "invalid_operations": participant_errors[:20]})
    check("cache CUDA timing uses complete native runtime launch flow", bool(writes) and not cuda_errors,
          {"operations": len(writes), "invalid_operations": cuda_errors[:20]})
    _assess_snapshots(expected.get("tensor_capture") or {}, normalized, profile, writes, layers, check, tensor_status=tensor_status)
    if require_prefix_snapshot:
        reused = []
        for snapshot in snapshots:
            for hit in prefix_hits:
                if snapshot["request_id"] == hit["request_id"] or snapshot["token_position"] >= hit["matched_token_count"]:
                    continue
                candidates = [binding for (step, request_id, group), values in bindings.items() if request_id == hit["request_id"] and group == snapshot["cache_group"] for binding in values]
                if any(allocation_identity(binding) == allocation_identity(snapshot) and binding["token_range_start"] <= snapshot["token_position"] < binding["token_range_end"] for binding in candidates):
                    reused.append({"snapshot_id": snapshot["snapshot_id"], "writer": snapshot["request_id"], "reader": hit["request_id"], "position": snapshot["token_position"], "allocation": allocation_identity(snapshot)})
        check("shared-prefix reader maps to the retained original-writer snapshot", bool(reused), reused[:12])


def _native_correlation_matches(activity: dict, activities: dict, native: list) -> bool:
    """Read native profiler IDs independently of normalized scope inheritance."""
    def raw(identity: Any) -> dict:
        if not isinstance(identity, str) or not identity.startswith("activity-"):
            return {}
        try:
            return native[int(identity.removeprefix("activity-"))]
        except (ValueError, IndexError, TypeError):
            return {}
    event = raw(activity.get("activity_id"))
    if event.get("cat") not in {"kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset"} or event.get("ph") != "X":
        return False
    if event.get("ts") != activity.get("start") or event.get("dur") != activity.get("duration") or event.get("name") != activity.get("name"):
        return False
    key = "correlation" if activity.get("scope", {}).get("correlation_source") == "native_correlation_id" else "External id"
    identity = (event.get("args") or {}).get(key)
    related = activity.get("related_activity_ids", [])
    if identity is None or not related:
        return False
    for related_id in related:
        candidate = raw(related_id)
        normalized = activities.get(related_id, {})
        if candidate.get("ph") != "X" or str((candidate.get("args") or {}).get(key)) != str(identity):
            return False
        if normalized.get("scope", {}).get("operation_id") != activity.get("scope", {}).get("operation_id"):
            return False
    return True


def _assess_snapshots(configuration: dict, normalized: dict, profile: dict, writes: list, layers: dict,
                      check: Any, *, tensor_status: str | None) -> None:
    capture = normalized.get("tensor_capture") or {}
    snapshots = normalized.get("tensor_snapshots", [])
    enabled = configuration.get("enabled", False)
    expected_status = tensor_status or ("completed" if enabled else "disabled")
    check("numerical capture status and manifest agree", capture == profile.get("tensor_capture")
          and capture.get("enabled") == enabled and capture.get("status") == expected_status
          and capture.get("max_bytes") == configuration.get("max_bytes", 1048576),
          {"expected_status": expected_status, "status": capture.get("status"), "reason": capture.get("reason")})
    selection = capture.get("selection") or {}
    expected_layers = configuration.get("layer_indices") or [0, 18, 35]
    head_indices = configuration.get("head_indices") or [0]
    position_rows = [row for write in writes if write.get("layer_index") in expected_layers for row in write.get("participants", [])]
    request_id = configuration.get("request_id") or (position_rows[0]["request_id"] if position_rows else None)
    positions = configuration.get("token_positions") or [row["token_position"] for row in next((write.get("participants", []) for write in writes if write.get("layer_index") in expected_layers and any(row["request_id"] == request_id for row in write.get("participants", []))), []) if row["request_id"] == request_id][:4]
    dimensions = configuration.get("dimension_count", 16)
    start = configuration.get("dimension_start", 0)
    if enabled:
        check("numerical selection retains requested request/token/layer/head slice", selection == {
            "request_id": request_id, "token_positions": positions, "layer_indices": expected_layers,
            "head_indices": head_indices, "dimension_start": start, "dimension_count": dimensions}, selection)
    writes_by_id = {write["operation_id"]: write for write in writes}
    expected_keys = set()
    for write in writes:
        if write["layer_index"] in expected_layers:
            for row in write.get("participants", []):
                if row["request_id"] == request_id and row["token_position"] in positions:
                    expected_keys.update((write["operation_id"], request_id, row["token_position"], write["layer_index"], head, kind) for head in head_indices for kind in ("K", "V"))
    errors, actual_keys = [], []
    for snapshot in snapshots:
        write = writes_by_id.get(snapshot.get("operation_id"), {})
        participant = next((row for row in write.get("participants", []) if row["request_id"] == snapshot.get("request_id") and row["token_position"] == snapshot.get("token_position")), None)
        key = tuple(snapshot.get(field) for field in ("operation_id", "request_id", "token_position", "layer_index", "kv_head", "kind"))
        actual_keys.append(key)
        valid = key in expected_keys and snapshot.get("step_id") == write.get("step_id") and participant is not None
        valid = valid and allocation_identity(snapshot) == allocation_identity(participant or {}) and snapshot.get("slot") == (participant or {}).get("slot")
        valid = valid and snapshot.get("device") == "cuda:0" and snapshot.get("dtype") == "bfloat16" and snapshot.get("shape") == [dimensions] and snapshot.get("dimension_start") == start
        valid = valid and snapshot.get("byte_count") == dimensions * 2 and len(snapshot.get("values", [])) == dimensions and all(finite(value) for value in snapshot.get("values", []))
        valid = valid and snapshot.get("observation_boundary") == SLICE_BOUNDARY and start + dimensions <= layers.get(snapshot.get("layer_index"), {}).get("head_dim", 0)
        if not valid:
            errors.append(snapshot.get("snapshot_id"))
    byte_count = sum(snapshot.get("byte_count", 0) for snapshot in snapshots)
    check("finite bounded numerical slices preserve native allocation identity", not errors and len(set(actual_keys)) == len(actual_keys)
          and len({snapshot.get("snapshot_id") for snapshot in snapshots}) == len(snapshots)
          and capture.get("retained_bytes") == byte_count <= capture.get("max_bytes", 0)
          and capture.get("observation_boundary") == SLICE_BOUNDARY,
          {"snapshot_count": len(snapshots), "retained_bytes": byte_count, "max_bytes": capture.get("max_bytes"), "invalid_snapshots": errors[:20]})
    if expected_status == "completed":
        check("every selected K/V slice is retained", bool(expected_keys) and set(actual_keys) == expected_keys,
              {"expected_slices": len(expected_keys), "actual_slices": len(actual_keys), "expected_bytes": len(expected_keys) * dimensions * 2})
    elif expected_status == "partial":
        check("byte-budget partial capture is explicit and bounded", bool(snapshots) and set(actual_keys) < expected_keys
              and "byte limit" in capture.get("reason", "") and byte_count == (capture.get("max_bytes", 0) // (dimensions * 2)) * dimensions * 2,
              {"expected_slices": len(expected_keys), "actual_slices": len(actual_keys), "reason": capture.get("reason")})
    elif expected_status == "disabled":
        check("disabled numerical capture retains no slices", not snapshots and byte_count == 0, len(snapshots))
    elif expected_status == "unavailable":
        check("unavailable numerical capture states a reason", not snapshots and bool(capture.get("reason")) and byte_count == 0, capture.get("reason"))


def assess_native_reference(directory: Path) -> dict:
    """Recheck saved source-input receipts independently of observer pass flags."""
    data = load_recording(directory)
    manifest = data.manifest
    reference = json.loads((directory / "native-reference-assessment.json").read_text())
    normalized = json.loads((directory / "profiling/activities.json").read_text())
    checks: list[dict] = []

    def check(label: str, ok: bool, detail: Any) -> None:
        checks.append({"check": label, "ok": bool(ok), "detail": detail})

    expected_identity = source_identity(ROOT)
    identity = (manifest.get("observability") or {}).get("code_identity") or {}
    check("observer and loaded production source match repaired assessment source", identity == expected_identity
          and reference.get("source_refs", {}).get("capture_source") == expected_identity, identity)
    helper = ROOT / "docs/qualification/kv-live-gpu/native_reference.py"
    check("observer helper source retained", hashlib.sha256(helper.read_bytes()).hexdigest() == reference.get("source_refs", {}).get("helper_sha256"),
          reference.get("source_refs", {}).get("helper_sha256"))
    check("direct native observer retains its distinct evidence boundary", manifest.get("execution_kind") == "gpu"
          and manifest.get("qualification_observer", {}).get("separate_from_unmodified_api_demo") is True
          and not (manifest.get("capture") or {}).get("http_boundary_observed")
          and manifest.get("completion_status") == "completed" and evaluate(data).ok, manifest.get("qualification_observer"))
    check("native observer retains pinned model/tokenizer", manifest.get("model", {}).get("id") == MODEL
          and manifest.get("model", {}).get("revision") == REVISION and manifest.get("tokenizer", {}).get("revision") == REVISION,
          {"model": manifest.get("model"), "tokenizer": manifest.get("tokenizer")})
    layers = {layer["layer_index"]: layer for layer in manifest.get("kv_cache", {}).get("layers", [])}
    check("native observer loads all 36 qualified BF16 cache layers", set(layers) == set(range(36))
          and all(layer.get("layout_status") == "qualified" and layer.get("dtype") == "bfloat16" and layer.get("device") == "cuda:0"
                  and layer.get("backend") == "FlashAttentionImpl" and layer.get("num_kv_heads") == 8 and layer.get("head_dim") == 128 for layer in layers.values()), len(layers))
    scheduled = [event for event in data.events if event["event_type"] == "step_scheduled"]
    _assess_profile(directory, manifest, data.events, scheduled, {step["step_id"]: step for step in scheduled}, layers,
                    manifest["profiling"], manifest["run_configuration"]["profiling"], check, tensor_status="completed",
                    expected_profile_steps=None, require_prefix_snapshot=False, prefix_hits=[], allow_cancelled=False)
    snapshots = {snapshot["snapshot_id"]: snapshot for snapshot in normalized.get("tensor_snapshots", [])}
    writes = {write["operation_id"]: write for write in normalized.get("cache_writes", [])}
    identities, comparison_errors = [], []
    for comparison in reference.get("comparisons", []):
        expected = comparison.get("expected") or {}
        snapshot = snapshots.get(comparison.get("snapshot_id"), {})
        source_values, cache_values = comparison.get("source_values_float32", []), snapshot.get("values", [])
        identities.append((expected.get("layer_index"), expected.get("token_position"), expected.get("kind")))
        valid = len(source_values) == len(cache_values) == 16 and all(finite(value) for value in source_values + cache_values)
        if valid:
            valid = struct.pack("<16f", *source_values) == struct.pack("<16f", *cache_values)
        valid = valid and comparison.get("cache_values_float32") == cache_values
        valid = valid and all(snapshot.get(key) == expected.get(key) for key in ("operation_id", "step_id", "request_id", "token_position", "layer_index", "kv_head", "kind", "pool", "cache_group", "block_id", "allocation_generation", "slot", "dtype", "device", "shape", "dimension_start"))
        address, manager, kernel = expected.get("native_address"), expected.get("manager_block_size"), expected.get("kernel_block_size")
        valid = valid and integer(address) and integer(manager, 1) and integer(kernel, 1)
        if valid:
            valid = divmod(address, manager) == (snapshot.get("block_id"), snapshot.get("slot")) and divmod(address, kernel) == (expected.get("kernel_block_id"), expected.get("kernel_slot"))
        participants = writes.get(expected.get("operation_id"), {}).get("participants", [])
        participant = next((row for row in participants if row.get("row") == expected.get("packed_row")), {})
        valid = valid and all(participant.get(key) == expected.get(key) for key in ("request_id", "token_position", "pool", "cache_group", "block_id", "allocation_generation", "slot", "kernel_block_id", "kernel_slot"))
        if not valid:
            comparison_errors.append(comparison.get("snapshot_id"))
    expected_identities = {(layer, position, kind) for layer in (0, 18, 35) for position in range(4) for kind in ("K", "V")}
    check("all 24 saved native-input slices equal actual production snapshots bitwise with exact native addresses", len(identities) == 24
          and set(identities) == expected_identities and not comparison_errors,
          {"comparisons": len(identities), "invalid_snapshots": comparison_errors})
    activities = {activity["activity_id"]: activity for activity in normalized["activities"]}
    scopes = [activity for activity in activities.values() if activity["kind"] == "cpu_scope" and activity["name"].startswith("qualification.reference_inputs/")]
    write_scopes = [activity for activity in activities.values() if activity["kind"] == "cpu_scope" and activity["name"].startswith("explorer.cache_write/")]
    separated = len(scopes) == 3 and all(scope["scope"].get("operation_id") is None and all(
        scope["start"] + scope["duration"] <= write["start"] or write["start"] + write["duration"] <= scope["start"]
        for write in write_scopes if (write["process_id"], write["thread_id"]) == (scope["process_id"], scope["thread_id"])) for scope in scopes)
    check("source-input clone CPU annotations are outside native cache-update scopes", separated,
          [scope["name"] for scope in scopes])
    launches = [activity for activity in activities.values() if activity["name"].startswith(("cudaLaunch", "cuLaunch", "cudaMemcpy", "cuMemcpy"))
                and any((activity["process_id"], activity["thread_id"]) == (scope["process_id"], scope["thread_id"])
                        and scope["start"] <= activity["start"] and activity["start"] + activity["duration"] <= scope["start"] + scope["duration"] for scope in scopes)]
    native = native_trace(directory, manifest["profiling"]["raw_artifacts"][0])["traceEvents"]
    gpu = [activity for activity in activities.values() if activity["kind"].startswith("cuda_") and any(launch["activity_id"] in activity.get("related_activity_ids", []) for launch in launches)]
    check("native source-clone CUDA flow stays outside cache-write timing", bool(launches) and bool(gpu)
          and all(activity["scope"].get("operation_id") is None and _native_correlation_matches(activity, activities, native) for activity in gpu)
          and {launch["activity_id"] for launch in launches}.issubset({identity for activity in gpu for identity in activity.get("related_activity_ids", [])}),
          {"reference_launches": len(launches), "correlated_reference_gpu_activities": len(gpu)})
    return {"run_id": manifest["run_id"], "kind": "independent_saved_native_reference_audit", "ok": all(item["ok"] for item in checks), "checks": checks,
            "limits": "This separately instrumented direct-engine observer validates selected numerical slices and their addresses. Its extra clone overhead and timings are distinct from the unmodified API demo; it does not verify HTTP or browser behavior."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path, help="Original JSON POST /api/runs payload, or a separately labeled expected configuration.")
    mode.add_argument("--native-reference", action="store_true", help="Independently recheck a saved direct-engine native input observer.")
    parser.add_argument("--request-provenance", choices=("submitted", "expected"), default="submitted")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensor-status", choices=("completed", "partial", "disabled", "unavailable"))
    parser.add_argument("--require-prefix", action="store_true")
    parser.add_argument("--require-mixed-step", action="store_true")
    parser.add_argument("--require-prefix-snapshot", action="store_true")
    parser.add_argument("--expected-profile-steps", type=int)
    parser.add_argument("--allow-cancelled", action="store_true", help="Expect a cancelled run with explicit terminal request outcomes and a retained partial profile.")
    args = parser.parse_args()
    try:
        result = assess_native_reference(args.recording.resolve()) if args.native_reference else assess(args.recording.resolve(), json.loads(args.request.read_text()), tensor_status=args.tensor_status,
                        require_prefix=args.require_prefix, require_mixed_step=args.require_mixed_step,
                        require_prefix_snapshot=args.require_prefix_snapshot, expected_profile_steps=args.expected_profile_steps,
                        allow_cancelled=args.allow_cancelled, request_provenance=args.request_provenance)
    except Exception as error:
        result = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
