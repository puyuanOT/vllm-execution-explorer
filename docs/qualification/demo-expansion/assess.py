#!/usr/bin/env python3
"""Read-only assessment of a real dense or hybrid six-request recording.

This inspects the production recording contract, actual native activities and
memory samples. Successful HTTP completion and legacy workload checks alone do
not qualify a demo whose observers or engine cleanup did not finalize.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "instrumentation")]
from runner.load import load_recording
from runner.configuration import ConfigurationError, _parse_artifact_source, _parse_model
from runner.observability import source_identity
from runner.report import evaluate
from runner.profile_codec import expand_profile_wire


def artifact_ok(directory: Path, metadata: dict[str, Any] | None) -> bool:
    if not metadata or not isinstance(metadata.get("path"), str):
        return False
    path = (directory / metadata["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size != metadata.get("size_bytes"):
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == metadata.get("sha256")


def precision_evidence(manifest: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Resolve precision from loaded tensors and native cache encoding evidence."""
    weight = (manifest.get("model") or {}).get("weight_storage") or {}
    requested_model = _parse_model(expected)
    requested_quant = requested_model["quantization"]["kind"]
    requested_compute = requested_model["dtype"]
    native_formats = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    expected_format = expected.get("expected_weight_format") or (native_formats.get(requested_compute)
                        if requested_quant in {None, "none"} else requested_quant)
    requested_kv = (expected.get("vllm_configuration") or {}).get("kv_cache_dtype", expected.get("kv_cache_dtype", "auto"))
    settings_kv = (manifest.get("engine_settings") or {}).get("kv_cache_dtype")
    cache = manifest.get("kv_cache") or {}
    layers = cache.get("layers", [])
    architecture = manifest.get("architecture") or {}
    attention_indices = architecture.get("attention_layer_indices") if architecture.get("model_type") == "qwen3_5_text" else list(range(architecture.get("num_layers", 0)))
    observed_indices = [row.get("layer_index") for row in layers]
    valid = cache.get("status") == "observed" and bool(layers)
    layer_coverage = bool(attention_indices) and len(set(observed_indices)) == len(observed_indices) and sorted(observed_indices) == sorted(attention_indices)
    for layer in layers:
        storage_dtype = layer.get("dtype")
        native_dtype = (layer.get("quantization") or {}).get("cache_dtype")
        if requested_kv.startswith("fp8"):
            encoding = "fp8_e5m2" if requested_kv == "fp8_e5m2" else "fp8_e4m3"
            valid &= native_dtype == requested_kv and (storage_dtype == encoding or
                       storage_dtype == "uint8" and layer.get("numerical_encoding") == encoding and layer.get("layout_status") == "qualified")
        elif requested_kv == "auto":
            valid &= native_dtype == "auto" and storage_dtype == weight.get("compute_dtype")
        else:
            valid &= native_dtype == requested_kv and storage_dtype == requested_kv
    checks = {
        "observed_requested_weight_format": weight.get("status") == "observed" and expected_format is not None and weight.get("format") == expected_format
            and bool(weight.get("tensor_dtypes")) and bool(weight.get("compute_dtype")),
        "observed_requested_compute_dtype": weight.get("compute_dtype") in native_formats and (requested_compute == "auto" or weight.get("compute_dtype") == requested_compute),
        "requested_kv_setting_reaches_engine": requested_kv == settings_kv,
        "actual_kv_precision_observed": valid,
        "all_attention_layers_kv_precision_observed": layer_coverage,
    }
    return {"checks": checks, "requested_weight_format": expected_format, "observed_weight_storage": weight,
            "requested_compute_dtype": requested_compute, "requested_kv_dtype": requested_kv, "engine_kv_dtype": settings_kv,
            "observed_kv": [{key: row.get(key) for key in ("layer_index", "dtype", "quantization", "numerical_encoding", "layout_status")} for row in layers],
            "semantics": "Compute dtype and stored weight format differ for quantized weights; uint8 storage alone is not FP8 evidence."}


def adapter_evidence(manifest: dict[str, Any], events: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    requests = expected.get("requests", [])
    expected_routes = {row["request_id"]: row.get("adapter_id") or "base" for row in requests}
    recorded = [row for row in events if row["event_type"] == "request_model_route"]
    routes = {row["request_id"]: row.get("adapter_id", "base") for row in recorded}
    adapters = {row["id"]: row for row in (manifest.get("model") or {}).get("lora_adapters", [])}
    configured = _parse_model(expected)["lora_adapters"]
    configured = {row["id"]: row for row in configured}
    selected = set(expected_routes.values()) - {"base"}
    def artifact_source_identity(raw: Any) -> dict[str, Any] | None:
        try:
            canonical = _parse_artifact_source(raw, field="assessment.adapter.source")
        except ConfigurationError:
            return None
        identity = {key: canonical.get(key) for key in ("kind", "locator", "revision")}
        if canonical["kind"] == "operator_local":
            identity["transfer_destination"] = canonical["transfer"].get("destination")
        return identity
    sources_valid = all(id in configured and id in adapters and artifact_source_identity(configured[id].get("source")) is not None
                        and artifact_source_identity(adapters[id].get("source")) == artifact_source_identity(configured[id].get("source")) for id in selected)
    artifacts_valid = True
    for id in selected:
        row = adapters.get(id, {})
        hashes = row.get("artifact_sha256") or {}
        modules, rank = row.get("target_modules"), row.get("rank")
        artifacts_valid &= row.get("configuration_status") == "observed" and type(rank) is int and rank > 0 and isinstance(modules, (list, str)) and bool(modules)
        artifacts_valid &= "adapter_config.json" in hashes and any(name.startswith("adapter_model.") and name.endswith((".safetensors", ".bin")) for name in hashes)
        artifacts_valid &= all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in hashes.values())
    native_valid = len(recorded) == len(expected_routes)
    for row in recorded:
        id = row.get("adapter_id", "base")
        if id == "base":
            native_valid &= row.get("route_kind") == "base_model" and row.get("lora_int_id") is None
        else:
            native_id = row.get("lora_int_id")
            native_valid &= row.get("route_kind") == "lora_adapter" and type(native_id) is int and native_id > 0 and native_id == adapters.get(id, {}).get("lora_int_id")
    return {"checks": {"actual_request_adapter_routes": routes == expected_routes and native_valid,
                        "configured_adapter_source_identity": sources_valid, "adapter_configuration_and_model_hashes": artifacts_valid},
            "routes": recorded, "selected_adapters": sorted(selected)}


def prefix_evidence(manifest: dict[str, Any], events: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    """Verify reused physical generations have known owners on the reader route."""
    routes = {row["request_id"]: row.get("adapter_id", "base") for row in events if row["event_type"] == "request_model_route"}
    groups = {row["cache_group"]: row for row in (manifest.get("cache_geometry") or {}).get("groups", [])}
    generations: dict[tuple[str, int], int] = {}
    null_blocks: set[tuple[str, int]] = set()
    writers: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    bindings: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    table_owners: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    def group_key(row: dict[str, Any]) -> tuple[str, int, int, int] | None:
        group = row.get("cache_group", next(iter(groups)) if len(groups) == 1 else None)
        generation = row.get("allocation_generation")
        return (row["pool"], row["block_id"], generation, group) if group in groups and type(generation) is int and generation > 0 and (row["pool"], row["block_id"]) not in null_blocks else None
    lookups = []
    for row in events:
        kind = row["event_type"]
        if kind == "cache_inventory":
            for pool in row.get("pools", []):
                for block in pool.get("blocks", []):
                    generations[(pool["pool"], block["block_id"])] = block["allocation_generation"]
                    if block.get("is_null") is True:
                        null_blocks.add((pool["pool"], block["block_id"]))
        elif kind == "block_change":
            generations[(row["pool"], row["block_id"])] = row["allocation_generation"]
        elif kind == "kv_block_written":
            key = group_key(row)
            if key is not None:
                writers[key].add(row["request_id"])
        elif kind == "kv_block_binding":
            key = group_key(row)
            if key is not None:
                bindings[key].add(row["request_id"])
        elif kind == "step_scheduled":
            # The native scheduler copies new/resumed/extended request block
            # tables for every cache group. Recurrent groups have structural
            # ownership here even when per-token KV write mapping is unsupported.
            for change in row.get("block_table_changes", []):
                if change.get("change") not in {"replace", "append"}:
                    continue
                for group, blocks in enumerate(change.get("block_ids", [])):
                    pool = groups.get(group, {}).get("pool")
                    for block in blocks:
                        generation = generations.get((pool, block))
                        if pool is not None and type(generation) is int and generation > 0 and (pool, block) not in null_blocks:
                            table_owners[(pool, block, generation, group)].add(change["request_id"])
        elif kind == "prefix_lookup":
            references = []
            for group, blocks in enumerate(row.get("block_ids", [])):
                metadata = groups.get(group, {})
                pool = metadata.get("pool")
                for block in blocks:
                    generation = generations.get((pool, block))
                    key = (pool, block, generation, group)
                    origins = sorted(writers.get(key, set()))
                    holders = sorted(bindings.get(key, set()))
                    owners = sorted(table_owners.get(key, set()))
                    origin_routes = sorted({routes.get(origin, "unknown") for origin in origins + holders + owners})
                    references.append({"pool": pool, "block_id": block, "allocation_generation": generation, "cache_group": group,
                        "cache_type": metadata.get("cache_type"), "token_slot_mapping": metadata.get("token_slot_mapping"),
                        "reserved_null": (pool, block) in null_blocks,
                        "prior_writers": origins, "prior_bindings": holders, "prior_block_table_owners": owners, "origin_routes": origin_routes,
                        "host_result_write_already_observed": bool(origins)})
            hit = row.get("matched_token_count", 0) > 0
            lookups.append({**row, "reader_route": routes.get(row["request_id"]), "reused_storage_origins": references,
                "same_route_storage_proven": hit and bool(references) and all((reference["prior_writers"] or reference["prior_bindings"] or reference["prior_block_table_owners"]) and reference["origin_routes"] == [routes.get(row["request_id"])] for reference in references),
                "observed_cross_route_origin": any(reference["origin_routes"] and reference["origin_routes"] != [routes.get(row["request_id"])] for reference in references)})
    hits = [row for row in lookups if row.get("matched_token_count", 0) > 0]
    # This recipe contains an identical long prefix on base and adapter routes.
    # A different adapter's first reader must not hit the base priming request.
    requests = expected.get("requests", [])
    first_route = routes.get(requests[0]["request_id"]) if requests else None
    different_route_checks = []
    for request in requests[1:]:
        route = routes.get(request["request_id"])
        if route == first_route:
            continue
        # A positive hit is legitimate after a prior request on this same route
        # has written it; the allocation lineage check above proves that case.
        rows = [row for row in lookups if row["request_id"] == request["request_id"]]
        different_route_checks.extend(row.get("matched_token_count", 0) == 0 or row["same_route_storage_proven"] for row in rows)
    return {"checks": {"observed_prefix_reuse_same_adapter_route": any(row["same_route_storage_proven"] for row in hits) and not any(row["observed_cross_route_origin"] for row in hits),
                        "cross_adapter_prefix_storage_isolated": all(different_route_checks),
                        "all_request_prefix_lookups_observed": {row["request_id"] for row in lookups} == {row["request_id"] for row in requests}},
            "lookups": lookups, "positive_hits": len(hits), "different_route_lookup_count": len(different_route_checks),
            "fully_proven_hit_count": sum(row["same_route_storage_proven"] for row in hits),
            "incomplete_host_lineage_hit_count": sum(not row["same_route_storage_proven"] for row in hits),
            "scope": "Native lookup results and prior owners of the same pool/block/allocation generation/cache group prove structural adapter isolation. Native returned scheduler block tables include opaque recurrent-state block ownership, while per-token writes and numerical recurrent contents remain unavailable unless separately recorded. A native lookup may precede a prior execution's host result-associated write event; those are separately reported. Cache hashes are not retained or inferred."}


def memory_evidence(manifest: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [row for row in events if row["event_type"] == "gpu_memory_sample"]
    observed = [row for row in samples if row.get("status") == "observed" and row.get("devices")]
    inventories = [row for row in events if row["event_type"] == "gpu_memory_inventory" and row.get("status") == "observed"]
    phases = dict(Counter(row.get("phase") for row in observed))
    required_phases = {"model_load", "cache_initialization", "warmup", "inference", "cleanup"}
    linked = [row for row in observed if row.get("framework") is not None]
    uuid_checks = [row["framework"].get("gpu_uuid") in {device["gpu_uuid"] for device in row["devices"]}
                   and row["framework"].get("worker_pid") == row.get("engine_pid") for row in linked]
    latest = inventories[-1] if inventories else {}
    storages = latest.get("storages", [])
    unique = len({(row["device"], row["storage_id"]) for row in storages}) == len(storages)
    total_valid = bool(storages) and latest.get("unique_storage_bytes") == sum(row["byte_count"] for row in storages)
    categories: dict[str, int] = defaultdict(int)
    for row in storages:
        categories[f"{row['device']}:{','.join(row.get('categories', []))}"] += row["byte_count"]
    gpu_weights = sum(row["byte_count"] for row in storages if row["device"].startswith("cuda") and "weights" in row.get("categories", []))
    gpu_kv = sum(row["byte_count"] for row in storages if row["device"].startswith("cuda") and "attention_kv" in row.get("categories", []))
    gpu_recurrent = sum(row["byte_count"] for row in storages if row["device"].startswith("cuda") and "recurrent_state" in row.get("categories", []))
    gpu_lora = sum(row["byte_count"] for row in storages if row["device"].startswith("cuda") and "lora" in row.get("categories", []))
    loaded_gpu_bytes = sum(row["byte_count"] for row in storages if row["device"].startswith("cuda"))
    adapter_used = any(row.get("adapter_id", "base") != "base" for row in events if row["event_type"] == "request_model_route")
    hybrid = (manifest.get("architecture") or {}).get("model_type") == "qwen3_5_text"
    timestamps = all(row.get("memory_clock_domain") == "time.monotonic_ns"
                     and row.get("sample_started_monotonic_ns", float("inf")) <= row.get("sample_completed_monotonic_ns", -1) <= row["monotonic_ns"] for row in samples)
    maxima: dict[str, int] = defaultdict(int)
    for row in observed:
        for device in row["devices"]:
            maxima[device["gpu_uuid"]] = max(maxima[device["gpu_uuid"]], device["used_bytes"])
    cleanup = [row for row in observed if row.get("phase") == "cleanup"]
    inference = [row for row in observed if row.get("phase") == "inference"]
    cleanup_releases = False
    if cleanup and inference and gpu_weights:
        final, before = cleanup[-1], inference[-1]
        final_process = {row["gpu_uuid"]: row.get("process_used_bytes") for row in final["devices"] if row.get("process_status") == "observed"}
        before_process = {row["gpu_uuid"]: row.get("process_used_bytes") for row in before["devices"] if row.get("process_status") == "observed"}
        # Context/driver allocations may remain while the recorder's process is
        # alive. Require a measured decrease, rather than claim a zero footprint.
        cleanup_releases = final.get("framework") is None and any(final_process.get(uuid) is not None and value is not None
                            and final_process[uuid] <= value - loaded_gpu_bytes * 0.9 for uuid, value in before_process.items())
    checks = {
        "memory_recorder_finalized": (manifest.get("gpu_memory") or {}).get("status") == "completed",
        "measured_memory_all_phases": required_phases.issubset(phases),
        "memory_host_timestamps_valid": bool(samples) and timestamps,
        "worker_framework_uuid_join": bool(linked) and all(uuid_checks),
        "storage_dedup_and_total_valid": unique and total_valid,
        "weights_and_attention_storage_measured": gpu_weights > 0 and gpu_kv > 0,
        "hybrid_recurrent_storage_measured": not hybrid or gpu_recurrent > 0,
        "routed_lora_storage_measured": not adapter_used or gpu_lora > 0,
        "cleanup_memory_decrease_consistent_with_storage_release": cleanup_releases,
    }
    return {"checks": checks, "sample_count": len(samples), "successful_samples": len(observed), "unavailable_samples": len(samples) - len(observed),
            "phases": phases, "framework_sample_count": len(linked), "inventory_categories_bytes": dict(categories),
            "sampled_device_maximum_bytes": dict(maxima), "final_cleanup_sample": cleanup[-1] if cleanup else None,
            "cleanup_loaded_storage_threshold_bytes": loaded_gpu_bytes * 0.9,
            "semantics": "Device/process/framework views overlap; categories deduplicate backing storage. Device maxima are sampled maxima, not allocator peaks. Cleanup requires framework detachment and a measured process decrease of at least 90% of unique loaded GPU storage; management samples do not identify individual freed allocations."}


def read_receipt(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def finalization_evidence(manifest: dict[str, Any], events: list[dict[str, Any]], final: dict[str, Any] | None) -> dict[str, Any]:
    """Finalization must be positively observed for this run and owner PID."""
    raw_manifest = (final or {}).get("manifest")
    receipt_manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
    process = manifest.get("execution_process") or {}
    samples = [row for row in events if row["event_type"] == "gpu_memory_sample"]
    pids = {row.get("engine_pid") for row in samples}
    started, exited = process.get("started_monotonic_ns"), process.get("exited_monotonic_ns")
    process_valid = process.get("kind") == "spawn_per_run" and process.get("status") == "exited" and process.get("exit_code") == 0
    process_valid &= type(process.get("pid")) is int and process["pid"] > 0 and pids == {process["pid"]}
    process_valid &= process.get("clock_domain") == "controller:time.monotonic_ns" and type(started) is int and type(exited) is int
    process_valid &= bool(samples) and type(started) is int and type(exited) is int and 0 < started <= min(row["sample_started_monotonic_ns"] for row in samples) <= max(row["monotonic_ns"] for row in events) <= exited
    checks = {
        "finalization_receipt_present": final is not None,
        "finalization_receipt_matches_run": isinstance(manifest.get("run_id"), str) and bool(manifest["run_id"])
            and (final or {}).get("run_id") == manifest["run_id"] == receipt_manifest.get("run_id"),
        # Archived trace responses have no current-controller message. The
        # persisted manifest, receipt flags and matching owner exit prove
        # finalization; presentation text remains diagnostic evidence below.
        "finalization_receipt_completed": (final or {}).get("finalized") is True and (final or {}).get("status") == "completed"
            and (final or {}).get("capture_state") == "completed" and (final or {}).get("provisional") is False,
        "finalization_receipt_matches_recording": receipt_manifest.get("completion_status") == "completed"
            and (receipt_manifest.get("event_completeness") or {}).get("status") == "complete"
            and receipt_manifest.get("execution_process") == process
            and (receipt_manifest.get("observability") or {}).get("code_identity") == (manifest.get("observability") or {}).get("code_identity"),
        "execution_owner_exited_successfully": process_valid,
    }
    return {"ok": all(checks.values()), "checks": checks, "execution_process": process,
            "message": (final or {}).get("message"), "receipt_run_id": (final or {}).get("run_id")}


def management_receipt(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse the qualification helper's nvidia-smi receipt, never inferred data."""
    if not value or value.get("status") != "observed" or value.get("clock_domain") != "qualification_host:time.monotonic_ns" \
            or value.get("host_scope") != "machine executing this qualification helper" \
            or type(value.get("observed_monotonic_ns")) is not int or value["observed_monotonic_ns"] <= 0:
        return None
    gpu, apps = value.get("gpus") or {}, value.get("compute_apps") or {}
    if gpu.get("fields") != ["uuid", "memory.used", "memory.free", "memory.total"] or apps.get("fields") != ["pid", "used_memory"] \
            or gpu.get("units") != "memory in MiB" or apps.get("units") != "memory in MiB" \
            or type(gpu.get("returncode")) is not int or type(apps.get("returncode")) is not int \
            or gpu.get("returncode") != 0 or apps.get("returncode") != 0 \
            or not isinstance(gpu.get("rows"), list) or not isinstance(apps.get("rows"), list):
        return None
    try:
        devices, pids = {}, set()
        for line in gpu["rows"]:
            uuid, used, free, total = next(csv.reader([line], skipinitialspace=True))
            uuid = uuid.strip()
            used, free, total = int(used), int(free), int(total)
            if not uuid.startswith(("GPU-", "MIG-")) or uuid in devices or not 0 <= used <= total or not 0 <= free <= total or total <= 0:
                return None
            devices[uuid] = {"used_bytes": used * 1048576, "free_bytes": free * 1048576, "total_bytes": total * 1048576}
        for line in apps["rows"]:
            pid, used = next(csv.reader([line], skipinitialspace=True))
            pid, used = int(pid), int(used)
            if pid <= 0 or used < 0 or pid in pids:
                return None
            pids.add(pid)
        return {"observed_monotonic_ns": value["observed_monotonic_ns"], "devices": devices, "compute_pids": sorted(pids)} if devices else None
    except (ValueError, TypeError, csv.Error):
        return None


def cleanup_evidence(manifest: dict[str, Any], events: list[dict[str, Any]], memory: dict[str, Any], finalization: dict[str, Any],
                     before_receipt: dict[str, Any] | None, after_receipt: dict[str, Any] | None, source_matches: bool) -> dict[str, Any]:
    """Distinguish pre-exit counter release from measured post-owner release."""
    before, after = management_receipt(before_receipt), management_receipt(after_receipt)
    samples = [row for row in events if row["event_type"] == "gpu_memory_sample" and row.get("status") == "observed"]
    run_devices = {device["gpu_uuid"]: device for row in samples for device in row.get("devices", [])}
    process = manifest.get("execution_process") or {}
    target = manifest.get("execution_target") or {}
    # The receipts are executed on the loopback runner's qualification host.
    # Unique physical UUIDs, native total size and ordered host clock boundaries
    # provide the cross-receipt join; a remote execution target cannot use it.
    local_host = target.get("kind") == "local" and target.get("engine_location") == "this-machine"
    local_host &= manifest.get("clock_domain") == "time.monotonic_ns" and (manifest.get("controller_origin") or {}).get("clock_domain") == "time.monotonic_ns"
    boundary_valid = before is not None and after is not None and finalization["ok"] and local_host and source_matches and bool(run_devices)
    if boundary_valid:
        boundary_valid = before["observed_monotonic_ns"] < process["started_monotonic_ns"] <= process["exited_monotonic_ns"] < after["observed_monotonic_ns"]
    identity_valid = before is not None and after is not None and bool(run_devices)
    if identity_valid:
        identity_valid = all(uuid in before["devices"] and uuid in after["devices"]
            and before["devices"][uuid]["total_bytes"] == after["devices"][uuid]["total_bytes"]
            and abs(before["devices"][uuid]["total_bytes"] - device.get("total_bytes", -1048576)) < 1048576 for uuid, device in run_devices.items())
    pid_absent = before is not None and after is not None and type(process.get("pid")) is int \
        and process["pid"] not in before["compute_pids"] and process["pid"] not in after["compute_pids"]
    baseline_tolerance = 64 * 1048576
    near_baseline = identity_valid and all(after["devices"][uuid]["used_bytes"] <= before["devices"][uuid]["used_bytes"] + baseline_tolerance for uuid in run_devices)
    post_checks = {"management_receipts_observed": before is not None and after is not None,
                   "same_local_host_clock_and_owner_boundaries": boundary_valid, "same_gpu_uuid_and_total": identity_valid,
                   "execution_owner_pid_absent_after_exit": pid_absent, "device_usage_returns_to_before_baseline": near_baseline}
    pre_exit = memory["checks"]["cleanup_memory_decrease_consistent_with_storage_release"]
    post_exit = all(post_checks.values())
    last_inference = next((row for row in reversed(samples) if row.get("phase") == "inference"), None)
    final_cleanup = memory.get("final_cleanup_sample")
    return {"ok": pre_exit or post_exit,
            "basis": "in_recording_process_counter_decrease" if pre_exit else "external_post_owner_exit_receipts" if post_exit else "unverified",
            "pre_exit_counter_decrease_verified": pre_exit, "post_owner_exit_verified": post_exit, "post_owner_checks": post_checks,
            "before": before, "after": after, "baseline_tolerance_bytes": baseline_tolerance,
            "last_inference_process_bytes": {row["gpu_uuid"]: row.get("process_used_bytes") for row in (last_inference or {}).get("devices", [])},
            "final_recorded_cleanup_process_bytes": {row["gpu_uuid"]: row.get("process_used_bytes") for row in (final_cleanup or {}).get("devices", [])},
            "semantics": "The pre-exit check remains the observed 90% counter decrease test. A separate post-exit proof requires finalized successful owner exit, matching PID/UUID/total, local host monotonic boundaries, source parity and device usage within 64 MiB of the pre-run baseline. Retained pre-exit bytes remain recorded; neither test identifies particular freed tensors or explains native references."}


def profile_evidence(directory: Path, manifest: dict[str, Any], scheduled: list[dict[str, Any]], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = manifest.get("profiling") or {}
    artifact = profile.get("normalized_artifact")
    valid = artifact_ok(directory, artifact)
    normalized = expand_profile_wire(json.loads((directory / artifact["path"]).read_text())) if valid else {}
    activities = normalized.get("activities", [])
    gpu = [row for row in activities if row["kind"].startswith("cuda_")]
    correlated = [row for row in gpu if row.get("scope", {}).get("correlation") == "runtime_flow"]
    layers = sorted({row["scope"]["layer_index"] for row in correlated if row["scope"].get("layer_index") is not None})
    components = dict(Counter(row["scope"].get("component") for row in correlated if row["scope"].get("component") is not None))
    architecture = manifest.get("architecture") or {}
    hybrid = architecture.get("model_type") == "qwen3_5_text"
    native_steps = normalized.get("steps", [])
    by_step = {row["step_id"]: row for row in scheduled}
    rows_valid = bool(native_steps)
    for step in native_steps:
        schedule = by_step.get(step["step_id"])
        if schedule is None or step.get("mapping_status") != "observed" or step.get("padding_status") != "observed":
            rows_valid = False
            continue
        ranges = {row["request_id"]: range(row["range_start"], row["range_start"] + row["range_length"]) for row in schedule["requests"]}
        packed = step.get("packed_rows", [])
        if step.get("unpadded_tokens") != sum(len(value) for value in ranges.values()) or step.get("padded_tokens") != len(packed) or [row.get("row") for row in packed] != list(range(len(packed))):
            rows_valid = False
        identities = [(row.get("request_id"), row.get("token_position")) for row in packed if not row.get("is_padding")]
        if len(set(identities)) != len(identities) or len(identities) != sum(len(value) for value in ranges.values()):
            rows_valid = False
        for row in packed:
            if not row.get("is_padding") and (row.get("request_id") not in ranges or row.get("token_position") not in ranges[row["request_id"]]):
                rows_valid = False
    recorded_mode = (manifest.get("run_configuration") or {}).get("profiling", {}).get("mode", "none")
    expected_profile = (expected.get("profiling_configuration") or expected.get("profiling") or {}) if expected is not None else {}
    expected_mode = expected_profile.get("mode", "off") if expected is not None else recorded_mode
    recorded_mode = "off" if recorded_mode == "none" else recorded_mode
    expected_mode = "off" if expected_mode == "none" else expected_mode
    requested = expected_mode == "eager_diagnostic"
    normalization = normalized.get("normalization") or {}
    retention = normalization.get("attachment_retention") or {}
    gpu_complete = normalization.get("status") == "completed" or retention.get("status") == "cpu_detail_bounded" and retention.get("gpu_activities_complete") is True and retention.get("omitted_gpu_activity_count") == 0
    checks = {
        "requested_profile_matches_recording": recorded_mode == expected_mode,
        "native_profile_finalized": not requested or profile.get("status") in {"completed", "partial"} and profile.get("attachment_status") == "available",
        "native_artifact_integrity": not requested or valid and bool(profile.get("raw_artifacts")) and all(artifact_ok(directory, row) for row in profile["raw_artifacts"]),
        "native_profile_real_cuda": not requested or bool(correlated) and normalized.get("run_id") == manifest.get("run_id"),
        "native_all_model_layers_observed": not requested or layers == list(range(architecture.get("num_layers", 0))),
        "native_component_coverage": not requested or components.get("attention", 0) > 0 and components.get("ffn", 0) > 0 and (not hybrid or components.get("recurrent", 0) > 0),
        "native_rows_match_host_schedule": not requested or rows_valid,
        "native_all_host_iterations_captured": not requested or {row["step_id"] for row in native_steps} == set(by_step),
        "native_cuda_covers_all_host_iterations": not requested or {row["scope"].get("step_id") for row in correlated if row["scope"].get("step_id") is not None} == set(by_step),
        "native_cuda_activity_retention_complete": not requested or gpu_complete,
    }
    return {"checks": checks, "requested": requested, "status": profile.get("status"), "completeness": profile.get("completeness"),
            "reason": profile.get("reason"), "cuda_activity_count": len(gpu), "correlated_activity_count": len(correlated),
            "layers": layers, "components": components, "captured_steps": len(native_steps), "host_iterations": len(scheduled),
            "coverage": "Hybrid recurrent compute scopes are captured separately from attention cache writes; recurrent state tensor values and expert routes are not inferred." if hybrid else "Dense attention/FFN/layer compute and qualified cache writes; no per-token or per-head kernel durations are inferred."}


def output_evidence(manifest: dict[str, Any], events: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    """A generation cap is not proof that natural-EOS output meets a demo target."""
    requests = expected.get("requests", [])
    accepted: Counter = Counter()
    completed = {row["request_id"]: row for row in events if row["event_type"] == "completion"}
    for event in events:
        if event["event_type"] == "tokens_accepted":
            accepted[event["request_id"]] += event["token_count"]
    output_ids = manifest.get("engine_output_token_ids") or {}
    rows = [{"request_id": row["request_id"], "configured_max_output_tokens": row["max_output_tokens"],
             "accepted_tokens": accepted[row["request_id"]], "recorded_token_ids": len(output_ids.get(row["request_id"], [])),
             "completion_reason": completed.get(row["request_id"], {}).get("reason")} for row in requests]
    checks = {"all_requests_returned_within_configured_output_limits": bool(rows) and all(
        0 < row["accepted_tokens"] <= row["configured_max_output_tokens"]
        and row["recorded_token_ids"] == row["accepted_tokens"]
        and row["completion_reason"] in {"length", "stop", "eos"} for row in rows)}
    target = expected.get("demo_qualification")
    if target is not None:
        minimum, maximum = target.get("minimum_output_tokens"), target.get("maximum_output_tokens")
        valid_range = type(minimum) is int and type(maximum) is int and 1 <= minimum <= maximum
        checks["demo_output_caps_match_qualification_range"] = valid_range and bool(rows) and all(
            minimum <= row["configured_max_output_tokens"] <= maximum for row in rows)
        checks["demo_actual_outputs_within_qualification_range"] = valid_range and bool(rows) and all(
            minimum <= row["accepted_tokens"] <= maximum for row in rows)
    return {"checks": checks, "qualification_target": target, "requests": rows,
            "semantics": "Configured maxima bound generation. Natural EOS remains allowed; short output fails qualification rather than being padded or forced."}


def prompt_evidence(content: tuple[dict[str, Any], ...], expected: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in content if row.get("kind") == "prompt_text"]
    recorded = {row.get("request_id"): row.get("text") for row in rows}
    authored = {row["request_id"]: row["prompt"] for row in expected.get("requests", [])}
    return {"checks": {"demo_recorded_prompts_match_authored_inputs": bool(authored) and len(rows) == len(authored) and recorded == authored},
            "recorded_request_ids": list(recorded), "semantics": "Exact runner input text is retained before tokenizer submission."}


def assess(directory: Path, expected: dict[str, Any], *, final_state: Path | None = None, source_root: Path | None = None,
           gpu_memory_before: Path | None = None, gpu_memory_after: Path | None = None) -> dict[str, Any]:
    data = load_recording(directory)
    manifest, events = data.manifest, list(data.events)
    report = evaluate(data)
    scheduled = [row for row in events if row["event_type"] == "step_scheduled"]
    admitted = {row["request_id"]: row["sequence"] for row in events if row["event_type"] == "engine_admitted"}
    completed = {row["request_id"]: row for row in events if row["event_type"] == "completion"}
    expected_requests = expected.get("requests", [])
    ids = [row["request_id"] for row in expected_requests]
    groups = (manifest.get("workload_plan") or {}).get("batches", [])
    expected_groups = (expected.get("workload_plan") or {}).get("batches", [])
    memberships = [[row["id"] for row in group.get("samples", [])] for group in groups]
    expected_memberships = [[row["id"] for row in group.get("samples", [])] for group in expected_groups]
    count = Counter()
    for event in events:
        if event["event_type"] == "tokens_accepted":
            count[event["request_id"]] += event["token_count"]
    routes = {row["request_id"]: row.get("adapter_id", "base") for row in events if row["event_type"] == "request_model_route"}
    lookups = [row for row in events if row["event_type"] == "prefix_lookup"]
    g1, g2, g3 = (memberships + [[], [], []])[:3]
    prime_done = max((completed.get(id, {}).get("sequence", float("inf")) for id in g1), default=float("inf"))
    reuse_admitted = min((admitted.get(id, -1) for id in g2), default=-1)
    overlap = [id for id in g2 if any(admitted.get(third, float("inf")) < completed.get(id, {}).get("sequence", -1) for third in g3)]
    model = manifest.get("model") or {}
    revision = model.get("revision")
    expected_model = _parse_model(expected)["source"]
    final_path = final_state or directory.parent / "final-state.json"
    final = read_receipt(final_path)
    finalization = finalization_evidence(manifest, events, final)
    captured_source = (manifest.get("observability") or {}).get("code_identity") or {}
    current_source = source_identity(source_root or ROOT)
    checks = {
        "production_recording_integrity": report.ok,
        "complete_real_gpu_capture": manifest.get("execution_kind") == "gpu" and (manifest.get("provenance") or {}).get("real_gpu") is True
            and manifest.get("completion_status") == "completed" and (manifest.get("event_completeness") or {}).get("status") == "complete",
        "six_requests_three_matching_arrival_groups": len(ids) == 6 and set(admitted) == set(completed) == set(ids) and memberships == expected_memberships and list(map(len, memberships)) == [1, 2, 3],
        "prime_finishes_before_second_group_admission": prime_done < reuse_admitted,
        "third_group_admitted_while_second_group_active": bool(overlap),
        "continuous_batching_multi_request_iterations": any(len(row["requests"]) > 1 for row in scheduled),
        "prefill_decode_share_iteration": any(any(item["prompt_tokens"] > 0 for item in row["requests"]) and any(item["generation_tokens"] > 0 for item in row["requests"]) for row in scheduled),
        "expected_model_and_pinned_revision": model.get("id") == expected_model["locator"] and revision == expected_model["revision"] and isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "clean_finalization_receipt": finalization["ok"],
        "capture_source_matches_assessed_source": captured_source.get("source_sha256") == current_source["source_sha256"]
            and captured_source.get("source_file_count") == current_source["source_file_count"],
    }
    memory, profile = memory_evidence(manifest, events), profile_evidence(directory, manifest, scheduled, expected)
    precision = precision_evidence(manifest, expected)
    prefix = prefix_evidence(manifest, events, expected)
    adapter = adapter_evidence(manifest, events, expected)
    output = output_evidence(manifest, events, expected)
    prompts = prompt_evidence(data.content, expected) if expected.get("demo_qualification") is not None else None
    checks.update(output["checks"])
    if prompts is not None:
        checks.update(prompts["checks"])
    # This counter threshold remains visible as diagnostic evidence. Actual
    # owner exit can release allocations that were retained before exit.
    checks.update({key: value for key, value in memory["checks"].items() if key != "cleanup_memory_decrease_consistent_with_storage_release"})
    cleanup = cleanup_evidence(manifest, events, memory, finalization,
        read_receipt(gpu_memory_before or directory.parent / "gpu-memory-before.json"),
        read_receipt(gpu_memory_after or directory.parent / "gpu-memory-after.json"), checks["capture_source_matches_assessed_source"])
    checks["cleanup_allocation_lifecycle_verified"] = cleanup["ok"]
    checks.update(profile["checks"])
    checks.update(precision["checks"])
    checks.update(prefix["checks"])
    checks.update(adapter["checks"])
    texts: dict[str, str] = defaultdict(str)
    for row in data.content:
        if row.get("kind") == "text":
            texts[row["request_id"]] += row.get("text_delta", "")
    return {"ok": all(bool(value) for value in checks.values()), "run_id": manifest["run_id"], "checks": {key: bool(value) for key, value in checks.items()},
            "production_integrity": report.text(), "model": model, "architecture": manifest.get("architecture"), "engine": manifest.get("engine"),
            "memory": memory, "cleanup": cleanup, "finalization": finalization, "profile": profile, "precision": precision, "prefix": prefix, "adapter": adapter,
            "output_length": output, "prompt_inputs": prompts,
            "source_identity": {"captured": captured_source, "assessed": current_source}, "arrival_groups": memberships, "admitted": admitted,
            "completed": {id: {"sequence": row["sequence"], "reason": row.get("reason")} for id, row in completed.items()},
            "third_arrival_overlaps_second_requests": overlap, "prefix_lookups": lookups, "request_routes": routes,
            "outputs": [{"request_id": id, "accepted_tokens": count[id], "text": texts[id]} for id in ids],
            "finalization_message": final.get("message") if final else None,
            "scope": "This assesses retained GPU recording evidence and actual outputs, not output quality, benchmark representativeness, replay UI behavior, or exhaustive tensor/expert capture."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--final-state", type=Path)
    parser.add_argument("--source-root", type=Path, help="Source checkout whose runner/instrumentation must match the capture; defaults to this checkout.")
    parser.add_argument("--gpu-memory-before", type=Path, help="External pre-run management receipt; defaults to the recording's sibling gpu-memory-before.json.")
    parser.add_argument("--gpu-memory-after", type=Path, help="External post-owner-exit management receipt; defaults to the recording's sibling gpu-memory-after.json.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(args.recording, json.loads(args.payload.read_text()), final_state=args.final_state, source_root=args.source_root,
                    gpu_memory_before=args.gpu_memory_before, gpu_memory_after=args.gpu_memory_after)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
