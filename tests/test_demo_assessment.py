"""Assessment rejects incomplete or mismatched evidence instead of blessing HTTP success."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from runner.serve import create_app


spec = importlib.util.spec_from_file_location("demo_assess", Path(__file__).resolve().parents[1] / "docs/qualification/demo-expansion/assess.py")
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


def artifact(directory, name, value):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value).encode()
    path.write_bytes(payload)
    return {"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_artifact_integrity_rejects_changed_or_outside_files(tmp_path):
    metadata = artifact(tmp_path, "profile.json", {"native": True})
    assert assessment.artifact_ok(tmp_path, metadata)
    (tmp_path / "profile.json").write_text("modified")
    assert not assessment.artifact_ok(tmp_path, metadata)
    assert not assessment.artifact_ok(tmp_path, {**metadata, "path": "../profile.json"})


def test_precision_requires_native_fp8_encoding_and_exact_requested_weight_format():
    manifest = {"architecture": {"model_type": "qwen3", "num_layers": 1}, "model": {"weight_storage": {"status": "observed", "format": "fp8", "compute_dtype": "bfloat16", "tensor_dtypes": ["float8_e4m3fn", "bfloat16"]}},
                "engine_settings": {"kv_cache_dtype": "fp8"},
                "kv_cache": {"status": "observed", "layers": [{"layer_index": 0, "dtype": "uint8", "quantization": {"cache_dtype": "fp8"}, "layout_status": "qualified", "numerical_encoding": "fp8_e4m3"}]}}
    expected = {"quantization": "fp8", "dtype": "bfloat16", "kv_cache_dtype": "fp8"}
    assert all(assessment.precision_evidence(manifest, expected)["checks"].values())
    del manifest["kv_cache"]["layers"][0]["numerical_encoding"]
    assert not assessment.precision_evidence(manifest, expected)["checks"]["actual_kv_precision_observed"]
    manifest["model"]["weight_storage"]["format"] = "bf16"
    assert not assessment.precision_evidence(manifest, expected)["checks"]["observed_requested_weight_format"]


def test_quantized_weights_do_not_hide_wrong_compute_dtype():
    manifest = {"model": {"weight_storage": {"status": "observed", "format": "fp8", "compute_dtype": "float16", "tensor_dtypes": ["float8_e4m3fn", "float16"]}},
                "engine_settings": {"kv_cache_dtype": "auto"}, "kv_cache": {"status": "observed", "layers": [{"dtype": "float16", "quantization": {"cache_dtype": "auto"}}]}}
    result = assessment.precision_evidence(manifest, {"quantization": "fp8", "dtype": "bfloat16", "kv_cache_dtype": "auto"})
    assert result["checks"]["observed_requested_weight_format"]
    assert not result["checks"]["observed_requested_compute_dtype"]


def test_nested_model_configuration_precision_overrides_legacy_defaults():
    manifest = {"architecture": {"model_type": "qwen3", "num_layers": 1}, "model": {"weight_storage": {"status": "observed", "format": "bf16", "compute_dtype": "bfloat16", "tensor_dtypes": ["bfloat16"]}},
                "engine_settings": {"kv_cache_dtype": "auto"}, "kv_cache": {"status": "observed", "layers": [{"layer_index": 0, "dtype": "bfloat16", "quantization": {"cache_dtype": "auto"}}]}}
    expected = {"dtype": "bfloat16", "quantization": "none", "kv_cache_dtype": "auto", "model_configuration": {"dtype": "float16"}}
    result = assessment.precision_evidence(manifest, expected)
    assert not result["checks"]["observed_requested_compute_dtype"] and not result["checks"]["observed_requested_weight_format"]
    expected["model_configuration"] = {"quantization": {"kind": "fp8"}}
    assert not assessment.precision_evidence(manifest, expected)["checks"]["observed_requested_weight_format"]


def test_kv_precision_requires_unique_complete_attention_layers_for_dense_and_hybrid():
    expected = {"dtype": "bfloat16", "quantization": "none", "kv_cache_dtype": "auto"}
    for architecture, indices in [({"model_type": "qwen3", "num_layers": 36}, list(range(36))),
                                   ({"model_type": "qwen3_5_text", "num_layers": 64, "attention_layer_indices": list(range(3, 64, 4))}, list(range(3, 64, 4)))]:
        manifest = {"architecture": architecture, "engine_settings": {"kv_cache_dtype": "auto"},
                    "model": {"weight_storage": {"status": "observed", "format": "bf16", "compute_dtype": "bfloat16", "tensor_dtypes": ["bfloat16"]}},
                    "kv_cache": {"status": "observed", "layers": [{"layer_index": index, "dtype": "bfloat16", "quantization": {"cache_dtype": "auto"}} for index in indices]}}
        assert all(assessment.precision_evidence(manifest, expected)["checks"].values())
        manifest["kv_cache"]["layers"].pop()
        assert not assessment.precision_evidence(manifest, expected)["checks"]["all_attention_layers_kv_precision_observed"]
        manifest["kv_cache"]["layers"].append(manifest["kv_cache"]["layers"][0])
        assert not assessment.precision_evidence(manifest, expected)["checks"]["all_attention_layers_kv_precision_observed"]


def test_adapter_qualification_requires_native_route_source_and_model_hash():
    source = {"kind": "execution_host", "locator": "/trained/adapter", "revision": None}
    adapter = {"id": "a", "source": source, "lora_int_id": 1, "configuration_status": "observed", "rank": 4, "target_modules": ["q_proj"],
               "artifact_sha256": {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}}
    manifest = {"model": {"lora_adapters": [adapter]}}
    events = [{"event_type": "request_model_route", "request_id": "r", "adapter_id": "a", "route_kind": "lora_adapter", "lora_int_id": 1}]
    expected = {"requests": [{"request_id": "r", "adapter_id": "a"}], "model_configuration": {"lora_adapters": [{"id": "a", "source": source}]}}
    assert all(assessment.adapter_evidence(manifest, events, expected)["checks"].values())
    wrong = deepcopy(events)
    wrong[0].update(route_kind="base_model", lora_int_id=None)
    assert not assessment.adapter_evidence(manifest, wrong, expected)["checks"]["actual_request_adapter_routes"]
    wrong_manifest = deepcopy(manifest)
    wrong_manifest["model"]["lora_adapters"][0]["source"]["locator"] = "/wrong/adapter"
    assert not assessment.adapter_evidence(wrong_manifest, events, expected)["checks"]["configured_adapter_source_identity"]
    del adapter["artifact_sha256"]["adapter_model.safetensors"]
    assert not assessment.adapter_evidence(manifest, events, expected)["checks"]["adapter_configuration_and_model_hashes"]


def test_adapter_source_identity_uses_production_canonicalization_and_excludes_auth_flags():
    for configured, observed in [({"kind": "execution_host", "path": "/trained/adapter"}, {"kind": "execution_host", "locator": "/trained/adapter", "revision": None}),
                                  ({"kind": "huggingface", "repository": "org/adapter", "revision": "a" * 40}, {"kind": "huggingface", "locator": "org/adapter", "revision": "a" * 40, "authenticated": True})]:
        expected = {"requests": [{"request_id": "r", "adapter_id": "a"}], "model_configuration": {"lora_adapters": [{"id": "a", "source": configured}]}}
        manifest = {"model": {"lora_adapters": [{"id": "a", "source": observed}]}}
        result = assessment.adapter_evidence(manifest, [], expected)
        assert result["checks"]["configured_adapter_source_identity"]
        manifest["model"]["lora_adapters"][0]["source"]["revision"] = "b" * 40
        assert not assessment.adapter_evidence(manifest, [], expected)["checks"]["configured_adapter_source_identity"]


def test_prefix_namespace_lineage_keeps_generation_and_adapter_routes_separate():
    manifest = {"cache_geometry": {"groups": [{"cache_group": 0, "pool": "pool-0"}]}}
    events = [{"event_type": "request_model_route", "request_id": "prime", "adapter_id": "base"},
              {"event_type": "request_model_route", "request_id": "reader", "adapter_id": "base"},
              {"event_type": "request_model_route", "request_id": "adapter", "adapter_id": "a"},
              {"event_type": "prefix_lookup", "request_id": "prime", "matched_token_count": 0, "block_ids": [[]]},
              {"event_type": "block_change", "pool": "pool-0", "block_id": 1, "allocation_generation": 1},
              {"event_type": "kv_block_binding", "request_id": "prime", "pool": "pool-0", "block_id": 1, "allocation_generation": 1},
              {"event_type": "prefix_lookup", "request_id": "reader", "matched_token_count": 16, "block_ids": [[1]]},
              {"event_type": "prefix_lookup", "request_id": "adapter", "matched_token_count": 0, "block_ids": [[]]}]
    expected = {"requests": [{"request_id": "prime"}, {"request_id": "reader"}, {"request_id": "adapter"}]}
    result = assessment.prefix_evidence(manifest, events, expected)
    assert all(result["checks"].values())
    assert result["lookups"][1]["reused_storage_origins"][0]["host_result_write_already_observed"] is False
    assert not assessment.prefix_evidence(manifest, events[:-1], expected)["checks"]["all_request_prefix_lookups_observed"]
    wrong = deepcopy(events)
    wrong[-1].update(matched_token_count=16, block_ids=[[1]])
    assert not assessment.prefix_evidence(manifest, wrong, expected)["checks"]["cross_adapter_prefix_storage_isolated"]
    recycled = deepcopy(events)
    recycled.insert(-2, {"event_type": "block_change", "pool": "pool-0", "block_id": 1, "allocation_generation": 2})
    assert not assessment.prefix_evidence(manifest, recycled, expected)["checks"]["observed_prefix_reuse_same_adapter_route"]


def test_hybrid_prefix_native_tables_prove_structural_owners_without_recurrent_token_writes():
    groups = [{"cache_group": i, "pool": "pool-0", "cache_type": "MambaSpec" if i < 3 else "FullAttentionSpec",
               "token_slot_mapping": "unavailable" if i < 3 else "qualified_full_attention"} for i in range(4)]
    manifest = {"cache_geometry": {"groups": groups}}
    expected = {"requests": [{"request_id": "prime"}, {"request_id": "reader"}]}
    events = [{"event_type": "request_model_route", "request_id": id, "adapter_id": "base"} for id in ("prime", "reader")]
    events.append({"event_type": "prefix_lookup", "request_id": "prime", "matched_token_count": 0, "block_ids": [[], [], [], []]})
    events.extend({"event_type": "block_change", "pool": "pool-0", "block_id": block, "allocation_generation": 1} for block in (1, 2, 3, 4))
    events.append({"event_type": "step_scheduled", "block_table_changes": [{"request_id": "prime", "change": "replace", "block_ids": [[4], [3], [2], [1]]}]})
    events.append({"event_type": "kv_block_written", "request_id": "prime", "pool": "pool-0", "block_id": 1, "allocation_generation": 1, "cache_group": 3})
    events.append({"event_type": "prefix_lookup", "request_id": "reader", "matched_token_count": 784, "block_ids": [[4], [3], [2], [1]]})
    result = assessment.prefix_evidence(manifest, events, expected)
    assert all(result["checks"].values())
    references = result["lookups"][-1]["reused_storage_origins"]
    assert all(row["prior_block_table_owners"] == ["prime"] for row in references)
    assert [row["host_result_write_already_observed"] for row in references] == [False, False, False, True]
    wrong = deepcopy(events)
    wrong[0]["adapter_id"] = "a"
    assert not assessment.prefix_evidence(manifest, wrong, expected)["checks"]["cross_adapter_prefix_storage_isolated"]
    missing = deepcopy(events)
    missing[-3]["block_table_changes"][0]["block_ids"][0] = []
    assert not assessment.prefix_evidence(manifest, missing, expected)["checks"]["observed_prefix_reuse_same_adapter_route"]
    aliased = deepcopy(missing)
    aliased[-1]["block_ids"][0] = [1]
    assert not assessment.prefix_evidence(manifest, aliased, expected)["checks"]["observed_prefix_reuse_same_adapter_route"]


def test_prefix_structural_tables_do_not_prove_unknown_generation_or_reserved_null():
    manifest = {"cache_geometry": {"groups": [{"cache_group": 0, "pool": "pool-0"}]}}
    expected = {"requests": [{"request_id": "prime"}, {"request_id": "reader"}]}
    events = [{"event_type": "request_model_route", "request_id": id, "adapter_id": "base"} for id in ("prime", "reader")]
    events.extend([{"event_type": "prefix_lookup", "request_id": "prime", "matched_token_count": 0, "block_ids": [[]]},
                   {"event_type": "step_scheduled", "block_table_changes": [{"request_id": "prime", "change": "replace", "block_ids": [[0]]}]},
                   {"event_type": "prefix_lookup", "request_id": "reader", "matched_token_count": 16, "block_ids": [[0]]}])
    assert not assessment.prefix_evidence(manifest, events, expected)["checks"]["observed_prefix_reuse_same_adapter_route"]
    events.insert(2, {"event_type": "cache_inventory", "pools": [{"pool": "pool-0", "blocks": [{"block_id": 0, "allocation_generation": 1, "is_null": True}]}]})
    result = assessment.prefix_evidence(manifest, events, expected)
    assert not result["checks"]["observed_prefix_reuse_same_adapter_route"]
    assert result["lookups"][-1]["reused_storage_origins"][0]["reserved_null"] is True


def test_hybrid_native_coverage_uses_actual_layer_count_and_all_scheduled_rows(tmp_path):
    scheduled = [{"step_id": "step-1", "requests": [{"request_id": "r", "range_start": 7, "range_length": 2}]}]
    activities = [{"kind": "cuda_kernel", "scope": {"correlation": "runtime_flow", "layer_index": i, "step_id": "step-1", "component": "recurrent" if i % 4 != 3 else "attention"}} for i in range(64)]
    activities.append({"kind": "cuda_kernel", "scope": {"correlation": "runtime_flow", "layer_index": 0, "step_id": "step-1", "component": "ffn"}})
    steps = [{"step_id": "step-1", "mapping_status": "observed", "padding_status": "observed", "unpadded_tokens": 2, "padded_tokens": 3,
              "packed_rows": [{"row": 0, "request_id": "r", "token_position": 7, "is_padding": False}, {"row": 1, "request_id": "r", "token_position": 8, "is_padding": False}, {"row": 2, "request_id": None, "token_position": None, "is_padding": True}]}]
    normalized = {"run_id": "gpu", "activities": activities, "steps": steps,
                  "normalization": {"status": "partial", "attachment_retention": {"status": "cpu_detail_bounded", "gpu_activities_complete": True, "omitted_gpu_activity_count": 0}}}
    manifest = {"run_id": "gpu", "architecture": {"model_type": "qwen3_5_text", "num_layers": 64}, "run_configuration": {"profiling": {"mode": "eager_diagnostic"}},
                "profiling": {"status": "partial", "completeness": "bounded", "attachment_status": "available", "raw_artifacts": [artifact(tmp_path, "raw.json", {})], "normalized_artifact": artifact(tmp_path, "normalized.json", normalized)}}
    result = assessment.profile_evidence(tmp_path, manifest, scheduled)
    assert all(result["checks"].values()) and len(result["layers"]) == 64
    normalized["steps"][0]["packed_rows"][1]["token_position"] = 7
    manifest["profiling"]["normalized_artifact"] = artifact(tmp_path, "normalized.json", normalized)
    assert not assessment.profile_evidence(tmp_path, manifest, scheduled)["checks"]["native_rows_match_host_schedule"]
    scheduled.append({"step_id": "step-2", "requests": []})
    result = assessment.profile_evidence(tmp_path, manifest, scheduled)
    assert not result["checks"]["native_all_host_iterations_captured"]
    assert not result["checks"]["native_cuda_covers_all_host_iterations"]
    normalized["normalization"]["attachment_retention"]["omitted_gpu_activity_count"] = 1
    manifest["profiling"]["normalized_artifact"] = artifact(tmp_path, "normalized.json", normalized)
    assert not assessment.profile_evidence(tmp_path, manifest, scheduled)["checks"]["native_cuda_activity_retention_complete"]


def test_expected_diagnostic_profile_cannot_be_dropped_from_manifest(tmp_path):
    result = assessment.profile_evidence(tmp_path, {}, [], {"profiling": {"mode": "eager_diagnostic"}})
    assert result["requested"]
    assert not result["checks"]["requested_profile_matches_recording"]
    assert not result["checks"]["native_profile_finalized"]


def test_completed_host_run_does_not_hide_recording_observers_or_unreleased_memory():
    weights, cache = 1_000_000_000, 128_000_000
    storages = [{"device": "cuda:0", "storage_id": "w", "byte_count": weights, "categories": ["weights"]},
                {"device": "cuda:0", "storage_id": "kv", "byte_count": cache, "categories": ["attention_kv"]}]
    inventory = {"event_type": "gpu_memory_inventory", "status": "observed", "storages": storages, "unique_storage_bytes": weights + cache}
    def sample(phase, process, framework=True):
        return {"event_type": "gpu_memory_sample", "status": "observed", "phase": phase, "engine_pid": 42, "monotonic_ns": 3,
                "sample_started_monotonic_ns": 1, "sample_completed_monotonic_ns": 2, "memory_clock_domain": "time.monotonic_ns",
                "framework": {"gpu_uuid": "GPU-valid", "worker_pid": 42} if framework else None,
                "devices": [{"gpu_uuid": "GPU-valid", "used_bytes": process + 100, "process_status": "observed", "process_used_bytes": process}]}
    events = [sample(phase, weights + cache) for phase in ("model_load", "cache_initialization", "warmup", "inference", "cleanup")]
    events.append(inventory)
    manifest = {"completion_status": "completed", "gpu_memory": {"status": "recording"}}
    result = assessment.memory_evidence(manifest, events)
    assert not result["checks"]["memory_recorder_finalized"] and not result["checks"]["cleanup_memory_decrease_consistent_with_storage_release"]
    manifest["gpu_memory"]["status"] = "completed"
    events[-2] = sample("cleanup", 100_000_000, framework=False)
    assert all(assessment.memory_evidence(manifest, events)["checks"].values())
    events[-2] = sample("cleanup", 700_000_000, framework=False)
    assert not assessment.memory_evidence(manifest, events)["checks"]["cleanup_memory_decrease_consistent_with_storage_release"]
    events[0]["framework"]["gpu_uuid"] = "GPU-other"
    assert not assessment.memory_evidence(manifest, events)["checks"]["worker_framework_uuid_join"]


def test_hybrid_requires_measured_recurrent_storage_separate_from_attention():
    manifest = {"architecture": {"model_type": "qwen3_5_text"}, "gpu_memory": {"status": "completed"}}
    events = [{"event_type": "gpu_memory_inventory", "status": "observed", "unique_storage_bytes": 1,
               "storages": [{"device": "cuda:0", "storage_id": "kv", "byte_count": 1, "categories": ["attention_kv"]}]}]
    assert not assessment.memory_evidence(manifest, events)["checks"]["hybrid_recurrent_storage_measured"]
    events[0]["storages"][0]["categories"].append("recurrent_state")
    assert assessment.memory_evidence(manifest, events)["checks"]["hybrid_recurrent_storage_measured"]


def cleanup_receipts():
    process = {"kind": "spawn_per_run", "status": "exited", "exit_code": 0, "pid": 42,
               "clock_domain": "controller:time.monotonic_ns", "started_monotonic_ns": 20, "exited_monotonic_ns": 100}
    manifest = {"run_id": "real", "completion_status": "completed", "event_completeness": {"status": "complete"},
                "execution_process": process, "gpu_memory": {"status": "completed"}, "clock_domain": "time.monotonic_ns",
                "controller_origin": {"clock_domain": "time.monotonic_ns"}, "execution_target": {"kind": "local", "engine_location": "this-machine"},
                "observability": {"code_identity": {"source_sha256": "verified"}}}
    events = [{"event_type": "gpu_memory_sample", "status": "observed", "phase": phase, "engine_pid": 42,
               "monotonic_ns": 50 + index, "sample_started_monotonic_ns": 40 + index, "sample_completed_monotonic_ns": 45 + index,
               "memory_clock_domain": "time.monotonic_ns", "framework": {"gpu_uuid": "GPU-valid", "worker_pid": 42} if phase != "cleanup" else None,
               "devices": [{"gpu_uuid": "GPU-valid", "total_bytes": 100 * 1048576, "used_bytes": 80 * 1048576,
                            "process_status": "observed", "process_used_bytes": 70 * 1048576 if phase != "cleanup" else 65 * 1048576}]}
              for index, phase in enumerate(("model_load", "cache_initialization", "warmup", "inference", "cleanup"))]
    events.append({"event_type": "gpu_memory_inventory", "status": "observed", "monotonic_ns": 55, "storages": [
        {"device": "cuda:0", "storage_id": "w", "byte_count": 60 * 1048576, "categories": ["weights"]},
        {"device": "cuda:0", "storage_id": "kv", "byte_count": 5 * 1048576, "categories": ["attention_kv"]}], "unique_storage_bytes": 65 * 1048576})
    final = {"run_id": "real", "finalized": True, "status": "completed", "capture_state": "completed", "provisional": False,
             "message": "Recording finalized.", "manifest": deepcopy(manifest)}
    before = {"clock_domain": "qualification_host:time.monotonic_ns", "host_scope": "machine executing this qualification helper",
              "observed_monotonic_ns": 10, "status": "observed", "gpus": {"fields": ["uuid", "memory.used", "memory.free", "memory.total"],
              "units": "memory in MiB", "returncode": 0, "rows": ["GPU-valid, 4, 96, 100"]},
              "compute_apps": {"fields": ["pid", "used_memory"], "units": "memory in MiB", "returncode": 0, "rows": []}}
    after = deepcopy(before)
    after["observed_monotonic_ns"] = 110
    return manifest, events, final, before, after


def test_finalization_requires_present_receipt_matching_run_and_successful_owner_exit():
    manifest, events, final, _, _ = cleanup_receipts()
    assert assessment.finalization_evidence(manifest, events, final)["ok"]
    assert not assessment.finalization_evidence(manifest, events, None)["ok"]
    for field, value in (("run_id", "other"), ("finalized", False), ("status", "finalizing"),
                         ("capture_state", "finalizing"), ("provisional", True)):
        wrong = deepcopy(final)
        wrong[field] = value
        assert not assessment.finalization_evidence(manifest, events, wrong)["ok"]
    wrong = deepcopy(manifest)
    wrong["execution_process"]["exit_code"] = 1
    assert not assessment.finalization_evidence(wrong, events, final)["ok"]
    wrong = deepcopy(final)
    wrong["manifest"]["execution_process"]["pid"] = 99
    assert not assessment.finalization_evidence(manifest, events, wrong)["ok"]
    wrong = deepcopy(final)
    wrong["manifest"] = "not a recording manifest"
    assert not assessment.finalization_evidence(manifest, events, wrong)["ok"]


def test_archived_trace_receipt_proves_finalization_without_current_controller_message(tmp_path):
    manifest, events, _, _, _ = cleanup_receipts()
    directory = tmp_path / manifest["run_id"]
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "events.jsonl").write_text("".join(json.dumps({**row, "sequence": index}) + "\n"
        for index, row in enumerate(events, 1)))
    # Read the real archived API path rather than authoring a replacement
    # response with the qualification helper's preferred display wording.
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/api/runs/{manifest['run_id']}/trace")
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["message"] == ""
    assert receipt["finalized"] is True and receipt["provisional"] is False
    result = assessment.finalization_evidence(manifest, events, receipt)
    assert result["ok"] and result["message"] == ""
    receipt["manifest"]["execution_process"]["pid"] = 99
    assert not assessment.finalization_evidence(manifest, events, receipt)["ok"]


def test_missing_final_state_path_cannot_qualify_completed_recording(tmp_path, monkeypatch):
    manifest, events, _, _, _ = cleanup_receipts()
    manifest.update(model={"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "a" * 40}, architecture={"num_layers": 1})
    monkeypatch.setattr(assessment, "load_recording", lambda _: SimpleNamespace(manifest=manifest, events=events, content=[]))
    monkeypatch.setattr(assessment, "evaluate", lambda _: SimpleNamespace(ok=True, text=lambda: "valid"))
    monkeypatch.setattr(assessment, "source_identity", lambda _: {"source_sha256": "verified", "source_file_count": 1})
    result = assessment.assess(tmp_path / "recording", {"model_revision": "a" * 40}, final_state=tmp_path / "missing-final-state.json")
    assert not result["ok"] and not result["checks"]["clean_finalization_receipt"]
    assert not result["finalization"]["checks"]["finalization_receipt_present"]


def test_post_owner_exit_verifies_lifecycle_without_overwriting_retained_pre_exit_memory():
    manifest, events, final, before, after = cleanup_receipts()
    memory = assessment.memory_evidence(manifest, events)
    assert not memory["checks"]["cleanup_memory_decrease_consistent_with_storage_release"]
    finalization = assessment.finalization_evidence(manifest, events, final)
    result = assessment.cleanup_evidence(manifest, events, memory, finalization, before, after, True)
    assert result["ok"] and result["post_owner_exit_verified"] and not result["pre_exit_counter_decrease_verified"]
    assert result["basis"] == "external_post_owner_exit_receipts"
    assert result["final_recorded_cleanup_process_bytes"] == {"GPU-valid": 65 * 1048576}
    assert memory["final_cleanup_sample"]["devices"][0]["process_used_bytes"] == 65 * 1048576
    assert not memory["checks"]["cleanup_memory_decrease_consistent_with_storage_release"]


def test_post_owner_proof_rejects_pid_uuid_clock_host_source_and_management_failures():
    manifest, events, final, before, after = cleanup_receipts()
    memory = assessment.memory_evidence(manifest, events)
    finalization = assessment.finalization_evidence(manifest, events, final)
    for mutate in (
        lambda row: row["compute_apps"].update(rows=["42, 65"]),
        lambda row: row["gpus"].update(rows=["GPU-other, 4, 96, 100"]),
        lambda row: row["gpus"].update(rows=["GPU-valid, 4, 96, 101"]),
        lambda row: row["gpus"].update(rows=["GPU-valid, 69, 31, 100"]),
        lambda row: row.update(observed_monotonic_ns=99),
        lambda row: row.update(clock_domain="unrelated:time.monotonic_ns"),
        lambda row: row.update(host_scope="other machine"),
        lambda row: row["compute_apps"].update(returncode=1),
        lambda row: row["compute_apps"].update(returncode=False),
    ):
        wrong = deepcopy(after)
        mutate(wrong)
        assert not assessment.cleanup_evidence(manifest, events, memory, finalization, before, wrong, True)["ok"]
    assert not assessment.cleanup_evidence(manifest, events, memory, finalization, before, after, False)["ok"]
    assert not assessment.cleanup_evidence(manifest, events, memory, finalization, None, after, True)["ok"]
    assert not assessment.cleanup_evidence(manifest, events, memory, finalization, before, None, True)["ok"]
    wrong = deepcopy(manifest)
    wrong["execution_target"]["kind"] = "remote"
    assert not assessment.cleanup_evidence(wrong, events, memory, finalization, before, after, True)["ok"]


def test_pre_exit_release_is_valid_without_external_receipts_but_final_receipt_is_still_required():
    manifest, events, final, _, _ = cleanup_receipts()
    events[-2]["devices"][0]["process_used_bytes"] = 10 * 1048576
    memory = assessment.memory_evidence(manifest, events)
    result = assessment.cleanup_evidence(manifest, events, memory, assessment.finalization_evidence(manifest, events, final), None, None, True)
    assert result["ok"] and result["basis"] == "in_recording_process_counter_decrease"
    assert not result["post_owner_exit_verified"]
