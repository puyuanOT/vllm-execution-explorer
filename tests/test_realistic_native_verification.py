"""Independent native audit detects lost records and corrupted wire identities."""

import importlib.util
from pathlib import Path

import pytest

from runner.profile_codec import encode_compact_profile, expand_profile_wire

spec = importlib.util.spec_from_file_location("independent_native_verifier", Path(__file__).resolve().parents[1] /
                                             "docs/qualification/realistic-demos/verify_native.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def evidence():
    native = [
        {"ph": "X", "cat": "user_annotation", "name": "explorer.step/step-1", "ts": 0, "dur": 20, "pid": 1, "tid": 9},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 2, "dur": 1, "pid": 1, "tid": 9, "args": {"correlation": 7}},
        {"ph": "X", "cat": "kernel", "name": "actual_kernel", "ts": 4, "dur": 3, "pid": 3, "tid": 7, "args": {"correlation": 7, "device": 0, "stream": 7}},
        {"ph": "X", "cat": "gpu_memset", "name": "unmapped_memset", "ts": 8, "dur": 1, "pid": 3, "tid": 7, "args": {"device": 0, "stream": "7"}},
    ]
    cpu_scope = {"step_id": "step-1", "layer_index": 0, "component": "attention", "correlation": "explicit_scope"}
    activities = [
        {"activity_id": "activity-0", "kind": "cpu_scope", "name": "explorer.step/step-1", "device": None, "stream": None,
         "process_id": 1, "thread_id": 9, "start": 0, "duration": 20, "scope": dict(cpu_scope), "related_activity_ids": []},
        {"activity_id": "activity-1", "kind": "cpu_scope", "name": "cudaLaunchKernel", "device": None, "stream": None,
         "process_id": 1, "thread_id": 9, "start": 2, "duration": 1, "scope": dict(cpu_scope), "related_activity_ids": []},
        {"activity_id": "activity-2", "kind": "cuda_kernel", "name": "actual_kernel", "device": 0, "stream": 7,
         "process_id": 3, "thread_id": 7, "start": 4, "duration": 3,
         "scope": {**cpu_scope, "correlation": "runtime_flow", "correlation_source": "native_correlation_id"}, "related_activity_ids": ["activity-1"]},
        {"activity_id": "activity-3", "kind": "cuda_memset", "name": "unmapped_memset", "device": 0, "stream": "7",
         "process_id": 3, "thread_id": 7, "start": 8, "duration": 1,
         "scope": {"step_id": None, "layer_index": None, "component": None, "correlation": "unmapped"}, "related_activity_ids": []},
    ]
    return native, activities


def test_every_compact_expanded_cuda_record_matches_native_including_unmapped_activity():
    native, activities = evidence()
    profile = {"activities": activities, "normalization": {"status": "completed"}, "steps": []}
    decoded = expand_profile_wire(encode_compact_profile(profile))
    result = verifier.verify_activity_evidence(native, decoded["activities"])
    assert result["ok"] and result["native_cuda_count"] == result["normalized_cuda_count"] == 2
    assert result["runtime_mapped_cuda_count"] == result["unmapped_cuda_count"] == 1
    assert result["verified_cpu_link_count"] == 1


@pytest.mark.parametrize("mutation,check", [
    (lambda native, activities: activities.pop(), "all_native_cuda_ids_and_order_preserved"),
    (lambda native, activities: activities.append({**activities[2], "activity_id": "activity-999"}), "all_native_cuda_ids_and_order_preserved"),
    (lambda native, activities: activities.__setitem__(slice(2, 4), activities[2:4][::-1]), "all_native_cuda_ids_and_order_preserved"),
    (lambda native, activities: activities[2].__setitem__("duration", 3.25), "all_native_cuda_fields_exact"),
    (lambda native, activities: activities[2].__setitem__("device", False), "all_native_cuda_fields_exact"),
    (lambda native, activities: activities[2].__setitem__("name", "kernel_name_guess"), "all_native_cuda_fields_exact"),
    (lambda native, activities: activities[2].__setitem__("related_activity_ids", []), "native_runtime_correlation_candidates_exact"),
    (lambda native, activities: activities[2].__setitem__("related_activity_ids", ["activity-0"]), "native_runtime_correlation_candidates_exact"),
    (lambda native, activities: activities[1].__setitem__("start", 99), "all_related_cpu_fields_exact"),
    (lambda native, activities: native[1]["args"].__setitem__("correlation", 8), "native_runtime_correlation_candidates_exact"),
    (lambda native, activities: activities[2]["scope"].__setitem__("layer_index", 1), "runtime_flow_scope_matches_related_cpu_scope"),
    (lambda native, activities: activities[3]["scope"].__setitem__("layer_index", 1), "runtime_flow_scope_matches_related_cpu_scope"),
])
def test_native_verifier_detects_corruption_without_reusing_normalization(mutation, check):
    native, activities = evidence()
    mutation(native, activities)
    result = verifier.verify_activity_evidence(native, activities, mismatch_limit=1)
    assert not result["ok"] and result["checks"][check] is False
    assert result["mismatch_count"] >= 1 and len(result["first_mismatches"]) == 1


def test_native_external_ids_are_verified_against_every_original_cpu_candidate():
    native, activities = evidence()
    native[1]["args"] = {"External id": 7}
    native[2]["args"] = {"External id": 7, "device": 0, "stream": 7}
    activities[2]["scope"]["correlation_source"] = "native_external_id"
    assert verifier.verify_activity_evidence(native, activities)["ok"]
    native.append({"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernel", "ts": 2, "dur": 1,
                   "pid": 1, "tid": 9, "args": {"External id": 7}})
    result = verifier.verify_activity_evidence(native, activities)
    assert not result["ok"] and not result["checks"]["native_runtime_correlation_candidates_exact"]


def test_empty_cuda_input_is_not_reported_as_successful_gpu_verification():
    result = verifier.verify_activity_evidence([], [])
    assert not result["ok"] and not result["checks"]["native_cuda_present"]
