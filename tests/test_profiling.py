"""Synthetic profiler evidence validates correlation boundaries without GPU calls."""

from __future__ import annotations

import gzip
import importlib.util
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from runner.configuration import ConfigurationError, apply_legacy_fields, launch_blockers, parse_run_configuration
from runner.export import export_structural
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.http_session import DIAGNOSTIC_FINALIZATION_TIMEOUT_SECONDS, VllmHttpSession
from runner.profile_trace import PROFILE_CLOCK, normalize_trace
from runner.profile_codec import encode_compact_profile, expand_profile_wire
from runner.profiling import DiagnosticProfile, _bounded_normalized_payload, _link_cache_writes, profile_record
from runner.serve import RunRequest, create_app
from runner.workload import parse_run_workload
from vllm_030.collector import Recording
from vllm_030.profiling import DiagnosticHooks, copy_packed_rows


def trace_fixture():
    def cpu(name, start, duration, **args):
        return {"ph": "X", "cat": "user_annotation" if name.startswith("explorer.") else "cuda_runtime", "name": name, "ts": start, "dur": duration, "pid": 1, "tid": 9, "args": args}
    return {"traceEvents": [
        cpu("explorer.step/step-1", 100, 100), cpu("explorer.layer/0", 110, 80),
        cpu("explorer.component/0/attention", 120, 30), cpu("cudaLaunchKernel", 135, 2, correlation=7),
        {"ph": "X", "cat": "kernel", "name": "native_kernel", "ts": 170, "dur": 12, "pid": 3, "tid": 7, "args": {"device": 0, "stream": 7, "correlation": 7}},
        {"ph": "X", "cat": "kernel", "name": "qwen_layer_3_guess_is_not_evidence", "ts": 185, "dur": 4, "pid": 3, "tid": 7, "args": {"device": 0, "stream": 7}},
    ]}


def test_native_timestamps_and_explicit_launch_correlation_stay_distinct():
    normalized = normalize_trace(trace_fixture(), run_id="synthetic", steps=[])
    measured, unknown = normalized["activities"][-2:]
    assert normalized["clock"] == PROFILE_CLOCK
    assert (measured["start"], measured["duration"]) == (170, 12)
    assert measured["scope"] == {"step_id": "step-1", "layer_index": 0, "component": "attention", "correlation": "runtime_flow", "correlation_source": "native_correlation_id"}
    assert measured["device"] == 0 and measured["stream"] == 7
    assert measured["related_activity_ids"] == ["activity-3"]
    assert unknown["scope"]["step_id"] is None and unknown["scope"]["correlation"] == "unmapped"
    assert "per-token" in normalized["evidence"]


def test_ambiguous_native_correlation_never_becomes_a_layer_claim():
    trace = trace_fixture()
    trace["traceEvents"] += [
        {"ph": "X", "cat": "user_annotation", "name": "explorer.step/step-2", "ts": 300, "dur": 100, "pid": 1, "tid": 9},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 320, "dur": 1, "pid": 1, "tid": 9, "args": {"correlation": 7}},
    ]
    normalized = normalize_trace(trace, run_id="ambiguous", steps=[])
    assert normalized["activities"][4]["scope"]["correlation"] == "unmapped"
    assert normalize_trace(trace, run_id="bounded", steps=[], max_activities=2)["normalization"]["status"] == "partial"


class Array(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Array(result) if isinstance(key, slice) else result

    def tolist(self):
        return list(self)


def packed_runner():
    return SimpleNamespace(input_batch=SimpleNamespace(req_ids=["decode-first", "prefill-second"], num_reqs=2, num_computed_tokens_cpu=Array([30, 0])),
                           req_indices=SimpleNamespace(np=Array([0, 1, 1])), query_pos=SimpleNamespace(np=Array([0, 0, 1])),
                           query_start_loc=SimpleNamespace(np=Array([0, 1, 3])))


def test_packed_rows_use_observed_post_reorder_buffers():
    runner = packed_runner()
    output = SimpleNamespace(total_num_scheduled_tokens=3)
    mapping = copy_packed_rows(runner, output, "step-1")
    assert mapping["packed_rows"] == [
        {"row": 0, "request_id": "decode-first", "token_position": 30, "is_padding": False},
        {"row": 1, "request_id": "prefill-second", "token_position": 0, "is_padding": False},
        {"row": 2, "request_id": "prefill-second", "token_position": 1, "is_padding": False},
    ]
    runner.req_indices.np[2] = 0
    invalid = copy_packed_rows(runner, output, "step-1")
    assert invalid["mapping_status"] == "unavailable" and not invalid["packed_rows"]


def test_hooks_capture_padding_scope_and_stop_row_metadata_at_bound():
    contexts = []
    @contextmanager
    def record(name):
        contexts.append(name)
        yield
    component = lambda: SimpleNamespace(forward=lambda *args, **kwargs: None)
    Qwen3DecoderLayer = type("Qwen3DecoderLayer", (), {})
    layer = Qwen3DecoderLayer()
    layer.self_attn, layer.mlp = component(), component()
    layer.forward = lambda: (layer.self_attn.forward(), layer.mlp.forward())
    runner = packed_runner()
    runner.model = SimpleNamespace(named_modules=lambda: [("model.layers.0", layer)])
    runner._prepare_inputs = lambda *_args: None
    runner._preprocess = lambda *_args: None
    def execute(output):
        runner._prepare_inputs(output, [1, 2])
        runner._preprocess(output, 4)
        layer.forward()
        return "returned"
    runner.execute_model = execute
    hooks = DiagnosticHooks(runner, record, 1, check_sources=False)
    hooks.install()
    hooks.active = True
    output = SimpleNamespace(total_num_scheduled_tokens=3, explorer_step_id="step-1")
    assert runner.execute_model(output) == "returned"
    assert hooks.steps[0]["packed_rows"][-1] == {"row": 3, "request_id": None, "token_position": None, "is_padding": True}
    assert hooks.steps[0]["padded_tokens"] == 4
    assert contexts == ["explorer.step/step-1", "explorer.layer/0", "explorer.component/0/attention", "explorer.component/0/ffn"]
    runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=3, explorer_step_id="step-2"))
    assert len(hooks.steps) == 1 and len(contexts) == 4
    hooks.uninstall()
    assert runner.execute_model is execute


def test_profile_config_is_bounded_and_fixture_cannot_claim_measurement():
    payload = {"requests": [{"request_id": "a", "prompt": "hello", "arrival_offset_ms": 0, "max_output_tokens": 1}],
               "profiling_configuration": {"mode": "eager_diagnostic", "max_steps": 4}}
    requests, plan = parse_run_workload(payload)
    config = parse_run_configuration(payload, requests, plan)
    body = RunRequest(**payload)
    apply_legacy_fields(body, config)
    assert body.profiling_configuration == config["profiling"] == {"mode": "eager_diagnostic", "max_steps": 4}
    assert any("no GPU" in blocker for blocker in launch_blockers(config, requests, engine="fixture"))
    payload["profiling_configuration"]["max_steps"] = 1024
    config = parse_run_configuration(payload, requests, plan)
    apply_legacy_fields(body, config)
    assert body.profiling_configuration == config["profiling"] == {"mode": "eager_diagnostic", "max_steps": 1024}
    payload["profiling_configuration"]["max_steps"] = 1025
    with pytest.raises(ConfigurationError, match="step limit"):
        parse_run_configuration(payload, requests, plan)


@pytest.mark.parametrize("architecture,parallel", [
    ({"supported": False, "model_type": "llama"}, 1),
    ({"supported": True, "model_type": "qwen3"}, 2),
])
def test_unsupported_diagnostic_layout_is_rejected_before_profiler_import(architecture, parallel):
    engine = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True),
                                                       parallel_config=SimpleNamespace(tensor_parallel_size=parallel)))
    with pytest.raises(RuntimeError, match="dense Qwen3|one GPU"):
        DiagnosticProfile(engine, object(), object(), {"mode": "eager_diagnostic", "max_steps": 8}, architecture)


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_saved_profile_artifact_api_and_export_provenance(tmp_path: Path, outcome: str):
    recording = Recording.open(tmp_path / "synthetic", run_id="synthetic")
    raw_dir = recording.directory / "profiling" / "raw"
    raw_dir.mkdir(parents=True)
    with gzip.open(raw_dir / "synthetic.pt.trace.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(trace_fixture(), stream)
    # Exercise finalization around authored native-shape evidence; no torch/GPU call.
    profile = DiagnosticProfile.__new__(DiagnosticProfile)
    profile.recording = recording
    profile.record = profile_record({"mode": "eager_diagnostic", "max_steps": 1})
    profile.hooks = SimpleNamespace(steps=[{"step_id": "step-1", "packed_rows": [{"row": 0, "request_id": "a", "token_position": 0, "is_padding": False}],
                                         "unpadded_tokens": 1, "padded_tokens": 1, "mapping_status": "observed", "mapping_observation_boundary": "synthetic_test"}])
    profile._save_attachment(bounded=True, cancelled=outcome == "cancelled", failed=outcome == "failed")
    assert profile.record["status"] == "partial"
    assert profile.record["completeness"] == "bounded"
    if outcome == "completed":
        assert "capture limit" in profile.record["reason"]
    assert profile.record["qualification"] == "unverified"
    assert len(profile.record["raw_artifacts"]) == 1
    manifest = manifest_skeleton("synthetic", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["profiling"] = profile.record
    recording.close(manifest)
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/recordings/synthetic/profile")
    assert response.status_code == 200 and response.json()["run_id"] == "synthetic"
    exported = export_structural(recording.directory, tmp_path / "exported")
    stripped = json.loads((exported / "manifest.json").read_text())["profiling"]
    assert stripped["attachment_status"] == "omitted_by_structural_export" and stripped["normalized_artifact"] is None
    assert not (exported / "profiling").exists()


def test_isolated_explicit_layer_scope_retains_layer_without_inventing_batch():
    from runner.profile_trace import normalize_trace
    output = normalize_trace({"traceEvents": [{"ph": "X", "name": "explorer.layer/0", "cat": "user_annotation", "pid": 1, "tid": 2, "ts": 100, "dur": 20}]}, run_id="test", steps=[])
    scope = output["activities"][0]["scope"]
    assert scope["correlation"] == "explicit_scope"
    assert scope["layer_index"] == 0
    assert scope["step_id"] is None


def cache_trace_fixture():
    trace = trace_fixture()
    trace["traceEvents"].insert(3, {"ph": "X", "cat": "user_annotation", "name": "explorer.cache_write/0/cache-write-0",
                                    "ts": 125, "dur": 20, "pid": 1, "tid": 9})
    return trace


def test_cache_write_linking_visits_activities_once_and_preserves_native_order():
    class CountedActivities(list):
        passes = 0

        def __iter__(self):
            self.passes += 1
            return super().__iter__()

    normalized = normalize_trace(cache_trace_fixture(), run_id="test", steps=[])
    source_gpu = normalized["activities"][-2]
    normalized["activities"].extend([
        {**source_gpu, "activity_id": "later-memcpy", "kind": "cuda_memcpy", "start": 190},
        {**source_gpu, "activity_id": "other-operation", "scope": {**source_gpu["scope"], "operation_id": "cache-write-1"}, "related_activity_ids": []},
        {**normalized["activities"][4], "activity_id": "uncovered-launch", "scope": {**source_gpu["scope"], "operation_id": "cache-write-1"}},
    ])
    counted = CountedActivities(normalized["activities"])
    normalized["activities"] = counted
    # Thousands of writes guard against restoring a scan per write; wall-clock assertions would be flaky.
    writes = [{"operation_id": f"cache-write-{index}", "status": "observed"} for index in range(2_000)]
    _link_cache_writes(normalized, writes)
    assert counted.passes == 1
    assert writes[0]["gpu_activity_ids"] == [source_gpu["activity_id"], "later-memcpy"]
    assert writes[0]["status"] == "observed"
    assert writes[1]["gpu_activity_ids"] == ["other-operation"]
    assert writes[1]["status"] == "unavailable" and "every observed GPU launch" in writes[1]["reason"]
    assert normalized["activities"][6]["activity_id"] not in writes[0]["gpu_activity_ids"], "Unmapped GPU activities cannot acquire a cache-write claim."


def test_cpu_detail_bound_retains_every_gpu_scope_launch_and_step(monkeypatch):
    trace = cache_trace_fixture()
    trace["traceEvents"].append({"ph": "X", "cat": "cpu_op", "name": "unrelated-cpu-" + "x" * 20_000,
                                 "ts": 250, "dur": 1, "pid": 1, "tid": 9})
    steps = [{"step_id": f"step-{index}"} for index in range(1, 45)]
    normalized = normalize_trace(trace, run_id="test", steps=steps)
    original_ids = [activity["activity_id"] for activity in normalized["activities"]]
    writes = [{"operation_id": "cache-write-0", "status": "observed"}]
    _link_cache_writes(normalized, writes)
    normalized.update(cache_writes=writes, tensor_snapshots=[{"snapshot_id": "retained-numerical-evidence"}])
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 10_000)
    payload, bounded = _bounded_normalized_payload(normalized)
    assert bounded and payload is not None and len(payload) <= 10_000
    retained = json.loads(payload)
    assert [activity["activity_id"] for activity in retained["activities"]] == original_ids[:-1]
    assert retained["steps"] == steps and retained["cache_writes"] == writes
    assert retained["tensor_snapshots"] == [{"snapshot_id": "retained-numerical-evidence"}]
    retention = retained["normalization"]["attachment_retention"]
    assert retained["normalization"]["status"] == "partial"
    assert retention["original_activity_count"] == 8 and retention["retained_activity_count"] == 7
    assert retention["omitted_cpu_activity_count"] == 1 and retention["omitted_gpu_activity_count"] == 0
    assert retention["gpu_activities_complete"] is True
    assert retained["activities"][-1]["scope"]["correlation"] == "unmapped", "Unmapped native GPU evidence must also survive."
    _link_cache_writes(retained, writes)
    assert writes[0]["status"] == "observed" and writes[0]["gpu_activity_ids"] == ["activity-5"]


def test_detail_reduction_never_repairs_truncated_gpu_evidence_or_claims_oversized_attachment(monkeypatch):
    trace = cache_trace_fixture()
    trace["traceEvents"].insert(0, {"ph": "X", "cat": "cpu_op", "name": "unrelated-cpu-" + "x" * 20_000,
                                   "ts": 1, "dur": 1, "pid": 1, "tid": 9})
    normalized = normalize_trace(trace, run_id="test", steps=[], max_activities=6)
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 10_000)
    payload, bounded = _bounded_normalized_payload(normalized)
    assert bounded and payload is not None
    assert normalized["normalization"]["attachment_retention"]["gpu_activities_complete"] is False
    write = {"operation_id": "cache-write-0", "status": "observed"}
    _link_cache_writes(normalized, [write])
    assert write["status"] == "unavailable"
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 1)
    assert _bounded_normalized_payload(normalized)[0] is None


def test_long_repeated_native_names_are_losslessly_shared_without_dropping_participants(monkeypatch):
    trace = cache_trace_fixture()
    trace["traceEvents"][5]["name"] = "native_cache_kernel<" + "exact-template," * 100 + ">"
    trace["traceEvents"] += [{**trace["traceEvents"][5], "ts": 170 + index / 100, "dur": index / 100 + 0.5}
                              for index in range(60)]
    normalized = normalize_trace(trace, run_id="test", steps=[{"step_id": "step-1", "packed_rows": [{"row": 0, "request_id": "request-1", "token_position": 2, "is_padding": False}]}])
    write = {"operation_id": "cache-write-0", "status": "observed", "participants": [{"row": 0, "request_id": "request-1", "token_position": 2, "pool": "pool-0", "block_id": 7, "slot": 2}]}
    _link_cache_writes(normalized, [write])
    normalized.update(cache_writes=[write], tensor_snapshots=[])
    original = json.loads(json.dumps(normalized))
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 40_000)
    payload, bounded = _bounded_normalized_payload(normalized)
    assert bounded and payload is not None and len(payload) <= 40_000
    encoded = json.loads(payload)
    assert len(json.dumps(original).encode()) > 40_000
    assert encoded["steps"] == original["steps"] and encoded["cache_writes"] == original["cache_writes"]
    assert encoded["normalization"]["attachment_encoding"]["kind"] == "activity_name_dictionary_v1"
    assert encoded["normalization"]["attachment_retention"]["omitted_gpu_activity_count"] == 0
    expanded = [{**{key: value for key, value in activity.items() if key != "name_id"}, "name": encoded["activity_names"][activity["name_id"]]} for activity in encoded["activities"]]
    assert expanded == original["activities"]
    assert normalized["activities"] == original["activities"], "Linking keeps exact expanded in-memory names."
    # The independent native-source assessor expands the same wire format before
    # checking kernel names, timestamps and launch/correlation identities.
    spec = importlib.util.spec_from_file_location("native_wire_assessor", Path(__file__).resolve().parents[1] / "docs/qualification/kv-live-gpu/assess.py")
    assessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assessor)
    assert assessor.expanded_activities(encoded) == original["activities"]
    for mutation in (lambda value: value["activities"][0].update(name="ambiguous"),
                     lambda value: value["activities"][0].update(name_id=len(value["activity_names"])),
                     lambda value: value["activities"][0].update(name_id=True),
                     lambda value: value.pop("activity_names")):
        bad = json.loads(payload)
        mutation(bad)
        with pytest.raises(ValueError):
            assessor.expanded_activities(bad)


def compact_profile_fixture():
    rows = [{"row": 0, "request_id": "request-1", "token_position": 2, "is_padding": False}]
    normalized = normalize_trace(cache_trace_fixture(), run_id="compact", steps=[{"step_id": "step-1", "packed_rows": rows}])
    # Native IDs are not array positions and can have arbitrary gaps.
    ids = {row["activity_id"]: f"native-{index * 7 + 3}" for index, row in enumerate(normalized["activities"])}
    for row in normalized["activities"]:
        row["activity_id"] = ids[row["activity_id"]]
        row["related_activity_ids"] = [ids[id] for id in row["related_activity_ids"]]
    native = normalized["activities"][5]
    normalized["activities"].append({**native, "activity_id": "native-later", "start": 188.125,
                                      "duration": 0, "stream": "7", "related_activity_ids": [ids["activity-4"], "missing-native-id"]})
    # A future activity field takes the verbatim path rather than disappearing.
    normalized["activities"][0]["observer_detail"] = {"label": "préfill", "value": 1.0}
    participant = {**rows[0], "pool": "pool-0", "cache_group": 0, "block_id": 7, "allocation_generation": 1,
                   "slot": 2, "kernel_block_id": 7, "kernel_slot": 2}
    normalized["cache_writes"] = [
        {"operation_id": "write-first", "step_id": "step-1", "participants": [participant], "gpu_activity_ids": [native["activity_id"]]},
        {"operation_id": "write-repeat", "step_id": "step-1", "participants": [dict(participant)], "gpu_activity_ids": ["native-later", "missing-native-id"]},
        {"operation_id": "write-generation", "step_id": "step-1", "participants": [{**participant, "allocation_generation": 2}], "gpu_activity_ids": []},
        {"operation_id": "write-extra", "step_id": "step-1", "participants": [{**participant, "observer_detail": "preserve"}], "gpu_activity_ids": []},
    ]
    return normalized


def test_compact_native_profile_round_trip_preserves_full_objects_order_and_independent_aliases():
    original = compact_profile_fixture()
    before = json.loads(json.dumps(original))
    wire = encode_compact_profile(original)
    assert original == before, "Encoding must not replace expanded evidence used by native linking."
    decoded = expand_profile_wire(json.loads(json.dumps(wire)))
    decoded["normalization"].pop("attachment_encoding")
    assert decoded == before
    assert wire["activities"][0] == before["activities"][0], "Unknown activity fields remain verbatim."
    assert wire["cache_writes"][0]["participant_ids"] == wire["cache_writes"][1]["participant_ids"]
    assert len(wire["cache_participants"]) == 3, "Generation changes and additional fields prevent interning aliases."
    assert wire["activities"][-1][10][-1] == "missing-native-id", "Unknown link identities must not be guessed."
    assert isinstance(wire["activities"][-1][10][0], int)
    assert wire["cache_writes"][1]["gpu_activity_refs"] == [len(before["activities"]) - 1, "missing-native-id"]
    decoded["cache_writes"][0]["participants"][0]["block_id"] = 999
    assert decoded["cache_writes"][1]["participants"][0]["block_id"] == 7
    assert wire["cache_participants"][0]["block_id"] == before["cache_writes"][0]["participants"][0]["block_id"] == 7
    decoded["activities"][5]["scope"]["component"] = "changed"
    assert decoded["activities"][-1]["scope"]["component"] == before["activities"][-1]["scope"]["component"]
    assert expand_profile_wire(before) == before


@pytest.mark.parametrize("mutation", [
    lambda value: value["activities"][1].__setitem__(2, True),
    lambda value: value["activities"][1].__setitem__(2, len(value["activity_names"])),
    lambda value: value["activities"][1].__setitem__(9, -1),
    lambda value: value["activities"][-1][10].__setitem__(0, len(value["activities"])),
    lambda value: value["activities"][-1][10].__setitem__(0, False),
    lambda value: value["activities"][1].pop(),
    lambda value: value["activities"][1].__setitem__(0, value["activities"][2][0]),
    lambda value: value["normalization"]["attachment_encoding"].__setitem__("encoded_activity_count", 0),
    lambda value: value["normalization"]["attachment_encoding"].__setitem__("encoded_cache_write_count", 0),
    lambda value: value["normalization"]["attachment_encoding"].__setitem__("unique_scope_count", True),
    lambda value: value["normalization"]["attachment_encoding"].__setitem__("kind", "unknown-codec"),
    lambda value: value.pop("activity_scopes"),
    lambda value: value["activity_names"].__setitem__(0, {}),
    lambda value: value["cache_writes"][0]["participant_ids"].__setitem__(0, True),
    lambda value: value["cache_writes"][0]["participant_ids"].__setitem__(0, len(value["cache_participants"])),
    lambda value: value["cache_writes"][0].__setitem__("participants", []),
    lambda value: value["cache_writes"][0].__setitem__("gpu_activity_ids", []),
])
def test_compact_profile_rejects_ambiguous_counts_and_invalid_references(mutation):
    wire = encode_compact_profile(compact_profile_fixture())
    mutation(wire)
    with pytest.raises(ValueError):
        expand_profile_wire(wire)


@pytest.mark.parametrize("table,count", [("activity_names", "unique_name_count"), ("activity_scopes", "unique_scope_count"),
                                        ("cache_participants", "unique_participant_count")])
def test_compact_profile_rejects_dictionary_aliases_even_with_matching_counts(table, count):
    wire = encode_compact_profile(compact_profile_fixture())
    wire[table].append(json.loads(json.dumps(wire[table][0])))
    wire["normalization"]["attachment_encoding"][count] += 1
    with pytest.raises(ValueError, match="duplicate"):
        expand_profile_wire(wire)


def test_compact_encoding_fits_required_evidence_without_weakening_byte_or_gpu_bounds(monkeypatch):
    normalized = normalize_trace(cache_trace_fixture(), run_id="compact-bound", steps=[])
    native = normalized["activities"][5]
    normalized["activities"].extend({**native, "activity_id": f"native-repeat-{index}", "start": 200 + index / 10}
                                     for index in range(120))
    participant = {"row": 0, "request_id": "request-1", "token_position": 2, "is_padding": False,
                   "pool": "pool-0", "cache_group": 0, "block_id": 7, "allocation_generation": 1, "slot": 2,
                   "kernel_block_id": 7, "kernel_slot": 2}
    normalized["cache_writes"] = [{"operation_id": f"write-{index}", "participants": [dict(participant)],
                                    "gpu_activity_ids": [native["activity_id"]]} for index in range(36)]
    before = json.loads(json.dumps(normalized))
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 20_000)
    payload, bounded = _bounded_normalized_payload(normalized)
    assert bounded and payload is not None and len(payload) <= 20_000
    wire = json.loads(payload)
    assert wire["normalization"]["attachment_encoding"]["kind"] == "native_profile_compact_v1"
    decoded = expand_profile_wire(wire)
    assert decoded["activities"] == before["activities"] and decoded["cache_writes"] == before["cache_writes"]
    assert normalized["activities"] == before["activities"], "In-memory records remain expanded."
    retention = wire["normalization"]["attachment_retention"]
    assert retention["omitted_gpu_activity_count"] == 0 and retention["gpu_activities_complete"] is True
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 1)
    assert _bounded_normalized_payload(normalized)[0] is None, "Required evidence never bypasses the byte bound."


def test_native_activity_bound_omits_cpu_detail_before_late_gpu_without_changing_attribution(monkeypatch):
    trace = cache_trace_fixture()
    trace["traceEvents"] += [{"ph": "X", "cat": "cpu_op", "name": f"compilation_detail_{index}", "ts": 210 + index / 1000,
                               "dur": 0.01, "pid": 1, "tid": 9} for index in range(500)]
    trace["traceEvents"] += [
        {"ph": "X", "cat": "user_annotation", "name": "explorer.step/step-2", "ts": 300, "dur": 100, "pid": 1, "tid": 9},
        {"ph": "X", "cat": "user_annotation", "name": "explorer.layer/0", "ts": 310, "dur": 70, "pid": 1, "tid": 9},
        {"ph": "X", "cat": "user_annotation", "name": "explorer.component/0/recurrent", "ts": 320, "dur": 50, "pid": 1, "tid": 9},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 325, "dur": 1, "pid": 1, "tid": 9, "args": {"correlation": 8}},
        {"ph": "X", "cat": "kernel", "name": "recurrent_native_kernel", "ts": 350, "dur": 10, "pid": 3, "tid": 7, "args": {"correlation": 8, "device": 0, "stream": 7}},
        # Preserve a second launch matching the first correlation in a different step;
        # the ambiguity must remain unmapped after reducing unrelated CPU detail.
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 335, "dur": 1, "pid": 1, "tid": 9, "args": {"correlation": 7}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cuLaunchKernel", "ts": 130, "dur": 1, "pid": 1, "tid": 9, "args": {"correlation": 99}},
    ]
    steps = [{"step_id": "step-1"}, {"step_id": "step-2"}]
    full = normalize_trace(trace, run_id="test", steps=steps, max_activities=1_000)
    bounded = normalize_trace(trace, run_id="test", steps=steps, max_activities=20)
    assert bounded["steps"] == steps
    assert bounded["normalization"]["status"] == "partial"
    retention = bounded["normalization"]["attachment_retention"]
    assert retention["gpu_activities_complete"] is True and retention["omitted_gpu_activity_count"] == 0
    assert retention["omitted_cpu_activity_count"] == 500
    assert retention["original_activity_count"] == len(full["activities"])
    assert len(bounded["activities"]) <= 20
    ids = [activity["activity_id"] for activity in bounded["activities"]]
    assert ids == [activity["activity_id"] for activity in full["activities"] if not activity["name"].startswith("compilation_detail_")]
    assert [activity for activity in bounded["activities"] if activity["kind"].startswith("cuda_")] == [
        activity for activity in full["activities"] if activity["kind"].startswith("cuda_")]
    late_gpu = next(activity for activity in bounded["activities"] if activity["name"] == "recurrent_native_kernel")
    assert late_gpu["scope"]["step_id"] == "step-2" and late_gpu["scope"]["component"] == "recurrent"
    for activity in bounded["activities"]:
        assert set(activity["related_activity_ids"]).issubset(ids)
    write = {"operation_id": "cache-write-0", "status": "observed"}
    _link_cache_writes(bounded, [write])
    assert write["status"] == "unavailable", "Uncovered launches and ambiguous correlation are not repaired by detail reduction."
    # The native activity bound propagates even when the serialized attachment is small.
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 20_000)
    payload, reduced = _bounded_normalized_payload(bounded)
    assert payload is not None and reduced
    assert bounded["normalization"]["attachment_retention"]["byte_limit"] == 20_000


def test_combined_native_and_byte_detail_bounds_keep_original_counts(monkeypatch):
    trace = cache_trace_fixture()
    trace["traceEvents"][5]["args"]["External id"] = 88
    trace["traceEvents"].append({"ph": "X", "cat": "cpu_op", "name": "unused_external_candidate_" + "x" * 20_000,
                                 "ts": 250, "dur": 1, "pid": 1, "tid": 9, "args": {"External id": 88}})
    trace["traceEvents"] += [{"ph": "X", "cat": "cpu_op", "name": f"unrelated_{index}", "ts": 260, "dur": 1, "pid": 1, "tid": 9} for index in range(50)]
    normalized = normalize_trace(trace, run_id="test", steps=[], max_activities=10)
    assert normalized["normalization"]["attachment_retention"]["omitted_cpu_activity_count"] == 50
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", 10_000)
    payload, bounded = _bounded_normalized_payload(normalized)
    assert bounded and payload is not None
    retention = normalized["normalization"]["attachment_retention"]
    assert retention["original_activity_count"] == 58
    assert retention["retained_activity_count"] == 7 and retention["omitted_cpu_activity_count"] == 51
    assert retention["omitted_gpu_activity_count"] == 0 and retention["gpu_activities_complete"] is True
    assert normalized["activities"][-2]["scope"]["correlation"] == "runtime_flow"


def test_profile_marks_finalizing_before_native_stop_and_uninstalls_after_attachment():
    profile = DiagnosticProfile.__new__(DiagnosticProfile)
    profile._started, profile._finished = True, False
    profile.record = profile_record({"mode": "eager_diagnostic", "max_steps": 4})
    order = []
    profile.hooks = SimpleNamespace(steps=[{"step_id": "step-1"}], uninstall=lambda: order.append("uninstall"))

    def stop():
        assert profile.record["status"] == "finalizing" and profile.record["captured_steps"] == 1
        order.append("stop")

    def save(**_kwargs):
        assert order == ["stop"]
        order.append("attachment")
        profile.record["status"] = "completed"

    profile.engine = SimpleNamespace(stop_profile=stop)
    profile._save_attachment = save
    profile.finish()
    assert order == ["stop", "attachment", "uninstall"]
    assert profile.record["status"] == "completed"


@pytest.mark.parametrize("enabled,status,alive,error", [
    (False, "disabled", False, None), (True, "completed", False, None), (True, "partial", False, None),
    (True, "recording", False, "did not finish finalizing"), (True, "finalizing", False, "did not finish finalizing"),
    (True, "finalizing", True, "cleanup deadline"), (True, "failed", False, "native finalization failed"),
])
def test_http_close_waits_for_diagnostic_finalization_and_rejects_pending_status(enabled, status, alive, error):
    session = VllmHttpSession.__new__(VllmHttpSession)
    session._tickets_lock, session._tickets, session._stop = threading.Lock(), {}, threading.Event()
    order, timeouts = [], []
    session._server = SimpleNamespace(shutdown=lambda: order.append("http-stop"), server_close=lambda: order.append("http-close"))
    session._http = SimpleNamespace(join=lambda **_kwargs: order.append("http-join"))
    session._owner = SimpleNamespace(join=lambda **kwargs: (order.append("owner-join"), timeouts.append(kwargs["timeout"])), is_alive=lambda: alive)
    session._clients = SimpleNamespace(shutdown=lambda **kwargs: order.append(("clients-stop", kwargs["wait"])))
    session._raise_owner_error = lambda: None
    session.profiling_record = {"enabled": enabled, "status": status, "reason": "native finalization failed"}
    if error:
        with pytest.raises(RuntimeError, match=error):
            session.close()
        assert session.profiling_record["status"] == "failed"
    else:
        session.close()
        assert session.profiling_record["status"] == status
    assert session._stop.is_set()
    assert timeouts == [DIAGNOSTIC_FINALIZATION_TIMEOUT_SECONDS if enabled else 30]
    assert order == ["http-stop", "http-close", "http-join", "owner-join", ("clients-stop", not alive)]


@pytest.mark.parametrize("attachment_limit", [10_000, 1])
def test_saved_attachment_bounds_cpu_detail_without_losing_cache_timing(tmp_path: Path, monkeypatch, attachment_limit):
    recording = Recording.open(tmp_path / "bounded", run_id="bounded")
    raw_dir = recording.directory / "profiling" / "raw"
    raw_dir.mkdir(parents=True)
    trace = cache_trace_fixture()
    trace["traceEvents"].append({"ph": "X", "cat": "cpu_op", "name": "unrelated-cpu-" + "x" * 20_000,
                                 "ts": 250, "dur": 1, "pid": 1, "tid": 9})
    with gzip.open(raw_dir / "synthetic.pt.trace.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(trace, stream)
    profile = DiagnosticProfile.__new__(DiagnosticProfile)
    profile.recording = recording
    profile.record = profile_record({"mode": "eager_diagnostic", "max_steps": 4})
    write = {"operation_id": "cache-write-0", "status": "observed"}
    cache = SimpleNamespace(finalize=lambda: None, writes=[write], snapshots=[], capture_record=lambda: {"status": "disabled"})
    profile.hooks = SimpleNamespace(steps=[{"step_id": "step-1", "mapping_status": "observed", "padded_tokens": 1}], cache_capture=cache)
    monkeypatch.setattr("runner.profiling.MAX_NORMALIZED_BYTES", attachment_limit)
    profile._save_attachment(bounded=False, cancelled=False, failed=False)
    assert profile.record["status"] == "partial" and profile.record["completeness"] == "bounded"
    if attachment_limit == 1:
        assert profile.record["attachment_status"] == "unavailable" and profile.record["normalized_artifact"] is None
        assert not (recording.directory / "profiling" / "activities.json").exists()
        assert len(profile.record["raw_artifacts"]) == 1 and profile.record["captured_steps"] == 1
        recording.close(manifest_skeleton("bounded", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
        return
    assert profile.record["attachment_status"] == "available" and profile.record["normalized_artifact"]["size_bytes"] <= 10_000
    assert "Unrelated CPU detail" in profile.record["reason"]
    attachment = json.loads((recording.directory / profile.record["normalized_artifact"]["path"]).read_text())
    assert attachment["cache_writes"][0]["status"] == "observed"
    assert attachment["cache_writes"][0]["gpu_activity_ids"] == ["activity-5"]
    manifest = manifest_skeleton("bounded", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["profiling"] = profile.record
    recording.close(manifest)
    assert TestClient(create_app(tmp_path)).get("/api/recordings/bounded/profile").json() == attachment
