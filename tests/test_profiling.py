"""Synthetic profiler evidence validates correlation boundaries without GPU calls."""

from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from runner.configuration import ConfigurationError, apply_legacy_fields, launch_blockers, parse_run_configuration
from runner.export import export_structural
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.profile_trace import PROFILE_CLOCK, normalize_trace
from runner.profiling import DiagnosticProfile, profile_record
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
    payload["profiling_configuration"]["max_steps"] = 17
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
    assert profile.record["status"] == ("completed" if outcome == "completed" else "partial")
    assert profile.record["completeness"] == "bounded"
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
