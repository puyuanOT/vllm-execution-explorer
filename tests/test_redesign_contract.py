"""Redesign contracts that can be certified without a CUDA host."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import runner.serve as serve_module
from runner.artifacts import ArtifactTransferManager
from runner.capabilities import CAPABILITY_STATUSES, capability_report
from runner.configuration import ConfigurationError, launch_blockers, parse_run_configuration
from runner.controller import drive_run
from runner.fixture_engine import ObservedChunk
from runner.http_session import VllmHttpSession
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.serve import RunRequest, RunState, _manifest_for, create_app
from runner.targets import SshTarget, TargetError
from runner.workload import PROVISIONAL_LIMITS, WorkloadError, parse_run_workload
from vllm_030.collector import Recording


def test_tens_of_requests_and_exact_synthetic_metadata_are_accepted():
    requests = [
        {
            "request_id": f"request-{index}",
            "prompt": f"prompt {index}",
            "arrival_offset_ms": index,
            "max_output_tokens": 256,
            "resolved_input_tokens": 2,
        }
        for index in range(40)
    ]
    parsed, plan = parse_run_workload({"requests": requests})
    assert len(parsed) == 40
    assert PROVISIONAL_LIMITS["max_requests"] >= 40
    assert plan["batches"][0]["samples"][0]["resolved_input_tokens"] == 2

    synthetic, _ = parse_run_workload(
        {
            "requests": [
                {
                    "request_id": "synthetic-1",
                    "prompt": "resolved tokenizer text",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 8,
                    "input_mode": "synthetic",
                    "target_input_tokens": 128,
                    "resolved_input_tokens": 128,
                }
            ]
        }
    )
    assert synthetic[0].target_input_tokens == synthetic[0].resolved_input_tokens == 128
    unresolved, _ = parse_run_workload(
        {
            "requests": [
                {
                    "request_id": "synthetic-unresolved",
                    "prompt": "",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 8,
                    "input_mode": "synthetic",
                    "target_input_tokens": 256,
                    "resolved_input_tokens": None,
                }
            ]
        }
    )
    assert unresolved[0].target_input_tokens == 256
    assert unresolved[0].resolved_input_tokens is None
    with pytest.raises(WorkloadError, match="must equal"):
        parse_run_workload(
            {
                "requests": [
                    {
                        "request_id": "bad",
                        "prompt": "not exact",
                        "arrival_offset_ms": 0,
                        "max_output_tokens": 8,
                        "input_mode": "synthetic",
                        "target_input_tokens": 128,
                        "resolved_input_tokens": 127,
                    }
                ]
            }
        )


def test_batch_adapter_defaults_and_request_overrides_are_canonical():
    raw_requests = [
        {"request_id": "a", "prompt": "one", "arrival_offset_ms": 0, "max_output_tokens": 8},
        {"request_id": "b", "prompt": "two", "arrival_offset_ms": 0, "max_output_tokens": 8, "adapter_id": "base"},
    ]
    plan = {
        "version": 1,
        "batches": [
            {
                "id": "batch-a",
                "label": "A",
                "arrival_offset_ms": 0,
                "adapter_id": "legal-lora",
                "samples": [
                    {"id": "a", "label": "input_1", "max_output_tokens": 8},
                    {"id": "b", "label": "input_2", "max_output_tokens": 8, "adapter_id": "base"},
                ],
            }
        ],
    }
    requests, canonical = parse_run_workload({"requests": raw_requests, "workload_plan": plan})
    assert [request.adapter_id for request in requests] == ["legal-lora", "base"]
    assert [sample["adapter_id"] for sample in canonical["batches"][0]["samples"]] == ["legal-lora", "base"]


def test_configuration_snapshot_is_secret_free_and_context_aware():
    requests, plan = parse_run_workload(
        {
            "requests": [
                {
                    "request_id": "a",
                    "prompt": "one",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 32,
                    "resolved_input_tokens": 64,
                    "adapter_id": "legal-lora",
                }
            ]
        }
    )
    payload = {
        "huggingface_token": "hf_do-not-record-this",
        "model_configuration": {
            "source": {"kind": "huggingface", "repository": "org/model", "revision": "model-sha"},
            "tokenizer": {"revision": "tokenizer-sha"},
            "dtype": "float16",
            "quantization": {"kind": "awq"},
            "lora_adapters": [
                {
                    "id": "legal-lora",
                    "source": {"kind": "execution_host", "path": "/models/legal-lora"},
                }
            ],
        },
        "vllm_configuration": {
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_scheduled_tokens": 2048,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.85,
            "fresh_prefix_cache": True,
        },
        "sampling": {"temperature": 0, "top_p": 1, "seed": 7, "stop": ["END"]},
    }
    snapshot = parse_run_configuration(payload, requests, plan)
    rendered = json.dumps(snapshot)
    assert "hf_do-not-record-this" not in rendered
    assert snapshot["model"]["source"]["authenticated"] is True
    assert snapshot["vllm"]["fresh_prefix_cache"] is True
    assert len(snapshot["snapshot_sha256"]) == 64
    assert launch_blockers(snapshot, requests, "vllm") == []

    too_small = dict(payload)
    too_small["vllm_configuration"] = {**payload["vllm_configuration"], "max_model_len": 80}
    with pytest.raises(ConfigurationError, match="exceeds"):
        parse_run_configuration(too_small, requests, plan)


def test_capability_statuses_disable_uncertified_controls():
    report = capability_report(platform="darwin", cuda_available=False, vllm_version=None)
    assert tuple(report["status_values"]) == CAPABILITY_STATUSES
    assert report["capture_boundaries"]["http_receipt"]["status"] == "gpu_validation_required"
    assert report["capture_boundaries"]["http_receipt"]["selectable"] is False
    assert report["workload"]["synthetic_exact_tokens"]["status"] == "gpu_validation_required"
    assert report["model_options"]["lora"]["status"] == "gpu_validation_required"
    assert report["model_options"]["lora"]["selectable"] is True
    assert report["model_options"]["quantization"]["awq"]["selectable"] is False
    assert report["model_options"]["quantization"]["bitsandbytes"]["selectable"] is False


def test_a10_quantization_capabilities_require_the_bitsandbytes_plugin():
    report = capability_report(
        platform="linux",
        cuda_available=True,
        vllm_version="0.30.0",
        cuda_compute_capability=(8, 6),
        bitsandbytes_plugin_available=False,
    )
    quantization = report["model_options"]["quantization"]
    assert quantization["awq"]["selectable"] is True
    assert quantization["gptq"]["selectable"] is True
    assert quantization["bitsandbytes"]["status"] == "blocked"
    assert quantization["bitsandbytes"]["selectable"] is False
    assert "vllm-bnb-plugin" in quantization["bitsandbytes"]["message"]
    assert quantization["fp8"]["selectable"] is True
    assert "weight-only W8A16" in quantization["fp8"]["message"]


def test_a10_bitsandbytes_is_selectable_when_the_plugin_is_present():
    report = capability_report(
        platform="linux",
        cuda_available=True,
        vllm_version="0.30.0",
        cuda_compute_capability=(8, 6),
        bitsandbytes_plugin_available=True,
    )
    capability = report["model_options"]["quantization"]["bitsandbytes"]
    assert capability["status"] == "gpu_validation_required"
    assert capability["selectable"] is True


def test_run_rejects_bitsandbytes_when_the_runtime_plugin_is_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "runner.serve.decide_target",
        lambda *args, **kwargs: SimpleNamespace(ok=True, message="ready"),
    )
    monkeypatch.setattr("runner.serve._cuda_available", lambda: True)
    monkeypatch.setattr("runner.serve._cuda_compute_capability", lambda: (8, 6))
    monkeypatch.setattr("runner.serve._vllm_version", lambda: "0.30.0")
    monkeypatch.setattr("runner.serve._bitsandbytes_plugin_available", lambda: False)
    client = TestClient(create_app(tmp_path / "runs"))
    response = client.post(
        "/api/runs",
        json={
            "engine": "vllm",
            "requests": [
                {
                    "request_id": "a",
                    "prompt": "hello",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 1,
                }
            ],
            "model_configuration": {
                "source": {
                    "kind": "huggingface",
                    "repository": "org/model",
                    "revision": "model-sha",
                },
                "tokenizer": {"revision": "tokenizer-sha"},
                "quantization": {"kind": "bitsandbytes"},
            },
        },
    )
    assert response.status_code == 409
    assert "vllm-bnb-plugin" in response.json()["detail"]


def test_bitsandbytes_probe_requires_the_registered_vllm_plugin(monkeypatch):
    class EntryPoints(list):
        def select(self, **criteria):
            return EntryPoints(
                item
                for item in self
                if all(getattr(item, key) == value for key, value in criteria.items())
            )

    monkeypatch.setattr(serve_module, "_module_available", lambda name: name == "vllm_bnb_plugin")
    monkeypatch.setattr(
        serve_module.importlib.metadata,
        "entry_points",
        lambda: EntryPoints([
            SimpleNamespace(
                group="vllm.general_plugins",
                name="bitsandbytes",
                value="some_other_module:register",
            )
        ]),
    )
    assert serve_module._bitsandbytes_plugin_available() is False

    monkeypatch.setattr(
        serve_module.importlib.metadata,
        "entry_points",
        lambda: EntryPoints([
            SimpleNamespace(
                group="vllm.general_plugins",
                name="bitsandbytes",
                value="vllm_bnb_plugin:register",
            )
        ]),
    )
    assert serve_module._bitsandbytes_plugin_available() is True


def test_fixture_rejects_attached_lora_even_when_requests_use_base_model():
    requests, plan = parse_run_workload(
        {"requests": [{"request_id": "a", "prompt": "one", "arrival_offset_ms": 0, "max_output_tokens": 1}]}
    )
    snapshot = parse_run_configuration(
        {
            "model_configuration": {
                "source": {"kind": "execution_host", "path": "/models/base"},
                "lora_adapters": [
                    {"id": "legal", "source": {"kind": "execution_host", "path": "/models/legal"}}
                ],
            }
        },
        requests,
        plan,
    )
    assert launch_blockers(snapshot, requests, "vllm") == []
    assert launch_blockers(snapshot, requests, "fixture") == ["The demo fixture does not execute LoRA adapters."]


def test_huggingface_lora_requires_a_pinned_revision():
    requests, plan = parse_run_workload(
        {"requests": [{"request_id": "a", "prompt": "one", "arrival_offset_ms": 0, "max_output_tokens": 1}]}
    )
    with pytest.raises(ConfigurationError, match="pinned revision"):
        parse_run_configuration(
            {
                "model_configuration": {
                    "source": {"kind": "execution_host", "path": "/models/base"},
                    "lora_adapters": [
                        {"id": "legal", "source": {"kind": "huggingface", "repository": "org/legal"}}
                    ],
                }
            },
            requests,
            plan,
        )

    snapshot = parse_run_configuration(
        {
            "huggingface_token": "hf_ephemeral",
            "model_configuration": {
                "source": {"kind": "execution_host", "path": "/models/base"},
                "lora_adapters": [
                    {
                        "id": "legal",
                        "source": {
                            "kind": "huggingface",
                            "repository": "org/legal",
                            "revision": "commit-sha",
                        },
                    }
                ],
            },
        },
        requests,
        plan,
    )
    assert snapshot["model"]["lora_adapters"][0]["source"]["authenticated"] is True
    assert "hf_ephemeral" not in json.dumps(snapshot)


def test_fixture_api_exposes_configuration_capture_and_provenance(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "runs"))
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    assert client.get("/api/targets/status").json()["phase"] == "idle"
    started = client.post(
        "/api/runs",
        json={
            "engine": "fixture",
            "requests": [
                {
                    "request_id": "a",
                    "prompt": "Hello",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 1,
                }
            ],
        },
    )
    assert started.status_code == 200
    assert len(started.json()["configuration_snapshot_sha256"]) == 64
    deadline = time.monotonic() + 5
    snapshot = {}
    while time.monotonic() < deadline:
        snapshot = client.get("/api/runs/current").json()
        if not snapshot["active"]:
            break
        time.sleep(0.01)
    assert snapshot["status"] == "completed"
    assert snapshot["capture"]["phase"] == "completed"
    manifest = client.get(f"/api/recordings/{snapshot['run_id']}/manifest").json()
    assert manifest["run_configuration"]["snapshot_sha256"] == started.json()["configuration_snapshot_sha256"]
    assert manifest["capture"]["transport"] == "fixture_direct"
    assert manifest["provenance"]["evidence_kind"] == "fixture"
    assert manifest["initial_cache_state"]["reset_verified"] is False
    assert "huggingface_token" not in json.dumps(manifest)


def test_completed_gpu_manifest_records_observed_acceptance_without_stale_required_claim():
    class Session:
        execution_kind = "gpu"
        execution_target = {
            "kind": "local",
            "engine_location": "this-machine",
            "client_platform": "linux",
            "runner_endpoint": "http://127.0.0.1:8765",
            "remote_bind": "127.0.0.1",
        }
        dispatch_transport = "instrumented_loopback_http"
        hardware_record = {"gpu": "NVIDIA A10", "memory_mib": 23028}
        engine_record = {"name": "vllm", "version": "0.30.0"}
        model_record = {"id": "facebook/opt-125m", "revision": "pinned"}
        tokenizer_record = {"id": "facebook/opt-125m", "revision": "pinned"}
        sampling_record = {"temperature": 0, "seed": 0}
        initial_cache_state = {"reset_verified": True}
        cache_geometry = None
        engine_settings = None
        preparation_observations = None

    manifest = _manifest_for(
        RunState(),
        "run-observed",
        RunRequest(engine="vllm", requests=[]),
        Session(),
        [{"request_id": "a", "http_receipt_observed": True}],
        {"a": [1]},
        {},
        "completed",
    )

    assert manifest["provenance"]["gpu_acceptance"] == "passed"
    assert manifest["provenance"]["real_gpu"] is True
    assert "still required" not in manifest["provenance"]["claim"]
    assert manifest["workload_limits"]["gpu_validated"] is False

    cancelled = _manifest_for(
        RunState(),
        "run-cancelled",
        RunRequest(engine="vllm", requests=[]),
        Session(),
        [],
        {},
        {},
        "cancelled",
    )
    assert cancelled["provenance"]["gpu_acceptance"] == "passed"
    assert cancelled["provenance"]["http_serving"] is False
    assert "were not observed" in cancelled["provenance"]["claim"]


def test_failed_gpu_preparation_is_not_mislabeled_as_real_gpu_and_keeps_redacted_error():
    state = RunState()
    state.message = "Could not load with hf_secret_value"
    manifest = _manifest_for(
        state,
        "run-failed",
        RunRequest(engine="vllm", requests=[], huggingface_token="hf_secret_value"),
        None,
        [],
        {},
        {},
        "failed",
    )

    assert manifest["execution_kind"] == "gpu"
    assert manifest["execution_target"]["client_platform"] != "fixture"
    assert manifest["provenance"]["evidence_kind"] == "instrumented_gpu_run"
    assert manifest["provenance"]["gpu_acceptance"] == "required"
    assert manifest["provenance"]["real_gpu"] is False
    assert manifest["capture"]["message"] == "Could not load with [Hugging Face token]"
    assert manifest["capture"]["transport"] == "in_process_direct_engine"
    assert "hf_secret_value" not in json.dumps(manifest)


def test_operator_local_artifact_transfer_has_bounded_retryable_status(tmp_path: Path):
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "weights.safetensors").write_bytes(b"weights")
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    manager = ArtifactTransferManager(run=run)
    prepared = manager.prepare(str(artifact), "/srv/models/example folder")
    assert prepared["status"] == "prepared"
    assert prepared["total_bytes"] == len(b"weights")
    assert str(artifact) not in json.dumps(prepared)
    target = SshTarget(
        user="ubuntu",
        host="192.0.2.10",
        identity_file=str(tmp_path / "identity.pem"),
        remote_workdir="/home/ubuntu/explorer",
    )
    manager.start(prepared["transfer_id"], target)
    deadline = time.monotonic() + 2
    status = manager.snapshot(prepared["transfer_id"])
    while status["status"] == "transferring" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot(prepared["transfer_id"])
    assert status["status"] == "completed"
    assert status["bytes_transferred"] == status["total_bytes"]
    assert any(call[0] == "rsync" and "--copy-links" in call for call in calls)
    rsync_call = next(call for call in calls if call[0] == "rsync")
    assert "--protect-args" not in rsync_call
    assert rsync_call[-1] == "ubuntu@192.0.2.10:'/srv/models/example folder'"


def test_operator_local_artifact_transfer_failure_redacts_local_paths(tmp_path: Path):
    artifact = tmp_path / "private" / "model"
    artifact.mkdir(parents=True)
    (artifact / "weights.safetensors").write_bytes(b"weights")
    identity = tmp_path / "identity.pem"

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == "rsync":
            return subprocess.CompletedProcess(argv, 1, "", f"could not read {artifact} with {identity}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    manager = ArtifactTransferManager(run=run)
    prepared = manager.prepare(str(artifact), "/srv/models/example")
    target = SshTarget(
        user="ubuntu",
        host="192.0.2.10",
        identity_file=str(identity),
        remote_workdir="/home/ubuntu/explorer",
    )
    manager.start(prepared["transfer_id"], target)
    deadline = time.monotonic() + 2
    status = manager.snapshot(prepared["transfer_id"])
    while status["status"] == "transferring" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot(prepared["transfer_id"])

    assert status["status"] == "failed"
    assert str(artifact) not in status["message"]
    assert str(identity) not in status["message"]
    assert "[local artifact]" in status["message"]
    assert "[identity file]" in status["message"]


def test_operator_local_artifact_rejects_symlinked_directories(tmp_path: Path):
    artifact = tmp_path / "model"
    artifact.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "uninspected.safetensors").write_bytes(b"weights")
    (artifact / "linked-weights").symlink_to(outside, target_is_directory=True)

    manager = ArtifactTransferManager()
    with pytest.raises(TargetError, match="symlinked directory"):
        manager.prepare(str(artifact), "/srv/models/example")


def test_operator_local_http_upload_is_streamed_to_retryable_staging(tmp_path: Path):
    client = TestClient(
        create_app(
            tmp_path / "runs",
            artifacts_dir=tmp_path / "artifacts",
        )
    )
    response = client.post(
        "/api/artifacts/operator-local/upload",
        data={"destination": "/srv/models/example"},
        files=[
            ("files", ("config.json", b"{}", "application/json")),
            ("files", ("weights.safetensors", b"weights", "application/octet-stream")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "prepared"
    assert body["upload"] == {
        "status": "staged",
        "file_count": 2,
        "total_bytes": 9,
        "retained_for_retry": True,
    }
    assert body["destination"] == "/srv/models/example"
    assert "config.json" not in json.dumps(body)
    assert len(list((tmp_path / "artifacts").glob("upload-*"))) == 1


def test_loopback_http_session_keeps_engine_calls_on_one_owner_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeEngineSession:
        execution_target = LOCAL_FIXTURE_TARGET
        cache_geometry = {"block_size": 16, "num_gpu_blocks": 8, "initial_free_blocks": 8}
        engine_settings = {"max_num_seqs": 4, "max_num_scheduled_tokens": 32}
        engine_record = {"name": "fake-vllm", "version": "0.30.0"}
        model_record = {"id": "fake"}
        tokenizer_record = {"id": "fake"}
        hardware_record = {"gpu": "fake"}
        sampling_record = {"temperature": 0}
        initial_cache_state = {"kind": "fresh", "reset_verified": True}
        preparation_observations = {"capture_ready_monotonic_ns": 1}

        def __init__(self) -> None:
            self.owner_threads: set[int] = set()
            self.queued: list[str] = []

        def add_work_request(self, request) -> None:
            self.owner_threads.add(threading.get_ident())
            self.queued.append(request.request_id)

        def has_unfinished(self) -> bool:
            self.owner_threads.add(threading.get_ident())
            return bool(self.queued)

        def step(self) -> list[ObservedChunk]:
            self.owner_threads.add(threading.get_ident())
            request_id = self.queued.pop(0)
            return [ObservedChunk(request_id, (7,), "x", "stop")]

        def cancel(self) -> None:
            self.owner_threads.add(threading.get_ident())
            self.queued.clear()

        def close(self) -> None:
            self.owner_threads.add(threading.get_ident())

    fake = FakeEngineSession()
    monkeypatch.setattr("runner.http_session.VllmSession.open", lambda _recording, _body: fake)
    recording = Recording.open(tmp_path / "http", run_id="http")
    session = VllmHttpSession.open(recording, object())
    requests, _plan = parse_run_workload(
        {
            "requests": [
                {"request_id": "a", "prompt": "one", "arrival_offset_ms": 0, "max_output_tokens": 1},
                {"request_id": "b", "prompt": "two", "arrival_offset_ms": 0, "max_output_tokens": 1},
            ]
        }
    )
    try:
        status, tokens, arrivals = drive_run(session, recording, requests, max_run_seconds=5)
    finally:
        session.close()
    recording.close(manifest_skeleton("http", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, arrivals=arrivals))
    assert status == "completed"
    assert tokens == {"a": [7], "b": [7]}
    assert all(item["http_receipt_observed"] for item in arrivals)
    assert all(item["engine_admission_observed"] for item in arrivals)
    assert len(fake.owner_threads) == 1
    events = [json.loads(line) for line in (tmp_path / "http" / "events.jsonl").read_text().splitlines()]
    event_types = {event["event_type"] for event in events}
    assert {"http_request_received", "http_engine_admission_correlated", "http_response_completed"} <= event_types
    for request_id in ("a", "b"):
        request_events = [event for event in events if event.get("request_id") == request_id]
        sequences = {event["event_type"]: event["sequence"] for event in request_events}
        assert sequences["request_dispatched"] < sequences["http_request_received"]
        assert sequences["http_request_received"] < sequences["http_engine_admission_correlated"]
        assert sequences["http_response_completed"] <= max(event["sequence"] for event in events)


def test_loopback_http_session_propagates_engine_owner_step_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class ExplodingSession:
        def __init__(self) -> None:
            self.admitted = False

        def add_work_request(self, _request) -> None:
            self.admitted = True
            return None

        def has_unfinished(self) -> bool:
            return self.admitted

        def step(self):
            raise RuntimeError("engine step exploded")

        def cancel(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr("runner.http_session.VllmSession.open", lambda _recording, _body: ExplodingSession())
    recording = Recording.open(tmp_path / "explode", run_id="explode")
    session = VllmHttpSession.open(recording, object())
    requests, _plan = parse_run_workload(
        {"requests": [{"request_id": "a", "prompt": "one", "arrival_offset_ms": 0, "max_output_tokens": 1}]}
    )
    with pytest.raises(RuntimeError, match="engine step exploded"):
        drive_run(session, recording, requests, max_run_seconds=5)
    with pytest.raises(RuntimeError, match="engine step exploded"):
        session.close()
    recording.close(manifest_skeleton("explode", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    events = [json.loads(line) for line in (tmp_path / "explode" / "events.jsonl").read_text().splitlines()]
    assert sum(event["event_type"] == "http_response_completed" for event in events) == 1
