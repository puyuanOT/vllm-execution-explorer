import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from runner.gpu_memory import GpuMemoryRecorder, NvidiaSmiMemorySource, framework_memory, loaded_storages, observed_weight_storage
from runner.configuration import ConfigurationError, apply_legacy_fields, launch_blockers, parse_run_configuration
from runner.workload import parse_run_workload
from runner.vllm_session import VllmSession, _lora_artifact_metadata
from vllm_030.collector import Recording


class Tensor:
    def __init__(self, pointer, byte_count, dtype="bfloat16", device="cuda:0"):
        self.pointer, self.byte_count, self.dtype, self.device = pointer, byte_count, dtype, device

    def numel(self):
        return self.byte_count

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.pointer, nbytes=lambda: self.byte_count)


def test_backing_storages_deduplicate_aliases_and_keep_shared_and_host_owners():
    weight, cache, shared = Tensor(1, 100), Tensor(2, 200), Tensor(3, 300)
    recurrent = type("QwenGatedDeltaNet", (), {})()
    recurrent.kv_cache = [Tensor(4, 400), Tensor(4, 400)]
    module = SimpleNamespace(kv_cache=cache, lora_a_stacked=[shared])
    model = SimpleNamespace(
        named_parameters=lambda **_kwargs: [('\"quoted\".weight', weight), ("tied.weight", weight), ("mixed.weight", shared), ("cpu.embeddings", Tensor(5, 500, device="cpu"))],
        named_buffers=lambda **_kwargs: [("scales", Tensor(6, 40, dtype="float32"))],
        named_modules=lambda: [("layers.0.attn", module), ("layers.1.attn", module), ("layers.2.gdn", recurrent)],
    )
    runner = SimpleNamespace(model=model, positions=Tensor(7, 20))
    result = loaded_storages(SimpleNamespace(model_runner=runner, rank=0, local_rank=0))
    assert result["unique_storage_bytes"] == 1560
    cache_rows = [row for row in result["storages"] if row["categories"] == ["attention_kv"]]
    assert len(cache_rows) == 1 and cache_rows[0]["byte_count"] == 200
    assert len(cache_rows[0]["owners"]) == 2
    assert next(row for row in result["storages"] if "lora" in row["categories"])["categories"] == ["weights", "lora"]
    assert sum(row["byte_count"] for row in result["storages"] if row["device"] == "cpu") == 500
    assert next(row for row in result["storages"] if "recurrent_state" in row["categories"])["byte_count"] == 400
    assert all("pointer" not in str(row) and "data_ptr" not in str(row) for row in result["storages"])


def test_framework_counters_use_worker_device_uuid_and_never_synchronize(monkeypatch):
    cuda = SimpleNamespace(is_initialized=lambda: True, get_device_properties=lambda device: SimpleNamespace(uuid="GPU-exact"),
        memory_allocated=lambda device: 20, memory_reserved=lambda device: 30,
        max_memory_allocated=lambda device: 25, max_memory_reserved=lambda device: 35,
        synchronize=lambda: pytest.fail("sampling must not synchronize"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    sample = framework_memory(SimpleNamespace(device="cuda:0", rank=4, local_rank=0), [])
    assert sample["gpu_uuid"] == "GPU-exact" and sample["rank"] == 4
    assert sample["allocated_bytes"] == 20 and sample["reserved_bytes"] == 30
    assert sample["peak_allocated_bytes"] == 25 and sample["peak_reserved_bytes"] == 35


def test_framework_uuid_normalizes_actual_torch_nvml_difference_without_merging_mig(monkeypatch):
    identity = "46d7f16b-4e67-1e98-39bf-e3179167c952"
    cuda = SimpleNamespace(is_initialized=lambda: True, get_device_properties=lambda _: SimpleNamespace(uuid=identity),
        memory_allocated=lambda _: 100, memory_reserved=lambda _: 200, max_memory_allocated=lambda _: 150, max_memory_reserved=lambda _: 250)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    devices = [{"gpu_uuid": f"GPU-{identity}", "physical_index": 0}]
    assert framework_memory(SimpleNamespace(device="cuda:0"), devices)["gpu_uuid"] == devices[0]["gpu_uuid"]
    devices[0]["gpu_uuid"] = f"MIG-{identity}"
    assert framework_memory(SimpleNamespace(device="cuda:0"), devices)["gpu_uuid"] == devices[0]["gpu_uuid"]
    devices.append({"gpu_uuid": f"GPU-{identity}", "physical_index": 0})
    assert framework_memory(SimpleNamespace(device="cuda:0"), devices)["gpu_uuid"] == identity


def test_framework_does_not_initialize_cuda_or_guess_hidden_device_identity(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    assert framework_memory(SimpleNamespace(device="cuda:0"), []) is None
    cuda = SimpleNamespace(is_initialized=lambda: True, get_device_properties=lambda _: SimpleNamespace(),
        memory_allocated=lambda _: 0, memory_reserved=lambda _: 0, max_memory_allocated=lambda _: 0, max_memory_reserved=lambda _: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-a")
    rows = [{"gpu_uuid": "GPU-a", "physical_index": 0}, {"gpu_uuid": "GPU-b", "physical_index": 1}]
    assert framework_memory(SimpleNamespace(device="cuda:0"), rows)["gpu_uuid"] == "GPU-b"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "missing")
    assert framework_memory(SimpleNamespace(device="cuda:0"), rows)["gpu_uuid"] is None


def test_samples_and_query_failures_are_preserved_in_actual_collector(tmp_path):
    class Source:
        name = "test_management_api"
        count = 0

        def sample(self, pid):
            self.count += 1
            if self.count == 2:
                raise PermissionError()
            return [{"gpu_uuid": "GPU-a", "physical_index": 0, "name": "Test", "total_bytes": 100,
                     "used_bytes": 40, "free_bytes": 60, "process_status": "observed", "process_used_bytes": 30}]

    recording = Recording(tmp_path / "run", run_id="memory")
    recorder = GpuMemoryRecorder(recording, source=Source(), interval_ms=100_000)
    recorder.start()
    recorder.set_phase("inference", "capture_ready")
    recorder.sample()
    recorder.set_phase("cleanup", "session.close")
    recorder.close()
    recording.close({"completion_status": "completed", "gpu_memory": recorder.record})
    events = [json.loads(line) for line in (recording.directory / "events.jsonl").read_text().splitlines()]
    samples = [event for event in events if event["event_type"] == "gpu_memory_sample"]
    assert [row["status"] for row in samples] == ["observed", "unavailable", "observed"]
    assert samples[1]["reason"] == "PermissionError"
    assert samples[0]["sample_started_monotonic_ns"] <= samples[0]["sample_completed_monotonic_ns"] <= samples[0]["monotonic_ns"]
    assert recorder.record["device_peak_kind"] == "highest_successful_sample"
    assert recorder.record["status"] == "completed" and recorder.record["sampler_status"] == "stopped"
    assert recorder.record["final_sample_status"] == "observed"
    assert samples[-1]["phase"] == "cleanup"


@pytest.mark.parametrize("management_in_flight", [False, True], ids=["idle-sampler", "busy-sampler"])
def test_slow_management_queries_never_block_model_iterations(tmp_path, monkeypatch, management_in_flight):
    """Exercise the real session, sampler, CLI source and persisted collector."""
    blocked, release, block_queries = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    query_threads, failures, results = [], [], []
    active_queries = peak_queries = engine_steps = 0

    def management_command(arguments, **kwargs):
        nonlocal active_queries, peak_queries
        query_threads.append(threading.current_thread().name)
        active_queries += 1
        peak_queries = max(peak_queries, active_queries)
        try:
            if block_queries.is_set():
                blocked.set()
                if not release.wait(5):
                    raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])
            text = "0, GPU-test, Test GPU, 1024, 512, 512\n" if arguments[1].startswith("--query-gpu=") else ""
            return SimpleNamespace(stdout=text)
        finally:
            active_queries -= 1

    def engine_step():
        nonlocal engine_steps
        engine_steps += 1
        return []

    def run_iterations():
        try:
            for _ in range(3):
                results.append(session.step())
        except BaseException as error:
            failures.append(error)
        finally:
            returned.set()

    monkeypatch.setattr(subprocess, "run", management_command)
    recording = Recording(tmp_path / "run", run_id="responsive-memory")
    recorder = GpuMemoryRecorder(recording, source=NvidiaSmiMemorySource(), interval_ms=100 if management_in_flight else 100_000)
    session = VllmSession(SimpleNamespace(step=engine_step), None, recording)
    session._memory = recorder
    owner = threading.Thread(target=run_iterations, name="test-engine-owner")
    try:
        recorder.start()
        recorder.set_phase("inference", "capture_ready")
        block_queries.set()
        if management_in_flight:
            assert blocked.wait(2), "the background sampler did not start its next management query"
            recorder.sample()  # A concurrent sample must skip the already active query.
        owner.start()
        assert returned.wait(0.5), "model iterations waited for an external GPU management query"
        assert engine_steps == 3 and results == [[], [], []] and not failures
        assert "test-engine-owner" not in query_threads
        if not management_in_flight:
            assert "gpu-memory-recorder" not in query_threads
    finally:
        block_queries.clear()
        release.set()
        if owner.ident is not None:
            owner.join(timeout=2)
        recorder.set_phase("cleanup", "session.close")
        recorder.close()
        recording.close({"completion_status": "completed", "gpu_memory": recorder.record})

    events = [json.loads(line) for line in (recording.directory / "events.jsonl").read_text().splitlines()]
    samples = [event for event in events if event["event_type"] == "gpu_memory_sample"]
    assert all(sample["status"] == "observed" for sample in samples)
    assert samples[0]["phase"] == "preparation" and samples[-1]["phase"] == "cleanup"
    assert [event["phase"] for event in events if event["event_type"] == "gpu_memory_phase"] == ["inference", "cleanup"]
    assert peak_queries == 1 and active_queries == 0
    assert recorder.record["status"] == "completed" and not recorder._thread.is_alive()
    assert "gpu-memory-recorder" in query_threads  # The bounded shutdown also samples on this worker.


@pytest.mark.parametrize("blocked_during", ["periodic", "final"])
def test_sampler_timeout_retires_publication_before_collector_closes(tmp_path, monkeypatch, blocked_during):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    collector_retired, late_publications, failures = threading.Event(), [], []

    class Source:
        name = "blocking_management_api"
        calls = 0

        def sample(self, pid):
            self.calls += 1
            if self.calls > 1:
                entered.set()
                assert release.wait(5)
            return []

    recording = Recording(tmp_path / "timeout", run_id="memory-timeout")
    event = recording.event

    def guarded_event(*args, **kwargs):
        if collector_retired.is_set():
            late_publications.append((args, kwargs))
        return event(*args, **kwargs)

    monkeypatch.setattr(recording, "event", guarded_event)
    monkeypatch.setattr("runner.gpu_memory.SAMPLER_SHUTDOWN_TIMEOUT_SECONDS", 0.02)
    source = Source()
    recorder = GpuMemoryRecorder(recording, source=source, interval_ms=10 if blocked_during == "periodic" else 100_000)
    session = VllmSession(SimpleNamespace(), None, recording)
    session._memory = recorder

    def close_session():
        try:
            session.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()

    closer = threading.Thread(target=close_session)
    try:
        recorder.start()
        if blocked_during == "periodic":
            assert entered.wait(2)
        closer.start()
        assert entered.wait(2), "the final management query did not run"
        assert closed.wait(0.5), "session cleanup waited for the blocked management query"
        assert not failures
        assert recorder.record["status"] == "partial" and recorder.record["sampler_status"] == "shutdown_timeout"
        assert recorder.record["final_sample_status"] == "not_recorded" and recorder.record["shutdown_timeout_seconds"] == 0.02
        assert "late results are discarded" in recorder.record["reason"]
        metadata = json.dumps(recorder.record, sort_keys=True)
        recorder.close()  # Repeated close must not start another query or change its terminal metadata.
        recorder.sample()
        assert source.calls == 2 and json.dumps(recorder.record, sort_keys=True) == metadata
        collector_retired.set()
        manifest = recording.close({"completion_status": "completed", "gpu_memory": session.gpu_memory_record})
        persisted = (recording.directory / "events.jsonl").read_bytes()
        assert manifest["gpu_memory"]["status"] == "partial"
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=2)
        if recorder._thread is not None:
            recorder._thread.join(timeout=2)
        recorder.close()
        if not recording._closed:
            recording.close({"completion_status": "failed", "gpu_memory": recorder.record})

    assert not late_publications and not recorder._thread.is_alive()
    assert (recording.directory / "events.jsonl").read_bytes() == persisted
    assert json.dumps(recorder.record, sort_keys=True) == metadata


def test_weight_format_needs_native_config_and_loaded_types():
    bf16 = {"storages": [{"categories": ["weights"], "dtypes": ["bfloat16"]}]}
    config = SimpleNamespace(dtype="torch.bfloat16", quantization=None)
    assert observed_weight_storage(config, bf16)["format"] == "bf16"
    config.quantization = "fp8"
    assert observed_weight_storage(config, bf16)["status"] == "unresolved"
    fp8 = {"storages": [{"categories": ["weights"], "dtypes": ["float8_e4m3fn", "bfloat16"]}]}
    assert observed_weight_storage(config, fp8)["format"] == "fp8"
    packed = {"storages": [{"categories": ["weights"], "dtypes": ["uint8"]}]}
    config.quantization = "compressed-tensors"
    assert observed_weight_storage(config, packed)["format"] is None


def test_lora_metadata_is_observed_from_resolved_artifact_and_identity_changes(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "target_modules": ["q_proj", "v_proj"], "rank_pattern": {"v_proj": 8}}))
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"test-version-one")
    first = _lora_artifact_metadata(str(tmp_path))
    assert first["configuration_status"] == "observed" and first["rank"] == 4
    assert first["target_modules"] == ["q_proj", "v_proj"] and first["rank_pattern"] == {"v_proj": 8}
    assert len(first["artifact_sha256"][weights.name]) == 64
    weights.write_bytes(b"test-version-two")
    assert _lora_artifact_metadata(str(tmp_path))["artifact_sha256"][weights.name] != first["artifact_sha256"][weights.name]
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": -1, "target_modules": ["q_proj"]}))
    assert _lora_artifact_metadata(str(tmp_path))["configuration_status"] == "unavailable"


@pytest.mark.parametrize("dtype", ["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"])
def test_kv_precision_is_preserved_in_canonical_and_engine_configuration(dtype):
    payload = {"requests": [{"request_id": "a", "prompt": "hello", "max_output_tokens": 1, "arrival_offset_ms": 0}],
               "kv_cache_dtype": "auto", "vllm_configuration": {"kv_cache_dtype": dtype}}
    requests, plan = parse_run_workload(payload)
    config = parse_run_configuration(payload, requests, plan)
    body = SimpleNamespace()
    apply_legacy_fields(body, config)
    assert body.kv_cache_dtype == config["vllm"]["kv_cache_dtype"] == dtype
    if dtype != "auto":
        assert any("KV precision" in reason for reason in launch_blockers(config, requests, "fixture"))
    payload["vllm_configuration"]["kv_cache_dtype"] = "fp4_guessed"
    with pytest.raises(ConfigurationError, match="KV cache dtype"):
        parse_run_configuration(payload, requests, plan)
