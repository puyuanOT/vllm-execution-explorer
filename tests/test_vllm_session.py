import os
import sys
import json
import shutil
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from runner.vllm_session import (
    VllmSession,
    _EngineSessionLease,
    _LoRARoute,
    _add_engine_request,
    _build_engine,
    _expose_venv_tools,
    _huggingface_auth,
    _make_lora_request,
    _resolve_lora_path,
    _release_unused_cuda_memory,
    _stable_lora_ids,
    _synthetic_token_ids,
)
from runner.workload import WorkRequest


def test_recorded_request_ids_stay_the_submitted_ids():
    assert os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] == "1"


def test_live_session_requires_a_model_before_revision_resolution():
    try:
        VllmSession.open(None, SimpleNamespace())
        raise AssertionError("expected missing model to fail")
    except RuntimeError as exc:
        assert "model repository" in str(exc)


def test_bad_inventory_preparation_closes_engine_without_claiming_capture_ready(monkeypatch):
    from vllm_030.compatibility import CompatibilityError
    calls = []
    events = []
    core = SimpleNamespace(scheduler=object())
    engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=core, shutdown=lambda: calls.append("shutdown")),
                             reset_prefix_cache=lambda: True, vllm_config=SimpleNamespace(speculative_config=None),
                             get_tokenizer_group=lambda: SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda _text: [1])))
    monkeypatch.setattr("runner.vllm_session._build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr("runner.vllm_session._warmup", lambda _engine: None)
    monkeypatch.setattr("runner.vllm_session._import_vllm_version", lambda: "0.30.0")
    def reject(*_args, **_kwargs):
        raise CompatibilityError("Initial physical KV inventory is incomplete.")
    from importlib import import_module
    monkeypatch.setattr(import_module("vllm_030.install"), "install", reject)
    recording = SimpleNamespace(directory=Path("/unused"), event=lambda event_type, **_fields: events.append(event_type))
    body = SimpleNamespace(model="org/model", model_revision="a" * 40, tokenizer_revision=None, tokenizer=None)
    try:
        VllmSession.open(recording, body)
        raise AssertionError("expected inventory rejection")
    except CompatibilityError as error:
        assert "incomplete" in str(error)
    assert calls == ["shutdown"]
    assert "capture_ready" not in events


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    """Expose cleanup order and the native cached, dynamically read switch."""
    from importlib import import_module

    calls, events = [], []
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    @cache
    def environment(name):
        assert name == "VLLM_USE_V2_MODEL_RUNNER"
        return os.environ.get(name) != "0"
    environment("VLLM_USE_V2_MODEL_RUNNER")
    monkeypatch.setitem(sys.modules, "vllm.envs", SimpleNamespace(__getattr__=environment))
    class Config:
        speculative_config = None
        @property
        def use_v2_model_runner(self):
            return environment("VLLM_USE_V2_MODEL_RUNNER")
    config = Config()
    core = SimpleNamespace(scheduler=SimpleNamespace(waiting=[], running=[]))
    engine = SimpleNamespace(vllm_config=config, reset_prefix_cache=lambda: True,
        engine_core=SimpleNamespace(engine_core=core, shutdown=lambda: calls.append(("shutdown", config.use_v2_model_runner))),
        _finalizer=lambda: calls.append(("finalizer", config.use_v2_model_runner)))
    def build(*_args, **_kwargs):
        calls.append(("build", config.use_v2_model_runner))
        return engine
    def describe(self, *_args):
        calls.append(("describe", config.use_v2_model_runner))
        self.architecture_record = {"supported": True, "model_type": "qwen3"}
    class Profile:
        def __init__(self, *_args):
            self.record = {"status": "ready"}
        def start(self):
            calls.append(("profile_start", config.use_v2_model_runner))
        def finish(self):
            calls.append(("profile_finish", config.use_v2_model_runner))
    adapter = SimpleNamespace(uninstall=lambda: calls.append(("uninstall", config.use_v2_model_runner)))
    monkeypatch.setattr("runner.vllm_session._build_engine", build)
    monkeypatch.setattr("runner.vllm_session._warmup", lambda _engine: calls.append(("warmup", config.use_v2_model_runner)))
    monkeypatch.setattr("runner.vllm_session._import_vllm_version", lambda: "0.30.0")
    monkeypatch.setattr("runner.vllm_session._loaded_tokenizer", lambda _engine: SimpleNamespace(decode=lambda ids: str(ids)))
    monkeypatch.setattr(import_module("vllm_030.install"), "install", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(VllmSession, "_describe", describe)
    monkeypatch.setattr("runner.profiling.DiagnosticProfile", Profile)
    monkeypatch.setattr("runner.vllm_session._release_unused_cuda_memory", lambda: calls.append(("release_memory", config.use_v2_model_runner)))
    body = SimpleNamespace(model="org/model", model_revision="a" * 40, tokenizer=None, tokenizer_revision=None,
        profiling_configuration={"mode": "eager_diagnostic", "max_steps": 8})
    recording = SimpleNamespace(directory=tmp_path, event=lambda kind, **_fields: events.append(kind))
    return SimpleNamespace(body=body, recording=recording, engine=engine, core=core, adapter=adapter,
        profile=Profile, calls=calls, events=events, environment=environment)


@pytest.mark.parametrize("stage", ["warmup", "reset_error", "reset_unverified", "core", "speculative",
    "version", "tokenizer", "install", "configuration", "sampling", "describe", "profiler_construct", "profiler_start", "capture_ready"])
def test_every_post_construction_preparation_failure_shuts_down_and_restores_selection(monkeypatch, preparation, stage):
    from importlib import import_module
    probe = preparation
    def fail(*_args, **_kwargs):
        raise RuntimeError(f"failed at {stage}")
    if stage in {"warmup", "core", "version", "tokenizer"}:
        name = {"warmup": "_warmup", "core": "_engine_core", "version": "_import_vllm_version", "tokenizer": "_loaded_tokenizer"}[stage]
        monkeypatch.setattr(f"runner.vllm_session.{name}", fail)
    elif stage.startswith("reset"):
        probe.engine.reset_prefix_cache = fail if stage == "reset_error" else lambda: False
    elif stage == "speculative":
        probe.engine.vllm_config.speculative_config = object()
    elif stage == "install":
        monkeypatch.setattr(import_module("vllm_030.install"), "install", fail)
    elif stage == "configuration":
        probe.body.requests = [{"request_id": "x", "prompt": "hello", "arrival_offset_ms": 0, "max_output_tokens": 1}]
        monkeypatch.setattr("runner.configuration.parse_run_configuration", fail)
    elif stage == "sampling":
        probe.body.sampling = {"temperature": -1}
    elif stage == "describe":
        monkeypatch.setattr(VllmSession, "_describe", fail)
    elif stage == "profiler_construct":
        monkeypatch.setattr("runner.profiling.DiagnosticProfile", fail)
    elif stage == "profiler_start":
        monkeypatch.setattr(probe.profile, "start", fail)
    else:
        def event(kind, **_fields):
            if kind == "capture_ready":
                fail()
            probe.events.append(kind)
        probe.recording.event = event
    with pytest.raises((RuntimeError, ValueError)):
        VllmSession.open(probe.recording, probe.body)
    assert probe.calls.count(("shutdown", False)) == 1
    assert probe.calls.count(("finalizer", False)) == 1
    assert probe.calls[-1] == ("release_memory", False)
    assert "capture_ready" not in probe.events
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert probe.environment("VLLM_USE_V2_MODEL_RUNNER") is True
    # Failed preparation relinquishes ownership, permitting a subsequent run.
    lease = _EngineSessionLease(False)
    lease.release()


@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_diagnostic_selection_is_owned_until_close_and_restores_exact_prior_environment(monkeypatch, preparation, previous):
    if previous is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", previous)
    session = VllmSession.open(preparation.recording, preparation.body)
    assert session.engine.vllm_config.use_v2_model_runner is False
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    with pytest.raises(RuntimeError, match="Another in-process"):
        _EngineSessionLease(False)
    session.close()
    assert session.engine is None and session._profile is None and session.adapter is None
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == previous
    assert preparation.environment("VLLM_USE_V2_MODEL_RUNNER") == (previous != "0")
    assert preparation.calls[-4:] == [("uninstall", False), ("shutdown", False), ("finalizer", False), ("release_memory", False)]
    session.close()
    assert preparation.calls.count(("shutdown", False)) == 1
    preparation.body.profiling_configuration = {"mode": "off"}
    subsequent = VllmSession.open(preparation.recording, preparation.body)
    assert subsequent.engine.vllm_config.use_v2_model_runner == (previous != "0")
    subsequent.close()
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == previous


@pytest.mark.parametrize("stage", ["profile_finish", "uninstall", "shutdown", "finalizer", "release_memory"])
def test_close_failures_still_release_engine_references_and_runner_lease(monkeypatch, preparation, stage):
    session = VllmSession.open(preparation.recording, preparation.body)
    def fail(*_args, **_kwargs):
        preparation.calls.append((stage, False))
        raise RuntimeError(stage)
    if stage == "profile_finish":
        session._profile.finish = fail
    elif stage == "uninstall":
        session.adapter.uninstall = fail
    elif stage == "shutdown":
        session.engine.engine_core.shutdown = fail
    elif stage == "finalizer":
        session.engine._finalizer = fail
    else:
        monkeypatch.setattr("runner.vllm_session._release_unused_cuda_memory", fail)
    with pytest.raises(RuntimeError, match=stage):
        session.close()
    assert session.engine is None and session._profile is None and session.adapter is None
    assert ("finalizer", False) in preparation.calls
    assert ("release_memory", False) in preparation.calls
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert preparation.environment("VLLM_USE_V2_MODEL_RUNNER") is True
    lease = _EngineSessionLease(False)
    lease.release()


def test_cuda_cache_release_resets_compiler_then_collects_cycles_without_initializing_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr("runner.vllm_session.gc.collect", lambda: calls.append("gc"))
    cuda = SimpleNamespace(is_initialized=lambda: False, empty_cache=lambda: calls.append("empty_cache"))
    compiler = SimpleNamespace(reset=lambda: calls.append("compiler_reset"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, compiler=compiler))
    _release_unused_cuda_memory()
    assert calls == ["gc"]
    cuda.is_initialized = lambda: True
    _release_unused_cuda_memory()
    assert calls == ["gc", "compiler_reset", "gc", "empty_cache"]


def test_compiler_reset_failure_still_collects_cycles_and_empties_unused_cuda_cache(monkeypatch):
    calls = []
    def reset():
        calls.append("compiler_reset")
        raise RuntimeError("compiler reset failed")
    monkeypatch.setattr("runner.vllm_session.gc.collect", lambda: calls.append("gc"))
    cuda = SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: calls.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, compiler=SimpleNamespace(reset=reset)))
    with pytest.raises(RuntimeError, match="compiler reset failed"):
        _release_unused_cuda_memory()
    assert calls == ["compiler_reset", "gc", "empty_cache"]


def test_engine_process_finds_venv_tools_even_when_python_is_a_symlink(monkeypatch, tmp_path):
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    executable = bindir / "python"
    executable.symlink_to(sys.executable)
    (bindir / "ninja").write_text("venv tool")
    (bindir / "ninja").chmod(0o700)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("PATH", "/usr/bin")
    _expose_venv_tools()
    assert shutil.which("ninja") == str(bindir / "ninja")
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bindir)


def test_synthetic_prompt_uses_exactly_the_requested_token_count():
    tokenizer = SimpleNamespace(encode=lambda _text, add_special_tokens=False: [17, 18])
    group = SimpleNamespace(tokenizer=tokenizer)
    engine = SimpleNamespace(get_tokenizer_group=lambda: group)
    assert _synthetic_token_ids(engine, 257) == [17] * 257


def test_lora_integer_ids_are_positive_and_stable_across_configuration_order():
    left = _stable_lora_ids([{"id": "zeta"}, {"id": "alpha"}])
    right = _stable_lora_ids([{"id": "alpha"}, {"id": "zeta"}])
    assert left == right == {"alpha": 1, "zeta": 2}


def test_lora_request_uses_the_installed_factory_signature():
    class Request:
        def __init__(self, *, lora_name: str, lora_int_id: int, lora_path: str) -> None:
            self.values = (lora_name, lora_int_id, lora_path)

    request = _make_lora_request(Request, "legal", 3, "/models/legal")
    assert request.values == ("legal", 3, "/models/legal")


def test_engine_admission_passes_only_configured_lora_request(monkeypatch):
    class SamplingParams:
        def __init__(self, **values) -> None:
            self.values = values

    package = SimpleNamespace()
    sampling_module = SimpleNamespace(SamplingParams=SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_module)

    tokenizer = SimpleNamespace(encode=lambda _text: [11, 12])
    calls: list[tuple[tuple, dict]] = []

    def add_request(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    engine = SimpleNamespace(
        add_request=add_request,
        get_tokenizer_group=lambda: SimpleNamespace(tokenizer=tokenizer),
    )
    events: list[tuple[str, dict]] = []
    recording = SimpleNamespace(event=lambda event_type, **values: events.append((event_type, values)))
    lora_request = object()
    route = _LoRARoute("legal", 1, "/models/legal", {"kind": "execution_host"}, lora_request)
    session = VllmSession(engine, None, recording, lora_routes={"legal": route})
    resolution = session.add_work_request(
        WorkRequest("a", "hello", 0, 3, adapter_id="legal")
    )
    session.add_work_request(WorkRequest("b", "base", 0, 3, adapter_id="base"))

    assert resolution["resolved_input_tokens"] == 2 and resolution["input_token_ids"] == [11, 12]
    assert resolution["request_configuration"] == {
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 3}, "max_output_tokens": 3,
        "adapter_id": "legal", "route_kind": "lora_adapter", "lora_int_id": 1,
        "lora_route": {"adapter_id": "legal", "lora_int_id": 1, "resolved_path": "/models/legal"},
        "priority": 0, "input_mode": "prompt", "target_input_tokens": None, "resolved_input_tokens": 2,
    }
    assert calls[0][1] == {"lora_request": lora_request}
    assert calls[1][1] == {}
    assert [event[1]["adapter_id"] for event in events] == ["legal", "base"]


def test_effective_request_settings_match_actual_sampling_route_and_synthetic_input(monkeypatch):
    parameters = []
    class SamplingParams:
        def __init__(self, **values):
            parameters.append(values)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(SamplingParams=SamplingParams))
    tokenizer = SimpleNamespace(encode=lambda _text, **_kwargs: [11, 12])
    engine = SimpleNamespace(add_request=lambda *args, **kwargs: None, get_tokenizer_group=lambda: SimpleNamespace(tokenizer=tokenizer))
    recording = SimpleNamespace(event=lambda *_args, **_kwargs: None)
    route = _LoRARoute("legal", 3, "/models/legal", {"kind": "execution_host"}, object())
    session = VllmSession(engine, None, recording, lora_routes={"legal": route})
    session._sampling = {"temperature": 0.3, "top_p": 0.9, "seed": 42, "stop": ["END"]}
    first = session.add_work_request(WorkRequest("a", "normal", 0, 5, adapter_id="legal", sampling={"temperature": 0.7, "top_p": 0.4, "seed": 7, "stop": ["NEXT"]}, priority=-2))
    second = session.add_work_request(WorkRequest("b", "", 0, 2, input_mode="synthetic", target_input_tokens=5))
    assert first["request_configuration"]["sampling"] == parameters[0] == {"temperature": 0.7, "top_p": 0.4, "seed": 7, "stop": ["NEXT"], "max_tokens": 5}
    assert first["request_configuration"]["priority"] == -2
    assert first["request_configuration"]["lora_route"]["lora_int_id"] == 3
    assert second["request_configuration"]["sampling"] == parameters[1] == {"temperature": 0.3, "top_p": 0.9, "seed": 42, "stop": ["END"], "max_tokens": 2}
    assert second["input_token_ids"] == [11] * 5
    assert second["request_configuration"]["input_mode"] == "synthetic"
    assert second["request_configuration"]["target_input_tokens"] == second["request_configuration"]["resolved_input_tokens"] == 5
    assert second["request_configuration"]["route_kind"] == "base_model" and second["request_configuration"]["lora_route"] is None


def test_unconfigured_lora_is_rejected_before_engine_admission(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=lambda **_values: object()),
    )
    engine = SimpleNamespace(add_request=lambda *_args, **_kwargs: None)
    session = VllmSession(engine, None, SimpleNamespace(event=lambda *_args, **_kwargs: None))
    try:
        session.add_work_request(WorkRequest("a", "hello", 0, 3, adapter_id="missing"))
        raise AssertionError("expected the unconfigured adapter to be rejected")
    except RuntimeError as exc:
        assert "unconfigured" in str(exc)


def test_huggingface_adapter_resolution_uses_ephemeral_token(monkeypatch, tmp_path: Path):
    resolved = tmp_path / "snapshot"
    resolved.mkdir()
    observed: dict[str, str | None] = {}

    def snapshot_download(*, repo_id: str, revision: str, token: str | None = None) -> str:
        observed.update(repo_id=repo_id, revision=revision, token=token)
        return str(resolved)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    path = _resolve_lora_path(
        {"kind": "huggingface", "locator": "org/legal", "revision": "commit-sha"},
        token="hf_ephemeral",
    )
    assert path == str(resolved.resolve())
    assert observed == {"repo_id": "org/legal", "revision": "commit-sha", "token": "hf_ephemeral"}


def test_huggingface_auth_is_restored_after_engine_construction(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "previous")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with _huggingface_auth("ephemeral"):
        assert os.environ["HF_TOKEN"] == "ephemeral"
        assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "ephemeral"
    assert os.environ["HF_TOKEN"] == "previous"
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ


def test_engine_rejects_lora_when_add_request_has_no_route_parameter():
    class Engine:
        def add_request(self, request_id, prompt, params) -> None:
            del request_id, prompt, params

    try:
        _add_engine_request(Engine(), "a", {}, object(), object())
        raise AssertionError("expected unsupported routing to fail")
    except RuntimeError as exc:
        assert "cannot route LoRA" in str(exc)


def test_engine_args_enable_lora_and_size_the_adapter_cache(monkeypatch, tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 24}), encoding="utf-8")
    captured: dict[str, object] = {}

    class EngineArgs:
        def __init__(
            self,
            *,
            enable_lora=False,
            max_loras=1,
            max_cpu_loras=None,
            max_lora_rank=16,
            **values,
        ) -> None:
            captured.update(
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_cpu_loras=max_cpu_loras,
                max_lora_rank=max_lora_rank,
                **values,
            )

    class LLMEngine:
        @staticmethod
        def from_engine_args(args, *, enable_multiprocessing):
            captured["engine_args"] = args
            captured["enable_multiprocessing"] = enable_multiprocessing
            return "engine"

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    engine_package = ModuleType("vllm.engine")
    engine_package.__path__ = []
    v1_package = ModuleType("vllm.v1")
    v1_package.__path__ = []
    v1_engine_package = ModuleType("vllm.v1.engine")
    v1_engine_package.__path__ = []
    args_module = ModuleType("vllm.engine.arg_utils")
    args_module.EngineArgs = EngineArgs
    llm_module = ModuleType("vllm.v1.engine.llm_engine")
    llm_module.LLMEngine = LLMEngine
    for name, module in {
        "vllm": vllm,
        "vllm.engine": engine_package,
        "vllm.engine.arg_utils": args_module,
        "vllm.v1": v1_package,
        "vllm.v1.engine": v1_engine_package,
        "vllm.v1.engine.llm_engine": llm_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    route = _LoRARoute("legal", 1, str(adapter), {"kind": "execution_host"}, object())
    body = SimpleNamespace(
        model="org/model",
        tokenizer="org/model",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        token_budget=128,
        quantization="none",
    )
    assert _build_engine(body, {"legal": route}) == "engine"
    assert captured["enable_lora"] is True
    assert captured["max_loras"] == 1
    assert captured["max_cpu_loras"] == 1
    assert captured["max_lora_rank"] == 32
    assert captured["enable_multiprocessing"] is False
    monkeypatch.setattr("runner.profiling.engine_profiler_config", lambda directory, steps: {"directory": str(directory), "max_steps": steps})
    body.profiling_configuration = {"mode": "eager_diagnostic", "max_steps": 3}
    assert _build_engine(body, {"legal": route}, profiling_directory=tmp_path) == "engine"
    assert captured["enforce_eager"] is True
    assert captured["profiler_config"] == {"directory": str(tmp_path), "max_steps": 3}
