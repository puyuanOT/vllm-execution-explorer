"""In-process vLLM 0.30 session for one CUDA Linux host.

The explorer process can be a Mac. This session is the engine process: vLLM
0.30.x, multiprocessing off, prefix cache reset checked before recording starts.
Speculative decoding stays off. A fixture run never constructs this object.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# vLLM 0.30 appends a random suffix to request ids. Record the id submitted for the run.
os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

from runner.fixture_engine import ObservedChunk
from runner.targets import LocalTarget, decide_target, execution_target_record
from runner.workload import WorkRequest, canonical_sampling


@dataclass(frozen=True)
class _LoRARoute:
    """One configured adapter after resolving it on the execution host."""

    adapter_id: str
    lora_int_id: int
    resolved_path: str
    source: dict[str, Any]
    request: Any


class VllmSession:
    execution_kind = "gpu"
    dispatch_transport = "in_process_direct_engine"

    def __init__(
        self,
        engine: Any,
        adapter: Any,
        recording: Any,
        *,
        lora_routes: dict[str, _LoRARoute] | None = None,
    ) -> None:
        self.engine = engine
        self.adapter = adapter
        self.recording = recording
        self._request_ids: list[str] = []
        self._seen_tokens: dict[str, list[int]] = {}
        self._seen_text: dict[str, str] = {}
        self._closed = False
        self.execution_target: dict[str, Any] | None = None
        self.cache_geometry: dict[str, Any] | None = None
        self.engine_settings: dict[str, Any] | None = None
        self.engine_record: dict[str, Any] | None = None
        self.model_record: dict[str, Any] | None = None
        self.tokenizer_record: dict[str, Any] | None = None
        self.hardware_record: dict[str, Any] | None = None
        self.sampling_record: dict[str, Any] | None = None
        self.initial_cache_state: dict[str, Any] | None = None
        self.preparation_observations: dict[str, Any] | None = None
        self._sampling = canonical_sampling(None)
        self._lora_routes = dict(lora_routes or {})

    @classmethod
    def open(cls, recording: Any, body: Any) -> VllmSession:
        revision = getattr(body, "model_revision", None)
        tokenizer_revision = getattr(body, "tokenizer_revision", None)
        if getattr(body, "model_source_kind", "huggingface") == "huggingface" and (
            not revision or not tokenizer_revision
        ):
            raise RuntimeError("A Hugging Face live run needs model_revision and tokenizer_revision.")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        _expose_venv_tools()
        preparation_started = time.monotonic_ns()
        lora_routes = _prepare_lora_routes(body)
        with _huggingface_auth(getattr(body, "huggingface_token", None)):
            engine = _build_engine(body, lora_routes)
        engine_ready = time.monotonic_ns()
        _warmup(engine)
        warmup_completed = time.monotonic_ns()
        reset_started = time.monotonic_ns()
        if engine.reset_prefix_cache() is not True:
            raise RuntimeError("prefix cache reset was not verified")
        reset_completed = time.monotonic_ns()
        core = _engine_core(engine)
        _reject_speculative(engine)
        version = _import_vllm_version()
        from vllm_030.install import install

        adapter = install(core, recording, vllm_version=version, check_sources=True)
        session = cls(engine, adapter, recording, lora_routes=lora_routes)
        session._sampling = canonical_sampling(getattr(body, "sampling", None))
        session.preparation_observations = {
            "clock_domain": "engine_process:time.monotonic_ns",
            "preparation_started_monotonic_ns": preparation_started,
            "engine_ready_monotonic_ns": engine_ready,
            "warmup_completed_monotonic_ns": warmup_completed,
            "cache_reset_started_monotonic_ns": reset_started,
            "cache_reset_completed_monotonic_ns": reset_completed,
            "capture_ready_monotonic_ns": time.monotonic_ns(),
        }
        session._describe(body, version)
        recording.event(
            "capture_ready",
            cache_reset_verified=True,
            warmup_completed_before_reset=warmup_completed <= reset_started,
            waiting_queue_size=_queue_size(getattr(core.scheduler, "waiting", None)),
            running_queue_size=_queue_size(getattr(core.scheduler, "running", None)),
            observation_boundary="after_instrumentation_install",
        )
        return session

    def add_request(self, request_id: str, prompt: str, max_output_tokens: int = 16) -> None:
        from vllm.sampling_params import SamplingParams

        if not isinstance(prompt, str):
            raise RuntimeError("the live engine takes text prompts")
        params = SamplingParams(**_sampling_kwargs(self._sampling, max_output_tokens))
        self.engine.add_request(request_id, prompt, params)
        self._request_ids.append(request_id)
        self._seen_tokens[request_id] = []
        self._seen_text[request_id] = ""

    def add_work_request(self, request: WorkRequest) -> dict[str, Any]:
        from vllm.sampling_params import SamplingParams

        adapter_id = request.adapter_id or "base"
        lora = None if adapter_id == "base" else self._lora_routes.get(adapter_id)
        if adapter_id != "base" and lora is None:
            raise RuntimeError(f"request selects unconfigured LoRA adapter {adapter_id!r}")
        sampling = dict(self._sampling)
        if request.sampling:
            sampling.update(request.sampling)
        params = SamplingParams(**_sampling_kwargs(sampling, request.max_output_tokens))
        if request.input_mode == "synthetic":
            if request.target_input_tokens is None:
                raise RuntimeError("a synthetic request is missing its target input length")
            prompt: Any = {
                "prompt_token_ids": _synthetic_token_ids(
                    self.engine,
                    request.target_input_tokens,
                    None if lora is None else lora.request,
                ),
            }
            resolved = request.target_input_tokens
        else:
            token_ids = _prompt_token_ids(
                self.engine,
                request.prompt,
                None if lora is None else lora.request,
            )
            resolved = len(token_ids)
            if request.resolved_input_tokens is not None and request.resolved_input_tokens != resolved:
                raise RuntimeError(
                    f"{request.request_id} measured {request.resolved_input_tokens} input tokens, "
                    f"but the execution-host tokenizer resolved {resolved}"
                )
            prompt = {"prompt_token_ids": token_ids}
        self.recording.event(
            "request_model_route",
            request_id=request.request_id,
            adapter_id=adapter_id,
            lora_int_id=None if lora is None else lora.lora_int_id,
            route_kind="base_model" if lora is None else "lora_adapter",
            observation_boundary="immediately_before_engine_add_request",
        )
        _add_engine_request(
            self.engine,
            request.request_id,
            prompt,
            params,
            None if lora is None else lora.request,
        )
        self._request_ids.append(request.request_id)
        self._seen_tokens[request.request_id] = []
        self._seen_text[request.request_id] = ""
        return {
            "resolved_input_tokens": resolved,
            "adapter_id": adapter_id,
            "lora_int_id": None if lora is None else lora.lora_int_id,
        }

    def step(self) -> list[ObservedChunk]:
        outputs = self.engine.step()
        return _chunks(self, outputs)

    def has_unfinished(self) -> bool:
        return bool(self.engine.has_unfinished_requests())

    def cancel(self) -> None:
        if self._request_ids:
            self.engine.abort_request(list(self._request_ids))

    def disabled_step_times(self, prompt: str, steps: int) -> list[int]:
        """Time host steps after the adapter is removed. This is not a benchmark."""
        if self.adapter is not None:
            self.adapter.uninstall()
            self.adapter = None
        if self._request_ids:
            self.engine.abort_request(list(self._request_ids))
        if self.engine.reset_prefix_cache() is not True:
            raise RuntimeError("prefix cache reset was not verified before the disabled timing run")
        self.add_request("overhead-disabled", prompt, max(steps, 1))
        timings: list[int] = []
        for _ in range(max(steps, 1)):
            if not self.has_unfinished():
                break
            started = time.monotonic_ns()
            self.engine.step()
            timings.append(time.monotonic_ns() - started)
        if not timings:
            raise RuntimeError("the disabled timing run produced no steps")
        return timings

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.adapter is not None:
            self.adapter.uninstall()
            self.adapter = None
        client = getattr(self.engine, "engine_core", None)
        shutdown = getattr(client, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def _describe(self, body: Any, version: str) -> None:
        decision = decide_target(
            LocalTarget(),
            client_platform=sys.platform,
            cuda_available=True,
            vllm_version=version,
        )
        if not decision.ok:
            raise RuntimeError(decision.message)
        self.execution_target = execution_target_record(decision)
        core = _engine_core(self.engine)
        scheduler = core.scheduler
        pool = _block_pool(scheduler)
        free = pool.get_num_free_blocks() if hasattr(pool, "get_num_free_blocks") else None
        block_size = getattr(scheduler, "block_size", None)
        if block_size is None:
            block_size = self.engine.vllm_config.cache_config.block_size
        num_blocks = getattr(pool, "num_gpu_blocks", None)
        if num_blocks is None or free is None or block_size is None:
            raise RuntimeError("the block pool did not expose cache geometry")
        self.cache_geometry = {
            "block_size": int(block_size),
            "num_gpu_blocks": int(num_blocks),
            "initial_free_blocks": int(free),
        }
        config = self.engine.vllm_config
        scheduled = getattr(config.scheduler_config, "max_num_scheduled_tokens", None)
        batched = getattr(config.scheduler_config, "max_num_batched_tokens", None)
        self.engine_settings = {
            "async_scheduling": bool(getattr(config.scheduler_config, "async_scheduling", False)),
            "enable_prefix_caching": bool(config.cache_config.enable_prefix_caching),
            "enable_lora": bool(self._lora_routes),
            "max_num_seqs": int(config.scheduler_config.max_num_seqs),
            "max_num_scheduled_tokens": int(scheduled if scheduled is not None else batched),
            "max_model_len": int(config.model_config.max_model_len),
            "enable_chunked_prefill": bool(getattr(config.scheduler_config, "enable_chunked_prefill", False)),
            "gpu_memory_utilization": float(getattr(config.cache_config, "gpu_memory_utilization", 0.0)),
        }
        model = str(getattr(body, "model", None) or config.model_config.model)
        revision = None if body.model_revision is None else str(body.model_revision)
        tokenizer_revision = None if body.tokenizer_revision is None else str(body.tokenizer_revision)
        self.engine_record = {"name": "vllm", "version": version, "upstream_revision": None}
        self.model_record = {
            "id": model,
            "revision": revision,
            "dtype": str(getattr(body, "dtype", "auto")),
            "quantization": str(getattr(body, "quantization", "none")),
            "lora_adapters": [
                {
                    "id": route.adapter_id,
                    "lora_int_id": route.lora_int_id,
                    "resolved_path": route.resolved_path,
                    "source": dict(route.source),
                }
                for route in sorted(self._lora_routes.values(), key=lambda item: item.lora_int_id)
            ],
            "note": "Loaded by the in-process vLLM 0.30 engine on this CUDA host.",
        }
        self.tokenizer_record = {
            "id": str(getattr(body, "tokenizer", None) or model),
            "revision": tokenizer_revision,
        }
        self.hardware_record = {"gpu": _gpu_name()}
        self.sampling_record = dict(self._sampling)
        self.initial_cache_state = {
            "kind": "fresh",
            "reset_required": True,
            "reset_verified": True,
            "warmup_completed_before_reset": True,
            "capture_ready_after_reset": True,
            "evidence": "vllm.reset_prefix_cache returned true after warmup",
            "clock_domain": "engine_process:time.monotonic_ns",
            **(self.preparation_observations or {}),
        }


def _expose_venv_tools() -> None:
    """torch.compile looks up ninja on PATH. A venv python does not put its bin directory there."""
    bindir = os.path.dirname(os.path.realpath(sys.executable))
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if parts[:1] != [bindir]:
        os.environ["PATH"] = os.pathsep.join([bindir, *[part for part in parts if part != bindir]])


def _build_engine(body: Any, lora_routes: dict[str, _LoRARoute] | None = None) -> Any:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    model = str(getattr(body, "model", None) or "Qwen/Qwen3-4B-Instruct-2507")
    tokenizer = str(getattr(body, "tokenizer", None) or model)
    budget = int(getattr(body, "token_budget", 2048))
    # Qwen3-4B advertises a 256k context. An A10 cannot reserve that KV cache.
    # The explorer's runs stay inside the token budget, so the context only needs to cover it.
    desired = {
        "model": model,
        "tokenizer": tokenizer,
        "dtype": str(getattr(body, "dtype", "bfloat16")),
        "seed": 0,
        "enable_prefix_caching": bool(getattr(body, "prefix_caching", True)),
        "max_num_seqs": int(getattr(body, "max_num_seqs", 4)),
        "max_num_batched_tokens": budget,
        "max_num_scheduled_tokens": budget,
        "max_model_len": int(getattr(body, "max_model_len", None) or max(budget, 4096)),
        "enable_chunked_prefill": bool(getattr(body, "chunked_prefill", True)),
        "gpu_memory_utilization": float(getattr(body, "gpu_memory_utilization", 0.9)),
        "async_scheduling": False,
        "disable_log_stats": True,
        "trust_remote_code": False,
    }
    if getattr(body, "model_revision", None):
        desired["revision"] = str(body.model_revision)
    if getattr(body, "tokenizer_revision", None):
        desired["tokenizer_revision"] = str(body.tokenizer_revision)
    routes = lora_routes or {}
    if routes:
        desired["enable_lora"] = True
        desired["max_loras"] = len(routes)
        desired["max_cpu_loras"] = len(routes)
        rank = _max_adapter_rank(routes)
        if rank is not None:
            desired["max_lora_rank"] = rank
    quantization = str(getattr(body, "quantization", "none"))
    if quantization != "none":
        desired["quantization"] = quantization
    hf_token = getattr(body, "huggingface_token", None)
    if hf_token:
        desired["hf_token"] = str(hf_token)
    signature = inspect.signature(EngineArgs)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if routes and not accepts_kwargs:
        required_lora_args = {"enable_lora", "max_loras", "max_cpu_loras"}
        missing = sorted(required_lora_args.difference(signature.parameters))
        if missing:
            raise RuntimeError(
                "the installed vLLM EngineArgs cannot configure LoRA: " + ", ".join(missing)
            )
    accepted = {
        key: value
        for key, value in desired.items()
        if key in signature.parameters or accepts_kwargs
    }
    return LLMEngine.from_engine_args(EngineArgs(**accepted), enable_multiprocessing=False)


def _prepare_lora_routes(body: Any) -> dict[str, _LoRARoute]:
    configured = getattr(body, "lora_adapters", None) or []
    if not isinstance(configured, list):
        raise RuntimeError("the canonical LoRA adapter configuration must be a list")
    if not configured:
        return {}
    try:
        from vllm.lora.request import LoRARequest
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("this vLLM runtime does not expose LoRARequest") from exc

    token = getattr(body, "huggingface_token", None)
    integer_ids = _stable_lora_ids(configured)
    routes: dict[str, _LoRARoute] = {}
    for item in configured:
        if not isinstance(item, dict):
            raise RuntimeError("each canonical LoRA adapter must be an object")
        adapter_id = str(item.get("id") or "").strip()
        source = item.get("source")
        if not isinstance(source, dict):
            raise RuntimeError(f"LoRA adapter {adapter_id!r} has no canonical source")
        resolved_path = _resolve_lora_path(source, token=token)
        integer_id = integer_ids[adapter_id]
        request = _make_lora_request(LoRARequest, adapter_id, integer_id, resolved_path)
        routes[adapter_id] = _LoRARoute(
            adapter_id=adapter_id,
            lora_int_id=integer_id,
            resolved_path=resolved_path,
            source=_public_lora_source(source),
            request=request,
        )
    return routes


def _stable_lora_ids(configured: list[dict[str, Any]]) -> dict[str, int]:
    names: list[str] = []
    for item in configured:
        if not isinstance(item, dict):
            raise RuntimeError("each canonical LoRA adapter must be an object")
        name = str(item.get("id") or "").strip()
        if not name or name == "base":
            raise RuntimeError("configured LoRA adapter ids must be non-empty and cannot be 'base'")
        names.append(name)
    if len(set(names)) != len(names):
        raise RuntimeError("configured LoRA adapter ids must be unique")
    return {name: index for index, name in enumerate(sorted(names), start=1)}


def _resolve_lora_path(source: dict[str, Any], *, token: str | None) -> str:
    kind = str(source.get("kind") or "")
    locator = str(source.get("locator") or "").strip()
    if not locator:
        raise RuntimeError("a LoRA adapter source is missing its locator")
    if kind == "huggingface":
        revision = source.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            raise RuntimeError(f"Hugging Face LoRA adapter {locator!r} needs a pinned revision")
        try:
            from huggingface_hub import snapshot_download
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("huggingface_hub is required to resolve a Hugging Face LoRA adapter") from exc
        desired: dict[str, Any] = {"repo_id": locator, "revision": revision.strip()}
        signature = inspect.signature(snapshot_download)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if token:
            if "token" in signature.parameters or accepts_kwargs:
                desired["token"] = token
            elif "use_auth_token" in signature.parameters:
                desired["use_auth_token"] = token
            else:
                raise RuntimeError("the installed huggingface_hub cannot accept an ephemeral access token")
        resolved = snapshot_download(**desired)
        return _require_adapter_directory(str(resolved), label=f"Hugging Face LoRA adapter {locator!r}")
    if kind == "operator_local":
        transfer = source.get("transfer") or {}
        destination = transfer.get("destination") if isinstance(transfer, dict) else None
        path = str(destination or locator)
        return _require_adapter_directory(path, label="transferred LoRA adapter")
    if kind == "execution_host":
        return _require_adapter_directory(locator, label="execution-host LoRA adapter")
    raise RuntimeError(f"unsupported LoRA source kind {kind!r}")


def _require_adapter_directory(value: str, *, label: str) -> str:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} path does not exist: {value}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"{label} path is not a directory: {value}")
    return str(resolved)


def _make_lora_request(factory: Any, name: str, integer_id: int, path: str) -> Any:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        try:
            return factory(name, integer_id, path)
        except TypeError as exc:
            raise RuntimeError("the installed vLLM LoRARequest signature is unsupported") from exc
    parameters = signature.parameters
    accepts_kwargs = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    aliases = {
        "name": ("lora_name", "name"),
        "integer_id": ("lora_int_id", "adapter_id", "int_id"),
        "path": ("lora_path", "lora_local_path", "path"),
    }
    values = {"name": name, "integer_id": integer_id, "path": path}
    kwargs: dict[str, Any] = {}
    for role, candidates in aliases.items():
        selected = next((candidate for candidate in candidates if candidate in parameters), None)
        if selected is None and accepts_kwargs:
            selected = candidates[0]
        if selected is None:
            try:
                return factory(name, integer_id, path)
            except TypeError as exc:
                raise RuntimeError("the installed vLLM LoRARequest signature is unsupported") from exc
        kwargs[selected] = values[role]
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise RuntimeError("the installed vLLM LoRARequest rejected the configured adapter") from exc


def _add_engine_request(
    engine: Any,
    request_id: str,
    prompt: Any,
    sampling_params: Any,
    lora_request: Any,
) -> None:
    if lora_request is None:
        engine.add_request(request_id, prompt, sampling_params)
        return
    try:
        signature = inspect.signature(engine.add_request)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if "lora_request" not in signature.parameters and not accepts_kwargs:
            raise RuntimeError("the installed vLLM engine cannot route LoRA requests")
    try:
        engine.add_request(request_id, prompt, sampling_params, lora_request=lora_request)
    except TypeError as exc:
        raise RuntimeError("the installed vLLM engine rejected LoRA request routing") from exc


def _max_adapter_rank(routes: dict[str, _LoRARoute]) -> int | None:
    ranks: list[int] = []
    for route in routes.values():
        config_path = Path(route.resolved_path) / "adapter_config.json"
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        rank = raw.get("r")
        if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
            ranks.append(rank)
        pattern = raw.get("rank_pattern")
        if isinstance(pattern, dict):
            ranks.extend(
                value
                for value in pattern.values()
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            )
    if not ranks:
        return None
    observed = max(ranks)
    for supported in (1, 8, 16, 32, 64, 128, 256, 320, 512):
        if supported >= observed:
            return supported
    return observed


def _public_lora_source(source: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in source.items()
        if key not in {"access_token", "token", "hf_token", "huggingface_token"}
    }
    transfer = public.get("transfer")
    if isinstance(transfer, dict):
        public["transfer"] = {
            key: value
            for key, value in transfer.items()
            if key not in {"access_token", "token", "hf_token", "huggingface_token"}
        }
    return public


@contextmanager
def _huggingface_auth(token: Any):
    value = str(token).strip() if token else ""
    if not value:
        yield
        return
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = value
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _warmup(engine: Any) -> None:
    from vllm.sampling_params import SamplingParams

    engine.add_request(
        "explorer-warmup",
        "warmup",
        SamplingParams(temperature=0, seed=0, max_tokens=1),
    )
    guard = 0
    while engine.has_unfinished_requests():
        guard += 1
        if guard > 32:
            raise RuntimeError("warmup did not finish")
        engine.step()
    engine.abort_request(["explorer-warmup"])


def _engine_core(engine: Any) -> Any:
    client = engine.engine_core
    core = getattr(client, "engine_core", None)
    if core is None or not hasattr(core, "scheduler"):
        raise RuntimeError("the in-process engine core is not available")
    return core


def _block_pool(scheduler: Any) -> Any:
    manager = scheduler.kv_cache_manager
    pool = getattr(manager, "block_pool", None)
    if pool is None:
        pools = getattr(manager, "block_pools", None) or []
        pool = pools[0] if pools else None
    if pool is None:
        raise RuntimeError("the scheduler has no block pool")
    return pool


def _reject_speculative(engine: Any) -> None:
    speculative = getattr(engine.vllm_config, "speculative_config", None)
    if speculative is not None:
        raise RuntimeError("speculative decoding is outside this explorer")


def _import_vllm_version() -> str:
    import vllm

    return str(getattr(vllm, "__version__", ""))


def _gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        return "unavailable"
    return str(torch.cuda.get_device_name(0))


def _sampling_kwargs(sampling: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "temperature": float(sampling.get("temperature", 0.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "seed": int(sampling.get("seed", 0)),
        "max_tokens": max_output_tokens,
    }
    stop = sampling.get("stop") or []
    if stop:
        result["stop"] = list(stop)
    return result


def _synthetic_token_ids(engine: Any, target: int, lora_request: Any = None) -> list[int]:
    """Build an exact-length valid token-id prompt using the loaded tokenizer."""
    tokenizer = _loaded_tokenizer(engine, lora_request)
    encode = tokenizer.encode
    try:
        encoded = list(encode(" synthetic", add_special_tokens=False))
    except TypeError:
        encoded = list(encode(" synthetic"))
    if not encoded:
        raise RuntimeError("the loaded tokenizer returned no token for synthetic input construction")
    return [int(encoded[0])] * target


def _prompt_token_ids(engine: Any, prompt: str, lora_request: Any = None) -> list[int]:
    tokenizer = _loaded_tokenizer(engine, lora_request)
    encoded = list(tokenizer.encode(prompt))
    if not encoded:
        raise RuntimeError("the loaded tokenizer returned no input tokens")
    return [int(token_id) for token_id in encoded]


def _loaded_tokenizer(engine: Any, lora_request: Any = None) -> Any:
    group = None
    getter = getattr(engine, "get_tokenizer_group", None)
    if callable(getter):
        group = getter()
    candidates = [
        getattr(group, "tokenizer", None),
        getattr(engine, "tokenizer", None),
        getattr(getattr(engine, "input_processor", None), "tokenizer", None),
    ]
    lora_getter = getattr(group, "get_lora_tokenizer", None)
    if callable(lora_getter):
        try:
            candidates.insert(0, lora_getter(lora_request))
        except Exception as exc:
            if lora_request is not None:
                raise RuntimeError("the configured LoRA tokenizer could not be resolved") from exc
    for tokenizer in candidates:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            return tokenizer
    raise RuntimeError("the loaded tokenizer is unavailable")


def _queue_size(value: Any) -> int | None:
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _chunks(session: VllmSession, outputs: Any) -> list[ObservedChunk]:
    chunks: list[ObservedChunk] = []
    for output in outputs or []:
        request_id = str(getattr(output, "request_id", ""))
        if request_id not in session._seen_tokens:
            continue
        completions = getattr(output, "outputs", None) or []
        if not completions:
            continue
        completion = completions[0]
        token_ids = [int(token) for token in (getattr(completion, "token_ids", None) or [])]
        previous_ids = session._seen_tokens[request_id]
        if token_ids[: len(previous_ids)] == previous_ids:
            new_ids = tuple(token_ids[len(previous_ids) :])
            session._seen_tokens[request_id] = token_ids
        else:
            new_ids = tuple(token_ids)
            session._seen_tokens[request_id] = previous_ids + token_ids
        text = str(getattr(completion, "text", "") or "")
        previous_text = session._seen_text[request_id]
        if text.startswith(previous_text):
            delta = text[len(previous_text) :]
            session._seen_text[request_id] = text
        else:
            delta = text
            session._seen_text[request_id] = previous_text + text
        finished = bool(getattr(output, "finished", False))
        reason = getattr(completion, "finish_reason", None) if finished else None
        if new_ids or delta or reason:
            chunks.append(
                ObservedChunk(
                    request_id,
                    new_ids,
                    delta,
                    None if reason is None else str(reason),
                )
            )
    return chunks
