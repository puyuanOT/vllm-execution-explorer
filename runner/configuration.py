"""Canonical, secret-free configuration snapshots for one experiment.

The browser may submit either the legacy flat fields or the nested redesign
contract.  This module resolves both forms to one immutable recording shape.
It deliberately describes configurations that still need GPU acceptance; the
capability document decides whether a configuration may be launched.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from runner.workload import WorkRequest, WorkloadError, canonical_sampling, require_int

CONFIGURATION_VERSION = 1
MODEL_SOURCE_KINDS = frozenset({"huggingface", "execution_host", "operator_local"})
DTYPES = frozenset({"auto", "float16", "bfloat16", "float32"})
QUANTIZATION_KINDS = frozenset({"none", "awq", "gptq", "bitsandbytes", "fp8"})


class ConfigurationError(ValueError):
    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def parse_run_configuration(
    payload: dict[str, Any],
    requests: list[WorkRequest],
    workload_plan: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical configuration without prompts or credentials."""
    try:
        vllm = _parse_vllm(payload)
        model = _parse_model(payload)
        sampling = canonical_sampling(payload.get("sampling"), field="sampling")
        profiling = _parse_profiling(payload)
    except WorkloadError as exc:
        raise ConfigurationError(str(exc), field=exc.field) from exc

    known_adapters = {"base", *(item["id"] for item in model["lora_adapters"])}
    for request in requests:
        adapter_id = request.adapter_id or "base"
        if vllm["scheduling_policy"] != "priority" and request.priority != 0:
            raise ConfigurationError("Nonzero request priorities require the priority scheduling policy.", field=f"requests.{request.request_id}.priority")
        if adapter_id not in known_adapters:
            raise ConfigurationError(
                f"{request.request_id} selects unattached adapter {adapter_id!r}",
                field=f"requests.{request.request_id}.adapter_id",
            )
        token_count = request.resolved_input_tokens or request.target_input_tokens
        if token_count is not None and token_count + request.max_output_tokens > vllm["max_model_len"]:
            raise ConfigurationError(
                f"{request.request_id} needs {token_count + request.max_output_tokens} tokens, "
                f"which exceeds max_model_len {vllm['max_model_len']}",
                field=f"requests.{request.request_id}.max_output_tokens",
            )

    snapshot = {
        "version": CONFIGURATION_VERSION,
        "model": model,
        "vllm": vllm,
        "sampling": sampling,
        "profiling": profiling,
        "workload": {
            "request_count": len(requests),
            "batch_count": len(workload_plan.get("batches") or []),
            "input_modes": sorted({request.input_mode for request in requests}),
            "adapter_ids": sorted({request.adapter_id or "base" for request in requests}),
            "token_validation": _token_validation(requests),
        },
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["snapshot_sha256"] = hashlib.sha256(encoded).hexdigest()
    return snapshot


def copy_configuration(value: dict[str, Any]) -> dict[str, Any]:
    """Make the state/manifest boundary explicit and resistant to later edits."""
    return deepcopy(value)


def apply_legacy_fields(body: Any, configuration: dict[str, Any]) -> None:
    """Populate fields consumed by the existing in-process vLLM adapter."""
    model = configuration["model"]
    source = model["source"]
    vllm = configuration["vllm"]
    body.model = source["locator"]
    body.model_source_kind = source["kind"]
    body.tokenizer = model["tokenizer"]["locator"]
    body.model_revision = source.get("revision")
    body.tokenizer_revision = model["tokenizer"].get("revision")
    body.max_num_seqs = vllm["max_num_seqs"]
    body.token_budget = vllm["max_num_scheduled_tokens"]
    body.prefix_caching = vllm["enable_prefix_caching"]
    body.max_model_len = vllm["max_model_len"]
    body.chunked_prefill = vllm["enable_chunked_prefill"]
    body.gpu_memory_utilization = vllm["gpu_memory_utilization"]
    body.scheduling_policy = vllm["scheduling_policy"]
    body.long_prefill_token_threshold = vllm["long_prefill_token_threshold"]
    body.kv_cache_memory_bytes = vllm["kv_cache_memory_bytes"]
    body.profiling_configuration = copy_configuration(configuration.get("profiling") or {"mode": "off", "max_steps": 8})
    body.dtype = model["dtype"]
    body.quantization = model["quantization"]["kind"]
    body.lora_adapters = copy_configuration(model["lora_adapters"])
    body.sampling = copy_configuration(configuration["sampling"])


def launch_blockers(configuration: dict[str, Any], requests: list[WorkRequest], engine: str) -> list[str]:
    """Report controls that the selected execution path cannot honestly apply."""
    blockers: list[str] = []
    model = configuration["model"]
    sampling = configuration["sampling"]
    if engine == "fixture":
        if (configuration.get("profiling") or {}).get("mode", "off") != "off":
            blockers.append("Measured profiling requires a live dense Qwen3 vLLM host; the teaching fixture has no GPU activities.")
        vllm = configuration["vllm"]
        if vllm["scheduling_policy"] != "fcfs" or vllm["long_prefill_token_threshold"] or vllm["kv_cache_memory_bytes"] is not None:
            blockers.append("The demo fixture does not apply priority scheduling, a long-prefill cap, or an explicit KV memory budget. Use a live vLLM host for these controls.")
        if any(request.input_mode == "synthetic" for request in requests):
            blockers.append("The demo fixture cannot verify synthetic token lengths with the selected tokenizer.")
        if sampling != canonical_sampling(None):
            blockers.append("The demo fixture supports greedy sampling only.")
        if any(request.sampling for request in requests):
            blockers.append("The demo fixture does not apply per-request sampling overrides.")
        if model["lora_adapters"] or any((request.adapter_id or "base") != "base" for request in requests):
            blockers.append("The demo fixture does not execute LoRA adapters.")
        if model["quantization"]["kind"] != "none":
            blockers.append("The demo fixture does not load quantized weights.")
    return blockers


def _parse_vllm(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("vllm_configuration") or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("vllm_configuration must be an object", field="vllm_configuration")
    seqs = require_int(
        raw.get("max_num_seqs", payload.get("max_num_seqs", 4)),
        "max_num_seqs",
        field="vllm_configuration.max_num_seqs",
        minimum=1,
        maximum=256,
    )
    budget = require_int(
        raw.get("max_num_scheduled_tokens", raw.get("token_budget", payload.get("token_budget", 2048))),
        "max_num_scheduled_tokens",
        field="vllm_configuration.max_num_scheduled_tokens",
        minimum=1,
        maximum=1_000_000,
    )
    context = require_int(
        raw.get("max_model_len", payload.get("max_model_len") or max(4096, budget)),
        "max_model_len",
        field="vllm_configuration.max_model_len",
        minimum=1,
        maximum=2_000_000,
    )
    prefix = raw.get("enable_prefix_caching", raw.get("prefix_caching", payload.get("prefix_caching", True)))
    chunked = raw.get("enable_chunked_prefill", raw.get("chunked_prefill", payload.get("chunked_prefill", True)))
    if not isinstance(prefix, bool):
        raise ConfigurationError("enable_prefix_caching must be true or false", field="vllm_configuration.enable_prefix_caching")
    if not isinstance(chunked, bool):
        raise ConfigurationError("enable_chunked_prefill must be true or false", field="vllm_configuration.enable_chunked_prefill")
    if budget < seqs:
        raise ConfigurationError("Tokens per iteration must be at least the maximum concurrent sequences.", field="vllm_configuration.max_num_scheduled_tokens")
    if not chunked and budget < context:
        raise ConfigurationError("With chunked prefill disabled, tokens per iteration must be at least the maximum context length.", field="vllm_configuration.max_num_scheduled_tokens")
    policy = raw.get("scheduling_policy", payload.get("scheduling_policy", "fcfs"))
    if policy not in {"fcfs", "priority"}:
        raise ConfigurationError("Scheduling policy must be fcfs or priority.", field="vllm_configuration.scheduling_policy")
    threshold = require_int(raw.get("long_prefill_token_threshold", payload.get("long_prefill_token_threshold", 0)), "long-prefill token threshold", field="vllm_configuration.long_prefill_token_threshold", minimum=0, maximum=context)
    if threshold and not chunked:
        raise ConfigurationError("A long-prefill token cap requires chunked prefill.", field="vllm_configuration.long_prefill_token_threshold")
    cache_bytes = raw.get("kv_cache_memory_bytes", payload.get("kv_cache_memory_bytes"))
    if cache_bytes is not None:
        cache_bytes = require_int(cache_bytes, "KV memory bytes", field="vllm_configuration.kv_cache_memory_bytes", minimum=1, maximum=2**53-1)
    utilization = raw.get("gpu_memory_utilization", payload.get("gpu_memory_utilization", 0.9))
    if isinstance(utilization, bool) or not isinstance(utilization, (int, float)) or not 0 < float(utilization) <= 1:
        raise ConfigurationError(
            "gpu_memory_utilization must be greater than 0 and at most 1",
            field="vllm_configuration.gpu_memory_utilization",
        )
    fresh = raw.get("fresh_prefix_cache", True)
    if fresh is not True:
        raise ConfigurationError(
            "every experiment must begin with a fresh prefix cache",
            field="vllm_configuration.fresh_prefix_cache",
        )
    return {
        "max_model_len": context,
        "max_num_seqs": seqs,
        "max_num_scheduled_tokens": budget,
        "enable_chunked_prefill": chunked,
        "enable_prefix_caching": prefix,
        "gpu_memory_utilization": float(utilization),
        "scheduling_policy": policy,
        "long_prefill_token_threshold": threshold,
        "kv_cache_memory_bytes": cache_bytes,
        "fresh_prefix_cache": True,
    }


def _parse_profiling(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("profiling_configuration")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("profiling_configuration must be an object", field="profiling_configuration")
    mode = raw.get("mode", "off")
    if mode not in {"off", "eager_diagnostic"}:
        raise ConfigurationError("Profiling mode must be off or eager_diagnostic.", field="profiling_configuration.mode")
    steps = require_int(raw.get("max_steps", 8), "profile step limit", field="profiling_configuration.max_steps", minimum=1, maximum=16)
    return {"mode": mode, "max_steps": steps}


def _parse_model(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("model_configuration") or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("model_configuration must be an object", field="model_configuration")
    legacy_model = str(payload.get("model") or "Qwen/Qwen3-4B-Instruct-2507").strip()
    source_raw = raw.get("source")
    if source_raw is None:
        source_raw = {
            "kind": "huggingface",
            "repository": legacy_model,
            "revision": payload.get("model_revision"),
        }
    source = _parse_artifact_source(
        source_raw,
        field="model_configuration.source",
        authenticated=bool(payload.get("huggingface_token")),
    )
    dtype = raw.get("dtype", payload.get("dtype", "bfloat16"))
    if dtype not in DTYPES:
        raise ConfigurationError(f"dtype must be one of {sorted(DTYPES)}", field="model_configuration.dtype")
    quant_raw = raw.get("quantization", payload.get("quantization") or "none")
    if isinstance(quant_raw, str):
        quant_raw = {"kind": quant_raw}
    if not isinstance(quant_raw, dict):
        raise ConfigurationError("quantization must be a string or object", field="model_configuration.quantization")
    quant_kind = quant_raw.get("kind", "none")
    if quant_kind not in QUANTIZATION_KINDS:
        raise ConfigurationError(
            f"quantization kind must be one of {sorted(QUANTIZATION_KINDS)}",
            field="model_configuration.quantization.kind",
        )
    adapters_raw = raw.get("lora_adapters") or []
    if not isinstance(adapters_raw, list):
        raise ConfigurationError("lora_adapters must be a list", field="model_configuration.lora_adapters")
    adapters: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(adapters_raw):
        field = f"model_configuration.lora_adapters[{index}]"
        if not isinstance(item, dict):
            raise ConfigurationError("each LoRA adapter must be an object", field=field)
        adapter_id = _nonempty(item.get("id"), f"{field}.id")
        if adapter_id == "base" or adapter_id in seen:
            raise ConfigurationError("LoRA adapter ids must be unique and cannot be 'base'", field=f"{field}.id")
        seen.add(adapter_id)
        adapter_source = _parse_artifact_source(
            item.get("source"),
            field=f"{field}.source",
            authenticated=bool(payload.get("huggingface_token")),
        )
        if adapter_source["kind"] == "huggingface" and not adapter_source.get("revision"):
            raise ConfigurationError(
                "a Hugging Face LoRA adapter needs a pinned revision",
                field=f"{field}.source.revision",
            )
        adapters.append({"id": adapter_id, "source": adapter_source})
    tokenizer_raw = raw.get("tokenizer") or {}
    if not isinstance(tokenizer_raw, dict):
        raise ConfigurationError("tokenizer must be an object", field="model_configuration.tokenizer")
    tokenizer = {
        "locator": str(tokenizer_raw.get("locator") or payload.get("tokenizer") or source["locator"]).strip(),
        "revision": tokenizer_raw.get("revision") or payload.get("tokenizer_revision") or None,
    }
    return {
        "source": source,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "quantization": {"kind": quant_kind},
        "lora_adapters": adapters,
    }


def _parse_artifact_source(raw: Any, *, field: str, authenticated: bool = False) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigurationError("artifact source must be an object", field=field)
    if raw.get("access_token"):
        raise ConfigurationError(
            "submit Hugging Face credentials in the ephemeral top-level huggingface_token field",
            field=f"{field}.access_token",
        )
    kind = raw.get("kind")
    if kind not in MODEL_SOURCE_KINDS:
        raise ConfigurationError(f"source kind must be one of {sorted(MODEL_SOURCE_KINDS)}", field=f"{field}.kind")
    if kind == "huggingface":
        locator = raw.get("repository", raw.get("repo_id", raw.get("locator")))
    else:
        locator = raw.get("path", raw.get("locator"))
    source: dict[str, Any] = {
        "kind": kind,
        "locator": _nonempty(locator, f"{field}.locator"),
        "revision": raw.get("revision"),
    }
    if kind == "huggingface":
        source["authenticated"] = bool(authenticated or raw.get("authenticated"))
    if kind == "operator_local":
        transfer = raw.get("transfer") or {}
        if not isinstance(transfer, dict):
            raise ConfigurationError("transfer must be an object", field=f"{field}.transfer")
        destination = transfer.get("destination")
        source["transfer"] = {
            "required_for_remote": True,
            "destination": None if destination is None else _nonempty(destination, f"{field}.transfer.destination"),
        }
    return source


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string", field=field)
    return value.strip()


def _token_validation(requests: list[WorkRequest]) -> dict[str, int]:
    reported = sum(request.resolved_input_tokens is not None for request in requests)
    return {
        "client_reported_request_count": reported,
        "engine_verified_request_count": 0,
        "pending_engine_verification_count": len(requests),
    }
