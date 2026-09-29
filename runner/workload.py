"""Workload authoring and executable-request validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

PROVISIONAL_LIMITS = {
    "max_requests": 64,
    "max_prompt_characters": 2_000_000,
    "max_prompt_bytes": 2_000_000,
    "max_prompt_tokens": 131_072,
    "max_output_tokens": 8_192,
    "max_run_seconds": 600,
    "max_stop_strings": 8,
    "max_stop_string_characters": 256,
    "limit_basis": "contract_safety_ceiling",
    "gpu_validated": False,
    "note": (
        "Supports tens of authored requests. Tokenizer, context, memory, and recording limits "
        "must still be validated on the selected GPU before capture."
    ),
}

PLAN_VERSION = 1
_FORBIDDEN_PLAN_KEYS = {
    "prompt",
    "prompt_text",
    "text",
    "text_delta",
    "token_ids",
    "output",
    "content",
}


class WorkloadError(ValueError):
    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class WorkRequest:
    request_id: str
    prompt: str
    arrival_offset_ms: int
    max_output_tokens: int
    input_mode: str = "prompt"
    target_input_tokens: int | None = None
    resolved_input_tokens: int | None = None
    adapter_id: str | None = None
    sampling: dict[str, Any] | None = None


def parse_workload(payload: dict) -> list[WorkRequest]:
    requests, _plan = parse_run_workload(payload)
    return requests


def parse_run_workload(payload: dict) -> tuple[list[WorkRequest], dict[str, Any]]:
    raw = payload.get("requests")
    if not isinstance(raw, list) or not raw:
        raise WorkloadError("a run needs at least one request", field="requests")
    if len(raw) > int(PROVISIONAL_LIMITS["max_requests"]):
        raise WorkloadError(
            f"a run accepts at most {PROVISIONAL_LIMITS['max_requests']} requests",
            field="requests",
        )
    requests: list[WorkRequest] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        request = parse_request_item(item, field_prefix=f"requests[{index}]")
        if request.request_id in seen:
            raise WorkloadError("each request needs its own id", field=f"requests[{index}].request_id")
        seen.add(request.request_id)
        requests.append(request)
    plan = canonical_plan(requests, payload.get("workload_plan", None))
    routes = {
        str(sample["id"]): str(sample.get("adapter_id") or "base")
        for batch in plan["batches"]
        for sample in batch["samples"]
    }
    requests = [replace(request, adapter_id=routes[request.request_id]) for request in requests]
    return requests, plan


def parse_request_item(item: object, *, field_prefix: str = "request") -> WorkRequest:
    if not isinstance(item, dict):
        raise WorkloadError("each request must be an object", field=field_prefix)
    request_id = item.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip() or request_id != request_id.strip():
        raise WorkloadError("each request needs its own id", field=f"{field_prefix}.request_id")
    if "/" in request_id or "\\" in request_id:
        raise WorkloadError("request id must be a single path segment", field=f"{field_prefix}.request_id")
    input_mode = item.get("input_mode", "prompt")
    if input_mode not in {"prompt", "synthetic"}:
        raise WorkloadError("input_mode must be prompt or synthetic", field=f"{field_prefix}.input_mode")
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or (input_mode == "prompt" and not prompt.strip()):
        raise WorkloadError(f"{request_id} has an empty prompt", field=f"{field_prefix}.prompt")
    if len(prompt) > int(PROVISIONAL_LIMITS["max_prompt_characters"]) or len(prompt.encode("utf-8")) > int(
        PROVISIONAL_LIMITS["max_prompt_bytes"]
    ):
        raise WorkloadError(f"{request_id} exceeds the prompt payload safety limit", field=f"{field_prefix}.prompt")
    if "arrival_offset_ms" not in item:
        raise WorkloadError(f"{request_id} needs an arrival offset in milliseconds", field=f"{field_prefix}.arrival_offset_ms")
    offset = require_int(
        item.get("arrival_offset_ms"),
        f"{request_id} arrival offset",
        field=f"{field_prefix}.arrival_offset_ms",
        minimum=0,
        maximum=int(PROVISIONAL_LIMITS["max_run_seconds"]) * 1000,
    )
    if "max_output_tokens" not in item:
        raise WorkloadError(f"{request_id} needs a maximum output token count", field=f"{field_prefix}.max_output_tokens")
    max_output = require_int(
        item.get("max_output_tokens"),
        f"{request_id} maximum output tokens",
        field=f"{field_prefix}.max_output_tokens",
        minimum=1,
        maximum=int(PROVISIONAL_LIMITS["max_output_tokens"]),
    )
    target = _optional_token_count(item.get("target_input_tokens"), field=f"{field_prefix}.target_input_tokens")
    resolved = _optional_token_count(
        item.get("resolved_input_tokens", item.get("measured_input_tokens")),
        field=f"{field_prefix}.resolved_input_tokens",
    )
    if input_mode == "synthetic":
        if target is None:
            raise WorkloadError(
                "a synthetic request needs target_input_tokens",
                field=f"{field_prefix}.target_input_tokens",
            )
        if resolved is not None and target != resolved:
            raise WorkloadError(
                "a synthetic request's resolved token length must equal its target",
                field=f"{field_prefix}.resolved_input_tokens",
            )
    elif target is not None:
        raise WorkloadError(
            "target_input_tokens is only valid for synthetic requests",
            field=f"{field_prefix}.target_input_tokens",
        )
    adapter = item.get("adapter_id")
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        raise WorkloadError("adapter_id must be a non-empty string", field=f"{field_prefix}.adapter_id")
    sampling = None
    if item.get("sampling") is not None:
        sampling = canonical_sampling(item.get("sampling"), field=f"{field_prefix}.sampling")
    return WorkRequest(
        request_id=request_id,
        prompt=prompt,
        arrival_offset_ms=offset,
        max_output_tokens=max_output,
        input_mode=input_mode,
        target_input_tokens=target,
        resolved_input_tokens=resolved,
        adapter_id=None if adapter is None else adapter.strip(),
        sampling=sampling,
    )


def _optional_token_count(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return require_int(
        value,
        "input token count",
        field=field,
        minimum=1,
        maximum=int(PROVISIONAL_LIMITS["max_prompt_tokens"]),
    )


def canonical_sampling(raw: object, *, field: str = "sampling") -> dict[str, Any]:
    if raw is None:
        return {"temperature": 0.0, "top_p": 1.0, "seed": 0, "stop": []}
    if not isinstance(raw, dict):
        raise WorkloadError("sampling must be an object", field=field)
    temperature = raw.get("temperature", 0.0)
    top_p = raw.get("top_p", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 10:
        raise WorkloadError("temperature must be from 0 to 10", field=f"{field}.temperature")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1:
        raise WorkloadError("top_p must be greater than 0 and at most 1", field=f"{field}.top_p")
    seed = require_int(raw.get("seed", 0), "seed", field=f"{field}.seed", minimum=0, maximum=2**63 - 1)
    stop = raw.get("stop", [])
    if not isinstance(stop, list) or len(stop) > int(PROVISIONAL_LIMITS["max_stop_strings"]):
        raise WorkloadError(
            f"stop must contain at most {PROVISIONAL_LIMITS['max_stop_strings']} strings",
            field=f"{field}.stop",
        )
    clean_stop: list[str] = []
    for index, value in enumerate(stop):
        if not isinstance(value, str) or not value or len(value) > int(PROVISIONAL_LIMITS["max_stop_string_characters"]):
            raise WorkloadError("each stop value must be a short non-empty string", field=f"{field}.stop[{index}]")
        clean_stop.append(value)
    return {"temperature": float(temperature), "top_p": float(top_p), "seed": seed, "stop": clean_stop}


def require_int(
    value: object,
    label: str,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkloadError(f"{label} must be an integer", field=field)
    if value < minimum or (maximum is not None and value > maximum):
        upper = "unlimited" if maximum is None else str(maximum)
        raise WorkloadError(f"{label} must be from {minimum} to {upper}", field=field)
    return value


def require_engine_settings(payload: dict[str, Any]) -> dict[str, Any]:
    seqs = require_int(
        payload.get("max_num_seqs"),
        "max_num_seqs",
        field="max_num_seqs",
        minimum=1,
        maximum=256,
    )
    budget = require_int(
        payload.get("token_budget"),
        "token budget",
        field="token_budget",
        minimum=1,
        maximum=1_000_000,
    )
    prefix = payload.get("prefix_caching")
    if not isinstance(prefix, bool):
        raise WorkloadError("prefix_caching must be true or false", field="prefix_caching")
    model = payload.get("model", "Qwen/Qwen3-4B-Instruct-2507")
    if not isinstance(model, str) or not model.strip():
        raise WorkloadError("model must be a non-empty string", field="model")
    return {
        "max_num_seqs": seqs,
        "token_budget": budget,
        "prefix_caching": prefix,
        "model": model.strip(),
    }


def canonical_plan(requests: list[WorkRequest], raw: object) -> dict[str, Any]:
    if raw is None:
        return synthesize_plan(requests)
    if not isinstance(raw, dict):
        raise WorkloadError("workload_plan must be an object", field="workload_plan")
    _reject_content_keys(raw, "workload_plan")
    if raw.get("version") != 1:
        raise WorkloadError("workload_plan version must be 1", field="workload_plan.version")
    batches = raw.get("batches")
    if not isinstance(batches, list) or not batches:
        raise WorkloadError("workload_plan needs at least one batch", field="workload_plan.batches")
    flat: list[dict[str, Any]] = []
    canonical_batches: list[dict[str, Any]] = []
    seen_batches: set[str] = set()
    seen_samples: set[str] = set()
    for batch_index, batch in enumerate(batches):
        prefix = f"workload_plan.batches[{batch_index}]"
        if not isinstance(batch, dict):
            raise WorkloadError("each plan batch must be an object", field=prefix)
        _reject_content_keys(batch, prefix)
        batch_id = batch.get("id")
        label = batch.get("label")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise WorkloadError("each plan batch needs an id", field=f"{prefix}.id")
        if batch_id in seen_batches:
            raise WorkloadError("plan batch ids must be unique", field=f"{prefix}.id")
        seen_batches.add(batch_id)
        if not isinstance(label, str) or not label.strip():
            raise WorkloadError("each plan batch needs a label", field=f"{prefix}.label")
        if "arrival_offset_ms" not in batch:
            raise WorkloadError("each plan batch needs an arrival offset", field=f"{prefix}.arrival_offset_ms")
        offset = require_int(
            batch.get("arrival_offset_ms"),
            "plan batch arrival offset",
            field=f"{prefix}.arrival_offset_ms",
            minimum=0,
            maximum=int(PROVISIONAL_LIMITS["max_run_seconds"]) * 1000,
        )
        samples = batch.get("samples")
        if not isinstance(samples, list) or not samples:
            raise WorkloadError("a plan batch needs at least one sample", field=f"{prefix}.samples")
        batch_adapter = batch.get("adapter_id", "base")
        if not isinstance(batch_adapter, str) or not batch_adapter.strip():
            raise WorkloadError("batch adapter_id must be a non-empty string", field=f"{prefix}.adapter_id")
        batch_adapter = batch_adapter.strip()
        canonical_samples: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(samples):
            sample_prefix = f"{prefix}.samples[{sample_index}]"
            if not isinstance(sample, dict):
                raise WorkloadError("each plan sample must be an object", field=sample_prefix)
            _reject_content_keys(sample, sample_prefix)
            sample_id = sample.get("id")
            sample_label = sample.get("label")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise WorkloadError("each plan sample needs an id", field=f"{sample_prefix}.id")
            if sample_id in seen_samples:
                raise WorkloadError("plan sample ids must be unique", field=f"{sample_prefix}.id")
            seen_samples.add(sample_id)
            if not isinstance(sample_label, str) or not sample_label.strip():
                raise WorkloadError("each plan sample needs a label", field=f"{sample_prefix}.label")
            if "max_output_tokens" not in sample:
                raise WorkloadError("each plan sample needs a maximum output count", field=f"{sample_prefix}.max_output_tokens")
            max_output = require_int(
                sample.get("max_output_tokens"),
                "plan sample maximum output tokens",
                field=f"{sample_prefix}.max_output_tokens",
                minimum=1,
                maximum=int(PROVISIONAL_LIMITS["max_output_tokens"]),
            )
            request = requests[len(flat)] if len(flat) < len(requests) else None
            sample_adapter = sample.get("adapter_id")
            if sample_adapter is not None and (not isinstance(sample_adapter, str) or not sample_adapter.strip()):
                raise WorkloadError("sample adapter_id must be a non-empty string", field=f"{sample_prefix}.adapter_id")
            effective_adapter = (
                sample_adapter.strip()
                if isinstance(sample_adapter, str)
                else (request.adapter_id if request is not None and request.adapter_id else batch_adapter)
            )
            order = len(flat)
            canonical_samples.append(
                {
                    "id": sample_id,
                    "label": sample_label,
                    "max_output_tokens": max_output,
                    "order": order,
                    "input_mode": request.input_mode if request is not None else "prompt",
                    "target_input_tokens": request.target_input_tokens if request is not None else None,
                    "resolved_input_tokens": request.resolved_input_tokens if request is not None else None,
                    "input_token_count_provenance": (
                        "client_submitted_unverified"
                        if request is not None and request.resolved_input_tokens is not None
                        else "pending_execution_host_tokenizer"
                    ),
                    "adapter_id": effective_adapter,
                    "sampling_override": request.sampling if request is not None else None,
                }
            )
            flat.append(
                {
                    "id": sample_id,
                    "arrival_offset_ms": offset,
                    "max_output_tokens": max_output,
                    "adapter_id": effective_adapter,
                }
            )
        canonical_batches.append(
            {
                "id": batch_id,
                "label": label.strip(),
                "arrival_offset_ms": offset,
                "adapter_id": batch_adapter,
                "samples": canonical_samples,
            }
        )
    if len(flat) != len(requests):
        raise WorkloadError("workload_plan must list exactly the submitted requests", field="workload_plan")
    for request, planned in zip(requests, flat, strict=True):
        if planned["id"] != request.request_id:
            raise WorkloadError(
                "workload_plan sample order must match the submitted requests",
                field="workload_plan",
            )
        if planned["arrival_offset_ms"] != request.arrival_offset_ms:
            raise WorkloadError(
                f"{request.request_id} plan offset does not match the request",
                field="workload_plan",
            )
        if planned["max_output_tokens"] != request.max_output_tokens:
            raise WorkloadError(
                f"{request.request_id} plan output limit does not match the request",
                field="workload_plan",
            )
        if request.adapter_id is not None and planned["adapter_id"] != request.adapter_id:
            raise WorkloadError(
                f"{request.request_id} plan adapter does not match the request override",
                field="workload_plan",
            )
    return {"version": PLAN_VERSION, "synthesized": False, "batches": canonical_batches}


def synthesize_plan(requests: list[WorkRequest]) -> dict[str, Any]:
    batches = []
    for index, request in enumerate(requests):
        batches.append(
            {
                "id": f"request:{request.request_id}",
                "label": request.request_id,
                "arrival_offset_ms": request.arrival_offset_ms,
                "adapter_id": request.adapter_id or "base",
                "samples": [
                    {
                        "id": request.request_id,
                        "label": request.request_id,
                        "max_output_tokens": request.max_output_tokens,
                        "order": index,
                        "input_mode": request.input_mode,
                        "target_input_tokens": request.target_input_tokens,
                        "resolved_input_tokens": request.resolved_input_tokens,
                        "input_token_count_provenance": (
                            "client_submitted_unverified"
                            if request.resolved_input_tokens is not None
                            else "pending_execution_host_tokenizer"
                        ),
                        "adapter_id": request.adapter_id or "base",
                        "sampling_override": request.sampling,
                    }
                ],
            }
        )
    return {"version": PLAN_VERSION, "synthesized": True, "batches": batches}


def append_request_to_plan(plan: dict[str, Any], request: WorkRequest) -> dict[str, Any]:
    """Keep an already accepted add-now request in the plan without merging it into an earlier batch."""
    batches = [dict(batch, samples=list(batch["samples"])) for batch in plan["batches"]]
    order = sum(len(batch["samples"]) for batch in batches)
    batches.append(
        {
            "id": f"added:{request.request_id}",
            "label": request.request_id,
            "arrival_offset_ms": request.arrival_offset_ms,
            "adapter_id": request.adapter_id or "base",
            "samples": [
                {
                    "id": request.request_id,
                    "label": request.request_id,
                    "max_output_tokens": request.max_output_tokens,
                    "order": order,
                    "input_mode": request.input_mode,
                    "target_input_tokens": request.target_input_tokens,
                    "resolved_input_tokens": request.resolved_input_tokens,
                    "input_token_count_provenance": (
                        "client_submitted_unverified"
                        if request.resolved_input_tokens is not None
                        else "pending_execution_host_tokenizer"
                    ),
                    "adapter_id": request.adapter_id or "base",
                    "sampling_override": request.sampling,
                }
            ],
        }
    )
    return {
        "version": PLAN_VERSION,
        "synthesized": bool(plan.get("synthesized")),
        "batches": batches,
    }


def _reject_content_keys(value: object, field: str) -> None:
    if isinstance(value, dict):
        overlap = _FORBIDDEN_PLAN_KEYS.intersection(value)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise WorkloadError(f"{field} cannot include prompt or token content ({names})", field=field)
        for key, item in value.items():
            _reject_content_keys(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_content_keys(item, f"{field}[{index}]")
