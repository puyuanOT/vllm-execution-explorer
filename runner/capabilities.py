"""Stable support reporting for controls exposed by the redesign."""

from __future__ import annotations

from typing import Any

CAPABILITY_CONTRACT_VERSION = 1
CAPABILITY_STATUSES = (
    "available",
    "blocked",
    "gpu_validation_required",
    "implementation_pending",
)


def capability_report(
    *,
    platform: str,
    cuda_available: bool,
    vllm_version: str | None,
    cuda_compute_capability: tuple[int, int] | None = None,
    bitsandbytes_plugin_available: bool = False,
) -> dict[str, Any]:
    live_environment = cuda_available and bool(vllm_version and vllm_version.startswith("0.30."))
    live_message = (
        "The pinned vLLM 0.30.x CUDA runtime is present; GPU acceptance is still required."
        if live_environment
        else "This machine does not have the pinned vLLM 0.30.x CUDA runtime. Use a CUDA-ready SSH host."
    )
    return {
        "contract_version": CAPABILITY_CONTRACT_VERSION,
        "status_values": list(CAPABILITY_STATUSES),
        "runtime": {
            "platform": platform,
            "cuda_available": cuda_available,
            "cuda_compute_capability": (
                list(cuda_compute_capability) if cuda_compute_capability is not None else None
            ),
            "vllm_version": vllm_version,
            "pinned_vllm": "0.30.x",
        },
        "execution_modes": {
            "fixture": _feature("available", True, "Local deterministic teaching fixture; never presented as GPU evidence."),
            "local_gpu": _feature(
                "gpu_validation_required" if live_environment else "blocked",
                live_environment,
                live_message,
            ),
            "ssh_gpu": _feature(
                "gpu_validation_required",
                True,
                "SSH setup and loopback tunnelling are implemented; the supplied GPU host must pass acceptance.",
            ),
        },
        "capture_boundaries": {
            "planned_dispatch": _feature("available", True, "Recorded in the workload clock domain."),
            "actual_dispatch": _feature("available", True, "Recorded around the active dispatch call."),
            "http_receipt": _feature(
                "gpu_validation_required",
                live_environment,
                "The loopback HTTP boundary is instrumented; end-to-end GPU acceptance is still required.",
            ),
            "engine_admission": _feature(
                "gpu_validation_required",
                live_environment,
                "The scheduler add boundary is instrumented and distinct from HTTP receipt.",
            ),
        },
        "workload": {
            "prompt_text": _feature("available", True, "Executable prompt text with optional measured token metadata."),
            "synthetic_exact_tokens": _feature(
                "gpu_validation_required",
                live_environment,
                "The engine owner constructs and submits an exact token-id prompt; GPU/model acceptance is required.",
            ),
            "batch_adapter_defaults": _feature(
                "gpu_validation_required",
                True,
                "Batch defaults and per-request overrides route configured LoRARequest objects; GPU acceptance is still required.",
            ),
        },
        "model_sources": {
            "huggingface": _feature("gpu_validation_required", True, "Resolved on the execution host; tokens are never recorded."),
            "execution_host": _feature("gpu_validation_required", True, "Uses a path that already exists on the execution host."),
            "operator_local": _feature(
                "available",
                True,
                "Local artifacts can be inspected and transferred over the active SSH connection with retry status.",
            ),
        },
        "model_options": {
            "lora": _feature(
                "gpu_validation_required",
                True,
                "Configured adapters are resolved on the execution host and routed per request; GPU acceptance is still required.",
            ),
            "quantization": _quantization_capabilities(
                live_environment=live_environment,
                cuda_compute_capability=cuda_compute_capability,
                bitsandbytes_plugin_available=bitsandbytes_plugin_available,
            ),
        },
        "fresh_prefix_cache": _feature(
            "gpu_validation_required",
            live_environment,
            "Warmup must finish, then a verified reset must occur before capture readiness and workload t=0.",
        ),
    }


def _feature(status: str, selectable: bool, message: str) -> dict[str, Any]:
    if status not in CAPABILITY_STATUSES:
        raise ValueError(f"unknown capability status {status}")
    return {"status": status, "selectable": selectable, "message": message}


def _quantization_capabilities(
    *,
    live_environment: bool,
    cuda_compute_capability: tuple[int, int] | None,
    bitsandbytes_plugin_available: bool,
) -> dict[str, dict[str, Any]]:
    unavailable_runtime = (
        "A CUDA GPU and the pinned vLLM 0.30.x runtime must be available on this execution host."
    )
    result = {
        "none": _feature("available", True, "Load the model without an explicit quantization method."),
    }
    if not live_environment:
        for kind in ("awq", "gptq", "bitsandbytes", "fp8"):
            result[kind] = _feature("blocked", False, unavailable_runtime)
        return result

    numeric_capability = _numeric_compute_capability(cuda_compute_capability)
    awq_supported = numeric_capability is None or numeric_capability >= 75
    gptq_supported = numeric_capability is None or numeric_capability >= 70
    fp8_supported = numeric_capability is None or numeric_capability >= 75

    result["awq"] = _feature(
        "gpu_validation_required" if awq_supported else "blocked",
        awq_supported,
        (
            "Use an AWQ-quantized checkpoint. This GPU family is supported; the selected model must still pass preparation."
            if awq_supported
            else "AWQ requires an NVIDIA GPU with compute capability 7.5 or newer."
        ),
    )
    result["gptq"] = _feature(
        "gpu_validation_required" if gptq_supported else "blocked",
        gptq_supported,
        (
            "Use a GPTQ-quantized checkpoint. This GPU family is supported; the selected model must still pass preparation."
            if gptq_supported
            else "GPTQ requires an NVIDIA GPU with compute capability 7.0 or newer."
        ),
    )

    bnb_supported = bitsandbytes_plugin_available and (
        numeric_capability is None or numeric_capability >= 70
    )
    if not bitsandbytes_plugin_available:
        bnb_message = (
            "bitsandbytes requires the out-of-tree vllm-bnb-plugin, which is not installed on this execution host."
        )
    elif not bnb_supported:
        bnb_message = "bitsandbytes requires an NVIDIA GPU with compute capability 7.0 or newer."
    else:
        bnb_message = (
            "The vllm-bnb-plugin is installed and this GPU family is supported. "
            "The selected model must still pass preparation."
        )
    result["bitsandbytes"] = _feature(
        "gpu_validation_required" if bnb_supported else "blocked",
        bnb_supported,
        bnb_message,
    )

    if fp8_supported and numeric_capability is not None and numeric_capability < 89:
        fp8_message = (
            "This GPU can attempt FP8 as weight-only W8A16 through Marlin; native FP8 W8A8 compute requires "
            "compute capability 8.9 or newer. Model architecture and runtime kernel acceptance are verified "
            "during preparation and may still fail."
        )
    elif fp8_supported:
        fp8_message = (
            "This GPU family supports FP8. Model architecture and runtime kernel acceptance are verified "
            "during preparation and may still fail."
        )
    else:
        fp8_message = "FP8 model execution requires an NVIDIA GPU with compute capability 7.5 or newer."
    result["fp8"] = _feature(
        "gpu_validation_required" if fp8_supported else "blocked",
        fp8_supported,
        fp8_message,
    )
    return result


def _numeric_compute_capability(value: tuple[int, int] | None) -> int | None:
    if value is None:
        return None
    return value[0] * 10 + value[1]
