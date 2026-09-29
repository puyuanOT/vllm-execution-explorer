"""Manifest fields shared by fixture recordings and live runs."""

from __future__ import annotations

from typing import Any

from vllm_030.constants import ADAPTER_NAME, ADAPTER_REVISION, CLOCK_DOMAIN, SCHEMA_VERSION

from runner.workload import PROVISIONAL_LIMITS


def manifest_skeleton(
    run_id: str,
    *,
    execution_kind: str,
    execution_target: dict[str, Any],
    arrivals: list[dict[str, Any]] | None = None,
    completion_status: str = "completed",
) -> dict[str, Any]:
    fixture = execution_kind == "fixture"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "execution_kind": execution_kind,
        "engine": {"name": "vllm", "version": "0.30.0", "upstream_revision": None},
        "adapter": {"name": ADAPTER_NAME, "revision": ADAPTER_REVISION},
        "model": {
            "id": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": None,
            "note": "Candidate model. A fixture recording does not load weights.",
        },
        "tokenizer": {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": None},
        "hardware": {"gpu": None},
        "engine_settings": {
            "async_scheduling": False,
            "enable_prefix_caching": True,
            "max_num_seqs": 4,
            "max_num_scheduled_tokens": 32,
        },
        "sampling": {"temperature": 0, "seed": 0},
        "cache_geometry": {"block_size": 16, "num_gpu_blocks": 8, "initial_free_blocks": 8},
        "initial_cache_state": {
            "kind": "not_applicable" if fixture else "unknown",
            "reset_required": not fixture,
            "reset_verified": False,
            "warmup_completed_before_reset": False,
            "capture_ready_after_reset": False,
            "evidence": "fixture_has_no_vllm_prefix_cache" if fixture else "not_yet_observed",
        },
        "workload_limits": PROVISIONAL_LIMITS,
        "arrivals": arrivals or [],
        "completion_status": completion_status,
        "clock_domain": CLOCK_DOMAIN,
        "content_retained": True,
        "execution_target": execution_target,
        "run_configuration": {
            "version": 1,
            "status": "not_provided",
            "snapshot_sha256": None,
        },
        "capture": {
            "contract_version": 1,
            "transport": "fixture_direct" if fixture else "in_process_direct_engine",
            "http_boundary_observed": False,
            "engine_admission_boundary": "fixture_scheduler" if fixture else "scheduler.add_request",
            "phase_history": [],
            "status": completion_status,
        },
        "provenance": {
            "evidence_kind": "fixture" if fixture else "instrumented_gpu_run",
            "real_gpu": not fixture,
            "http_serving": False,
            "gpu_acceptance": "not_applicable" if fixture else "required",
            "claim": (
                "Deterministic demo evidence; not a vLLM or GPU observation."
                if fixture
                else "Instrumented direct-engine evidence. HTTP serving boundaries were not observed."
            ),
        },
    }


LOCAL_FIXTURE_TARGET = {
    "kind": "local",
    "engine_location": "this-machine",
    "client_platform": "fixture",
    "runner_endpoint": "http://127.0.0.1:8765",
    "remote_bind": "127.0.0.1",
}
