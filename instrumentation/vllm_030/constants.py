"""Shared recording contract for the vLLM 0.30.0 explorer."""

SCHEMA_VERSION = "1.0.0"
PRODUCER_ID = "vllm-030-adapter"
ADAPTER_NAME = "vllm_030"
ADAPTER_REVISION = "0.2.0"
CLOCK_DOMAIN = "time.monotonic_ns"
PINNED_VLLM_PREFIX = "0.30."

MANIFEST_REQUIRED = (
    "schema_version",
    "run_id",
    "execution_kind",
    "engine",
    "adapter",
    "model",
    "tokenizer",
    "hardware",
    "engine_settings",
    "sampling",
    "cache_geometry",
    "initial_cache_state",
    "workload_limits",
    "arrivals",
    "completion_status",
    "event_completeness",
    "clock_domain",
    "content_retained",
    "execution_target",
    "run_configuration",
    "capture",
    "provenance",
)

EVENT_REQUIRED = (
    "schema_version",
    "run_id",
    "producer_id",
    "sequence",
    "monotonic_ns",
    "event_type",
)

# Structural events must not carry recoverable prompt or output content.
STRUCTURAL_FORBIDDEN_FIELDS = frozenset(
    {
        "token_ids",
        "prompt_token_ids",
        "text",
        "text_delta",
        "logprobs",
        "prompt",
    }
)

EVENT_TYPES = frozenset(
    {
        "request_received",
        "request_tokenized",
        "request_queued",
        "request_dispatched",
        "request_dispatch_completed",
        "http_request_received",
        "http_engine_admission_correlated",
        "http_response_completed",
        "request_model_route",
        "engine_prompt_observed",
        "engine_admitted",
        "capture_ready",
        "step_scheduled",
        "step_submitted",
        "step_result_received",
        "prefix_lookup",
        "block_change",
        "tokens_accepted",
        "text_emitted",
        "preemption",
        "cancellation",
        "completion",
        "trace_gap",
    }
)
