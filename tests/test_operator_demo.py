"""The authored workload reaches every capture variant without changing engine routes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("operator_prepare", ROOT / "docs/qualification/realistic-demos/prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
capture = prepare.capture
assessment = capture.assessment


@pytest.mark.parametrize("variant", prepare.VARIANTS)
def test_every_variant_parses_corpus_and_preserves_precision_model_and_adapter_routes(variant):
    source = prepare.ARCHIVED_PLANS / f"{variant}-request.json"
    before = source.read_bytes()
    archived = json.loads(before)
    payload = prepare.plans()[variant]
    corpus = json.loads(capture.OPERATOR_CORPUS.read_text())
    requests, plan = capture.qualification.parse_run_workload(payload)
    configuration = capture.qualification.parse_run_configuration(payload, requests, plan)
    assert source.read_bytes() == before
    assert payload["model"] == archived["model"]
    assert payload["model_revision"] == archived["model_revision"]
    assert payload["tokenizer_revision"] == archived["tokenizer_revision"]
    assert payload["dtype"] == archived["dtype"]
    assert payload["quantization"] == archived["quantization"]
    assert payload["kv_cache_dtype"] == archived["kv_cache_dtype"]
    assert payload.get("model_configuration") == archived.get("model_configuration")
    old_routes = {row["request_id"]: row.get("adapter_id") or "base" for row in archived["requests"]}
    assert {row.request_id: row.adapter_id for row in requests} == old_routes
    assert {sample["id"]: sample["adapter_id"] for group in plan["batches"] for sample in group["samples"]} == old_routes
    assert [len(group["samples"]) for group in plan["batches"]] == [1, 2, 3]
    second = 40000 if variant in {"lora-all", "lora-mixed"} else 20000
    assert [group["arrival_offset_ms"] for group in plan["batches"]] == [0, second, second + 25]
    assert [row.arrival_offset_ms for row in requests] == [0, second, second, second + 25, second + 25, second + 25]
    for request, authored in zip(requests, corpus["requests"]):
        assert request.request_id == authored["id"]
        assert request.prompt == capture.qualification.chat(f'{corpus["documents"][authored["document"]]}\n\nTask\n{authored["task"]}')
        assert request.max_output_tokens == authored["max_output_tokens"]
    assert [row.max_output_tokens for row in requests] == [64, 96, 128, 160, 192, 256]
    assert configuration["vllm"]["max_model_len"] == archived["max_model_len"]
    assert configuration["vllm"]["kv_cache_memory_bytes"] == archived["kv_cache_memory_bytes"]
    assert configuration["profiling"]["max_steps"] == 1024
    assert configuration["vllm"]["max_num_scheduled_tokens"] == (1024 if variant == "qwen38-27b" else 64)
    assert configuration["vllm"]["long_prefill_token_threshold"] == (784 if variant == "qwen38-27b" else 64)
    assert payload["profiling_configuration"]["tensor_capture"] == archived["profiling_configuration"]["tensor_capture"]
    assert "demo_qualification" not in configuration
    assert "min_tokens" not in json.dumps(configuration) and "ignore_eos" not in json.dumps(configuration)


def test_shared_document_prefix_and_custom_offsets_survive_capture_default_and_variant_plans():
    workload = capture.workload(second_ms=30000, third_ms=30100)
    prepared = prepare.plans(second_ms=30000, third_ms=30100)
    assert workload["requests"] == prepared["bf16"]["requests"]
    assert workload["workload_plan"] == prepared["bf16"]["workload_plan"]
    assert [row["arrival_offset_ms"] for row in workload["requests"]] == [0, 30000, 30000, 30100, 30100, 30100]
    for payload in prepared.values():
        assert [batch["arrival_offset_ms"] for batch in payload["workload_plan"]["batches"]] == [0, 30000, 30100]
        assert [row["arrival_offset_ms"] for row in payload["requests"]] == [0, 30000, 30000, 30100, 30100, 30100]
    corpus = json.loads(capture.OPERATOR_CORPUS.read_text())
    shared_ids = corpus["qualification"]["shared_request_ids"]
    shared = [row for row in corpus["requests"] if row["id"] in shared_ids]
    assert len({row["document"] for row in shared}) == 1
    assert len({row["task"] for row in shared}) == len(shared)


def test_arrival_overrides_require_both_values():
    with pytest.raises(ValueError, match="both second_ms and third_ms"):
        prepare.plans(second_ms=30000)
    with pytest.raises(ValueError, match="both second_ms and third_ms"):
        prepare.plans(third_ms=30100)


def output_case(*, actual=64, cap=96, reason="eos", target=True):
    expected = {"requests": [{"request_id": "request-1", "max_output_tokens": cap}]}
    if target:
        expected["demo_qualification"] = {"minimum_output_tokens": 64, "maximum_output_tokens": 256}
    manifest = {"engine_output_token_ids": {"request-1": list(range(actual))}}
    events = [{"event_type": "tokens_accepted", "request_id": "request-1", "token_count": actual},
              {"event_type": "completion", "request_id": "request-1", "reason": reason}]
    return manifest, events, expected


def test_output_qualification_rejects_early_eos_without_requiring_full_cap():
    assert all(assessment.output_evidence(*output_case(actual=64, cap=96))["checks"].values())
    short = assessment.output_evidence(*output_case(actual=63, cap=96))
    assert short["checks"]["all_requests_returned_within_configured_output_limits"]
    assert not short["checks"]["demo_actual_outputs_within_qualification_range"]
    assert not assessment.output_evidence(*output_case(actual=97, cap=96))["checks"]["all_requests_returned_within_configured_output_limits"]
    assert all(assessment.output_evidence(*output_case(actual=256, cap=256, reason="length"))["checks"].values())
    assert not assessment.output_evidence(*output_case(actual=257, cap=256))["checks"]["demo_actual_outputs_within_qualification_range"]


def test_output_evidence_keeps_legacy_six_token_recordings_assessable_and_checks_ids_and_completion():
    manifest, events, expected = output_case(actual=6, cap=6, reason="length", target=False)
    assert all(assessment.output_evidence(manifest, events, expected)["checks"].values())
    assert not assessment.output_evidence(manifest, events[:-1], expected)["checks"]["all_requests_returned_within_configured_output_limits"]
    wrong_ids = deepcopy(manifest)
    wrong_ids["engine_output_token_ids"]["request-1"].pop()
    assert not assessment.output_evidence(wrong_ids, events, expected)["checks"]["all_requests_returned_within_configured_output_limits"]
    assert not assessment.output_evidence(*output_case(actual=0))["checks"]["all_requests_returned_within_configured_output_limits"]


def test_qualification_checks_exact_recorded_inputs_not_only_ids():
    payload = prepare.plans()["bf16"]
    content = tuple({"kind": "prompt_text", "request_id": row["request_id"], "text": row["prompt"]} for row in payload["requests"])
    assert all(assessment.prompt_evidence(content, payload)["checks"].values())
    changed = deepcopy(content)
    changed[0]["text"] = "older six-token demo prompt"
    assert not all(assessment.prompt_evidence(changed, payload)["checks"].values())
    assert not all(assessment.prompt_evidence(content[1:], payload)["checks"].values())
    assert not all(assessment.prompt_evidence(content + (content[0],), payload)["checks"].values())
