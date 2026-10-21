"""Learning controls and observations retain execution/content boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from enum import Enum, IntEnum
from types import SimpleNamespace

import pytest

from runner.architecture import describe_architecture, fixture_architecture
from runner.configuration import ConfigurationError, apply_legacy_fields, parse_run_configuration
from runner.controller import drive_run
from runner.export import export_structural
from runner.fixture_engine import FixtureEngine, FixtureSession
from runner.load import load_recording
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.report import evaluate
from runner.revisions import resolve_model_revisions
from runner.serve import RunRequest
from runner.vllm_session import _add_engine_request
from runner.workload import WorkloadError, WorkRequest, parse_run_workload
from vllm_030.collector import Recording
from vllm_030.install import install
from vllm_030.compatibility import CompatibilityError
from vllm_030.inventory import token_slot_mapping


def configuration(**settings):
    payload = {"requests": [{"request_id": "a", "prompt": "hello", "arrival_offset_ms": 0, "max_output_tokens": 3}],
               "vllm_configuration": settings}
    requests, plan = parse_run_workload(payload)
    return parse_run_configuration(payload, requests, plan)


def test_hub_resolution_pins_one_snapshot_and_defaults_tokenizer_to_it():
    calls = []
    body = SimpleNamespace(model="org/model", model_revision=None, tokenizer_revision=None,
                           tokenizer=None, model_source_kind="huggingface", huggingface_token="private")
    def info(repository, **kwargs):
        calls.append((repository, kwargs))
        return SimpleNamespace(sha="a" * 40)
    result = resolve_model_revisions(body, model_info=info)
    assert len(calls) == 1
    assert calls[0][1]["revision"] == "main"
    assert body.model_revision == body.tokenizer_revision == "a" * 40
    assert body.model_configuration["tokenizer"]["revision"] == "a" * 40
    assert result["model"]["requested_revision"] == "main"
    assert "private" not in json.dumps(result)


def test_hub_resolution_honors_distinct_tokenizer_and_never_records_auth_errors():
    body = SimpleNamespace(model="org/model", model_revision="release", tokenizer="org/tokenizer",
                           tokenizer_revision="v2", model_source_kind="huggingface", huggingface_token="private")
    result = resolve_model_revisions(body, model_info=lambda repo, **kwargs: SimpleNamespace(sha=("a" if repo == "org/model" else "b") * 40))
    assert result["tokenizer"]["resolved_revision"] == "b" * 40
    assert body.model_revision != body.tokenizer_revision
    body.model_revision = "missing"
    def fail(*args, **kwargs):
        raise ValueError("response contained private")
    with pytest.raises(RuntimeError, match="org/model.*ValueError") as error:
        resolve_model_revisions(body, model_info=fail)
    assert "private" not in str(error.value)


def test_explicit_commit_and_host_paths_do_not_need_hub_network():
    def fail(*args, **kwargs):
        raise AssertionError("network was unnecessary")
    body = SimpleNamespace(model="org/model", model_revision="C" * 40, tokenizer_revision=None, tokenizer=None)
    resolve_model_revisions(body, model_info=fail)
    assert body.tokenizer_revision == "c" * 40
    local = SimpleNamespace(model="/models/qwen", model_source_kind="execution_host")
    assert resolve_model_revisions(local, model_info=fail)["source"] == "execution_host_artifact"


@pytest.mark.parametrize("values,message", [
    ({"max_num_seqs": 4, "max_num_scheduled_tokens": 2}, "concurrent sequences"),
    ({"enable_chunked_prefill": False, "max_num_scheduled_tokens": 128, "max_model_len": 256}, "context length"),
    ({"long_prefill_token_threshold": 257, "max_model_len": 256}, "threshold"),
    ({"enable_chunked_prefill": False, "long_prefill_token_threshold": 16, "max_model_len": 128}, "requires chunked"),
    ({"kv_cache_memory_bytes": 0}, "KV memory bytes"),
])
def test_configuration_rejects_exact_runtime_relationships(values, message):
    with pytest.raises(ConfigurationError, match=message):
        configuration(**values)


def test_priority_and_learning_settings_survive_configuration_and_engine_admission():
    raw = {"requests": [{"request_id": "a", "prompt": "hi", "arrival_offset_ms": 0, "max_output_tokens": 3, "priority": -2}],
           "vllm_configuration": {"scheduling_policy": "priority", "long_prefill_token_threshold": 32,
                                  "kv_cache_memory_bytes": 1048576}}
    requests, plan = parse_run_workload(raw)
    snapshot = parse_run_configuration(raw, requests, plan)
    body = RunRequest(**raw)
    apply_legacy_fields(body, snapshot)
    assert body.scheduling_policy == "priority"
    assert body.long_prefill_token_threshold == 32 and body.kv_cache_memory_bytes == 1048576
    assert plan["batches"][0]["samples"][0]["priority"] == -2
    calls = []
    engine = SimpleNamespace(add_request=lambda *args, **kwargs: calls.append(kwargs))
    _add_engine_request(engine, "a", {}, object(), None, priority=requests[0].priority)
    assert calls == [{"priority": -2}]
    with pytest.raises(ConfigurationError, match="require.*priority"):
        parse_run_configuration({**raw, "vllm_configuration": {}}, requests, plan)
    raw["requests"][0]["priority"] = True
    with pytest.raises(WorkloadError, match="priority"):
        parse_run_workload(raw)


def test_architecture_uses_actual_head_dimension_and_declares_unsupported_models():
    reference = fixture_architecture()
    assert reference["source"] == "pinned_model_config_reference"
    assert reference["num_layers"] == 36 and reference["num_attention_heads"] == 32
    assert reference["num_key_value_heads"] == 8 and reference["head_dim"] == 128
    assert reference["hidden_size"] == 2560  # Qwen3's head dimension is not hidden/query-heads.
    unknown = describe_architecture(SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_moe")))
    assert unknown["supported"] is False and "dense Qwen3" in unknown["reason"]


def test_reservation_is_not_a_write_and_initial_inventory_includes_null_blocks(tmp_path):
    recording = Recording.open(tmp_path / "reserved", run_id="reserved")
    engine = FixtureEngine(blocks=3)
    engine.pool.blocks[2].is_null = True
    adapter = install(engine, recording, check_sources=False)
    engine.add_request("a", [11, 12, 13])
    output = engine.scheduler.schedule()
    adapter.uninstall()
    recording.close(manifest_skeleton("reserved", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="cancelled"))
    events = load_recording(tmp_path / "reserved").events
    inventory = next(event for event in events if event["event_type"] == "cache_inventory")
    assert inventory["pools"][0]["inventory_complete"] is True
    assert [block["block_id"] for block in inventory["pools"][0]["blocks"]] == [0, 1, 2]
    assert inventory["pools"][0]["blocks"][2]["is_null"] is True
    step = next(event for event in events if event["event_type"] == "step_scheduled")
    assert step["step_id"] == output.explorer_step_id
    assert [item["request_id"] for item in step["queue_before"]["waiting"]] == ["a"]
    assert [item["request_id"] for item in step["queue_after"]["running"]] == ["a"]
    assert any(event["event_type"] == "kv_block_binding" for event in events)
    assert not any(event["event_type"] == "kv_block_written" for event in events)


def test_local_teaching_capture_keeps_final_sample_out_of_kv_and_exports_no_content(tmp_path):
    recording = Recording.open(tmp_path / "teaching", run_id="teaching")
    session = FixtureSession(recording, max_num_seqs=1, token_budget=32, step_interval_s=0)
    status, tokens, arrivals = drive_run(session, recording, [WorkRequest("a", "one two three", 0, 2)])
    session.close()
    manifest = manifest_skeleton("teaching", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, arrivals=arrivals)
    manifest.update(architecture=session.architecture_record, cache_geometry=session.cache_geometry, engine_output_token_ids=tokens)
    recording.close(manifest)
    data = load_recording(tmp_path / "teaching")
    assert status == "completed" and len(tokens["a"]) == 2
    assert evaluate(data).ok, evaluate(data).text()
    writes = [event for event in data.events if event["event_type"] == "kv_block_written"]
    assert [(event["token_range_start"], event["token_range_end"]) for event in writes] == [(0, 3), (3, 4)]
    assert max(event["token_range_end"] for event in writes) == 4  # Prompt3 + only previous output1.
    assert any(event["event_type"] == "block_change" and event["operation"] == "release" for event in data.events)
    kv = [item for item in data.content if item["kind"] == "kv_tokens"]
    assert kv[1]["token_ids"] == tokens["a"][:1]
    assert all(item["step_id"] and item["cache_group"] == 0 for item in kv)
    export_structural(tmp_path / "teaching", tmp_path / "export")
    structural = load_recording(tmp_path / "export")
    assert structural.events == data.events and structural.manifest["content_retained"] is False
    assert not structural.content


class _FinishStatus(Enum):
    FINISHED_ABORTED = "abort"
    FINISHED_STOPPED = "stop"
    FINISHED_LENGTH_CAPPED = "length"
    FINISHED_ERROR = "error"


def test_native_abort_boundary_records_zero_output_prefill_after_cleanup(tmp_path):
    recording = Recording.open(tmp_path / "abort", run_id="abort")
    engine = FixtureEngine(authored=True, max_num_seqs=1, max_num_scheduled_tokens=2)
    returned = []

    def finish_requests(request_ids, finished_status):
        assert request_ids == ["a", "already-finished", "unknown"]
        request = engine.scheduler.requests["a"]
        request.status = finished_status
        engine.pool.free_blocks(engine.scheduler._block_tables.pop("a"))
        engine.scheduler.queue.remove("a")
        returned.append(request)  # Native return excludes completed/unknown IDs.
        return returned

    engine.scheduler.finish_requests = finish_requests
    adapter = install(engine, recording, check_sources=False)
    engine.add_request("a", [11, 12, 13, 14, 15, 16], max_output_tokens=3)
    assert engine.step() == []  # Partial prefill has not sampled an output.
    assert engine.scheduler.finish_requests(
        ["a", "already-finished", "unknown"], _FinishStatus.FINISHED_ABORTED,
    ) is returned
    adapter.uninstall()
    assert engine.scheduler.finish_requests is finish_requests
    manifest = manifest_skeleton("abort", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="cancelled")
    manifest["engine_output_token_ids"] = {"a": []}
    recording.close(manifest)
    data = load_recording(tmp_path / "abort")
    cancellations = [event for event in data.events if event["event_type"] == "cancellation"]
    assert len(cancellations) == 1
    event = cancellations[0]
    assert event["request_id"] == "a" and event["status"] == event["reason"] == "abort"
    assert event["request_status"] == "FINISHED_ABORTED"
    assert event["observation_boundary"] == "scheduler.finish_requests.returned"
    assert "step_id" not in event  # Abort is not a model execution/result claim.
    assert next(item for item in data.events if item["event_type"] == "block_change" and item["operation"] == "release")["sequence"] < event["sequence"]
    assert not any(item["event_type"] == "tokens_accepted" for item in data.events)
    assert evaluate(data).ok, evaluate(data).text()


@pytest.mark.parametrize("outcome", ["no_valid_requests", "exception", "stop", "length", "error"])
def test_finish_boundary_never_infers_abort_from_requested_ids_or_other_finishes(tmp_path, outcome):
    recording = Recording.open(tmp_path / outcome, run_id=outcome)
    engine = FixtureEngine()
    request = SimpleNamespace(request_id="a", status=_FinishStatus.FINISHED_ABORTED)
    returned = [] if outcome == "no_valid_requests" else [request]

    def finish_requests(request_ids, finished_status):
        assert request_ids == ["a", "unknown"]
        if outcome == "exception":
            # Even a status mutation does not prove successful abort cleanup.
            raise RuntimeError("cleanup failed")
        request.status = finished_status
        return returned

    engine.scheduler.finish_requests = finish_requests
    adapter = install(engine, recording, check_sources=False)
    status = next((item for item in _FinishStatus if item.value == outcome), _FinishStatus.FINISHED_ABORTED)
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="cleanup failed"):
            engine.scheduler.finish_requests(["a", "unknown"], status)
    else:
        assert engine.scheduler.finish_requests(["a", "unknown"], finished_status=status) is returned
    adapter.uninstall()
    recording.close(manifest_skeleton(outcome, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    assert not any(event["event_type"] in {"cancellation", "completion"} for event in load_recording(tmp_path / outcome).events)


@pytest.mark.parametrize("corruption,detail", [
    ("write_before_result", "preceding observed result"),
    ("stale_generation", "stale block occupant"),
    ("unscheduled_position", "not scheduled"),
    ("missing_inventory_slot", "every physical block"),
])
def test_learning_integrity_rejects_unobserved_kv_claims(tmp_path, corruption, detail):
    recording = Recording.open(tmp_path / "observed", run_id="observed")
    session = FixtureSession(recording, max_num_seqs=1, token_budget=32, step_interval_s=0)
    _status, tokens, arrivals = drive_run(session, recording, [WorkRequest("a", "one two three", 0, 2)])
    session.close()
    manifest = manifest_skeleton("observed", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, arrivals=arrivals)
    manifest["engine_output_token_ids"] = tokens
    recording.close(manifest)
    data = load_recording(tmp_path / "observed")
    events = [dict(event) for event in data.events]
    write_index = next(index for index, event in enumerate(events) if event["event_type"] == "kv_block_written")
    write = events[write_index]
    if corruption == "write_before_result":
        result_index = next(index for index, event in enumerate(events) if event["event_type"] == "step_result_received")
        events.insert(result_index, events.pop(write_index))
    elif corruption == "stale_generation":
        write["allocation_generation"] += 1
    elif corruption == "unscheduled_position":
        write.update(token_range_start=10, token_range_end=11)
    else:
        inventory = next(event for event in events if event["event_type"] == "cache_inventory")
        inventory["pools"][0]["blocks"].pop()
    finding = next(item for item in evaluate(replace(data, events=tuple(events))).findings if item.code == "kv_observations")
    assert not finding.ok and detail in finding.detail


def test_tokenizer_fragments_remain_optional_content(tmp_path):
    recording = Recording.open(tmp_path / "fragments", run_id="fragments")
    engine = FixtureEngine()
    adapter = install(engine, recording, check_sources=False, token_pieces=lambda ids: [f"decoded:{token}" for token in ids])
    engine.add_request("a", [11, 12, 13])
    output = engine.scheduler.schedule()
    result = engine.model_executor.execute_model(output).result()
    engine.scheduler.update_from_output(output, result)
    adapter.uninstall()
    recording.close(manifest_skeleton("fragments", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    data = load_recording(tmp_path / "fragments")
    assert all("token_pieces" not in event for event in data.events)
    for attachment in data.content:
        if attachment["kind"] in {"prompt", "accepted_tokens", "kv_tokens"}:
            assert attachment["token_pieces"] == [f"decoded:{token}" for token in attachment["token_ids"]]
            assert "not original-text slices" in attachment["token_piece_semantics"]
    export_structural(tmp_path / "fragments", tmp_path / "no-fragments")
    assert not load_recording(tmp_path / "no-fragments").content


class _NativeFinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return ("stop", "length", "abort")[self.value]


@pytest.mark.parametrize("reason,event_type,label", [
    (None, None, None),
    (_NativeFinishReason.STOP, "completion", "stop"),
    (_NativeFinishReason.LENGTH, "completion", "length"),
    (_NativeFinishReason.ABORT, "cancellation", "abort"),
])
def test_returned_native_finish_reason_zero_is_terminal_while_none_is_unfinished(tmp_path, reason, event_type, label):
    recording = Recording.open(tmp_path / "native-finish", run_id="native-finish")
    engine = FixtureEngine()
    returned = {0: SimpleNamespace(outputs=[SimpleNamespace(request_id="a", new_token_ids=[101], finish_reason=reason)])}
    engine.scheduler.update_from_output = lambda *_args, **_kwargs: returned
    adapter = install(engine, recording, check_sources=False)
    engine.add_request("a", [11, 12, 13])
    scheduler_output = engine.scheduler.schedule()
    result = engine.model_executor.execute_model(scheduler_output).result()
    assert engine.scheduler.update_from_output(scheduler_output, result) is returned
    adapter.uninstall()
    recording.close(manifest_skeleton("native-finish", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    events = load_recording(tmp_path / "native-finish").events
    accepted = next(event for event in events if event["event_type"] == "tokens_accepted")
    terminal = [event for event in events if event["event_type"] in {"completion", "cancellation"}]
    assert accepted["token_count"] == 1
    if event_type is None:
        assert terminal == []
    else:
        assert len(terminal) == 1
        assert terminal[0]["event_type"] == event_type
        assert terminal[0]["reason"] == terminal[0]["status"] == label
        assert terminal[0]["sequence"] > accepted["sequence"]
        assert terminal[0]["step_id"] == accepted["step_id"]


@pytest.mark.parametrize("corruption", ["absent", "missing_cell", "duplicate_cell", "nonrange_cell"])
def test_install_rejects_bad_initial_inventory_before_patching(tmp_path, corruption):
    recording = Recording.open(tmp_path / corruption, run_id=corruption)
    engine = FixtureEngine(blocks=3)
    before = engine.scheduler.schedule
    if corruption == "absent":
        engine.pool.blocks = None
    elif corruption == "missing_cell":
        engine.pool.blocks.pop()
    elif corruption == "duplicate_cell":
        engine.pool.blocks[-1].block_id = 1
    else:
        engine.pool.blocks[-1].block_id = 4
    with pytest.raises(CompatibilityError, match="every physical block"):
        install(engine, recording, check_sources=False)
    assert engine.scheduler.schedule == before
    recording.close(manifest_skeleton(corruption, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="failed"))
    assert not load_recording(tmp_path / corruption).events


@pytest.mark.parametrize("omit_untouched", [False, True])
def test_report_rejects_incomplete_inventory_even_without_block_usage(tmp_path, omit_untouched):
    recording = Recording.open(tmp_path / "incomplete", run_id="incomplete")
    adapter = install(FixtureEngine(blocks=3), recording, check_sources=False)
    adapter.uninstall()
    recording.close(manifest_skeleton("incomplete", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="cancelled"))
    data = load_recording(tmp_path / "incomplete")
    events = list(data.events)
    inventory = events[0]
    inventory["pools"][0]["inventory_complete"] = False
    if omit_untouched:
        inventory["pools"][0]["blocks"].pop()
    finding = next(item for item in evaluate(replace(data, events=tuple(events))).findings if item.code == "kv_observations")
    assert not finding.ok and "incomplete" in finding.detail


@pytest.fixture
def cache_specs(monkeypatch):
    class FullAttentionSpec(SimpleNamespace):
        pass
    class UniformTypeKVCacheSpecs(SimpleNamespace):
        pass
    monkeypatch.setattr("vllm_030.inventory._spec_types", lambda: (FullAttentionSpec, UniformTypeKVCacheSpecs))
    def full(**values):
        return FullAttentionSpec(**{"block_size": 16, "tokens_per_state": 1, "uses_slot_mapping": True,
                                    "sliding_window": None, "attention_chunk_size": None, "non_causal": False, **values})
    scheduler = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1), dcp_world_size=1, pcp_world_size=1)
    return full, FullAttentionSpec, UniformTypeKVCacheSpecs, scheduler


@pytest.mark.parametrize("wrapper", [False, True])
def test_exact_dense_full_attention_and_uniform_wrapper_are_qualified(cache_specs, wrapper):
    full, _type, uniform, scheduler = cache_specs
    spec = uniform(block_size=16, kv_cache_specs={"a": full(), "b": full()}) if wrapper else full()
    status, _reason = token_slot_mapping(scheduler, spec, 16, authored_fixture=False, pool_count=1)
    assert status == "qualified_full_attention"


@pytest.mark.parametrize("alteration", ["mla", "hidden_state", "rswa", "sink", "compressed", "no_slot_mapping", "sliding", "chunked", "noncausal", "inner_size", "empty_wrapper", "dcp", "pcp", "two_gpu"])
def test_specialized_cache_layouts_are_not_assigned_token_cells(cache_specs, alteration):
    full, full_type, uniform, scheduler = cache_specs
    first = full()
    if alteration in {"mla", "hidden_state", "rswa", "sink"}:
        first = type(alteration, (full_type,), {})(**vars(first))
    elif alteration in {"compressed", "no_slot_mapping", "sliding", "chunked", "noncausal", "inner_size"}:
        field, value = {"compressed": ("tokens_per_state", 2), "no_slot_mapping": ("uses_slot_mapping", False),
                        "sliding": ("sliding_window", 128), "chunked": ("attention_chunk_size", 128),
                        "noncausal": ("non_causal", True), "inner_size": ("block_size", 8)}[alteration]
        setattr(first, field, value)
    elif alteration == "dcp":
        scheduler.dcp_world_size = 2
    elif alteration == "pcp":
        scheduler.pcp_world_size = 2
    elif alteration == "two_gpu":
        scheduler.parallel_config.tensor_parallel_size = 2
    spec = uniform(block_size=16, kv_cache_specs={} if alteration == "empty_wrapper" else {"good": full(), "unsupported": first})
    status, reason = token_slot_mapping(scheduler, spec, 16, authored_fixture=False, pool_count=1)
    assert status == "unavailable" and reason


def test_mixed_groups_keep_physical_inventory_but_emit_only_qualified_token_ranges(tmp_path, cache_specs):
    full, full_type, _uniform, scheduler = cache_specs
    unsupported = type("SinkAttentionSpec", (full_type,), {})(**vars(full()))
    engine = FixtureEngine(blocks=3)
    engine.explorer_token_slot_mapper = None
    engine.scheduler.parallel_config = scheduler.parallel_config
    engine.scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=full(), layer_names=["full"]), SimpleNamespace(kv_cache_spec=unsupported, layer_names=["sink"])])
    original_blocks = engine.scheduler.kv_cache_manager.get_blocks
    def both_groups(request_id):
        observed = original_blocks(request_id)
        groups = observed.blocks
        return SimpleNamespace(blocks=groups * 2, get_block_ids=lambda: observed.get_block_ids() * 2)
    engine.scheduler.kv_cache_manager.get_blocks = both_groups
    recording = Recording.open(tmp_path / "mixed", run_id="mixed")
    adapter = install(engine, recording, check_sources=False)
    engine.add_request("a", [11, 12, 13])
    engine.step()
    adapter.uninstall()
    recording.close(manifest_skeleton("mixed", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="cancelled"))
    data = load_recording(tmp_path / "mixed")
    inventory = next(event for event in data.events if event["event_type"] == "cache_inventory")
    assert len(inventory["pools"][0]["blocks"]) == 3
    assert [group["token_slot_mapping"] for group in inventory["groups"]] == ["qualified_full_attention", "unavailable"]
    ranges = [event for event in data.events if event["event_type"] in {"kv_block_binding", "kv_block_written"}]
    assert ranges and {event["cache_group"] for event in ranges} == {0}
    unsupported_claim = {**ranges[0], "cache_group": 1}
    report = evaluate(replace(data, events=data.events + (unsupported_claim,)))
    finding = next(item for item in report.findings if item.code == "kv_observations")
    assert not finding.ok and "mapping is unavailable" in finding.detail


def test_hybrid_architecture_keeps_recurrent_state_separate_from_attention_kv():
    config = SimpleNamespace(model_type="qwen3_5_text", architectures=["Qwen3_5ForConditionalGeneration"],
        num_hidden_layers=4, hidden_size=5120, intermediate_size=17408, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4)
    result = describe_architecture(SimpleNamespace(hf_text_config=config))
    assert result["supported"] is True
    assert result["attention_layer_indices"] == [3]
    assert result["recurrent_layer_indices"] == [0, 1, 2]
    assert result["recurrent"]["value_heads"] == 48
    config.layer_types = ["full_attention"]
    assert describe_architecture(SimpleNamespace(hf_text_config=config))["supported"] is False
