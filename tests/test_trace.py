"""Adapter snapshots, fixture traces, and the report."""

import json
from types import SimpleNamespace

from vllm_030.capture import classify_token_range, snapshot_scheduled_step
from vllm_030.collector import Recording
from vllm_030.constants import MANIFEST_REQUIRED
from vllm_030.install import install

from runner.export import export_structural
from runner.fixture_engine import FixtureEngine
from runner.lessons import write_catalog
from runner.load import UnsupportedRecordingError, load_recording
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.report import evaluate, instrumentation_overhead


def test_scheduled_cursor_comes_from_the_scheduler_output():
    live = SimpleNamespace(num_computed_tokens=99, num_prompt_tokens=4)
    new_request = SimpleNamespace(req_id="a", num_computed_tokens=0, prompt_len=4, block_ids=([1],))
    output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            num_computed_tokens=[],
            new_block_ids=[],
        ),
        num_scheduled_tokens={"a": 4},
        total_num_scheduled_tokens=4,
        preempted_req_ids=set(),
    )
    snapshot = snapshot_scheduled_step(output, {"a": live}, token_budget=16)
    request = snapshot["requests"][0]
    assert request["range_start"] == 0
    assert request["prompt_tokens"] == 4
    assert request["generation_tokens"] == 0
    assert "prompt_token_ids" not in json.dumps(snapshot)


def test_a_range_can_cross_the_prompt_boundary():
    assert classify_token_range(3, 2, 4) == (1, 1)


def test_result_stays_with_the_step_that_was_scheduled(tmp_path):
    recording = Recording.open(tmp_path / "steps", run_id="steps")
    engine = FixtureEngine()
    adapter = install(engine, recording, check_sources=False)
    engine.add_request("request-a", [1, 2, 3, 4])
    first = engine.scheduler.schedule()
    second = engine.scheduler.schedule()
    engine.scheduler.update_from_output(first, SimpleNamespace())
    adapter.uninstall()
    # This fixture deliberately stops with its second scheduled step pending.
    recording.close({**_manifest("steps"), "completion_status": "cancelled"})
    data = load_recording(tmp_path / "steps")
    results = [event for event in data.events if event["event_type"] == "step_result_received"]
    assert results[0]["step_id"] == first.explorer_step_id
    assert results[0]["step_id"] != second.explorer_step_id
    report = evaluate(data)
    assert report.ok, report.text()


def test_block_reassignment_gets_a_new_generation(tmp_path):
    recording = Recording.open(tmp_path / "blocks", run_id="blocks")
    engine = FixtureEngine(blocks=1)
    adapter = install(engine, recording, check_sources=False)
    pool = engine.pool
    block = pool.get_new_blocks(1)[0]
    block.block_hash = object()
    pool.free_blocks([block])
    pool.touch([block])
    pool.free_blocks([block])
    pool.get_new_blocks(1)
    adapter.uninstall()
    recording.close(_manifest("blocks"))
    data = load_recording(tmp_path / "blocks")
    report = evaluate(data)
    assert report.ok, report.text()
    allocates = [
        event["allocation_generation"]
        for event in data.events
        if event["event_type"] == "block_change" and event["operation"] == "allocate"
    ]
    assert allocates == [1, 2]
    cached = [
        event
        for event in data.events
        if event["event_type"] == "block_change" and event["cached_at_zero_refs"]
    ]
    assert cached
    assert "cached" in report.text()


def test_preemption_event_has_no_invented_reason(tmp_path):
    recording = Recording.open(tmp_path / "preempt", run_id="preempt")
    engine = FixtureEngine()
    adapter = install(engine, recording, check_sources=False)
    engine.add_request("request-a", [1, 2, 3, 4])
    request = engine.scheduler.requests["request-a"]
    request.num_computed_tokens = 4
    engine.scheduler._preempt_request(request, 0.0)
    adapter.uninstall()
    recording.close(_manifest("preempt"))
    data = load_recording(tmp_path / "preempt")
    event = next(item for item in data.events if item["event_type"] == "preemption")
    assert "reason" not in event
    assert event["computed_tokens_after"] == 0
    assert event["preemption_count"] == 1


def test_catalog_lessons_pass_integrity_and_reject_an_unknown_schema(tmp_path):
    write_catalog(tmp_path / "catalog")
    index = json.loads((tmp_path / "catalog" / "index.json").read_text())
    assert {item["id"] for item in index["recordings"]} >= {
        "continuous-batching",
        "chunked-prefill",
        "prefix-reuse",
        "prefix-miss",
        "preemption",
        "cancellation",
        "incomplete-capture",
    }
    for item in index["recordings"]:
        data = load_recording(tmp_path / "catalog" / item["id"])
        report = evaluate(data)
        assert report.ok, item["id"] + "\n" + report.text()
        assert data.manifest["execution_kind"] == "fixture"
    batch = load_recording(tmp_path / "catalog" / "continuous-batching")
    feasibility = evaluate(batch, profile="feasibility")
    assert not feasibility.ok
    assert any(finding.code == "gpu_execution" and not finding.ok for finding in feasibility.findings)
    exported = export_structural(tmp_path / "catalog" / "single-request", tmp_path / "structural")
    structural = load_recording(exported)
    assert structural.manifest["content_retained"] is False
    assert not (exported / "content.jsonl").exists()
    assert structural.events == load_recording(tmp_path / "catalog" / "single-request").events
    broken = dict(batch.manifest)
    broken["schema_version"] = "9.0.0"
    (tmp_path / "catalog" / "continuous-batching" / "manifest.json").write_text(json.dumps(broken))
    try:
        load_recording(tmp_path / "catalog" / "continuous-batching")
        raise AssertionError("expected unsupported schema")
    except UnsupportedRecordingError as exc:
        assert "unsupported recording schema" in str(exc)


def test_schema_file_lists_the_manifest_contract():
    schema = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1] / "trace-schema" / "recording-v1.schema.json"
        ).read_text()
    )
    assert tuple(schema["$defs"]["manifest"]["required"]) == MANIFEST_REQUIRED


def test_overhead_note_is_labeled():
    note = instrumentation_overhead([100, 120], [80, 90])
    assert note["label"] == "instrumentation overhead"
    assert "not a performance benchmark" in note["disclaimer"]


def _manifest(run_id: str) -> dict:
    return manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
