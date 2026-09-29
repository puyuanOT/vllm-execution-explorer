"""Collector buffer rules."""

import time

import pytest

from vllm_030.collector import Recording, RecordingError
from vllm_030.constants import MANIFEST_REQUIRED

from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton


def _manifest(run_id: str) -> dict:
    return manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)


def test_buffer_exhaustion_keeps_earlier_events_and_marks_the_run_incomplete(tmp_path):
    recording = Recording.open(tmp_path / "gap", run_id="gap", capacity=3, autostart=False)
    first = recording.event("request_queued", request_id="a", status="queued")
    second = recording.event("request_queued", request_id="b", status="queued")
    third = recording.event("request_queued", request_id="c", status="queued")
    fourth = recording.event("request_queued", request_id="d", status="queued")
    assert first == 1
    assert second == 2
    assert third is None
    assert fourth is None
    manifest = recording.close(_manifest("gap"))
    events = [
        __import__("json").loads(line)
        for line in (tmp_path / "gap" / "events.jsonl").read_text().splitlines()
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert events[2]["event_type"] == "trace_gap"
    assert events[2]["missing_sequence_start"] == 4
    assert events[0]["event_type"] == "request_queued"
    completeness = manifest["event_completeness"]
    assert completeness["status"] == "incomplete"
    assert completeness["gap"]["missing_sequence_end"] == 5
    assert manifest["completion_status"] == "incomplete"
    assert set(MANIFEST_REQUIRED).issubset(manifest)


def test_structural_events_reject_token_text(tmp_path):
    recording = Recording.open(tmp_path / "bad", run_id="bad", autostart=False)
    with pytest.raises(RecordingError, match="content fields"):
        recording.event("tokens_accepted", token_ids=[1])
    recording.close(_manifest("bad"))


def test_a_full_buffer_does_not_block(tmp_path):
    recording = Recording.open(tmp_path / "fast", run_id="fast", capacity=3, autostart=False)
    started = time.monotonic()
    for _ in range(1000):
        recording.event("request_queued", request_id="a", status="queued")
    assert time.monotonic() - started < 1
    recording.close(_manifest("fast"))
