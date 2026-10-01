"""Completion claims must be supported by the retained event stream."""

from dataclasses import replace
from pathlib import Path

import pytest

from runner.load import RecordingData, load_recording
from runner.report import evaluate


def _event(sequence: int, kind: str, **fields: object) -> dict:
    return {
        "schema_version": "1.0.0",
        "run_id": "report-test",
        "producer_id": "test",
        "sequence": sequence,
        "monotonic_ns": sequence,
        "event_type": kind,
        **fields,
    }


def _scheduled(sequence: int = 2) -> dict:
    return _event(
        sequence,
        "step_scheduled",
        step_id="step-1",
        total_scheduled_tokens=1,
        requests=[{
            "request_id": "a",
            "range_start": 0,
            "range_length": 1,
            "prompt_tokens": 1,
            "generation_tokens": 0,
        }],
    )


def _recording(events: list[dict], **manifest: object) -> RecordingData:
    return RecordingData(
        manifest={
            "completion_status": "completed",
            "event_completeness": {"status": "complete"},
            **manifest,
        },
        events=tuple(events),
        content=(),
        directory=Path("."),
    )


def _completed_events() -> list[dict]:
    return [
        _event(1, "request_received", request_id="a"),
        _scheduled(),
        _event(3, "step_result_received", step_id="step-1", causal_event_sequence=2),
        _event(4, "completion", request_id="a", reason="stop"),
    ]


def test_truncated_acceptance_stream_cannot_keep_its_completed_claim() -> None:
    bundle = Path(__file__).resolve().parents[1] / "docs/acceptance/run-42f3805f07c3"
    recording = load_recording(bundle)
    first_step = next(index for index, event in enumerate(recording.events) if event["event_type"] == "step_scheduled")
    truncated = replace(recording, events=recording.events[: first_step + 1])

    report = evaluate(truncated)

    assert not report.ok
    assert "a completed run is missing" in report.text()
    assert "results for step-1" in report.text()
    assert "terminal events for" in report.text()


def test_completed_run_needs_a_terminal_for_every_declared_sample() -> None:
    recording = _recording(
        [_event(1, "capture_ready")],
        workload_plan={"batches": [{"samples": [{"id": "missing"}]}]},
    )
    assert "terminal events for missing" in evaluate(recording).text()
    assert not evaluate(recording).ok


def test_completed_run_needs_results_even_when_requests_are_terminal() -> None:
    events = _completed_events()
    events.pop(2)
    events[-1]["sequence"] = 3
    report = evaluate(_recording(events))
    assert not report.ok
    assert "results for step-1" in report.text()


@pytest.mark.parametrize("status_code, reason, ok", [(200, "stop", True), (200, None, False), (500, "stop", False)])
def test_only_successful_http_response_with_finish_reason_is_terminal(status_code: int, reason: str | None, ok: bool) -> None:
    events = _completed_events()
    events[-1] = _event(4, "http_response_completed", request_id="a", status_code=status_code, finish_reason=reason)
    report = evaluate(_recording(events))
    assert report.ok is ok, report.text()


def test_complete_stream_cannot_contain_a_trace_gap() -> None:
    report = evaluate(_recording([_event(1, "trace_gap", reason="buffer_exhausted")]))
    assert not report.ok
    assert "a complete event stream contains a trace gap" in report.text()


@pytest.mark.parametrize("completion", ["cancelled", "failed", "incomplete"])
def test_explicit_partial_run_can_retain_pending_work(completion: str) -> None:
    events = _completed_events()[:2]
    completeness = {"status": "complete"}
    if completion == "incomplete":
        events.append(_event(3, "trace_gap", reason="buffer_exhausted"))
        completeness = {"status": "incomplete"}
    report = evaluate(_recording(events, completion_status=completion, event_completeness=completeness))
    assert report.ok, report.text()


def test_empty_scheduler_iteration_does_not_require_a_model_result() -> None:
    report = evaluate(_recording([_event(1, "step_scheduled", step_id="empty", requests=[], total_scheduled_tokens=0)]))
    assert report.ok, report.text()
