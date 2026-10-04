"""Real spawned-process lifetime and controller transactions, without a GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from runner.configuration import parse_run_configuration
from runner.isolation import ExperimentIsolation, _execute_child, _valid_jsonl_prefix
from runner.load import load_recording
from runner.report import evaluate
from runner.serve import RunRequest, _finish_intake, _take_extra
from runner.state import RunState
from runner.workload import WorkRequest, parse_run_workload


def delayed_exit(state, arguments):
    _execute_child(state, arguments)
    time.sleep(0.4)


def unexpected_exit(state, arguments):
    directory = arguments["recordings_dir"] / arguments["run_id"]
    directory.mkdir(parents=True)
    event = {"schema_version": "1.0.0", "run_id": arguments["run_id"], "producer_id": "test-child",
             "sequence": 1, "monotonic_ns": time.monotonic_ns(), "event_type": "engine_admitted", "request_id": "first"}
    (directory / "events.jsonl").write_text(json.dumps(event) + '\n{"interrupted":')
    (directory / "content.jsonl").write_text("")
    os._exit(23)


def stuck_worker(state, _arguments):
    with state.lock:
        state.phase = "test_worker_ready"
    time.sleep(30)


def death_inside_transaction(state, _arguments):
    with state.lock:
        state.status, state.phase = "capturing", "transaction_in_progress"
        os._exit(24)


def stuck_inside_transaction(state, _arguments):
    with state.lock:
        state.phase = "transaction_in_progress"
        time.sleep(30)


def unrelated_worker():
    time.sleep(30)


@pytest.fixture
def isolated():
    isolation = ExperimentIsolation(max_lifetime_seconds=10)
    state = isolation.share(RunState())
    try:
        yield isolation, state
    finally:
        isolation.close()


def prepare(state, directory: Path, run_id: str, *, offset: int = 0):
    body = RunRequest(engine="fixture", requests=[{"request_id": "first", "prompt": "hello",
        "arrival_offset_ms": offset, "max_output_tokens": 8}], max_num_seqs=2, token_budget=64)
    requests, plan = parse_run_workload(body.model_dump())
    with state.lock:
        state.run_id, state.status, state.phase = run_id, "preparing", "configuration_accepted"
        state.configuration = parse_run_configuration(body.model_dump(), requests, plan)
        state.plan, state.text = plan, {"first": ""}
        state.extra, state.arrivals, state.phase_history = [], [], []
        state.cancelled, state.intake_open, state.finalized = False, True, False
    return {"recordings_dir": directory, "run_id": run_id, "requests": requests, "body": body}


def wait_for(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for isolated worker state")


def test_repeated_captures_use_distinct_joined_processes_and_persist_recordings(isolated, tmp_path):
    isolation, state = isolated
    pids = []
    for run_id in ("first-run", "second-run"):
        arguments = prepare(state, tmp_path, run_id)
        isolation.start(state, arguments)
        wait_for(lambda: not state.execution_process_alive)
        snapshot = state.snapshot()
        assert snapshot["status"] == "completed" and snapshot["finalized"] is True
        assert snapshot["active"] is False
        process = snapshot["execution_process"]
        assert process["status"] == "exited" and process["exit_code"] == 0
        assert process["pid"] != os.getpid()
        pids.append(process["pid"])
        data = load_recording(tmp_path / run_id)
        assert data.manifest["execution_process"] == process
        assert evaluate(data).ok
        assert data.manifest["engine_output_token_ids"]["first"]
    assert len(set(pids)) == 2


def test_public_readiness_stays_gated_until_child_is_joined(isolated, tmp_path):
    isolation, state = isolated
    arguments = prepare(state, tmp_path, "delayed-exit")
    isolation.start(state, arguments, target=delayed_exit)
    wait_for(lambda: state.finalized)
    assert (tmp_path / "delayed-exit" / "manifest.json").exists()
    snapshot = state.snapshot()
    assert snapshot["status"] == "finalizing"
    assert snapshot["capture"]["phase"] == "reaping_execution_process"
    assert snapshot["active"] is True and snapshot["finalized"] is False
    with pytest.raises(RuntimeError, match="still exiting"):
        isolation.start(state, arguments)
    wait_for(lambda: not state.execution_process_alive)
    assert state.snapshot()["finalized"] is True


def test_late_intake_and_cancellation_cross_the_shared_lock_and_preserve_evidence(isolated, tmp_path):
    isolation, state = isolated
    arguments = prepare(state, tmp_path, "late-intake", offset=150)
    isolation.start(state, arguments)
    wait_for(lambda: state.phase == "dispatching_workload")
    with state.lock:
        state.extra.append(WorkRequest("late", "a late prompt", 0, 8))
    wait_for(lambda: any(arrival["request_id"] == "late" for arrival in state.arrivals))
    with state.lock:
        state.cancelled, state.intake_open = True, False
    wait_for(lambda: not state.execution_process_alive)
    data = load_recording(tmp_path / "late-intake")
    assert data.manifest["completion_status"] == "cancelled"
    assert any(event.get("request_id") == "late" and event["event_type"] == "engine_admitted" for event in data.events)
    assert not any(event.get("request_id") == "first" and event["event_type"] == "engine_admitted" for event in data.events)
    # FixtureSession.cancel clears its queue; it does not claim a native abort observation.
    assert any(sample["id"] == "late" for batch in data.manifest["workload_plan"]["batches"] for sample in batch["samples"])
    assert state.snapshot()["intake_open"] is False
    assert evaluate(data).ok


def test_shared_intake_finish_does_not_drop_an_already_accepted_extra(isolated):
    _isolation, state = isolated
    with state.lock:
        state.intake_open = True
        state.extra.append(WorkRequest("late", "hello", 0, 1))
        assert _finish_intake(state) is False
        assert state.intake_open is True
    assert [item.request_id for item in _take_extra(state)] == ["late"]
    assert dict(state.text) == {"late": ""}
    assert _finish_intake(state) is True
    assert state.intake_open is False


def test_unexpected_worker_death_preserves_prefix_and_finalizes_unknown_failed_evidence(isolated, tmp_path):
    isolation, state = isolated
    arguments = prepare(state, tmp_path, "crashed")
    isolation.start(state, arguments, target=unexpected_exit)
    wait_for(lambda: not state.execution_process_alive)
    snapshot = state.snapshot()
    assert snapshot["status"] == "failed" and snapshot["finalized"] is True
    assert snapshot["execution_process"]["exit_code"] == 23
    data = load_recording(tmp_path / "crashed")
    assert data.events[0]["event_type"] == "engine_admitted"
    assert data.events[-1]["event_type"] == "trace_gap"
    assert data.events[-1]["reason"] == "execution_process_exited"
    assert data.manifest["completion_status"] == "failed"
    assert data.manifest["event_completeness"]["status"] == "incomplete"
    assert data.manifest["provenance"]["real_gpu"] is False
    assert evaluate(data).ok
    isolation.start(state, prepare(state, tmp_path, "after-crash"))
    wait_for(lambda: not state.execution_process_alive)
    assert state.snapshot()["status"] == "completed"


def test_lifetime_timeout_terminates_only_the_owned_worker_and_unblocks_next_capture(tmp_path):
    isolation = ExperimentIsolation(max_lifetime_seconds=0.6)
    state = isolation.share(RunState())
    try:
        arguments = prepare(state, tmp_path, "timeout")
        isolation.start(state, arguments, target=stuck_worker)
        wait_for(lambda: not state.execution_process_alive)
        assert state.snapshot()["status"] == "failed"
        assert "bounded lifetime" in state.message
        assert load_recording(tmp_path / "timeout").manifest["execution_process"]["exit_code"] != 0
    finally:
        isolation.close()


def test_worker_death_inside_transaction_does_not_orphan_the_controller_lock(isolated, tmp_path):
    isolation, state = isolated
    isolation.start(state, prepare(state, tmp_path, "locked-death"), target=death_inside_transaction)
    wait_for(lambda: not state.execution_process_alive)
    assert state.lock.acquire(timeout=0.2)
    state.lock.release()
    assert state.snapshot()["execution_process"]["exit_code"] == 24
    assert state.snapshot()["finalized"] is True
    assert load_recording(tmp_path / "locked-death").manifest["event_completeness"]["status"] == "incomplete"
    isolation.start(state, prepare(state, tmp_path, "after-locked-death"))
    wait_for(lambda: not state.execution_process_alive)
    assert state.snapshot()["status"] == "completed"


def test_controller_shutdown_reaps_owned_worker_even_when_it_holds_state_lock(tmp_path):
    isolation = ExperimentIsolation(max_lifetime_seconds=30, shutdown_grace_seconds=0.05,
                                    termination_grace_seconds=0.5)
    state = isolation.share(RunState())
    unrelated = isolation._context.Process(target=unrelated_worker)
    unrelated.start()
    try:
        isolation.start(state, prepare(state, tmp_path, "shutdown"), target=stuck_inside_transaction)
        wait_for(lambda: state.phase == "transaction_in_progress")
        owned = isolation._process
        isolation.close()
        assert not owned.is_alive() and owned.exitcode != 0
        assert unrelated.is_alive()
        assert load_recording(tmp_path / "shutdown").manifest["completion_status"] == "failed"
        assert isolation._manager is None
    finally:
        unrelated.terminate()
        unrelated.join(timeout=2)
        isolation.close()


def test_unconfirmed_worker_exit_keeps_handle_and_readiness_gate(tmp_path):
    class UnreapableProcess:
        pid, exitcode = 123, None

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

        def terminate(self):
            raise RuntimeError("Termination unavailable")

    isolation, state = ExperimentIsolation(), RunState()
    arguments = prepare(state, tmp_path, "unreaped")
    state.execution_process_alive = True
    process = UnreapableProcess()
    isolation._process = process
    isolation._supervise(state, arguments, process)
    assert isolation._process is process
    assert state.snapshot()["active"] is True and state.snapshot()["finalized"] is False
    assert state.execution_process["status"] == "exit_unconfirmed"
    with pytest.raises(RuntimeError, match="still exiting"):
        isolation.start(state, arguments)


@pytest.mark.parametrize("tail", [b'{"unfinished":', b'{"text":"\xf0\x9f', b""])
def test_interrupted_jsonl_preserves_original_bytes_and_a_parseable_prefix(tmp_path, tail):
    path = tmp_path / "events.jsonl"
    first = b'{"sequence":1,"event_type":"engine_admitted"}'
    original = first + (b"\n" + tail if tail else b"")
    path.write_bytes(original)
    rows = _valid_jsonl_prefix(path)
    with path.open("ab") as stream:
        stream.write(b'{"sequence":2,"event_type":"trace_gap"}\n')
    assert rows[0]["event_type"] == "engine_admitted"
    assert [json.loads(line)["sequence"] for line in path.read_text().splitlines()] == [1, 2]
    assert path.with_name("events.jsonl.interrupted").read_bytes() == original


def test_interior_corruption_is_not_rewritten_as_a_valid_prefix(tmp_path):
    path = tmp_path / "events.jsonl"
    original = b'{"sequence":1}\n\xf0\n{"sequence":3}\n'
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="interior record"):
        _valid_jsonl_prefix(path)
    assert path.read_bytes() == original
    assert not path.with_name("events.jsonl.interrupted").exists()


def test_rejected_production_start_cannot_replace_an_active_fixture_state(tmp_path, monkeypatch):
    import runner.serve as serve

    shares = []

    class TrackingIsolation(ExperimentIsolation):
        def share(self, state):
            shares.append(state)
            return super().share(state)

    monkeypatch.setattr(serve, "ExperimentIsolation", TrackingIsolation)
    monkeypatch.setattr(serve, "decide_target", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    monkeypatch.setattr(serve, "_cuda_available", lambda: True)
    monkeypatch.setattr(serve, "_vllm_version", lambda: "0.30.0")
    monkeypatch.setattr(serve, "_cuda_compute_capability", lambda: (8, 6))
    monkeypatch.setattr(serve, "_bitsandbytes_plugin_available", lambda: False)
    monkeypatch.setattr(serve, "capability_report", lambda **_kwargs: {
        "model_options": {"quantization": {"none": {"selectable": True}}}})
    with TestClient(serve.create_app(tmp_path)) as client:
        started = client.post("/api/runs", json={"engine": "fixture", "requests": [{
            "request_id": "original", "prompt": "Hello", "arrival_offset_ms": 400, "max_output_tokens": 1}]})
        assert started.status_code == 200
        rejected = client.post("/api/runs", json={"engine": "vllm", "requests": [{
            "request_id": "replacement", "prompt": "Hello", "arrival_offset_ms": 0, "max_output_tokens": 1}]})
        assert rejected.status_code == 409 and shares == [], rejected.json()
        wait_for(lambda: client.get("/api/runs/current").json()["finalized"])
        snapshot = client.get("/api/runs/current").json()
        assert snapshot["run_id"] == started.json()["run_id"]
        assert snapshot["status"] == "completed" and snapshot["text"]["original"]
