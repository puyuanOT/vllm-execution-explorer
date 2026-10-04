"""Real spawned-process lifetime and controller transactions, without a GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
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


def delayed_initialize(state, arguments):
    while state.phase != "permit_recording_initialization":
        if state.cancelled:
            return
        time.sleep(0.01)
    _execute_child(state, arguments)


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


def exit_when_released(state, _arguments):
    with state.lock:
        state.phase = "waiting_to_exit"
    while state.phase != "permit_exit":
        time.sleep(0.01)


class ObservedMutex:
    """Expose one acquisition decision without changing mutual exclusion."""

    def __init__(self, observing_thread):
        self._lock = threading.Lock()
        self.observing_thread = observing_thread
        self.decided = threading.Event()
        self.blocked = None

    def __enter__(self):
        if threading.current_thread().name == self.observing_thread:
            acquired = self._lock.acquire(blocking=False)
            self.blocked = not acquired
            self.decided.set()
            if not acquired:
                self._lock.acquire()
        else:
            self._lock.acquire()
        return self

    def __exit__(self, *_error):
        self._lock.release()


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


def test_previous_finalizer_cannot_clear_a_newly_admitted_owned_handle(tmp_path, monkeypatch):
    isolation, state = ExperimentIsolation(), RunState()
    gate_released, resume_finalizer = threading.Event(), threading.Event()
    admitted, stop_supervisor = threading.Event(), threading.Event()
    errors = []
    original_lock = state.lock

    class PausedGateLock:
        def __enter__(self):
            original_lock.acquire()
            return self

        def __exit__(self, *_error):
            original_lock.release()
            if threading.current_thread().name == "previous-finalizer" and not state.execution_process_alive:
                gate_released.set()
                assert resume_finalizer.wait(3)

        def acquire(self, *args, **kwargs):
            return original_lock.acquire(*args, **kwargs)

        def release(self):
            original_lock.release()

    class OwnedHandle:
        def __init__(self, *, alive):
            self.pid, self.exitcode = 123, None if alive else 0
            self.alive, self.terminated = alive, False

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            pass

        def terminate(self):
            self.alive, self.terminated, self.exitcode = False, True, -15

    arguments = prepare(state, tmp_path, "previous")
    directory = tmp_path / "previous"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    state.status, state.execution_process_alive = "completed", True
    state.lock = PausedGateLock()
    previous, new = OwnedHandle(alive=False), OwnedHandle(alive=True)
    isolation._state, isolation._process = state, previous
    lifecycle_lock = ObservedMutex("new-admission")
    isolation._lock = lifecycle_lock
    isolation._context = SimpleNamespace(Process=lambda **_kwargs: new)
    monkeypatch.setattr(isolation, "_supervise", lambda *_args: stop_supervisor.wait(3))

    def previous_finalizer():
        try:
            isolation._finalize_exit(state, arguments, previous, "")
        except BaseException as error:
            errors.append(error)

    def next_admission():
        try:
            isolation.start(state, prepare(state, tmp_path, "next"))
            admitted.set()
        except BaseException as error:
            errors.append(error)

    old_thread = threading.Thread(target=previous_finalizer, name="previous-finalizer")
    next_thread = threading.Thread(target=next_admission, name="new-admission")
    try:
        old_thread.start()
        assert gate_released.wait(3)
        next_thread.start()
        assert lifecycle_lock.decided.wait(3)
        if not lifecycle_lock.blocked:
            # The original bug admits here, then clears this new handle when
            # the previous finalizer resumes. No timing-based sleep is needed.
            assert admitted.wait(3)
        resume_finalizer.set()
        old_thread.join(timeout=3)
        next_thread.join(timeout=3)
        assert not errors and admitted.is_set()
        assert isolation._process is new and state.execution_process_alive
        stop_supervisor.set()
        isolation._thread.join(timeout=3)
        isolation.close()
        assert new.terminated  # The retained handle remains usable by shutdown.
    finally:
        resume_finalizer.set()
        stop_supervisor.set()
        old_thread.join(timeout=3)
        if next_thread.ident is not None:
            next_thread.join(timeout=3)
        if isolation._thread is not None:
            isolation._thread.join(timeout=3)
        isolation.close()


def test_shutdown_cannot_poll_between_native_waitpid_and_recorded_exit_code(tmp_path, monkeypatch):
    isolation = ExperimentIsolation(max_lifetime_seconds=30, shutdown_grace_seconds=0.05,
                                    termination_grace_seconds=0.5)
    state = isolation.share(RunState())
    lifecycle_lock = ObservedMutex("controller-shutdown")
    isolation._lock = lifecycle_lock
    reaped, publish_exit = threading.Event(), threading.Event()
    errors = []
    original_waitpid = os.waitpid
    shutdown_thread = None
    try:
        isolation.start(state, prepare(state, tmp_path, "native-reap"), target=exit_when_released)
        wait_for(lambda: state.phase == "waiting_to_exit")
        owned = isolation._process

        def paused_waitpid(pid, options):
            result = original_waitpid(pid, options)
            if result[0] == owned.pid and threading.current_thread().name == "vllm-experiment-supervisor":
                # The OS has reaped the real spawned child, but CPython has
                # not yet copied its return code into the Process handle.
                reaped.set()
                assert publish_exit.wait(3)
            return result

        monkeypatch.setattr(os, "waitpid", paused_waitpid)
        with state.lock:
            state.phase = "permit_exit"
        assert reaped.wait(3)

        def shutdown():
            try:
                isolation.close()
            except BaseException as error:
                errors.append(error)

        shutdown_thread = threading.Thread(target=shutdown, name="controller-shutdown")
        shutdown_thread.start()
        assert lifecycle_lock.decided.wait(3)
        assert lifecycle_lock.blocked is True
        publish_exit.set()
        shutdown_thread.join(timeout=3)
        assert not shutdown_thread.is_alive() and not errors
        assert owned.exitcode == 0 and not owned.is_alive()
        assert isolation._manager is None and isolation._process is None
        assert load_recording(tmp_path / "native-reap").manifest["execution_process"]["exit_code"] == 0
    finally:
        publish_exit.set()
        if shutdown_thread is not None:
            shutdown_thread.join(timeout=3)
        if isolation._thread is not None:
            isolation._thread.join(timeout=3)
        monkeypatch.setattr(os, "waitpid", original_waitpid)
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


def test_accepted_spawn_returns_empty_provisional_trace_before_worker_creates_files(tmp_path, monkeypatch):
    import runner.serve as serve

    isolations = []

    class DelayedFixtureIsolation(ExperimentIsolation):
        def __init__(self):
            super().__init__(max_lifetime_seconds=10)
            isolations.append(self)

        def start(self, state, arguments):
            # Exercise the production spawn/API contract with an explicit
            # fixture engine in the child; this test loads no model or GPU.
            arguments = {**arguments, "body": arguments["body"].model_copy(update={"engine": "fixture"})}
            super().start(state, arguments, target=delayed_initialize)

    monkeypatch.setattr(serve, "ExperimentIsolation", DelayedFixtureIsolation)
    monkeypatch.setattr(serve, "decide_target", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    monkeypatch.setattr(serve, "_cuda_available", lambda: True)
    monkeypatch.setattr(serve, "_vllm_version", lambda: "0.30.0")
    monkeypatch.setattr(serve, "_cuda_compute_capability", lambda: (8, 6))
    monkeypatch.setattr(serve, "_bitsandbytes_plugin_available", lambda: False)
    monkeypatch.setattr(serve, "capability_report", lambda **_kwargs: {
        "model_options": {"quantization": {"none": {"selectable": True}}}})
    with TestClient(serve.create_app(tmp_path)) as client:
        started = client.post("/api/runs", json={"engine": "vllm", "requests": [{
            "request_id": "first", "prompt": "Hello", "arrival_offset_ms": 0, "max_output_tokens": 1}]})
        assert started.status_code == 200, started.json()
        run_id = started.json()["run_id"]
        assert not (tmp_path / run_id).exists()
        response = client.get(f"/api/runs/{run_id}/trace?after_sequence=5&after_content_sequence=7")
        assert response.status_code == 200
        trace = response.json()
        assert trace["status"] == "preparing"
        assert trace["capture"]["phase"] == "configuration_accepted"
        assert trace["provisional"] is True and trace["finalized"] is False
        assert trace["events"] == [] and trace["content"] == [] and trace["manifest"] is None
        assert trace["next_event_sequence"] == 5 and trace["next_content_sequence"] == 7
        assert trace["events_truncated"] is False and trace["content_truncated"] is False
        assert trace["workload_plan"] == client.get("/api/runs/current").json()["workload_plan"]
        assert client.get("/api/runs/unknown/trace").status_code == 404
        with isolations[0]._state.lock:
            isolations[0]._state.phase = "permit_recording_initialization"
        wait_for(lambda: client.get("/api/runs/current").json()["finalized"])
        finalized = client.get(f"/api/runs/{run_id}/trace").json()
        assert finalized["provisional"] is False and finalized["finalized"] is True
        assert finalized["manifest"]["execution_process"]["status"] == "exited"
        assert finalized["events"] and finalized["content"]
