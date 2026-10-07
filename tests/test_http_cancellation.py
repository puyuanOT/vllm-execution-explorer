"""Actual HTTP owner/driver cancellation races with a controlled native step.

The engine boundary is synthetic; this suite does not establish GPU evidence.
The transport, controller, recorder and final manifest are production code.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from runner.controller import drive_run
from runner.fixture_engine import ObservedChunk
from runner.http_session import VllmHttpSession
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.serve import RunRequest, RunState, _manifest_for
from runner.workload import WorkRequest
from vllm_030.collector import Recording


class BlockedNativeSession:
    """Return one accepted chunk only once cancellation has been requested."""

    def __init__(self, recording, *, cancel_failure=False):
        self.recording = recording
        self.cancel_failure = cancel_failure
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[str, int]] = []
        self.active = False
        self.native_ids = [95456]
        self.finish_reason = None

    def _call(self, name):
        self.calls.append((name, threading.get_ident()))

    def add_work_request(self, request):
        self._call("admit")
        self.active = True
        return {"resolved_input_tokens": 4, "input_token_ids": [1, 2, 3, 4]}

    def has_unfinished(self):
        self._call("has_unfinished")
        return self.active

    def step(self):
        self._call("step")
        self.entered.set()
        assert self.release.wait(5), "the test did not release the in-flight native step"
        self.recording.content(kind="accepted_tokens", request_id="a", token_ids=self.native_ids, output_position_start=0)
        self.recording.event("tokens_accepted", request_id="a", step_id="step-1", output_position_start=0,
                             token_count=1, content_ref="a")
        self.active = False
        return [ObservedChunk("a", tuple(self.native_ids), " returned token", self.finish_reason)]

    def cancel(self):
        self._call("cancel")
        self.active = False
        if self.cancel_failure:
            raise RuntimeError("native cancellation cleanup failed")

    def close(self):
        self._call("close")


def open_transport(tmp_path, monkeypatch, *, cancel_failure=False):
    recording = Recording.open(tmp_path / "race", run_id="race")
    base = BlockedNativeSession(recording, cancel_failure=cancel_failure)
    monkeypatch.setattr("runner.http_session.VllmSession.open", lambda *_args: base)
    session = VllmHttpSession.open(recording, SimpleNamespace(model="controlled-test"))
    return recording, base, session


@pytest.mark.parametrize("stop_reason", ["cancel", "deadline", "cancel_failure"])
def test_in_flight_native_chunk_survives_cancellation_finalization(tmp_path: Path, monkeypatch, stop_reason):
    recording, base, session = open_transport(tmp_path, monkeypatch, cancel_failure=stop_reason == "cancel_failure")
    requested = threading.Event()
    result, failures, published, notes = [], [], [], {}
    request = WorkRequest("a", "readable prompt", 0, 4)

    def drive():
        try:
            result.append(drive_run(session, recording, [request], cancelled=requested.is_set if stop_reason != "deadline" else lambda: False,
                clock=(lambda: 10.0 if requested.is_set() else 0.0) if stop_reason == "deadline" else time.monotonic,
                max_run_seconds=5, notes=notes, on_status=lambda text: published.append(dict(text))))
        except BaseException as error:
            failures.append(error)

    driver = threading.Thread(target=drive, name="test-workload-driver")
    driver.start()
    try:
        assert base.entered.wait(3)
        requested.set()
        assert session._cancelled.wait(3), "driver must request cancel while native step is blocked"
        assert not session._cancellation_acknowledged.is_set()
        assert driver.is_alive(), "driver must wait for the in-flight step"
        base.release.set()
        driver.join(5)
        assert not driver.is_alive()
        assert not failures
        status, tokens, arrivals = result[0]
        assert status == ("cancelled" if stop_reason == "cancel" else "failed")
        assert tokens == {"a": base.native_ids}
        assert published[-1] == {"a": " returned token"}
        assert session.drain_cancelled_chunks() == [], "the final chunk is consumed exactly once"
        if stop_reason == "cancel_failure":
            assert "native cancellation cleanup failed" in notes["message"]
            with pytest.raises(RuntimeError, match="native cancellation cleanup failed"):
                session.close()
        else:
            session.close()
        state = RunState()
        manifest = _manifest_for(state, "race", RunRequest(engine="fixture", requests=[]), session,
                                 arrivals, tokens, {}, status)
        recording.close(manifest)
        content = [json.loads(line) for line in (recording.directory / "content.jsonl").read_text().splitlines()]
        saved_manifest = json.loads((recording.directory / "manifest.json").read_text())
        accepted = [token for row in content if row["kind"] == "accepted_tokens" for token in row["token_ids"]]
        emitted = [row for row in content if row["kind"] == "text"]
        assert accepted == tokens["a"] == saved_manifest["engine_output_token_ids"]["a"] == base.native_ids
        assert len(emitted) == 1 and emitted[0]["text_delta"] == " returned token"
        assert emitted[0]["token_range"] == [0, 1]
        assert {thread_id for _name, thread_id in base.calls} == {session._owner.ident}
        assert [name for name, _ in base.calls].count("step") == 1
        assert [name for name, _ in base.calls].count("cancel") == 1
    finally:
        base.release.set()
        if driver.is_alive():
            driver.join(5)
        if session._owner.is_alive():
            session.close()


def test_early_cancellation_does_not_admit_or_step(tmp_path: Path, monkeypatch):
    recording, base, session = open_transport(tmp_path, monkeypatch)
    status, tokens, arrivals = drive_run(session, recording, [WorkRequest("a", "prompt", 0, 4)], cancelled=lambda: True)
    assert status == "cancelled" and tokens == {"a": []} and arrivals == []
    assert session.drain_cancelled_chunks() == []
    with pytest.raises(RuntimeError, match="has been cancelled"):
        session.add_work_requests([WorkRequest("later", "prompt", 0, 4)])
    session.close()
    recording.close(manifest_skeleton("race", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status=status))
    assert "step" not in [name for name, _ in base.calls]
    assert "admit" not in [name for name, _ in base.calls]
    assert {thread_id for _name, thread_id in base.calls} == {session._owner.ident}


def test_final_chunk_is_published_before_request_becomes_idle(tmp_path: Path, monkeypatch):
    class HeldQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            self.put_entered = threading.Event()
            self.release_put = threading.Event()

        def put(self, item, *args, **kwargs):
            self.put_entered.set()
            assert self.release_put.wait(3)
            return super().put(item, *args, **kwargs)

    recording, base, session = open_transport(tmp_path, monkeypatch)
    base.finish_reason = "stop"
    held = HeldQueue()
    session._chunks = held
    try:
        session.add_work_requests([WorkRequest("a", "prompt", 0, 4)])
        assert base.entered.wait(3)
        base.release.set()
        assert held.put_entered.wait(3)
        assert session.has_unfinished(), "the final chunk is still awaiting publication"
        held.release_put.set()
        chunks = session.step()
        assert chunks == [ObservedChunk("a", (95456,), " returned token", "stop")]
    finally:
        base.release.set()
        held.release_put.set()
        session.close()
        recording.close(manifest_skeleton("race", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))


def test_cancellation_acknowledgment_timeout_is_bounded(tmp_path: Path, monkeypatch):
    recording, base, session = open_transport(tmp_path, monkeypatch)
    monkeypatch.setattr("runner.http_session.CANCELLATION_TIMEOUT_SECONDS", 0.01)
    session.add_work_requests([WorkRequest("a", "prompt", 0, 4)])
    assert base.entered.wait(3)
    try:
        with pytest.raises(RuntimeError, match="did not acknowledge cancellation within 0.01s"):
            session.cancel()
        assert not session._cancellation_acknowledged.is_set()
    finally:
        base.release.set()
        assert session._cancellation_acknowledged.wait(3)
        session.close()
        recording.close(manifest_skeleton("race", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="failed"))


def test_cancellation_never_steps_a_direct_session(tmp_path: Path):
    class DirectSession:
        def cancel(self):
            return None

        def step(self):
            raise AssertionError("cancellation must not step a synchronous engine")

    recording = Recording.open(tmp_path / "direct", run_id="direct")
    status, tokens, _ = drive_run(DirectSession(), recording, [WorkRequest("a", "prompt", 0, 4)], cancelled=lambda: True)
    recording.close(manifest_skeleton("direct", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status=status))
    assert status == "cancelled" and tokens == {"a": []}
