"""Controller clock and the loopback API."""

import threading
import time

from fastapi.testclient import TestClient

from runner.controller import drive_run
from runner.fixture_engine import ObservedChunk
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.serve import create_app, parse_serve_args
from runner.targets import TargetError
from runner.workload import WorkRequest
from vllm_030.collector import Recording


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Scripted:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.queue: list[str] = []
        self.cancelled = False

    def add_request(self, request_id: str, prompt: str, max_output_tokens: int = 16) -> None:
        del prompt, max_output_tokens
        self.added.append(request_id)
        self.queue.append(request_id)

    def step(self) -> list[ObservedChunk]:
        request_id = self.queue.pop(0)
        return [ObservedChunk(request_id, (7,), "x", "stop")]

    def has_unfinished(self) -> bool:
        return bool(self.queue)

    def cancel(self) -> None:
        self.cancelled = True
        self.queue.clear()


def test_later_offset_is_admitted_after_the_controller_clock_advances(tmp_path):
    recording = Recording.open(tmp_path / "clock", run_id="clock")
    clock = Clock()
    session = Scripted()
    status, tokens, arrivals = drive_run(
        session,
        recording,
        [
            WorkRequest("a", "one", 0, 8),
            WorkRequest("b", "two", 1000, 8),
        ],
        clock=clock,
        sleeper=clock.sleep,
    )
    recording.close(manifest_skeleton("clock", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    assert status == "completed"
    assert session.added == ["a", "b"]
    assert tokens == {"a": [7], "b": [7]}
    assert arrivals[0]["requested_offset_ns"] == 0
    assert arrivals[1]["requested_offset_ns"] == 1_000_000_000
    assert "delay_ns" not in arrivals[0]


def test_cancel_stops_before_a_later_request(tmp_path):
    recording = Recording.open(tmp_path / "cancel", run_id="cancel")
    session = Scripted()
    status, _tokens, arrivals = drive_run(
        session,
        recording,
        [WorkRequest("a", "one", 0, 8)],
        cancelled=lambda: True,
    )
    recording.close(
        manifest_skeleton(
            "cancel",
            execution_kind="fixture",
            execution_target=LOCAL_FIXTURE_TARGET,
            completion_status="cancelled",
        )
    )
    assert status == "cancelled"
    assert session.cancelled
    assert arrivals == []


def test_api_preflight_and_fixture_run(tmp_path):
    app = create_app(tmp_path / "runs")
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"] is True
    preflight = client.get("/api/preflight").json()
    assert preflight["ok"] is False
    refused = client.post(
        "/api/runs",
        json={"engine": "vllm", "requests": [{"request_id": "a", "prompt": "Hello"}]},
    )
    assert refused.status_code == 409
    started = client.post(
        "/api/runs",
        json={
            "engine": "fixture",
            "requests": [
                {
                    "request_id": "a",
                    "prompt": "Hello from the fixture",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 2,
                },
                {
                    "request_id": "b",
                    "prompt": "Second request joins",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 2,
                },
            ],
        },
    )
    assert started.status_code == 200
    deadline = time.monotonic() + 5
    snapshot = {}
    active = {"preparing", "capturing", "finalizing", "running"}
    while time.monotonic() < deadline:
        snapshot = client.get("/api/runs/current").json()
        if snapshot["status"] not in active:
            break
        time.sleep(0.02)
    assert snapshot["status"] == "completed"
    assert snapshot["text"]["a"]
    run_id = snapshot["run_id"]
    manifest = __import__("json").loads((tmp_path / "runs" / run_id / "manifest.json").read_text())
    assert manifest["execution_kind"] == "fixture"
    assert manifest["lesson"]["evidence"] == "fixture"
    events = client.get(f"/api/recordings/{run_id}/events")
    assert events.status_code == 200
    assert "step_scheduled" in events.text
    content = client.get(f"/api/recordings/{run_id}/content")
    assert content.status_code == 200
    assert "prompt_text" in content.text
    assert client.get("/api/recordings/not-a-run/manifest").status_code == 404


def test_ssh_response_omits_the_private_key(tmp_path):
    seen: list[list[str]] = []

    class Process:
        def terminate(self) -> None:
            return None

    def popen(argv: list[str]) -> Process:
        seen.append(argv)
        return Process()

    client = TestClient(create_app(tmp_path / "runs", popen=popen))
    response = client.post(
        "/api/targets/ssh",
        json={
            "user": "ubuntu",
            "host": "192.0.2.10",
            "identity_file": "/keys/test-key.pem",
            "remote_workdir": "/opt/explorer",
            "start_tunnel": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["identity_file_used"] is True
    assert "test-key.pem" not in json_dump(body)
    assert seen and any("192.0.2.10" in arg for arg in seen[0])


def test_runner_rejects_a_public_bind():
    try:
        parse_serve_args(["--bind", "0.0.0.0"])
        raise AssertionError("expected loopback rejection")
    except TargetError as exc:
        assert "loopback" in str(exc)


def test_request_accepted_during_final_queue_check_is_drained(tmp_path):
    checking_empty = threading.Event()
    allow_completion = threading.Event()

    class FinishingSession(Scripted):
        def has_unfinished(self) -> bool:
            if self.added == ["a"] and not self.queue:
                checking_empty.set()
                assert allow_completion.wait(5)
            return super().has_unfinished()

        def close(self) -> None:
            pass

    session = FinishingSession()
    client = TestClient(create_app(tmp_path / "runs", session_factory=lambda *_args: session))
    started = client.post("/api/runs", json={
        "engine": "fixture",
        "requests": [{"request_id": "a", "prompt": "first", "arrival_offset_ms": 0, "max_output_tokens": 1}],
    })
    assert started.status_code == 200
    assert checking_empty.wait(5)
    try:
        added = client.post("/api/runs/current/requests", json={
            "request_id": "b", "prompt": "accepted while finishing", "max_output_tokens": 1,
        })
        assert added.status_code == 200
    finally:
        allow_completion.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = client.get("/api/runs/current").json()
        if not snapshot["active"]:
            break
        time.sleep(0.01)
    assert snapshot["status"] == "completed"
    assert session.added == ["a", "b"]
    assert snapshot["text"] == {"a": "x", "b": "x"}
    manifest = client.get(f"/api/recordings/{started.json()['run_id']}/manifest").json()
    assert [item["request_id"] for item in manifest["arrivals"]] == ["a", "b"]
    assert client.post("/api/runs/current/requests", json={
        "request_id": "c", "prompt": "too late", "max_output_tokens": 1,
    }).status_code == 409


def json_dump(value: object) -> str:
    return __import__("json").dumps(value)
