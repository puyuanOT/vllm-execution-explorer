"""Arrival order, workload plans, demo fixture settings, and live trace cursors."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from runner.controller import drive_run
from runner.export import export_structural
from runner.fixture_engine import FixtureEngine, FixtureSession, ObservedChunk
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.serve import create_app
from runner.trace_index import JsonlIndex
from runner.workload import WorkRequest, WorkloadError, parse_run_workload
from vllm_030.collector import Recording


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Scripted:
    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self.added: list[tuple[str, int, float]] = []
        self.queue: list[str] = []
        self.step_starts: list[float] = []
        self.cancelled = False
        self.outputs = 1

    def add_request(self, request_id: str, prompt: str, max_output_tokens: int = 16) -> None:
        del prompt, max_output_tokens
        self.added.append((request_id, len(self.step_starts), self.clock.now))
        self.queue.append(request_id)

    def step(self) -> list[ObservedChunk]:
        self.step_starts.append(self.clock.now)
        if self.step_starts and len(self.step_starts) == 1:
            self.clock.now += 0.15
        request_id = self.queue.pop(0)
        return [ObservedChunk(request_id, (7,), "x", "stop")]

    def has_unfinished(self) -> bool:
        return bool(self.queue)

    def cancel(self) -> None:
        self.cancelled = True
        self.queue.clear()

    def close(self) -> None:
        return None


def test_equal_offsets_keep_submission_order(tmp_path: Path):
    recording = Recording.open(tmp_path / "ties", run_id="ties")
    clock = Clock()
    session = Scripted(clock)
    session.step = _instant_step(session)  # type: ignore[method-assign]
    status, _tokens, arrivals = drive_run(
        session,
        recording,
        [
            WorkRequest("z-sample", "alpha prompt", 0, 4),
            WorkRequest("a-sample", "beta prompt", 0, 4),
            WorkRequest("m-sample", "gamma prompt", 0, 4),
        ],
        clock=clock,
        sleeper=clock.sleep,
        max_run_seconds=10,
    )
    recording.close(manifest_skeleton("ties", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    assert status == "completed"
    assert [item[0] for item in session.added] == ["z-sample", "a-sample", "m-sample"]
    assert [item["request_id"] for item in arrivals] == ["z-sample", "a-sample", "m-sample"]


def test_long_step_admits_every_due_request_before_the_next_step(tmp_path: Path):
    recording = Recording.open(tmp_path / "late", run_id="late")
    clock = Clock()
    session = Scripted(clock)
    status, _tokens, arrivals = drive_run(
        session,
        recording,
        [
            WorkRequest("early", "one prompt", 0, 4),
            WorkRequest("middle", "two prompt", 100, 4),
            WorkRequest("last", "three prompt", 120, 4),
        ],
        clock=clock,
        sleeper=clock.sleep,
        max_run_seconds=10,
    )
    recording.close(manifest_skeleton("late", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    assert status == "completed"
    assert [(item[0], item[1], round(item[2], 3)) for item in session.added] == [
        ("early", 0, 0.0),
        ("middle", 1, 0.15),
        ("last", 1, 0.15),
    ]
    assert [item["request_id"] for item in arrivals] == ["early", "middle", "last"]


def test_closed_intake_still_drains_accepted_future_requests(tmp_path: Path):
    recording = Recording.open(tmp_path / "drain", run_id="drain")
    clock = Clock()
    session = Scripted(clock)
    session.step = _instant_step(session)  # type: ignore[method-assign]
    queued = [WorkRequest("later", "future prompt", 80, 4)]

    def pull() -> list[WorkRequest]:
        items = list(queued)
        queued.clear()
        return items

    status, _tokens, arrivals = drive_run(
        session,
        recording,
        [WorkRequest("now", "current prompt", 0, 4)],
        clock=clock,
        sleeper=clock.sleep,
        intake_open=lambda: False,
        pull_extra=pull,
        max_run_seconds=10,
    )
    recording.close(manifest_skeleton("drain", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET))
    assert status == "completed"
    assert [item[0] for item in session.added] == ["now", "later"]
    assert [item["request_id"] for item in arrivals] == ["now", "later"]


def test_idle_wait_is_interruptible(tmp_path: Path):
    recording = Recording.open(tmp_path / "idle", run_id="idle")
    clock = Clock()
    session = Scripted(clock)
    session.step = _instant_step(session)  # type: ignore[method-assign]
    sleeps: list[float] = []
    flags = {"cancel": False}

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds
        flags["cancel"] = True

    status, _tokens, arrivals = drive_run(
        session,
        recording,
        [
            WorkRequest("now", "current prompt", 0, 4),
            WorkRequest("much-later", "future prompt", 5_000, 4),
        ],
        clock=clock,
        sleeper=sleeper,
        cancelled=lambda: flags["cancel"],
        max_run_seconds=30,
        idle_slice_s=0.02,
    )
    recording.close(manifest_skeleton("idle", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="cancelled"))
    assert status == "cancelled"
    assert session.cancelled
    assert [item[0] for item in session.added] == ["now"]
    assert [item["request_id"] for item in arrivals] == ["now"]
    assert sleeps
    assert max(sleeps) <= 0.02


def test_run_deadline_stops_a_waiting_controller(tmp_path: Path):
    recording = Recording.open(tmp_path / "deadline", run_id="deadline")
    clock = Clock()
    session = Scripted(clock)
    session.step = _instant_step(session)  # type: ignore[method-assign]
    notes: dict[str, str] = {}
    status, _tokens, arrivals = drive_run(
        session,
        recording,
        [WorkRequest("later", "future prompt", 10_000, 4)],
        clock=clock,
        sleeper=clock.sleep,
        notes=notes,
        max_run_seconds=0.05,
        idle_slice_s=0.02,
    )
    recording.close(manifest_skeleton("deadline", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, completion_status="failed"))
    assert status == "failed"
    assert arrivals == []
    assert "0s" in notes["message"] or "exceeded" in notes["message"]
    assert session.cancelled


def test_workload_rejects_fractional_offsets_and_zero_output_limits():
    base = {"request_id": "sample-1", "prompt": "Name a color."}
    for payload in (
        {**base, "arrival_offset_ms": 1.5, "max_output_tokens": 4},
        {**base, "arrival_offset_ms": True, "max_output_tokens": 4},
        {**base, "arrival_offset_ms": "100", "max_output_tokens": 4},
        {**base, "arrival_offset_ms": 0, "max_output_tokens": 0},
        {**base, "arrival_offset_ms": 0, "max_output_tokens": False},
        {**base, "arrival_offset_ms": -1, "max_output_tokens": 4},
    ):
        try:
            parse_run_workload({"requests": [payload]})
        except WorkloadError as exc:
            assert exc.field
        else:
            raise AssertionError(f"accepted invalid request {payload}")


def test_plan_must_match_requests_and_omits_prompt_text():
    requests = [
        {"request_id": "sample-1", "prompt": "Name a primary color.", "arrival_offset_ms": 0, "max_output_tokens": 8},
        {"request_id": "sample-2", "prompt": "Name a capital city.", "arrival_offset_ms": 0, "max_output_tokens": 8},
        {"request_id": "sample-3", "prompt": "Count to three.", "arrival_offset_ms": 100, "max_output_tokens": 8},
    ]
    plan = {
        "version": 1,
        "batches": [
            {
                "id": "batch-a",
                "label": "A",
                "arrival_offset_ms": 0,
                "samples": [
                    {"id": "sample-1", "label": "input_1", "max_output_tokens": 8},
                    {"id": "sample-2", "label": "input_2", "max_output_tokens": 8},
                ],
            },
            {
                "id": "batch-b",
                "label": "B",
                "arrival_offset_ms": 100,
                "samples": [{"id": "sample-3", "label": "input_3", "max_output_tokens": 8}],
            },
        ],
    }
    parsed, canonical = parse_run_workload({"requests": requests, "workload_plan": plan})
    assert [item.arrival_offset_ms for item in parsed] == [0, 0, 100]
    assert canonical["batches"][0]["samples"][0]["label"] == "input_1"
    assert "Name a primary color." not in json.dumps(canonical)
    try:
        parse_run_workload({"requests": requests, "workload_plan": {**plan, "prompt": "hidden"}})
    except WorkloadError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("plan accepted prompt content")


def test_failure_keeps_the_plan_and_a_terminal_status(tmp_path: Path):
    def explode(recording, body):  # noqa: ARG001
        raise RuntimeError("engine failed during preparation")

    client = TestClient(create_app(tmp_path / "runs", session_factory=explode))
    started = client.post("/api/runs", json=_fixture_body())
    assert started.status_code == 200
    snapshot = _wait(client)
    assert snapshot["status"] == "failed"
    assert snapshot["active"] is False
    assert "engine failed" in snapshot["message"]
    manifest = json.loads((tmp_path / "runs" / snapshot["run_id"] / "manifest.json").read_text())
    assert manifest["workload_plan"]["batches"][0]["label"] == "A"
    assert manifest["completion_status"] == "failed"
    assert "Name a primary color." not in json.dumps(manifest["workload_plan"])
    content = (tmp_path / "runs" / snapshot["run_id"] / "content.jsonl").read_text()
    assert "Name a primary color." in content


def test_live_failure_message_redacts_ephemeral_huggingface_token(tmp_path: Path):
    token = "hf_secret_value"

    def explode(recording, body):  # noqa: ARG001
        raise RuntimeError(f"engine rejected {body.huggingface_token}")

    client = TestClient(create_app(tmp_path / "runs", session_factory=explode))
    body = _fixture_body()
    body["huggingface_token"] = token
    started = client.post("/api/runs", json=body)
    assert started.status_code == 200
    snapshot = _wait(client)
    assert token not in snapshot["message"]
    assert "[Hugging Face token]" in snapshot["message"]
    manifest = json.loads((tmp_path / "runs" / snapshot["run_id"] / "manifest.json").read_text())
    assert token not in json.dumps(manifest)


def test_recording_initialization_failure_does_not_stay_active(tmp_path: Path):
    def opener(*_args, **_kwargs):
        raise RuntimeError("disk full")

    client = TestClient(create_app(tmp_path / "runs", recording_opener=opener))
    started = client.post("/api/runs", json=_fixture_body())
    assert started.status_code == 200
    snapshot = _wait(client)
    assert snapshot["status"] == "failed"
    assert snapshot["active"] is False
    assert snapshot["finalized"] is False
    assert "disk full" in snapshot["message"]


def test_cleanup_failure_still_saves_the_manifest(tmp_path: Path):
    class Boom:
        def add_request(self, request_id: str, prompt: str, max_output_tokens: int = 16) -> None:
            del request_id, prompt, max_output_tokens

        def step(self) -> list[ObservedChunk]:
            return []

        def has_unfinished(self) -> bool:
            return False

        def cancel(self) -> None:
            return None

        def close(self) -> None:
            raise RuntimeError("cleanup exploded")

    client = TestClient(create_app(tmp_path / "runs", session_factory=lambda recording, body: Boom()))
    started = client.post("/api/runs", json=_fixture_body())
    assert started.status_code == 200
    snapshot = _wait(client)
    assert snapshot["status"] == "completed"
    assert snapshot["active"] is False
    assert "cleanup exploded" in snapshot["message"]
    manifest = json.loads((tmp_path / "runs" / snapshot["run_id"] / "manifest.json").read_text())
    assert manifest["workload_plan"]["batches"]
    assert manifest["controller_origin"]["provenance"]


def test_demo_fixture_honors_output_limits_and_records_effective_settings(tmp_path: Path):
    recording = Recording.open(tmp_path / "demo", run_id="demo")
    clock = Clock()
    session = FixtureSession(
        recording,
        max_num_seqs=1,
        token_budget=8,
        step_interval_s=0.04,
        sleeper=clock.sleep,
    )
    try:
        status, tokens, arrivals = drive_run(
            session,
            recording,
            [
                WorkRequest("input_1", "Name a color", 0, 2),
                WorkRequest("input_2", "Name a city", 0, 2),
            ],
            clock=clock,
            sleeper=clock.sleep,
            max_run_seconds=10,
        )
    finally:
        session.close()
    manifest = manifest_skeleton("demo", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["engine_settings"] = session.engine_settings
    manifest["engine_output_token_ids"] = tokens
    manifest["arrivals"] = arrivals
    recording.close(manifest)
    assert status == "completed"
    assert tokens["input_1"] and len(tokens["input_1"]) == 2
    assert tokens["input_2"] and len(tokens["input_2"]) == 2
    assert session.engine_settings["enable_prefix_caching"] is False
    assert session.engine_settings["max_num_seqs"] == 1
    assert session.engine_settings["teaching_pace_ms"] == 40
    assert session.engine_settings["demo_fixture"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "demo" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    scheduled = [event for event in events if event["event_type"] == "step_scheduled"]
    assert scheduled
    assert all(len(event["requests"]) <= 1 for event in scheduled)
    for request_id in ("input_1", "input_2"):
        request_events = [event for event in events if event.get("request_id") == request_id]
        sequences = {event["event_type"]: event["sequence"] for event in request_events}
        assert sequences["request_dispatched"] < sequences["engine_prompt_observed"]
        assert sequences["engine_admitted"] < sequences["request_dispatch_completed"]
    assert clock.now > 0


def test_lesson_engine_is_not_the_authored_scheduler():
    engine = FixtureEngine()
    assert engine.authored is False
    engine.add_request("request-a", [1, 2, 3, 4])
    output = engine.scheduler.schedule()
    assert output.num_scheduled_tokens["request-a"] == 4


def test_structural_export_keeps_labels_and_drops_all_content(tmp_path: Path):
    recording = Recording.open(tmp_path / "source", run_id="source")
    clock = Clock()
    session = FixtureSession(recording, max_num_seqs=2, token_budget=32, step_interval_s=0, sleeper=clock.sleep)
    try:
        _status, tokens, arrivals = drive_run(
            session,
            recording,
            [WorkRequest("sample-1", "Secret prompt text", 0, 1)],
            clock=clock,
            sleeper=clock.sleep,
            max_run_seconds=5,
        )
    finally:
        session.close()
    manifest = manifest_skeleton("source", execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET, arrivals=arrivals)
    manifest["engine_output_token_ids"] = tokens
    manifest["capture_debug"] = {
        "generated_text": "Secret completion text",
        "safe_count": 1,
    }
    manifest["workload_plan"] = {
        "version": 1,
        "synthesized": False,
        "batches": [
            {
                "id": "batch-a",
                "label": "A",
                "arrival_offset_ms": 0,
                "samples": [{"id": "sample-1", "label": "input_1", "max_output_tokens": 1, "order": 0}],
            }
        ],
    }
    manifest["controller_origin"] = {
        "controller_started_monotonic_ns": 10,
        "clock_domain": "time.monotonic_ns",
        "provenance": "injected_monotonic",
    }
    recording.close(manifest)
    destination = export_structural(tmp_path / "source", tmp_path / "exported")
    exported = json.loads((destination / "manifest.json").read_text())
    assert exported["workload_plan"]["batches"][0]["samples"][0]["label"] == "input_1"
    assert exported["capture_debug"] == {"safe_count": 1}
    assert "engine_output_token_ids" not in exported
    assert "Secret prompt text" not in json.dumps(exported)
    assert "Secret completion text" not in json.dumps(exported)
    assert not (destination / "content.jsonl").exists()


def test_trace_cursors_are_independent_and_ignore_partial_lines(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"")
    index = JsonlIndex(path, "sequence")
    path.write_bytes(b'{"sequence": 1, "event_type": "request_received"}\n{"sequence": 2')
    first, cursor, truncated = index.page(0, 10)
    assert [item["sequence"] for item in first] == [1]
    assert cursor == 1
    assert truncated is False
    again, again_cursor, _truncated = index.page(0, 1)
    assert again[0]["sequence"] == 1
    assert again_cursor == 1
    path.write_bytes(
        b'{"sequence": 1, "event_type": "request_received"}\n'
        b'{"sequence": 2, "event_type": "request_queued"}\n'
        b'{"sequence": 2, "event_type": "request_queued"}\n'
    )
    second, next_cursor, second_truncated = index.page(1, 10)
    assert [item["sequence"] for item in second] == [2]
    assert next_cursor == 2
    assert second_truncated is False
    page, page_cursor, page_truncated = index.page(0, 1)
    assert page_cursor == 1
    assert page_truncated is True


def test_trace_endpoint_pages_and_then_returns_the_manifest(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "runs"))
    started = client.post(
        "/api/runs",
        json={
            "engine": "fixture",
            "max_num_seqs": 2,
            "token_budget": 32,
            "prefix_caching": True,
            "requests": [
                {
                    "request_id": "sample-1",
                    "prompt": "Name a fruit.",
                    "arrival_offset_ms": 0,
                    "max_output_tokens": 1,
                }
            ],
            "workload_plan": {
                "version": 1,
                "batches": [
                    {
                        "id": "batch-a",
                        "label": "A",
                        "arrival_offset_ms": 0,
                        "samples": [{"id": "sample-1", "label": "input_1", "max_output_tokens": 1}],
                    }
                ],
            },
        },
    )
    assert started.status_code == 200
    run_id = started.json()["run_id"]
    snapshot = _wait(client)
    assert snapshot["status"] == "completed"
    first = client.get(f"/api/runs/{run_id}/trace", params={"limit": 1}).json()
    assert first["finalized"] is True
    assert first["events_truncated"] is True
    assert len(first["events"]) == 1
    retry = client.get(
        f"/api/runs/{run_id}/trace",
        params={"after_sequence": 0, "after_content_sequence": 0, "limit": 1},
    ).json()
    assert retry["events"][0]["sequence"] == first["events"][0]["sequence"]
    collected = []
    cursor = 0
    content_cursor = 0
    for _ in range(20):
        page = client.get(
            f"/api/runs/{run_id}/trace",
            params={"after_sequence": cursor, "after_content_sequence": content_cursor, "limit": 2},
        ).json()
        collected.extend(page["events"])
        cursor = page["next_event_sequence"]
        content_cursor = page["next_content_sequence"]
        if page["finalized"] and not page["events_truncated"] and not page["content_truncated"]:
            break
    else:
        raise AssertionError("trace pages did not drain")
    assert page["manifest"]["workload_plan"]["batches"][0]["samples"][0]["label"] == "input_1"
    assert page["manifest"]["controller_origin"]["controller_started_monotonic_ns"] > 0
    assert page["manifest"]["engine_settings"]["enable_prefix_caching"] is False
    assert page["manifest"]["demo_fixture"] is True
    sequences = [event["sequence"] for event in collected]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))
    full = client.get(f"/api/recordings/{run_id}/events").text
    full_sequences = [json.loads(line)["sequence"] for line in full.splitlines() if line.strip()]
    assert sequences == full_sequences
    assert any(item["kind"] == "prompt_text" for item in client.get(
        f"/api/runs/{run_id}/trace",
        params={"after_sequence": sequences[-1], "limit": 50},
    ).json()["content"])


def _instant_step(session: Scripted):
    def step() -> list[ObservedChunk]:
        session.step_starts.append(session.clock.now)
        request_id = session.queue.pop(0)
        return [ObservedChunk(request_id, (7,), "x", "stop")]

    return step


def _fixture_body() -> dict:
    return {
        "engine": "fixture",
        "requests": [
            {
                "request_id": "sample-1",
                "prompt": "Name a primary color.",
                "arrival_offset_ms": 0,
                "max_output_tokens": 1,
            }
        ],
        "workload_plan": {
            "version": 1,
            "batches": [
                {
                    "id": "batch-a",
                    "label": "A",
                    "arrival_offset_ms": 0,
                    "samples": [{"id": "sample-1", "label": "input_1", "max_output_tokens": 1}],
                }
            ],
        },
    }


def _wait(client: TestClient) -> dict:
    import time

    deadline = time.monotonic() + 5
    snapshot: dict = {}
    while time.monotonic() < deadline:
        snapshot = client.get("/api/runs/current").json()
        if snapshot["status"] not in {"preparing", "capturing", "finalizing", "running"}:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"run stayed active: {snapshot}")
