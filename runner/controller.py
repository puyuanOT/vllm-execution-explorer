"""Admit requests on the controller clock while an engine is stepping."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from vllm_030.collector import Recording

from runner.fixture_engine import ObservedChunk
from runner.workload import PROVISIONAL_LIMITS, WorkRequest


class Session(Protocol):
    def add_request(self, request_id: str, prompt: str, max_output_tokens: int = 16) -> Any: ...

    def step(self) -> list[ObservedChunk]: ...

    def has_unfinished(self) -> bool: ...

    def cancel(self) -> None: ...


def drive_run(
    session: Session,
    recording: Recording,
    requests: list[WorkRequest],
    *,
    clock: Callable[[], float] = time.monotonic,
    clock_ns: Callable[[], int] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    intake_open: Callable[[], bool] = lambda: True,
    cancelled: Callable[[], bool] = lambda: False,
    pull_extra: Callable[[], list[WorkRequest]] = lambda: [],
    on_status: Callable[[dict[str, str]], None] | None = None,
    on_arrival: Callable[[dict[str, object]], None] | None = None,
    origin_out: dict[str, object] | None = None,
    notes: dict[str, str] | None = None,
    prompted: set[str] | None = None,
    max_run_seconds: float | None = None,
    idle_slice_s: float = 0.02,
) -> tuple[str, dict[str, list[int]], list[dict[str, object]]]:
    """Admit due work in submission order.

    ``intake_open`` does not drop requests that were already accepted. Closing
    intake is the API's job: it rejects new adds, and this loop still drains
    the accepted list. Equal offsets keep submission order; they are not
    reordered by request id.
    """
    del intake_open
    if max_run_seconds is None:
        max_run_seconds = float(PROVISIONAL_LIMITS["max_run_seconds"])
    pending = list(requests)
    tokens: dict[str, list[int]] = {item.request_id: [] for item in requests}
    text: dict[str, str] = {item.request_id: "" for item in requests}
    emissions: dict[str, int] = {}
    arrivals: list[dict[str, object]] = []
    prompted_ids = set(prompted or ())
    started = clock()
    started_ns = _now_ns(clock, clock_ns)
    origin = {
        "controller_started_monotonic_ns": started_ns,
        "clock_domain": "time.monotonic_ns",
        "provenance": "injected_monotonic" if clock is not time.monotonic or clock_ns is not None else "time.monotonic_ns",
    }
    if origin_out is not None:
        origin_out.clear()
        origin_out.update(origin)

    def publish() -> None:
        if on_status is None:
            return
        on_status({request_id: "".join(value) for request_id, value in text.items()})

    def remember_prompt(request: WorkRequest) -> None:
        if request.request_id in prompted_ids:
            return
        prompted_ids.add(request.request_id)
        if request.input_mode == "synthetic":
            recording.content(
                kind="synthetic_input",
                request_id=request.request_id,
                target_input_tokens=request.target_input_tokens,
                resolved_input_tokens=request.resolved_input_tokens,
            )
        else:
            recording.content(kind="prompt_text", request_id=request.request_id, text=request.prompt)

    for request in requests:
        remember_prompt(request)
    _sort_pending(pending)

    def elapsed_ms() -> float:
        return (clock() - started) * 1000

    def deadline_reached() -> bool:
        return (clock() - started) >= max_run_seconds

    while True:
        if cancelled():
            session.cancel()
            return "cancelled", tokens, arrivals
        if deadline_reached():
            session.cancel()
            if notes is not None:
                notes["message"] = f"Stopped: the run exceeded {max_run_seconds:.0f}s."
            return "failed", tokens, arrivals
        extra = pull_extra()
        for request in extra:
            tokens.setdefault(request.request_id, [])
            text.setdefault(request.request_id, "")
            remember_prompt(request)
            pending.append(request)
        if extra:
            _sort_pending(pending)
        due = elapsed_ms()
        due_requests: list[WorkRequest] = []
        while pending and pending[0].arrival_offset_ms <= due:
            due_requests.append(pending.pop(0))
        observations: dict[str, dict[str, Any]] = {}
        batch_dispatch = getattr(session, "add_work_requests", None)
        if due_requests and callable(batch_dispatch):
            observations = batch_dispatch(due_requests)
        for request in due_requests:
            if cancelled() or deadline_reached():
                break
            observed = observations.get(request.request_id, {})
            if not callable(batch_dispatch):
                dispatch_started = _now_ns(clock, clock_ns)
                dispatch_transport = str(getattr(session, "dispatch_transport", "direct_session"))
                dispatch_event_sequence = recording.event(
                    "request_dispatched",
                    request_id=request.request_id,
                    planned_dispatch_offset_ns=request.arrival_offset_ms * 1_000_000,
                    dispatch_started_monotonic_ns=dispatch_started,
                    dispatch_clock_domain=origin["clock_domain"],
                    dispatch_transport=dispatch_transport,
                    phase="started",
                )
                add_work_request = getattr(session, "add_work_request", None)
                boundary = add_work_request(request) if callable(add_work_request) else session.add_request(
                    request.request_id,
                    request.prompt,
                    request.max_output_tokens,
                )
                dispatch_completed = _now_ns(clock, clock_ns)
                observed = boundary if isinstance(boundary, dict) else {}
                observed = {**observed, "dispatch_event_recorded": True}
                recording.event(
                    "request_dispatch_completed",
                    request_id=request.request_id,
                    dispatch_completed_monotonic_ns=dispatch_completed,
                    dispatch_clock_domain=origin["clock_domain"],
                    dispatch_transport=dispatch_transport,
                    causal_event_sequence=dispatch_event_sequence,
                )
            else:
                dispatch_started = int(observed.get("dispatch_started_monotonic_ns") or _now_ns(clock, clock_ns))
                dispatch_completed = int(observed.get("dispatch_completed_monotonic_ns") or dispatch_started)
            arrival = {
                "request_id": request.request_id,
                "planned_dispatch_offset_ns": request.arrival_offset_ms * 1_000_000,
                "dispatch_started_monotonic_ns": dispatch_started,
                "dispatch_completed_monotonic_ns": dispatch_completed,
                "dispatch_clock_domain": str(observed.get("dispatch_clock_domain") or origin["clock_domain"]),
                "dispatch_transport": str(getattr(session, "dispatch_transport", "direct_session")),
                "http_receipt_observed": bool(observed.get("http_receipt_observed", False)),
                "http_received_monotonic_ns": observed.get("http_received_monotonic_ns"),
                "http_receipt_clock_domain": observed.get("http_receipt_clock_domain"),
                "engine_admission_observed": bool(observed.get("engine_admission_observed", False)),
                "engine_admitted_monotonic_ns": observed.get("engine_admitted_monotonic_ns"),
                "engine_admission_clock_domain": observed.get("engine_admission_clock_domain"),
                "resolved_input_tokens": observed.get("resolved_input_tokens", request.resolved_input_tokens),
                "adapter_id": str(observed.get("adapter_id") or request.adapter_id or "base"),
                "lora_int_id": observed.get("lora_int_id"),
                # Kept for recording-v1 readers. These are controller observations,
                # not HTTP receipt or engine-admission timestamps.
                "requested_offset_ns": request.arrival_offset_ms * 1_000_000,
                "requested_offset_policy": "controller_offset",
                "controller_add_returned_monotonic_ns": dispatch_completed,
            }
            if not observed.get("dispatch_event_recorded"):
                recording.event(
                    "request_dispatched",
                    request_id=request.request_id,
                    planned_dispatch_offset_ns=arrival["planned_dispatch_offset_ns"],
                    dispatch_started_monotonic_ns=dispatch_started,
                    dispatch_completed_monotonic_ns=dispatch_completed,
                    dispatch_clock_domain=arrival["dispatch_clock_domain"],
                    dispatch_transport=arrival["dispatch_transport"],
                    http_receipt_observed=arrival["http_receipt_observed"],
                    http_received_monotonic_ns=arrival["http_received_monotonic_ns"],
                    http_receipt_clock_domain=arrival["http_receipt_clock_domain"],
                    engine_admission_observed=arrival["engine_admission_observed"],
                    engine_admitted_monotonic_ns=arrival["engine_admitted_monotonic_ns"],
                    engine_admission_clock_domain=arrival["engine_admission_clock_domain"],
                )
            arrivals.append(arrival)
            if on_arrival is not None:
                on_arrival(dict(arrival))
        if cancelled():
            session.cancel()
            return "cancelled", tokens, arrivals
        if deadline_reached():
            session.cancel()
            if notes is not None:
                notes["message"] = f"Stopped: the run exceeded {max_run_seconds:.0f}s."
            return "failed", tokens, arrivals
        if session.has_unfinished():
            for chunk in session.step():
                start = len(tokens[chunk.request_id])
                tokens[chunk.request_id].extend(chunk.new_token_ids)
                if chunk.text_delta:
                    index = emissions.get(chunk.request_id, 0)
                    end = len(tokens[chunk.request_id])
                    recording.text_emitted(
                        chunk.request_id,
                        emission_index=index,
                        text_delta=chunk.text_delta,
                        token_range=(start, end) if chunk.new_token_ids else None,
                    )
                    text[chunk.request_id] += chunk.text_delta
                    emissions[chunk.request_id] = index + 1
            publish()
            continue
        if pending:
            delay_ms = pending[0].arrival_offset_ms - elapsed_ms()
            if delay_ms > 0:
                sleeper(min(idle_slice_s, delay_ms / 1000))
            continue
        return "completed", tokens, arrivals


def _sort_pending(pending: list[WorkRequest]) -> None:
    """Sort by offset only. Python's sort is stable, so equal offsets keep submission order."""
    pending.sort(key=lambda item: item.arrival_offset_ms)


def _now_ns(clock: Callable[[], float], clock_ns: Callable[[], int] | None) -> int:
    if clock_ns is not None:
        return int(clock_ns())
    if clock is time.monotonic:
        return time.monotonic_ns()
    return int(round(clock() * 1_000_000_000))
