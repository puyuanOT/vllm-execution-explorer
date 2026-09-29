"""Arrival control for one staggered pair, independent of the playback clock."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from vllm_030.collector import Recording

from runner.fixture_engine import ObservedChunk


class Steppable(Protocol):
    def add_request(self, request_id: str, prompt: Any) -> None: ...

    def step(self) -> list[ObservedChunk]: ...

    def has_unfinished(self) -> bool: ...


@dataclass
class StaggerResult:
    arrivals: list[dict[str, Any]]
    engine_output_token_ids: dict[str, list[int]]
    step_times_ns: list[int] = field(default_factory=list)


def run_staggered(
    engine: Steppable,
    recording: Recording,
    prompt_a: Any,
    prompt_b: Any,
    *,
    max_steps: int = 32,
) -> StaggerResult:
    """Hold request B until request A has accepted one output token."""
    tokens: dict[str, list[int]] = {"request-a": [], "request-b": []}
    emissions: dict[str, int] = {}
    arrivals: list[dict[str, Any]] = []
    step_times: list[int] = []

    def admit(request_id: str, prompt: Any, policy: str, requested_offset_ns: int | None) -> None:
        started = time.monotonic_ns()
        engine.add_request(request_id, prompt)
        completed = time.monotonic_ns()
        planned = 0 if requested_offset_ns is None else requested_offset_ns
        recording.event(
            "request_dispatched",
            request_id=request_id,
            planned_dispatch_offset_ns=planned,
            dispatch_started_monotonic_ns=started,
            dispatch_completed_monotonic_ns=completed,
            dispatch_clock_domain="controller:time.monotonic_ns",
            dispatch_transport="direct_session",
            http_receipt_observed=False,
            engine_admission_observed=False,
        )
        arrivals.append(
            {
                "request_id": request_id,
                "planned_dispatch_offset_ns": planned,
                "dispatch_started_monotonic_ns": started,
                "dispatch_completed_monotonic_ns": completed,
                "dispatch_clock_domain": "controller:time.monotonic_ns",
                "dispatch_transport": "direct_session",
                "http_receipt_observed": False,
                "http_received_monotonic_ns": None,
                "http_receipt_clock_domain": None,
                "engine_admission_observed": False,
                "engine_admitted_monotonic_ns": None,
                "engine_admission_clock_domain": None,
                "requested_offset_ns": requested_offset_ns,
                "requested_offset_policy": policy,
                "controller_add_returned_monotonic_ns": completed,
            }
        )

    def observe(chunks: list[ObservedChunk]) -> None:
        for chunk in chunks:
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
                emissions[chunk.request_id] = index + 1

    admit("request-a", prompt_a, "immediate", 0)
    steps = 0
    while len(tokens["request-a"]) < 1:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("request A produced no accepted token")
        started = time.monotonic_ns()
        observe(engine.step())
        step_times.append(time.monotonic_ns() - started)
    admit("request-b", prompt_b, "after_request_a_first_accepted_token", None)
    while engine.has_unfinished():
        steps += 1
        if steps > max_steps:
            raise RuntimeError("staggered run exceeded its step limit")
        started = time.monotonic_ns()
        observe(engine.step())
        step_times.append(time.monotonic_ns() - started)
    return StaggerResult(arrivals, tokens, step_times)
