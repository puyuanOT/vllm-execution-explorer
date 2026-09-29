"""Bundled fixture lessons. Each one is labeled as a fixture, not a GPU trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from vllm_030.collector import Recording
from vllm_030.install import install

from runner.fixture_engine import FixtureEngine
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.stagger import run_staggered


def write_catalog(destination: Path) -> list[dict[str, str]]:
    destination.mkdir(parents=True, exist_ok=True)
    lessons = [
        _continuous_batching(destination),
        _single_request(destination),
        _chunked_prefill(destination),
        _prefix_reuse(destination),
        _prefix_miss(destination),
        _preemption(destination),
        _cancellation(destination),
        _incomplete(destination),
    ]
    (destination / "index.json").write_text(
        json.dumps({"recordings": lessons}, indent=2) + "\n",
        encoding="utf-8",
    )
    return lessons


def _lesson(title: str, summary: str) -> dict[str, Any]:
    return {"title": title, "summary": summary, "evidence": "fixture"}


def _continuous_batching(root: Path) -> dict[str, str]:
    run_id = "continuous-batching"
    recording = Recording.open(root / run_id, run_id=run_id)
    engine = FixtureEngine()
    adapter = install(engine, recording, check_sources=False)
    try:
        result = run_staggered(engine, recording, [1, 2, 3, 4], [5, 6, 7, 8])
    finally:
        adapter.uninstall()
    manifest = manifest_skeleton(
        run_id,
        execution_kind="fixture",
        execution_target=LOCAL_FIXTURE_TARGET,
        arrivals=result.arrivals,
    )
    manifest["engine_output_token_ids"] = result.engine_output_token_ids
    manifest["lesson"] = _lesson(
        "Continuous batching",
        "Request B is admitted after A has accepted an output token. "
        "A later step lists both request ids. This file was produced by the fixture engine.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _single_request(root: Path) -> dict[str, str]:
    run_id = "single-request"
    recording = Recording.open(root / run_id, run_id=run_id)
    engine = FixtureEngine()
    adapter = install(engine, recording, check_sources=False)
    tokens: list[int] = []
    try:
        engine.add_request("request-a", [1, 2, 3, 4])
        recording.content(kind="prompt_text", request_id="request-a", text="one request baseline")
        emissions = 0
        while engine.has_unfinished():
            for chunk in engine.step():
                start = len(tokens)
                tokens.extend(chunk.new_token_ids)
                if chunk.text_delta:
                    recording.text_emitted(
                        chunk.request_id,
                        emission_index=emissions,
                        text_delta=chunk.text_delta,
                        token_range=(start, len(tokens)),
                    )
                    emissions += 1
    finally:
        adapter.uninstall()
    manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["engine_output_token_ids"] = {"request-a": tokens}
    manifest["lesson"] = _lesson(
        "Single request",
        "One request keeps a single scheduled cursor, accepted token ids, emitted text, and a terminal status.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _chunked_prefill(root: Path) -> dict[str, str]:
    run_id = "chunked-prefill"
    recording = Recording.open(root / run_id, run_id=run_id)
    _requests(recording, [("long", "a long prompt split across scheduler steps"), ("short", "short")])
    first, first_result = _step(
        recording,
        "step-1",
        [
            _range("long", 0, 4, prompt_tokens=4, prompt_length=8),
            _range("short", 0, 2, prompt_tokens=2, prompt_length=2),
        ],
    )
    _accept(recording, "short", first, first_result, [11], 0)
    recording.text_emitted("short", emission_index=0, text_delta="Hi", token_range=(0, 1))
    second, second_result = _step(
        recording,
        "step-2",
        [
            _range("long", 4, 4, prompt_tokens=4, prompt_length=8),
            _range("short", 2, 1, prompt_tokens=0, prompt_length=2),
        ],
    )
    _accept(recording, "long", second, second_result, [21], 0)
    _accept(recording, "short", second, second_result, [12], 1)
    recording.text_emitted("long", emission_index=0, text_delta="Done", token_range=(0, 1))
    recording.event("completion", request_id="long", step_id=second, status="stop", reason="stop")
    recording.event("completion", request_id="short", step_id=second, status="stop", reason="stop")
    manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["lesson"] = _lesson(
        "Chunked prefill",
        "The long prompt is scheduled as two ranges. The second step also gives the short request a generation token.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _prefix_reuse(root: Path) -> dict[str, str]:
    run_id = "prefix-reuse"
    recording = Recording.open(root / run_id, run_id=run_id)
    _requests(recording, [("first", "shared prefix"), ("second", "shared prefix plus more")])
    step_id, result = _step(recording, "step-1", [_range("first", 0, 4, prompt_tokens=4, prompt_length=4)])
    _accept(recording, "first", step_id, result, [31], 0)
    recording.event("completion", request_id="first", step_id=step_id, status="stop", reason="stop")
    recording.event(
        "prefix_lookup",
        request_id="second",
        matched_token_count=4,
        block_ids=[[1]],
        shared_prefix_boundary=4,
    )
    later, later_result = _step(
        recording,
        "step-2",
        [_range("second", 4, 1, prompt_tokens=0, prompt_length=6)],
    )
    _accept(recording, "second", later, later_result, [32], 0)
    recording.text_emitted("second", emission_index=0, text_delta="Next", token_range=(0, 1))
    recording.event("completion", request_id="second", step_id=later, status="stop", reason="stop")
    manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["lesson"] = _lesson(
        "Prefix reuse",
        "The second request matches four prefix tokens, so this step starts at token 4. "
        "Matching a prefix skips that prompt work. It does not skip ordinary decoding.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _prefix_miss(root: Path) -> dict[str, str]:
    run_id = "prefix-miss"
    recording = Recording.open(root / run_id, run_id=run_id)
    _requests(recording, [("changed", "a different prefix")])
    recording.event(
        "prefix_lookup",
        request_id="changed",
        matched_token_count=0,
        block_ids=[],
        shared_prefix_boundary=0,
    )
    step_id, result = _step(recording, "step-1", [_range("changed", 0, 4, prompt_tokens=4, prompt_length=4)])
    _accept(recording, "changed", step_id, result, [41], 0)
    recording.text_emitted("changed", emission_index=0, text_delta="Miss", token_range=(0, 1))
    recording.event("completion", request_id="changed", step_id=step_id, status="stop", reason="stop")
    manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["lesson"] = _lesson(
        "Prefix miss",
        "A changed prefix matches zero tokens, so the full prompt range is scheduled.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _preemption(root: Path) -> dict[str, str]:
    run_id = "preemption"
    recording = Recording.open(root / run_id, run_id=run_id)
    _requests(recording, [("crowded", "needs a block again")])
    recording.event(
        "block_change",
        pool="pool-0",
        block_id=3,
        allocation_generation=1,
        operation="allocate",
        ref_count_before=0,
        ref_count_after=1,
        has_hash_before=False,
        has_hash_after=False,
        cached_at_zero_refs=False,
    )
    step_id, result = _step(recording, "step-1", [_range("crowded", 0, 4, prompt_tokens=4, prompt_length=4)])
    _accept(recording, "crowded", step_id, result, [51], 0)
    recording.event(
        "preemption",
        request_id="crowded",
        status="preempted",
        computed_tokens_before=4,
        computed_tokens_after=0,
        preemption_count=1,
    )
    recording.event(
        "block_change",
        pool="pool-0",
        block_id=3,
        allocation_generation=1,
        operation="release",
        ref_count_before=1,
        ref_count_after=0,
        has_hash_before=True,
        has_hash_after=True,
        cached_at_zero_refs=True,
        release_source="free_blocks",
    )
    recording.event(
        "block_change",
        pool="pool-0",
        block_id=3,
        allocation_generation=1,
        operation="evict",
        ref_count_before=0,
        ref_count_after=0,
        has_hash_before=True,
        has_hash_after=False,
        cached_at_zero_refs=False,
    )
    recording.event(
        "block_change",
        pool="pool-0",
        block_id=3,
        allocation_generation=2,
        operation="allocate",
        ref_count_before=0,
        ref_count_after=1,
        has_hash_before=False,
        has_hash_after=False,
        cached_at_zero_refs=False,
    )
    again, again_result = _step(recording, "step-2", [_range("crowded", 0, 4, prompt_tokens=4, prompt_length=4)])
    _accept(recording, "crowded", again, again_result, [52], 1)
    recording.text_emitted("crowded", emission_index=0, text_delta="Again", token_range=(1, 2))
    recording.event("completion", request_id="crowded", step_id=again, status="stop", reason="stop")
    manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["lesson"] = _lesson(
        "Preemption and recomputation",
        "The scheduled cursor returns to 0 and the same block id comes back as a new allocation generation. "
        "The trace records the transition. It does not record a decision reason.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _cancellation(root: Path) -> dict[str, str]:
    run_id = "cancellation"
    recording = Recording.open(root / run_id, run_id=run_id)
    _requests(recording, [("kept", "finishes"), ("dropped", "cancelled while queued")])
    step_id, result = _step(recording, "step-1", [_range("kept", 0, 2, prompt_tokens=2, prompt_length=2)])
    _accept(recording, "kept", step_id, result, [61], 0)
    recording.text_emitted("kept", emission_index=0, text_delta="Stay", token_range=(0, 1))
    recording.event("completion", request_id="kept", step_id=step_id, status="stop", reason="stop")
    recording.event("cancellation", request_id="dropped", status="abort", reason="abort")
    manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
    manifest["lesson"] = _lesson(
        "Cancellation",
        "One request completes. The other is cancelled. Those are different terminal events.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _incomplete(root: Path) -> dict[str, str]:
    run_id = "incomplete-capture"
    recording = Recording.open(root / run_id, run_id=run_id, capacity=8, autostart=False)
    _requests(recording, [("partial", "capture stopped")])
    _step(recording, "step-1", [_range("partial", 0, 2, prompt_tokens=2, prompt_length=2)])
    recording.event("step_submitted", step_id="step-missing", causal_event_sequence=1, observation="before_execute_model")
    manifest = manifest_skeleton(
        run_id,
        execution_kind="fixture",
        execution_target=LOCAL_FIXTURE_TARGET,
        completion_status="completed",
    )
    manifest["lesson"] = _lesson(
        "Incomplete capture",
        "The collector filled its buffer, wrote one trace gap, and kept the events already queued. "
        "State after the gap is unknown.",
    )
    recording.close(manifest)
    return {"id": run_id, **manifest["lesson"]}


def _requests(recording: Recording, pairs: list[tuple[str, str]]) -> None:
    for request_id, prompt in pairs:
        recording.event("request_received", request_id=request_id, arrival_clock="request.arrival_time")
        recording.content(kind="prompt_text", request_id=request_id, text=prompt)
        recording.event("request_tokenized", request_id=request_id, prompt_token_count=len(prompt.split()) or 1)
        recording.event("request_queued", request_id=request_id, status="queued")


def _range(
    request_id: str,
    start: int,
    length: int,
    *,
    prompt_tokens: int,
    prompt_length: int,
) -> dict[str, int | str]:
    return {
        "request_id": request_id,
        "range_start": start,
        "range_length": length,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": length - prompt_tokens,
        "num_prompt_tokens": prompt_length,
    }


def _step(
    recording: Recording,
    step_id: str,
    requests: list[dict[str, int | str]],
) -> tuple[str, int]:
    scheduled = recording.event(
        "step_scheduled",
        step_id=step_id,
        token_budget=32,
        total_scheduled_tokens=sum(int(item["range_length"]) for item in requests),
        requests=requests,
        preempted_request_ids=[],
        block_table_changes=[],
    )
    assert scheduled is not None
    recording.event(
        "step_submitted",
        step_id=step_id,
        causal_event_sequence=scheduled,
        observation="before_execute_model",
    )
    result = recording.event(
        "step_result_received",
        step_id=step_id,
        causal_event_sequence=scheduled,
        observation="future.result",
    )
    assert result is not None
    return step_id, result


def _accept(
    recording: Recording,
    request_id: str,
    step_id: str,
    result_sequence: int,
    token_ids: list[int],
    position: int,
) -> None:
    recording.content(
        kind="accepted_tokens",
        request_id=request_id,
        step_id=step_id,
        output_position_start=position,
        token_ids=token_ids,
    )
    recording.event(
        "tokens_accepted",
        request_id=request_id,
        step_id=step_id,
        output_position_start=position,
        token_count=len(token_ids),
        content_ref=request_id,
        causal_event_sequence=result_sequence,
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    destination = Path(args[0] if args else "web/public/recordings")
    lessons = write_catalog(destination)
    print(f"wrote {len(lessons)} fixture lessons to {destination}")


if __name__ == "__main__":
    main()
