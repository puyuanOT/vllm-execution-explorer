"""Checks a recording against the vLLM 0.30 execution contract."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_030.constants import EVENT_TYPES, STRUCTURAL_FORBIDDEN_FIELDS
from vllm_030.compatibility import CompatibilityError
from vllm_030.inventory import validate_cache_inventory

from runner.load import RecordingData


@dataclass(frozen=True)
class Finding:
    code: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return all(finding.ok for finding in self.findings)

    def text(self) -> str:
        lines = ["Recording report", ""]
        for finding in self.findings:
            mark = "pass" if finding.ok else "fail"
            lines.append(f"[{mark}] {finding.code}: {finding.detail}")
        lines.append("")
        lines.append("Result: pass" if self.ok else "Result: fail")
        return "\n".join(lines) + "\n"


def instrumentation_overhead(enabled_step_ns: list[int], disabled_step_ns: list[int]) -> dict[str, object]:
    if not enabled_step_ns or not disabled_step_ns:
        raise ValueError("both runs need step timings")
    enabled_mean = sum(enabled_step_ns) / len(enabled_step_ns)
    disabled_mean = sum(disabled_step_ns) / len(disabled_step_ns)
    return {
        "label": "instrumentation overhead",
        "enabled_step_count": len(enabled_step_ns),
        "disabled_step_count": len(disabled_step_ns),
        "enabled_mean_ns": round(enabled_mean),
        "disabled_mean_ns": round(disabled_mean),
        "mean_difference_ns": round(enabled_mean - disabled_mean),
        "disclaimer": (
            "Host-side difference in engine step time for one bounded workload. "
            "This is instrumentation overhead, not a performance benchmark."
        ),
    }


def evaluate(data: RecordingData, *, profile: str = "integrity") -> Report:
    findings = list(_integrity(data))
    if profile == "feasibility":
        findings.extend(_feasibility(data))
    elif profile != "integrity":
        raise ValueError(f"unknown report profile {profile}")
    return Report(tuple(findings))


def _integrity(data: RecordingData) -> list[Finding]:
    events = data.events
    manifest = data.manifest
    findings = [
        _sequences(events, manifest),
        _known_types(events),
        _content_separated(events),
        _step_identity(events),
        _cursors(events),
        _blocks(events),
        _arrivals(manifest),
        _completeness_agreement(manifest, events),
    ]
    if any(event.get("event_type") == "cache_inventory" for event in events):
        findings.append(_kv_observations(events))
    return findings


def _kv_observations(events: tuple[dict, ...]) -> Finding:
    slots: dict[tuple[str, int], tuple[int, bool]] = {}
    results: dict[str, int] = {}
    groups: dict[int, dict] = {}
    scheduled: dict[str, dict] = {}
    for event in events:
        kind = event.get("event_type")
        if kind == "trace_gap":
            break  # The authoritative prefix ends here; never validate inferred future state.
        if kind == "cache_inventory":
            try:
                validate_cache_inventory(event)
            except CompatibilityError as exc:
                return Finding("kv_observations", False, str(exc))
            groups = {int(group["cache_group"]): group for group in event.get("groups", [])}
            for pool in event.get("pools", []):
                blocks = pool.get("blocks", [])
                for block in blocks:
                    slots[(str(pool["pool"]), int(block["block_id"]))] = (int(block["allocation_generation"]), bool(block.get("is_null")))
        elif kind == "block_change" and event.get("operation") == "allocate":
            key = (str(event["pool"]), int(event["block_id"]))
            if key not in slots:
                return Finding("kv_observations", False, "Allocation refers to a block absent from the initial inventory.")
            slots[key] = (int(event["allocation_generation"]), slots[key][1])
        elif kind == "step_scheduled":
            scheduled[str(event["step_id"])] = event
        elif kind == "step_result_received":
            results[str(event["step_id"])] = int(event["sequence"])
        elif kind in {"kv_block_binding", "kv_block_written"}:
            key = (str(event["pool"]), int(event["block_id"]))
            if key not in slots or slots[key] != (int(event["allocation_generation"]), False):
                return Finding("kv_observations", False, "KV range refers to an unknown, reserved-null, or stale block occupant.")
            start, end = int(event["token_range_start"]), int(event["token_range_end"])
            group = groups.get(int(event["cache_group"]))
            if group is None or start < 0 or end <= start:
                return Finding("kv_observations", False, "KV range has invalid token positions or an unobserved cache group.")
            if "token_slot_mapping" in group and group.get("token_slot_mapping") not in {"qualified_full_attention", "authored_fixture"}:
                return Finding("kv_observations", False, "KV token range claims a cache group whose token-slot mapping is unavailable.")
            size = group.get("block_size")
            if isinstance(size, int) and start // size != (end - 1) // size:
                return Finding("kv_observations", False, "KV range crosses a physical block boundary.")
            step = str(event["step_id"])
            if step not in scheduled:
                return Finding("kv_observations", False, "KV range has no corresponding scheduled step.")
            if kind == "kv_block_written":
                if step not in results or event.get("causal_event_sequence") != results[step]:
                    return Finding("kv_observations", False, "KV computation has no preceding observed result for its step.")
                request = next((item for item in scheduled[step]["requests"] if item["request_id"] == event["request_id"]), None)
                if request is None or start < request["range_start"] or end > request["range_start"] + request["range_length"]:
                    return Finding("kv_observations", False, "Computed KV range was not scheduled for this request.")
    return Finding("kv_observations", True, "Physical generations and token ranges match observed inventories, scheduler work, and returned results.")


def _feasibility(data: RecordingData) -> list[Finding]:
    kind = data.manifest.get("execution_kind")
    gpu = Finding(
        "gpu_execution",
        kind == "gpu",
        "captured from a vLLM process on a CUDA host"
        if kind == "gpu"
        else f"execution_kind is {kind}; a feasibility proof has to be a GPU run",
    )
    return [
        gpu,
        _batched_step(data.events),
        _token_match(data),
        _text_stream(data),
        _no_gap(data.manifest),
        _overhead(data.manifest),
    ]


def _sequences(events: tuple[dict, ...], manifest: dict) -> Finding:
    if not events:
        return Finding("sequences", False, "recording has no events")
    expected = 1
    gap_at = None
    for event in events:
        sequence = event.get("sequence")
        if sequence != expected:
            return Finding("sequences", False, f"expected sequence {expected}, found {sequence}")
        if event.get("event_type") == "trace_gap":
            gap_at = sequence
            break
        expected += 1
    if gap_at is not None and events[-1].get("sequence") != gap_at:
        return Finding("sequences", False, "events continue after the trace gap")
    return Finding("sequences", True, f"{len(events)} recorded events are ordered")


def _known_types(events: tuple[dict, ...]) -> Finding:
    unknown = sorted({str(event.get("event_type")) for event in events} - EVENT_TYPES)
    if unknown:
        return Finding("event_types", False, "unsupported event types: " + ", ".join(unknown))
    return Finding("event_types", True, "every event type is in the v1 contract")


def _content_separated(events: tuple[dict, ...]) -> Finding:
    leaked = sorted(
        {
            field
            for event in events
            for field in STRUCTURAL_FORBIDDEN_FIELDS
            if field in event
        }
    )
    if leaked:
        return Finding("content_separated", False, "structural events contain " + ", ".join(leaked))
    return Finding("content_separated", True, "prompt text, token ids, and text deltas stay in the content attachment")


def _step_identity(events: tuple[dict, ...]) -> Finding:
    scheduled: dict[str, dict] = {}
    for event in events:
        if event.get("event_type") == "step_scheduled":
            step_id = event.get("step_id")
            if step_id in scheduled:
                return Finding("step_identity", False, f"{step_id} was scheduled twice")
            scheduled[step_id] = event
    for event in events:
        if event.get("event_type") not in {"step_submitted", "step_result_received", "tokens_accepted"}:
            continue
        step_id = event.get("step_id")
        origin = scheduled.get(step_id)
        if origin is None:
            return Finding("step_identity", False, f"{event.get('event_type')} names unknown step {step_id}")
        causal = event.get("causal_event_sequence")
        if event.get("event_type") != "tokens_accepted" and causal != origin["sequence"]:
            return Finding(
                "step_identity",
                False,
                f"{step_id} {event.get('event_type')} does not point at its step_scheduled event",
            )
        if event.get("event_type") == "tokens_accepted" and causal is not None:
            result = next(
                (
                    item
                    for item in events
                    if item.get("event_type") == "step_result_received" and item.get("step_id") == step_id
                ),
                None,
            )
            if result is None or causal != result["sequence"]:
                return Finding("step_identity", False, f"tokens for {step_id} do not point at that step's result")
    if not scheduled:
        return Finding("step_identity", True, "no scheduler steps were recorded")
    return Finding("step_identity", True, f"{len(scheduled)} steps keep their own result association")


def _cursors(events: tuple[dict, ...]) -> Finding:
    ranges: dict[str, list[tuple[int, int]]] = {}
    reset: dict[str, int | None] = {}
    for event in events:
        if event.get("event_type") == "preemption":
            request_id = str(event["request_id"])
            reset[request_id] = int(event["computed_tokens_after"])
            continue
        if event.get("event_type") != "step_scheduled":
            continue
        for request in event.get("requests") or []:
            request_id = str(request["request_id"])
            start = int(request["range_start"])
            length = int(request["range_length"])
            previous = ranges.get(request_id)
            if previous:
                expect = reset.get(request_id)
                if expect is None:
                    expect = previous[-1][0] + previous[-1][1]
                if start != expect:
                    return Finding(
                        "cursors",
                        False,
                        f"{request_id} scheduled at {start} after the cursor was left at {expect}",
                    )
                reset[request_id] = None
            prompt = int(request["prompt_tokens"])
            generation = int(request["generation_tokens"])
            if prompt + generation != length:
                return Finding("cursors", False, f"{request_id} prompt and generation counts do not add up")
            ranges.setdefault(request_id, []).append((start, length))
    return Finding("cursors", True, "scheduled ranges meet the previous cursor, including after preemption")


def _blocks(events: tuple[dict, ...]) -> Finding:
    generations: dict[tuple[str, int], int] = {}
    seen = 0
    for event in events:
        if event.get("event_type") != "block_change":
            continue
        seen += 1
        pool = str(event["pool"])
        block_id = int(event["block_id"])
        key = (pool, block_id)
        operation = event["operation"]
        before = int(event["ref_count_before"])
        after = int(event["ref_count_after"])
        generation = int(event["allocation_generation"])
        cached = bool(event["cached_at_zero_refs"])
        has_hash_after = bool(event["has_hash_after"])
        if cached != (after == 0 and has_hash_after):
            return Finding("blocks", False, f"block {block_id} cached flag does not match its reference count and hash")
        if operation == "allocate":
            if before != 0 or after != before + 1:
                return Finding("blocks", False, f"allocation of block {block_id} is {before}->{after}")
            previous = generations.get(key, 0)
            if generation != previous + 1:
                return Finding(
                    "blocks",
                    False,
                    f"block {block_id} occupant generation {generation} does not follow {previous}",
                )
            generations[key] = generation
        elif operation == "reuse":
            if after != before + 1 or generation != generations.get(key, 0):
                return Finding("blocks", False, f"reuse of block {block_id} changed occupant or reference count unexpectedly")
        elif operation == "release":
            if after != before - 1 or after < 0 or generation != generations.get(key, 0):
                return Finding("blocks", False, f"release of block {block_id} is {before}->{after}")
        elif operation == "evict":
            if after != before or has_hash_after:
                return Finding("blocks", False, f"eviction of block {block_id} still has a hash or changed references")
        elif operation == "cache":
            if after != before or generation != generations.get(key, 0) or not has_hash_after:
                return Finding("blocks", False, f"cache publication of block {block_id} changed references or occupant")
        else:
            return Finding("blocks", False, f"unsupported block operation {operation}")
        if after == 0 and has_hash_after:
            # A zero-reference hashed block remains a cached occupant.
            pass
    if seen == 0:
        return Finding("blocks", True, "no block transitions were recorded")
    cached_releases = [
        event
        for event in events
        if event.get("event_type") == "block_change" and event.get("cached_at_zero_refs")
    ]
    if cached_releases:
        return Finding(
            "blocks",
            True,
            f"{seen} block transitions; a zero-reference block that still has a hash stays cached",
        )
    return Finding("blocks", True, f"{seen} block transitions follow reference counts and allocation generations")


def _arrivals(manifest: dict) -> Finding:
    arrivals = manifest.get("arrivals") or []
    forbidden = {"delay_ns", "offset_error_ns", "skew_ns", "arrival_delta_ns"}
    for arrival in arrivals:
        overlap = forbidden.intersection(arrival)
        if overlap:
            return Finding("arrivals", False, "arrival record subtracts clocks via " + ", ".join(sorted(overlap)))
        if "requested_offset_policy" not in arrival or "controller_add_returned_monotonic_ns" not in arrival:
            return Finding("arrivals", False, "arrival is missing the requested policy or the controller receipt time")
        if "planned_dispatch_offset_ns" in arrival:
            required = {
                "dispatch_started_monotonic_ns",
                "dispatch_completed_monotonic_ns",
                "dispatch_clock_domain",
                "http_receipt_observed",
                "engine_admission_observed",
            }
            missing = sorted(required.difference(arrival))
            if missing:
                return Finding("arrivals", False, "dispatch observation is missing " + ", ".join(missing))
            if arrival.get("http_receipt_observed") and arrival.get("http_received_monotonic_ns") is None:
                return Finding("arrivals", False, "HTTP receipt is marked observed without a timestamp")
            if arrival.get("engine_admission_observed") and arrival.get("engine_admitted_monotonic_ns") is None:
                return Finding("arrivals", False, "engine admission is marked observed without a timestamp")
    return Finding(
        "arrivals",
        True,
        "planned dispatch, client dispatch, HTTP receipt, and engine admission remain named observations",
    )


def _completeness_agreement(manifest: dict, events: tuple[dict, ...]) -> Finding:
    completeness = manifest.get("event_completeness") or {}
    status = completeness.get("status")
    completion = manifest.get("completion_status")
    if status == "incomplete" and completion == "completed":
        return Finding("completeness", False, "a completed run still has an incomplete event stream")
    if status not in {"complete", "incomplete"}:
        return Finding("completeness", False, "event completeness is missing")
    if status == "complete" and any(event.get("event_type") == "trace_gap" for event in events):
        return Finding("completeness", False, "a complete event stream contains a trace gap")
    if status == "complete" and completion == "completed":
        # Contiguous sequence numbers only prove that the retained prefix is
        # ordered. A truncated file can still be contiguous, so a completed
        # run must also retain the terminal evidence for its declared work.
        requests = {
            str(event["request_id"])
            for event in events
            if event.get("request_id") is not None
        }
        requests.update(
            str(arrival["request_id"])
            for arrival in manifest.get("arrivals") or []
            if arrival.get("request_id") is not None
        )
        plan = manifest.get("workload_plan") or {}
        for batch in plan.get("batches") or []:
            requests.update(str(sample["id"]) for sample in batch.get("samples") or [] if sample.get("id"))
        scheduled = set()
        results = set()
        terminal = set()
        for event in events:
            kind = event.get("event_type")
            if kind == "step_scheduled":
                participants = event.get("requests") or []
                requests.update(str(request["request_id"]) for request in participants)
                # An empty scheduler iteration need not invoke the executor.
                if participants or event.get("total_scheduled_tokens", 0) > 0:
                    scheduled.add(event.get("step_id"))
            elif kind == "step_result_received":
                results.add(event.get("step_id"))
            elif kind in {"completion", "cancellation"}:
                terminal.add(str(event.get("request_id")))
            elif kind == "http_response_completed":
                code = event.get("status_code")
                reason = event.get("finish_reason")
                if isinstance(code, int) and 200 <= code < 300 and isinstance(reason, str) and reason:
                    terminal.add(str(event.get("request_id")))
        missing_results = sorted(str(step) for step in scheduled - results)
        missing_terminals = sorted(requests - terminal)
        if missing_results or missing_terminals:
            missing = []
            if missing_results:
                missing.append("results for " + ", ".join(missing_results))
            if missing_terminals:
                missing.append("terminal events for " + ", ".join(missing_terminals))
            return Finding("completeness", False, "a completed run is missing " + "; ".join(missing))
    return Finding("completeness", True, f"event stream is {status}")


def _batched_step(events: tuple[dict, ...]) -> Finding:
    for event in events:
        if event.get("event_type") != "step_scheduled":
            continue
        requests = event.get("requests") or []
        if len(requests) >= 2 and all(int(item["range_length"]) > 0 for item in requests):
            names = ", ".join(str(item["request_id"]) for item in requests)
            return Finding("batching", True, f"{event.get('step_id')} scheduled {names} together")
    return Finding("batching", False, "no step scheduled more than one request")


def _token_match(data: RecordingData) -> Finding:
    expected = (data.manifest.get("engine_output_token_ids") or {})
    if not expected:
        return Finding("token_match", False, "manifest has no engine output token ids to compare")
    actual: dict[str, list[int]] = {str(key): [] for key in expected}
    rows = sorted(data.content, key=lambda item: int(item["content_sequence"]))
    for row in rows:
        if row.get("kind") != "accepted_tokens":
            continue
        request_id = str(row["request_id"])
        actual.setdefault(request_id, []).extend(int(token) for token in row["token_ids"])
    if actual != {str(key): list(value) for key, value in expected.items()}:
        return Finding("token_match", False, "accepted token ids differ from the engine output")
    return Finding("token_match", True, "accepted token ids match the engine output for each request")


def _text_stream(data: RecordingData) -> Finding:
    text = [row for row in data.content if row.get("kind") == "text"]
    structural = [event for event in data.events if event.get("event_type") == "text_emitted"]
    if not text or not structural:
        return Finding("text_stream", False, "emitted text is missing from the content attachment or the event stream")
    return Finding("text_stream", True, "emitted text is a separate stream from accepted token ids")


def _no_gap(manifest: dict) -> Finding:
    status = (manifest.get("event_completeness") or {}).get("status")
    if status != "complete":
        return Finding("trace_gap", False, "feasibility run contains a trace gap")
    return Finding("trace_gap", True, "the success recording contains no trace gap")


def _overhead(manifest: dict) -> Finding:
    overhead = manifest.get("instrumentation_overhead")
    if not isinstance(overhead, dict):
        return Finding("overhead", False, "instrumentation overhead was not measured")
    if overhead.get("label") != "instrumentation overhead":
        return Finding("overhead", False, "overhead note is missing its label")
    disclaimer = str(overhead.get("disclaimer") or "")
    if "not a performance benchmark" not in disclaimer:
        return Finding("overhead", False, "overhead note is missing the benchmark disclaimer")
    return Finding("overhead", True, "host-side step-time difference is labeled as instrumentation overhead")
