"""Normalize native profiler timestamps and explicit scope/runtime correlations."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


PROFILE_CLOCK = {"domain": "torch.profiler.chrome_trace", "unit": "microseconds", "origin": "trace_local"}
MAX_ACTIVITIES = 4_000_000
_STEP = re.compile(r"^explorer\.step/(step-[0-9]+)$")
_LAYER = re.compile(r"^explorer\.layer/([0-9]+)$")
_COMPONENT = re.compile(r"^explorer\.component/([0-9]+)/(attention|recurrent|ffn)$")
_CACHE_WRITE = re.compile(r"^explorer\.cache_write/([0-9]+)/(cache-write-[0-9]+)$")


def normalize_trace(trace: dict[str, Any], *, run_id: str, steps: list[dict[str, Any]], max_activities: int = MAX_ACTIVITIES) -> dict[str, Any]:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("The native profiler trace has no traceEvents array.")
    timed = [(index, event) for index, event in enumerate(events) if isinstance(event, dict) and event.get("ph") == "X" and _number(event.get("ts")) and _number(event.get("dur")) and event["dur"] >= 0]
    truncated = len(timed) > max_activities
    scope_events = timed
    retention = None
    if truncated:
        required = _required_native_events(timed)
        if len(required) <= max_activities:
            # Resolve scope inheritance and ambiguity from the complete native CPU stream.
            # Only output detail is reduced; activity IDs remain original trace-event indices.
            retention = {
                "status": "cpu_detail_bounded", "activity_limit": max_activities,
                "original_activity_count": len(timed), "retained_activity_count": len(required),
                "omitted_cpu_activity_count": len(timed) - len(required), "omitted_gpu_activity_count": 0,
                "gpu_activities_complete": True,
                "preserved": "All native CUDA activities, CPU correlation candidates, launch records and explicit diagnostic scopes; source activity IDs and ordering are unchanged.",
            }
            timed = required
        else:
            # Required evidence itself cannot fit. Preserve the historical bounded window
            # without claiming complete GPU coverage or referring to omitted CPU records.
            timed = timed[:max_activities]
            scope_events = timed
    cpu = [(index, event) for index, event in scope_events if not _gpu_kind(event)]
    by_thread: dict[tuple[Any, Any], list[tuple[int, dict]]] = defaultdict(list)
    for index, event in cpu:
        by_thread[(event.get("pid"), event.get("tid"))].append((index, event))
    scopes: dict[int, dict[str, Any]] = {}
    correlations: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for thread_events in by_thread.values():
        active: list[tuple[float, dict[str, Any]]] = []
        for index, event in sorted(thread_events, key=lambda value: (value[1]["ts"], -value[1]["dur"], value[0])):
            start, end = event["ts"], event["ts"] + event["dur"]
            active = [(stop, scope) for stop, scope in active if stop >= end]
            scope = _empty_scope()
            for _stop, inherited in active:
                scope.update(inherited)
            explicit = _explicit_scope(str(event.get("name", "")))
            if explicit:
                scope.update(explicit)
                active.append((end, dict(scope)))
            if any(scope[key] is not None for key in ("step_id", "layer_index", "component")):
                scope["correlation"] = "explicit_scope"
            scopes[index] = scope
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            for key in ("External id", "correlation"):
                value = args.get(key)
                if value is not None:
                    correlations[(key, str(value))].append((index, scope))
    activities = []
    for index, event in timed:
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        kind = _gpu_kind(event)
        scope = dict(scopes.get(index) or _empty_scope())
        related: list[str] = []
        if kind:
            matches: list[tuple[int, dict[str, Any], str]] = []
            for key in ("correlation", "External id"):
                if args.get(key) is not None:
                    matches.extend((cpu_index, candidate, key) for cpu_index, candidate in correlations.get((key, str(args[key])), []))
                    if matches:
                        break  # Prefer direct native launch correlation over ancestor external IDs.
            signatures = {_scope_identity(candidate) for _cpu_index, candidate, _key in matches}
            if len(signatures) == 1 and matches and matches[0][1]["step_id"] is not None:
                scope = {**matches[0][1], "correlation": "runtime_flow"}
                scope["correlation_source"] = "native_correlation_id" if any(key == "correlation" for _index, _scope, key in matches) else "native_external_id"
                related = sorted({f"activity-{cpu_index}" for cpu_index, _candidate, _key in matches})
        activities.append({
            "activity_id": f"activity-{index}", "kind": kind or _cpu_kind(event), "name": str(event.get("name", "unnamed")),
            "device": _identity(args.get("device")) if kind else None,
            "stream": _identity(args.get("stream")) if kind else None,
            "process_id": _identity(event.get("pid")), "thread_id": _identity(event.get("tid")),
            "start": event["ts"], "duration": event["dur"], "scope": scope,
            "related_activity_ids": related,
        })
    cuda = [activity for activity in activities if activity["kind"].startswith("cuda_")]
    normalization = {"status": "partial" if truncated else "completed", "source_activity_count": len([event for event in events if isinstance(event, dict) and event.get("ph") == "X"]),
                     "activity_limit": max_activities, "gpu_activity_count": len(cuda),
                     "correlated_gpu_activity_count": sum(activity["scope"]["correlation"] == "runtime_flow" for activity in cuda)}
    if retention is not None:
        normalization["attachment_retention"] = retention
    return {
        "contract_version": 1, "run_id": run_id, "mode": "eager_diagnostic", "execution_mode": "eager",
        "backend": "torch.profiler", "clock": dict(PROFILE_CLOCK), "steps": steps, "activities": activities,
        "normalization": normalization,
        "evidence": "Native profiler durations describe batch/layer CPU scopes and individually observed CUDA activities. Packed rows identify participating tokens; no per-token or per-head latency is measured.",
    }


def _required_native_events(timed: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    """Keep every CUDA event and both native-correlation candidate sets, including ambiguity."""
    keys: set[tuple[str, str]] = set()
    for _index, event in timed:
        if _gpu_kind(event):
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            keys.update((key, str(args[key])) for key in ("correlation", "External id") if args.get(key) is not None)
    required = []
    for index, event in timed:
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        if _gpu_kind(event) or str(event.get("name", "")).startswith(("explorer.", "cudaLaunch", "cuLaunch")) or any(
            (key, str(args[key])) in keys for key in ("correlation", "External id") if args.get(key) is not None):
            required.append((index, event))
    return required


def _explicit_scope(name: str) -> dict[str, Any] | None:
    if match := _STEP.fullmatch(name):
        return {"step_id": match[1], "layer_index": None, "component": None}
    if match := _LAYER.fullmatch(name):
        return {"layer_index": int(match[1]), "component": None}
    if match := _COMPONENT.fullmatch(name):
        return {"layer_index": int(match[1]), "component": match[2]}
    if match := _CACHE_WRITE.fullmatch(name):
        return {"layer_index": int(match[1]), "component": "attention", "operation": "kv_cache_write", "operation_id": match[2]}
    return None


def _empty_scope() -> dict[str, Any]:
    return {"step_id": None, "layer_index": None, "component": None, "correlation": "unmapped"}


def _scope_identity(scope: dict[str, Any]) -> tuple[Any, ...]:
    return scope["step_id"], scope["layer_index"], scope["component"], scope.get("operation"), scope.get("operation_id")


def _gpu_kind(event: dict[str, Any]) -> str | None:
    category = str(event.get("cat", "")).lower()
    if category in {"kernel", "gpu_kernel"}:
        return "cuda_kernel"
    if category == "gpu_memcpy":
        return "cuda_memcpy"
    if category == "gpu_memset":
        return "cuda_memset"
    return None


def _cpu_kind(event: dict[str, Any]) -> str:
    return "cpu_scope" if str(event.get("cat", "")).lower() in {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver", "python_function"} else "other"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _identity(value: Any) -> str | int | None:
    return value if isinstance(value, (str, int)) and not isinstance(value, bool) else None
