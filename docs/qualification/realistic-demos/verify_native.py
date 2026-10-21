#!/usr/bin/env python3
"""Independently compare every saved CUDA activity with original native input.

Only the production wire decoder is reused. Native fields and correlation
candidates are read directly; layer-scope inheritance is not reconstructed.
This helper does not execute inference or change recording files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runner.profile_codec import expand_profile_wire

MAX_RAW_BYTES = 4 * 1024**3
MAX_ATTACHMENT_BYTES = 512 * 1024**2
CUDA_CATEGORIES = {"kernel": "cuda_kernel", "gpu_kernel": "cuda_kernel", "gpu_memcpy": "cuda_memcpy", "gpu_memset": "cuda_memset"}
CPU_CATEGORIES = {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver", "python_function"}
SOURCE_ID = re.compile(r"^activity-(0|[1-9][0-9]*)$")
FIELDS = ("activity_id", "kind", "name", "device", "stream", "process_id", "thread_id", "start", "duration")


def signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def valid_timed(event: Any) -> bool:
    return isinstance(event, dict) and event.get("ph") == "X" and all(
        type(event.get(key)) in {int, float} and math.isfinite(event[key]) for key in ("ts", "dur")) and event["dur"] >= 0


def native_fields(index: int, event: dict[str, Any]) -> dict[str, Any]:
    category = str(event.get("cat", "")).lower()
    kind = CUDA_CATEGORIES.get(category)
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    def identity(value: Any) -> str | int | None:
        return value if type(value) in {str, int} else None
    return {"activity_id": f"activity-{index}", "kind": kind or ("cpu_scope" if category in CPU_CATEGORIES else "other"),
            "name": str(event.get("name", "unnamed")), "device": identity(args.get("device")) if kind else None,
            "stream": identity(args.get("stream")) if kind else None, "process_id": identity(event.get("pid")),
            "thread_id": identity(event.get("tid")), "start": event["ts"], "duration": event["dur"]}


def verify_activity_evidence(native: list[Any], activities: list[dict[str, Any]], *, mismatch_limit: int = 20) -> dict[str, Any]:
    checks = {name: True for name in ("native_cuda_present", "native_cuda_timestamps_valid", "normalized_activity_ids_unique",
        "all_native_cuda_ids_and_order_preserved", "all_native_cuda_fields_exact", "all_related_cpu_fields_exact",
        "native_runtime_correlation_candidates_exact", "runtime_flow_scope_matches_related_cpu_scope")}
    mismatches = []
    mismatch_count = 0
    def fail(check: str, detail: dict[str, Any]) -> None:
        nonlocal mismatch_count
        checks[check] = False
        mismatch_count += 1
        if len(mismatches) < mismatch_limit:
            mismatches.append({"check": check, **detail})

    expected = []
    requested_correlations = set()
    for index, event in enumerate(native):
        if not isinstance(event, dict) or str(event.get("cat", "")).lower() not in CUDA_CATEGORIES or event.get("ph") != "X":
            continue
        if not valid_timed(event):
            fail("native_cuda_timestamps_valid", {"native_index": index})
            continue
        expected.append(native_fields(index, event))
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        requested_correlations.update((key, str(args[key])) for key in ("correlation", "External id") if args.get(key) is not None)
    if not expected:
        fail("native_cuda_present", {"reason": "No valid native CUDA timed records are present."})
    candidates = defaultdict(set)
    for index, event in enumerate(native):
        if not valid_timed(event) or str(event.get("cat", "")).lower() in CUDA_CATEGORIES:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        for key in ("correlation", "External id"):
            if args.get(key) is not None and (key, str(args[key])) in requested_correlations:
                candidates[(key, str(args[key]))].add(f"activity-{index}")
    by_id = {}
    gpu = []
    for activity in activities:
        id = activity.get("activity_id")
        if not isinstance(id, str) or id in by_id:
            fail("normalized_activity_ids_unique", {"activity_id": id})
        else:
            by_id[id] = activity
        if str(activity.get("kind", "")).startswith("cuda_"):
            gpu.append(activity)
    native_ids = [row["activity_id"] for row in expected]
    gpu_ids = [row.get("activity_id") for row in gpu]
    if gpu_ids != native_ids:
        fail("all_native_cuda_ids_and_order_preserved", {"native_count": len(native_ids), "normalized_count": len(gpu_ids),
            "missing_ids": sorted(set(native_ids) - set(gpu_ids))[:mismatch_limit],
            "additional_ids": sorted(set(gpu_ids) - set(native_ids))[:mismatch_limit], "same_set": set(native_ids) == set(gpu_ids)})
    mapped = unmapped = link_count = 0
    for source in expected:
        id = source["activity_id"]
        activity = by_id.get(id)
        if activity is None or signature({field: activity.get(field) for field in FIELDS}) != signature(source):
            fail("all_native_cuda_fields_exact", {"activity_id": id, "native": source,
                "normalized": {field: activity.get(field) for field in FIELDS} if activity else None})
        if activity is None:
            continue
        scope = activity.get("scope") if isinstance(activity.get("scope"), dict) else {}
        related = activity.get("related_activity_ids")
        if scope.get("correlation") == "unmapped":
            unmapped += 1
            if related != []:
                fail("native_runtime_correlation_candidates_exact", {"activity_id": id, "reason": "Unmapped activity has runtime links."})
            if any(scope.get(key) is not None for key in ("step_id", "layer_index", "component", "operation", "operation_id")):
                fail("runtime_flow_scope_matches_related_cpu_scope", {"activity_id": id, "reason": "Unmapped CUDA activity has an attributed execution scope."})
            continue
        mapped += 1
        key = {"native_correlation_id": "correlation", "native_external_id": "External id"}.get(scope.get("correlation_source"))
        source_match = SOURCE_ID.fullmatch(id)
        args = native[int(source_match[1])].get("args", {})
        if scope.get("correlation") != "runtime_flow" or key is None or not isinstance(args, dict) or args.get(key) is None or not isinstance(related, list):
            fail("native_runtime_correlation_candidates_exact", {"activity_id": id, "reason": "Invalid runtime-flow source or references."})
            continue
        expected_links = candidates[(key, str(args[key]))]
        if any(not isinstance(value, str) for value in related) or len(set(related)) != len(related) or set(related) != expected_links or not expected_links:
            fail("native_runtime_correlation_candidates_exact", {"activity_id": id,
                "expected_cpu_ids": sorted(expected_links)[:mismatch_limit], "related_cpu_ids": related[:mismatch_limit]})
        for cpu_id in related:
            link_count += 1
            match = SOURCE_ID.fullmatch(cpu_id) if isinstance(cpu_id, str) else None
            index = int(match[1]) if match else len(native)
            event = native[index] if index < len(native) else None
            cpu = by_id.get(cpu_id) if isinstance(cpu_id, str) else None
            if not valid_timed(event) or str(event.get("cat", "")).lower() in CUDA_CATEGORIES or cpu is None:
                fail("all_related_cpu_fields_exact", {"activity_id": id, "related_id": cpu_id, "reason": "Related native CPU record is unavailable."})
                continue
            if signature({field: cpu.get(field) for field in FIELDS}) != signature(native_fields(index, event)):
                fail("all_related_cpu_fields_exact", {"activity_id": id, "related_id": cpu_id, "reason": "Related native CPU fields differ."})
            cpu_scope = cpu.get("scope") if isinstance(cpu.get("scope"), dict) else {}
            if signature({field: value for field, value in scope.items() if field not in {"correlation", "correlation_source"}}) != signature(
                    {field: value for field, value in cpu_scope.items() if field not in {"correlation", "correlation_source"}}):
                fail("runtime_flow_scope_matches_related_cpu_scope", {"activity_id": id, "related_id": cpu_id})
    return {"ok": all(checks.values()), "checks": checks, "native_cuda_count": len(expected), "normalized_cuda_count": len(gpu),
            "runtime_mapped_cuda_count": mapped, "unmapped_cuda_count": unmapped, "verified_cpu_link_count": link_count,
            "mismatch_count": mismatch_count, "first_mismatches": mismatches,
            "scope": "Every raw valid CUDA timed record, original source ID/order and exact fields, and every declared CPU runtime correlation candidate. Unmapped CUDA remains valid; native layer inheritance and individual token/head timing are not independently inferred."}


def verified_path(directory: Path, metadata: dict[str, Any], *, limit: int) -> Path:
    path = (directory / metadata["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size != metadata["size_bytes"] or path.stat().st_size > limit:
        raise ValueError("Artifact path/size is invalid or exceeds its read bound.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError("Artifact SHA256 differs from its manifest.")
    return path


def verify_recording(directory: Path) -> dict[str, Any]:
    started = time.monotonic()
    verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_text())
    profile = manifest.get("profiling") or {}
    if len(profile.get("raw_artifacts", [])) != 1 or profile.get("attachment_status") != "available" or not profile.get("normalized_artifact"):
        raise ValueError("One native trace and an available normalized attachment are required.")
    raw = verified_path(directory, profile["raw_artifacts"][0], limit=MAX_RAW_BYTES)
    attachment = verified_path(directory, profile["normalized_artifact"], limit=MAX_ATTACHMENT_BYTES)
    opener = gzip.open if raw.suffix == ".gz" else open
    with opener(raw, "rb") as handle:
        payload = handle.read(MAX_RAW_BYTES + 1)
    if len(payload) > MAX_RAW_BYTES:
        raise ValueError("Decompressed native trace exceeds the 4 GiB read bound.")
    raw_size = len(payload)
    native = json.loads(payload)
    del payload
    wire = json.loads(attachment.read_bytes())
    decoded = expand_profile_wire(wire)
    result = verify_activity_evidence(native["traceEvents"], decoded["activities"])
    result["checks"]["normalized_run_identity_matches_manifest"] = decoded.get("run_id") == manifest["run_id"]
    result["ok"] = all(result["checks"].values())
    result.update(run_id=manifest["run_id"], execution_kind=manifest.get("execution_kind"),
                  artifact_hashes_verified=True, raw_compressed_bytes=raw.stat().st_size, raw_decompressed_bytes=raw_size,
                  normalized_bytes=attachment.stat().st_size, native_event_count=len(native["traceEvents"]),
                  normalized_activity_count=len(decoded["activities"]), captured_steps=profile.get("captured_steps"),
                  verifier_sha256=verifier_sha, elapsed_seconds=time.monotonic() - started)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_recording(args.recording.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"ok": False, "error": str(error), "scope": "Read-only independent native verification; no successful result is claimed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
