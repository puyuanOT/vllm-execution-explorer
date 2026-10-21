#!/usr/bin/env python3
"""Read native evidence offline; measure envelopes without claiming lost hook data."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

if Path(__file__).with_name("profile_trace.py").exists():
    from profile_trace import _gpu_kind, _number, _required_native_events, normalize_trace
    from profile_codec import encode_compact_profile, expand_profile_wire
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from runner.profile_trace import _gpu_kind, _number, _required_native_events, normalize_trace
    from runner.profile_codec import encode_compact_profile, expand_profile_wire


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.recording / "manifest.json").read_text())
    raw = manifest["profiling"]["raw_artifacts"]
    if len(raw) != 1:
        raise ValueError("Exactly one raw native trace is required.")
    path = args.recording / raw[0]["path"]
    compressed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != raw[0]["size_bytes"] or compressed_sha != raw[0]["sha256"]:
        raise ValueError("Raw native trace does not match its manifest.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        payload = handle.read()
    decompressed_size = len(payload)
    decompressed_sha = hashlib.sha256(payload).hexdigest()
    trace = json.loads(payload)
    del payload
    timed = [(index, event) for index, event in enumerate(trace["traceEvents"])
             if isinstance(event, dict) and event.get("ph") == "X" and _number(event.get("ts"))
             and _number(event.get("dur")) and event["dur"] >= 0]
    required = _required_native_events(timed)
    counts = Counter(_gpu_kind(event) or "cpu" for _, event in timed)
    scopes = {event["name"].split("/", 1)[1] for _, event in timed if str(event.get("name", "")).startswith("explorer.step/")}
    result = {"run_id": manifest["run_id"], "recording_source_identity": (manifest.get("observability") or {}).get("code_identity"),
              "compressed_bytes": path.stat().st_size, "compressed_sha256": compressed_sha,
              "decompressed_bytes": decompressed_size, "decompressed_sha256": decompressed_sha,
              "native_event_count": len(trace["traceEvents"]), "valid_timed_activity_count": len(timed),
              "required_native_activity_count": len(required), "native_kind_counts": dict(counts),
              "native_step_scope_count": len(scopes), "manifest_captured_steps": manifest["profiling"].get("captured_steps"),
              "raw_bound_192_mib_exceeded": decompressed_size > 192 * 1024 * 1024,
              "required_activity_bound_250000_exceeded": len(required) > 250_000}
    del timed, required
    # The original hook rows/writes were never exported after the native-byte
    # failure. Only actual native activities can be measured from this source.
    normalized = normalize_trace(trace, run_id=manifest["run_id"], steps=[],
                                 max_activities=max(result["required_native_activity_count"], 1))
    del trace
    activities = normalized["activities"]
    correlated = {id for row in activities if row["kind"].startswith("cuda_") for id in row["related_activity_ids"]}
    retained = [row for row in activities if row["kind"].startswith("cuda_") or row["activity_id"] in correlated
                or row["name"].startswith(("explorer.", "cudaLaunch", "cuLaunch"))]
    result["native_required_normalization"] = normalized["normalization"]
    result["browser_byte_policy_activity_count"] = len(retained)
    normalized["activities"] = retained
    encoded = encode_compact_profile(normalized)
    wire = (json.dumps(encoded, separators=(",", ":")) + "\n").encode()
    restored = expand_profile_wire(json.loads(wire))
    result.update(compact_native_activity_bytes=len(wire), compact_round_trip_exact=restored["activities"] == retained,
                  compact_encoding=encoded["normalization"]["attachment_encoding"],
                  compact_cuda_count=sum(row["kind"].startswith("cuda_") for row in retained),
                  cuda_correlated_native_step_count=len({row["scope"]["step_id"] for row in retained
                    if row["kind"].startswith("cuda_") and row["scope"].get("correlation") == "runtime_flow"}),
                  elapsed_seconds=time.monotonic() - started,
                  scope="Read-only native activity sizing. Lost in-memory hook packed rows/cache-write participants and numerical snapshots are not reconstructed, and this is not a qualified capture attachment.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
