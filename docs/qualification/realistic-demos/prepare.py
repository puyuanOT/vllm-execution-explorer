#!/usr/bin/env python3
"""Write validated, pending-GPU plans without replacing qualified recordings."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_PLANS = ROOT / "docs/qualification/demo-expansion"
VARIANTS = ("bf16", "fp8-kv", "fp8-weights", "lora-all", "lora-mixed", "qwen38-27b")
spec = importlib.util.spec_from_file_location("realistic_capture", ARCHIVED_PLANS / "capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def plans(*, second_ms: int | None = None, third_ms: int | None = None) -> dict[str, dict]:
    if (second_ms is None) != (third_ms is None):
        raise ValueError("Provide both second_ms and third_ms to override variant arrival timings.")
    result = {}
    for variant in VARIANTS:
        payload = json.loads((ARCHIVED_PLANS / f"{variant}-request.json").read_text(encoding="utf-8"))
        routes = {row["request_id"]: row.get("adapter_id") or "base" for row in payload["requests"]}
        second = second_ms if second_ms is not None else (40000 if variant in {"lora-all", "lora-mixed"} else 20000)
        third = third_ms if third_ms is not None else second + 25
        payload.update(capture.corpus_workload(second_ms=second, third_ms=third, adapter_routes=routes))
        payload["profiling_configuration"]["max_steps"] = 1024
        capture.qualification.RunRequest(**payload)
        requests, plan = capture.qualification.parse_run_workload(payload)
        capture.qualification.parse_run_configuration(payload, requests, plan)
        result[variant] = payload
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("plans"))
    parser.add_argument("--second-ms", type=int, help="Override second arrival for every variant; also provide --third-ms.")
    parser.add_argument("--third-ms", type=int, help="Override third arrival for every variant; also provide --second-ms.")
    args = parser.parse_args()
    try:
        prepared = plans(second_ms=args.second_ms, third_ms=args.third_ms)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    for variant, payload in prepared.items():
        capture.qualification.save(args.output / f"{variant}-request.json", payload)
    print(f"Validated {len(prepared)} plans: {args.output}. GPU qualification remains pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
