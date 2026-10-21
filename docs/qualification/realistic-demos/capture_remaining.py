#!/usr/bin/env python3
"""Capture selected remaining variants serially on the authorized execution host.

Run only on the authorized execution host. Every case uses capture.py's actual
API/export/assessment path. Attempts stay intact; strict passes are copied to
qualified directories. Independent native audits and UI checks remain separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
VARIANTS = ("fp8-kv", "fp8-weights", "lora-all", "lora-mixed", "qwen38-27b")
EXPECTED_SOURCE = "db403bda651d8fd6fe8308a5889f0d465b18af2e63a1974ba5ee1aae4017107b"
sys.path.insert(0, str(ROOT))
from runner.observability import source_identity


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def current() -> dict:
    with urlopen("http://127.0.0.1:8765/api/runs/current", timeout=15) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", choices=VARIANTS, required=True)
    args = parser.parse_args()
    progress_path = EVIDENCE / "capture-progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {"cases": []}
    if progress_path.exists():
        archive = EVIDENCE / "capture-progress-initial.json"
        if not archive.exists():
            shutil.copy2(progress_path, archive)
    progress = {"status": "running", "cases": previous["cases"], "source_identity": source_identity(ROOT)}
    save(EVIDENCE / "capture-progress.json", progress)
    try:
        for variant in args.variant:
            identity = source_identity(ROOT)
            if identity["source_sha256"] != EXPECTED_SOURCE or identity["source_file_count"] != 41:
                raise RuntimeError("Frozen capture source changed; refusing another GPU case.")
            qualified = EVIDENCE / f"{variant}-qualified"
            if qualified.exists():
                raise RuntimeError(f"Already-qualified case is retained: {qualified}")
            attempt = 1
            while (EVIDENCE / f"{variant}-attempt-{attempt}").exists():
                attempt += 1
            output = EVIDENCE / f"{variant}-attempt-{attempt}"
            state = current()
            if state.get("active") or not state.get("finalized") or state.get("execution_process", {}).get("status") != "exited":
                raise RuntimeError("Previous GPU owner has not finalized/exited.")
            command = [sys.executable, "-u", str(ROOT / "docs/qualification/demo-expansion/capture.py"),
                       "--execute", "--payload", str(EVIDENCE / f"plans/{variant}-request.json"),
                       "--output", str(output), "--recordings", str(ROOT / "recordings/runs"),
                       "--gpu-memory", "--timeout", "1800"]
            print(f"Starting {variant}, attempt {attempt}", flush=True)
            with (EVIDENCE / f"{variant}-attempt-{attempt}.log").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{variant} helper failed with exit code {completed.returncode}; retained at {output}")
            assessment = json.loads((output / "six-request/assessment.json").read_text())
            workload = json.loads((output / "expansion-assessment.json").read_text())
            if not assessment["ok"] or not workload["ok"]:
                raise RuntimeError(f"{variant} did not pass strict assessment/workload checks.")
            shutil.copytree(output, qualified)
            result = {"variant": variant, "run_id": assessment["run_id"], "strict_assessment_ok": True,
                      "workload_ok": True, "qualified_directory": str(qualified),
                      "accepted_output_tokens": {row["request_id"]: row["accepted_tokens"] for row in assessment["outputs"]},
                      "cleanup_basis": assessment["cleanup"]["basis"], "profile": assessment["profile"],
                      "independent_native_audit": "pending separate verifier"}
            progress["cases"].append(result)
            save(EVIDENCE / "capture-progress.json", progress)
            print(json.dumps(result), flush=True)
        progress["status"] = "completed"
        save(EVIDENCE / "capture-progress.json", progress)
        return 0
    except Exception as error:
        progress.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        save(EVIDENCE / "capture-progress.json", progress)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
