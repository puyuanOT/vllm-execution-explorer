"""One spawned process owns an experiment's CUDA and compilation lifetime."""

from __future__ import annotations

import json
import multiprocessing
from multiprocessing.connection import wait as wait_for_process_exit
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from runner.state import ACTIVE_STATUSES, RunState, SharedRunState
from runner.workload import PROVISIONAL_LIMITS
from vllm_030.constants import CLOCK_DOMAIN, PRODUCER_ID, SCHEMA_VERSION


def _execute_child(state: SharedRunState, arguments: dict[str, Any]) -> None:
    from runner.serve import _execute

    with state.lock:
        state.execution_process = {**(state.execution_process or {}), "pid": os.getpid(), "status": "running"}
    _execute(state=state, **arguments)


def _valid_jsonl_prefix(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        path.write_text("", encoding="utf-8")
        return []
    original = path.read_bytes()
    lines, rows = original.splitlines(keepends=True), []
    prefix = original
    for index, line in enumerate(lines):
        try:
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise RuntimeError(f"Corrupt interior record in {path.name}; refusing to rewrite evidence.")
            prefix = b"".join(lines[:index])
            break
        rows.append(value)
    if prefix and not prefix.endswith(b"\n"):
        prefix += b"\n"
    if prefix != original:
        # Preserve every original byte, including an interrupted UTF-8 codepoint.
        backup = path.with_name(path.name + ".interrupted")
        with backup.open("xb") as stream:
            stream.write(original)
        path.write_bytes(prefix)
    return rows


def _save_manifest(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name("manifest.finalizing.json")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finish_failed_process(state: SharedRunState, arguments: dict[str, Any], error: str) -> None:
    """Retain the authoritative prefix; unobserved work after death stays unknown."""
    from runner.serve import _manifest_for

    directory = arguments["recordings_dir"] / arguments["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    events = _valid_jsonl_prefix(directory / "events.jsonl")
    _valid_jsonl_prefix(directory / "content.jsonl")
    gap = next((event for event in events if event.get("event_type") == "trace_gap"), None)
    if gap is None:
        gap = {"schema_version": SCHEMA_VERSION, "run_id": arguments["run_id"], "producer_id": PRODUCER_ID,
               "sequence": int(events[-1]["sequence"]) + 1 if events else 1,
               "monotonic_ns": time.monotonic_ns(), "event_type": "trace_gap", "reason": "execution_process_exited"}
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(gap, separators=(",", ":")) + "\n")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _manifest_for(state, arguments["run_id"], arguments["body"], None,
                                 list(state.arrivals), {}, state.origin or {}, "failed")
    manifest.update(schema_version=SCHEMA_VERSION, run_id=arguments["run_id"], clock_domain=CLOCK_DOMAIN,
                    completion_status="failed", execution_process=state.execution_process)
    manifest["interrupted_evidence_files"] = [path.name for path in directory.glob("*.interrupted")]
    manifest["event_completeness"] = {"status": "incomplete", "gap": {"sequence": gap["sequence"], "reason": gap["reason"]},
                                    "dropped_record_count": None, "dropped_event_count": None, "dropped_content_count": None,
                                    "unwritten_record_count": "unknown_after_process_exit"}
    manifest.setdefault("capture", {}).update(status="failed", message=error)
    profiling = manifest.get("profiling")
    if isinstance(profiling, dict) and profiling.get("enabled"):
        profiling.update(status="partial", completeness="unknown", qualification="unverified",
                         reason="The execution process exited unexpectedly; retained profiler files may be incomplete.")
    _save_manifest(manifest_path, manifest)


class ExperimentIsolation:
    def __init__(self, *, max_lifetime_seconds: float = 900 + int(PROVISIONAL_LIMITS["max_run_seconds"]) + 120,
                 shutdown_grace_seconds: float = 30, termination_grace_seconds: float = 10) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._manager: Any = None
        self._state: SharedRunState | None = None
        self._process: Any = None
        self._arguments: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._lock_directory: Any = None
        self._closing = False
        self._max_lifetime_seconds = max_lifetime_seconds
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._termination_grace_seconds = termination_grace_seconds

    def share(self, state: RunState) -> SharedRunState:
        with self._lock:
            if self._state is None:
                if self._closing:
                    raise RuntimeError("The controller is shutting down.")
                self._manager = self._context.Manager()
                self._lock_directory = tempfile.TemporaryDirectory(prefix="vllm-experiment-state-")
                self._state = SharedRunState(self._manager, state, Path(self._lock_directory.name) / "transaction.lock")
            return self._state

    def start(self, state: SharedRunState, arguments: dict[str, Any], *, target: Any = _execute_child) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("The controller is shutting down.")
            with state.lock:
                if state.execution_process_alive:
                    raise RuntimeError("The previous execution process is still exiting.")
                state.execution_process_alive = True
                state.execution_process = {"kind": "spawn_per_run", "status": "starting", "pid": None,
                    "started_monotonic_ns": time.monotonic_ns(), "clock_domain": "controller:time.monotonic_ns"}
            self._arguments = arguments
            self._process = self._context.Process(target=target, args=(state, arguments),
                                                  name=f"explorer-{arguments['run_id']}")
            self._thread = threading.Thread(target=self._supervise, args=(state, arguments, self._process),
                                            name="vllm-experiment-supervisor", daemon=True)
            self._thread.start()

    def _supervise(self, state: SharedRunState, arguments: dict[str, Any], process: Any) -> None:
        failure = ""
        try:
            with self._lock:
                process.start()
                sentinel = process.sentinel
            # Waiting on the sentinel never reaps the child. Shutdown may
            # terminate/reap it while this wait is pending; Process operations
            # themselves remain serialized under the lifecycle mutex.
            if not wait_for_process_exit([sentinel], timeout=self._max_lifetime_seconds):
                self._stop_owned_process(process)
                failure = f"Execution process exceeded its {self._max_lifetime_seconds:g}s bounded lifetime."
            with self._lock:
                process.join()
                exit_code = process.exitcode
            manifest = arguments["recordings_dir"] / arguments["run_id"] / "manifest.json"
            if not failure and (exit_code != 0 or not manifest.is_file() or state.status in ACTIVE_STATUSES):
                failure = f"Execution process exited before a finalized capture (exit code {exit_code})."
        except Exception as error:
            failure = f"Execution process supervision failed ({type(error).__name__})."
        finally:
            with self._lock:
                alive = process.pid is not None and process.is_alive()
            if alive:
                try:
                    self._stop_owned_process(process)
                except Exception:
                    with self._lock:
                        with state.lock:
                            state.status, state.phase = "failed", "execution_process_exit_unconfirmed"
                            state.message = "The owned execution process has not exited; new captures remain blocked."
                            state.intake_open, state.finalized = False, False
                            state.execution_process = {**(state.execution_process or {}), "pid": process.pid,
                                                       "status": "exit_unconfirmed", "exit_code": process.exitcode}
                    # Keep the handle and readiness gate for shutdown's retry.
                    return
            with self._lock:
                if process.pid is not None:
                    process.join()
            self._finalize_exit(state, arguments, process, failure)

    def _finalize_exit(self, state: SharedRunState, arguments: dict[str, Any], process: Any, failure: str) -> None:
        try:
            with self._lock:
                with state.lock:
                    state.execution_process = {**(state.execution_process or {}), "pid": process.pid,
                        "status": "failed" if failure else "exited", "exit_code": process.exitcode,
                        "exited_monotonic_ns": time.monotonic_ns()}
                    if failure:
                        state.status, state.phase, state.message = "failed", "failed", failure
                        state.intake_open = False
            if failure:
                _finish_failed_process(state, arguments, failure)
            else:
                path = arguments["recordings_dir"] / arguments["run_id"] / "manifest.json"
                document = json.loads(path.read_text(encoding="utf-8"))
                document["execution_process"] = state.execution_process
                _save_manifest(path, document)
            with state.lock:
                state.finalized = True
        except Exception as error:
            with state.lock:
                state.status, state.phase = "failed", "failed"
                state.message = f"Execution process evidence finalization failed ({type(error).__name__})."
                state.finalized = False
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
                    with state.lock:
                        state.execution_process_alive = False

    def _stop_owned_process(self, process: Any) -> None:
        with self._lock:
            if not process.is_alive():
                process.join()
                return
            process.terminate()
            process.join(timeout=self._termination_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join(timeout=self._termination_grace_seconds)
            if process.is_alive():
                raise RuntimeError("The owned execution process did not exit after bounded termination.")

    def close(self) -> None:
        with self._lock:
            self._closing = True
        state = self._state
        if state is not None and state.execution_process_alive:
            acquired = state.lock.acquire(timeout=min(self._shutdown_grace_seconds, 1))
            if acquired:
                try:
                    state.cancelled = True
                    state.intake_open = False
                finally:
                    state.lock.release()
            if self._thread is not None:
                self._thread.join(timeout=self._shutdown_grace_seconds)
            with self._lock:
                process = self._process
                alive = process is not None and process.is_alive()
            if alive:
                self._stop_owned_process(process)
                if self._thread is not None:
                    self._thread.join(timeout=self._termination_grace_seconds)
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Execution supervision has not exited; state manager remains available.")
            with self._lock:
                needs_finalization = state.execution_process_alive and process is not None and not process.is_alive()
                if needs_finalization:
                    process.join()
            if needs_finalization:
                self._finalize_exit(state, self._arguments or {}, process, "Execution process was reaped during controller shutdown.")
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._state = None
        if self._lock_directory is not None:
            self._lock_directory.cleanup()
            self._lock_directory = None
