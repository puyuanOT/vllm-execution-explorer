"""Controller state with the same atomic transactions across a worker process."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any

from runner.configuration import copy_configuration

ACTIVE_STATUSES = {"preparing", "capturing", "finalizing"}


class RunState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.run_id: str | None = None
        self.status = "idle"
        self.intake_open = False
        self.cancelled = False
        self.text: dict[str, str] = {}
        self.message = ""
        self.extra: list[Any] = []
        self.plan: dict[str, Any] | None = None
        self.origin: dict[str, Any] | None = None
        self.arrivals: list[dict[str, Any]] = []
        self.finalized = False
        self.phase = "idle"
        self.phase_history: list[dict[str, Any]] = []
        self.configuration: dict[str, Any] | None = None
        self.execution_process_alive = False
        self.execution_process: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            status, phase = self.status, self.phase
            finalized = self.finalized and not self.execution_process_alive
            if self.execution_process_alive and status not in ACTIVE_STATUSES:
                status, phase = "finalizing", "reaping_execution_process"
            return {
                "run_id": self.run_id, "status": status,
                "active": status in ACTIVE_STATUSES or self.execution_process_alive,
                "intake_open": self.intake_open, "text": dict(self.text),
                "message": self.message, "workload_plan": self.plan,
                "controller_origin": self.origin, "finalized": finalized,
                "run_configuration": copy_configuration(self.configuration) if self.configuration else None,
                "execution_process": self.execution_process,
                "capture": {"status": status, "phase": phase,
                    "history": [dict(item) for item in self.phase_history], "finalized": finalized},
            }


class FileTransactionLock:
    """A dead worker cannot orphan the POSIX lock; threads remain reentrant."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._descriptor: int | None = None
        self._owner: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["path"])

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        import fcntl

        deadline = time.monotonic() + timeout if timeout >= 0 else None
        if not self._thread_lock.acquire(blocking, timeout):
            return False
        if self._depth:
            self._depth += 1
            return True
        descriptor = None
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._descriptor, self._depth, self._owner = descriptor, 1, threading.get_ident()
                    return True
                except BlockingIOError:
                    if not blocking or (deadline is not None and time.monotonic() >= deadline):
                        os.close(descriptor)
                        self._thread_lock.release()
                        return False
                    time.sleep(0.005)
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            self._thread_lock.release()
            raise

    def release(self) -> None:
        import fcntl

        if self._depth < 1 or self._owner != threading.get_ident():
            raise RuntimeError("Cannot release an unlocked transaction.")
        self._depth -= 1
        if self._depth == 0:
            descriptor, self._descriptor, self._owner = self._descriptor, None, None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        self._thread_lock.release()

    def __enter__(self) -> FileTransactionLock:
        self.acquire()
        return self

    def __exit__(self, *_exception: Any) -> None:
        self.release()


class SharedRunState(RunState):
    """Only small control/state data crosses IPC; recording events stay in child files."""

    def __init__(self, manager: Any, state: RunState, lock_path: Path) -> None:
        object.__setattr__(self, "lock", FileTransactionLock(lock_path))
        mutable = {
            "text": manager.dict(dict(state.text)), "extra": manager.list(list(state.extra)),
            "arrivals": manager.list(list(state.arrivals)), "phase_history": manager.list(list(state.phase_history)),
        }
        object.__setattr__(self, "_mutable", mutable)
        object.__setattr__(self, "_values", manager.dict({
            name: value for name, value in vars(state).items() if name != "lock" and name not in mutable
        }))

    def __getattr__(self, name: str) -> Any:
        mutable = self.__dict__.get("_mutable", {})
        if name in mutable:
            return mutable[name]
        values = self.__dict__.get("_values")
        if values is not None and name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        mutable = self.__dict__.get("_mutable", {})
        if name in mutable:
            target = mutable[name]
            if name == "text":
                target.clear()
                target.update(value)
            else:
                target[:] = list(value)
        elif name.startswith("_") or name == "lock":
            object.__setattr__(self, name, value)
        else:
            self._values[name] = value
