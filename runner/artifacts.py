"""Bounded operator-local artifact inspection and SSH transfer state."""

from __future__ import annotations

import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from runner.targets import SshTarget, TargetError, ssh_argv, ssh_transport

MAX_ARTIFACT_BYTES = 4 * 1024**4
MAX_ARTIFACT_FILES = 200_000
HTTP_UPLOAD_LIMIT_BYTES = 64 * 1024**3
HTTP_UPLOAD_MAX_FILES = 10_000
TRANSFER_STATUSES = ("prepared", "transferring", "completed", "failed")


@dataclass(frozen=True)
class LocalArtifact:
    source: Path
    name: str
    kind: str
    total_bytes: int
    file_count: int


class ArtifactTransferManager:
    def __init__(self, *, run: Callable[..., Any] | None = None) -> None:
        self._run = run or subprocess.run
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, Path] = {}

    def prepare(self, source_path: str, destination: str) -> dict[str, Any]:
        artifact = inspect_local_artifact(source_path)
        clean_destination = validate_remote_destination(destination)
        transfer_id = f"transfer-{uuid.uuid4().hex[:12]}"
        record = {
            "transfer_id": transfer_id,
            "status": "prepared",
            "source_name": artifact.name,
            "artifact_kind": artifact.kind,
            "destination": clean_destination,
            "total_bytes": artifact.total_bytes,
            "bytes_transferred": 0,
            "file_count": artifact.file_count,
            "attempt": 0,
            "message": "Artifact inspected and ready to transfer.",
            "progress_basis": "completion_confirmed",
        }
        with self._lock:
            self._records[transfer_id] = record
            self._sources[transfer_id] = artifact.source
        return dict(record)

    def snapshot(self, transfer_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(transfer_id)
            if record is None:
                raise KeyError(transfer_id)
            return dict(record)

    def start(self, transfer_id: str, target: SshTarget) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(transfer_id)
            source = self._sources.get(transfer_id)
            if record is None or source is None:
                raise KeyError(transfer_id)
            if record["status"] == "transferring":
                return dict(record)
            record["status"] = "transferring"
            record["attempt"] += 1
            record["bytes_transferred"] = 0
            record["message"] = "Transfer in progress."
            snapshot = dict(record)
        threading.Thread(
            target=self._copy,
            args=(transfer_id, source, target),
            name=f"artifact-{transfer_id}",
            daemon=True,
        ).start()
        return snapshot

    def _copy(self, transfer_id: str, source: Path, target: SshTarget) -> None:
        try:
            with self._lock:
                destination = str(self._records[transfer_id]["destination"])
            if target.identity_file is None:
                raise TargetError("Reconnect the SSH host before transferring model artifacts.")
            parent = str(PurePosixPath(destination).parent)
            _check(
                self._run(
                    ssh_argv(target, f"mkdir -p {shlex.quote(parent)}"),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                ),
                target.identity_file,
            )
            remote_shell = " ".join(shlex.quote(part) for part in ssh_transport(target))
            source_arg = str(source.resolve()) + ("/" if source.is_dir() else "")
            remote_arg = f"{target.user}@{target.host}:{shlex.quote(destination)}"
            _check(
                self._run(
                    [
                        "rsync",
                        "-az",
                        "--partial",
                        "--copy-links",
                        "-e",
                        remote_shell,
                        source_arg,
                        remote_arg,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=24 * 60 * 60,
                ),
                target.identity_file,
            )
        except Exception as exc:  # noqa: BLE001 - status must retain a retryable failure
            message = _redact(str(exc), target.identity_file)
            message = message.replace(str(source.resolve()), "[local artifact]")
            with self._lock:
                record = self._records[transfer_id]
                record["status"] = "failed"
                record["message"] = message
            return
        with self._lock:
            record = self._records[transfer_id]
            record["status"] = "completed"
            record["bytes_transferred"] = record["total_bytes"]
            record["message"] = "Artifact transfer completed."


def inspect_local_artifact(source_path: str) -> LocalArtifact:
    if not isinstance(source_path, str) or not source_path.strip():
        raise TargetError("Choose a local model file or directory.")
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise TargetError("The selected local model artifact does not exist.")
    if source.is_file():
        size = source.stat().st_size
        files = 1
        kind = "file"
    elif source.is_dir():
        size = 0
        files = 0
        for item in source.rglob("*"):
            if item.is_symlink() and item.is_dir():
                raise TargetError("The selected model artifact contains a symlinked directory.")
            if not item.is_file():
                continue
            files += 1
            if files > MAX_ARTIFACT_FILES:
                raise TargetError(f"A model artifact may contain at most {MAX_ARTIFACT_FILES} files.")
            size += item.stat().st_size
            if size > MAX_ARTIFACT_BYTES:
                raise TargetError("The selected model artifact exceeds the transfer safety limit.")
        kind = "directory"
    else:
        raise TargetError("Choose a regular model file or directory.")
    if size > MAX_ARTIFACT_BYTES:
        raise TargetError("The selected model artifact exceeds the transfer safety limit.")
    return LocalArtifact(source, source.name, kind, size, files)


def validate_remote_destination(destination: str) -> str:
    if not isinstance(destination, str) or not destination.strip():
        raise TargetError("Choose an absolute destination on the execution host.")
    if any(ord(char) < 32 for char in destination):
        raise TargetError("The artifact destination contains invalid characters.")
    path = PurePosixPath(destination.strip())
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise TargetError("Choose a specific absolute artifact destination on the execution host.")
    return str(path)


def _check(completed: Any, identity: str) -> None:
    if int(getattr(completed, "returncode", 1)) == 0:
        return
    detail = f"{getattr(completed, 'stderr', '') or ''}{getattr(completed, 'stdout', '') or ''}".strip()
    raise TargetError(_redact(detail[-2000:] or "artifact transfer failed", identity))


def _redact(text: str, identity: str | None) -> str:
    return text.replace(identity, "[identity file]") if identity else text
