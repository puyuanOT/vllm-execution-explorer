"""Explicit experiment/request trace handles, independent of engine threads."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any


def source_identity(source_root: Path | None = None) -> dict[str, Any]:
    """Fingerprint the distributed capture source, including uncommitted changes."""
    root = source_root or Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    files = sorted(path for folder in (root / "runner", root / "instrumentation")
        for path in folder.rglob("*") if path.is_file() and path.suffix in {".py", ".json"})
    for path in files:
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {"source_sha256": digest.hexdigest(), "source_file_count": len(files),
        "source_scope": "runner and instrumentation .py/.json files on disk at trace creation",
        "declared_revision": os.environ.get("VLLM_EXPLORER_CODE_REVISION") or None}


class ExperimentTrace:
    """Optional LangSmith tracing; recording never depends on a tracing service."""

    def __init__(self, directory: Path, model: str, configuration: dict[str, Any]) -> None:
        self._root: Any = None
        self._capture: Any = None
        self._requests: dict[str, Any] = {}
        self._status = "unconfigured"
        self._error: str | None = None
        self._configuration = configuration
        try:
            self._code_identity = source_identity()
        except OSError:
            self._code_identity = {"source_sha256": None, "status": "unavailable",
                "reason": "Capture source files could not be read for fingerprinting."}
        if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
            return
        try:
            from langsmith.run_trees import RunTree

            self._root = RunTree(
                name=f"vLLM experiment: {directory.name}",
                run_type="chain",
                project_name=os.environ.get("LANGSMITH_PROJECT", "vllm-execution-explorer"),
                inputs={"model": model, "configuration": configuration},
                extra={"metadata": {"recording_id": directory.name, "model": model,
                    "code_identity": self._code_identity}},
            )
            self._root.post()
            self._capture = self._root.create_child(name="Serving requests", run_type="chain", inputs={"model": model})
            self._capture.post()
            self._status = "started"
        except Exception as error:
            self._failed(error)

    def start_request(self, request_id: str, prompt: str, token_ids: list[int] | None = None,
        request_configuration: dict[str, Any] | None = None) -> None:
        if self._capture is None or request_id in self._requests:
            return
        try:
            child = self._capture.create_child(
                name=f"Generate: {request_id}", run_type="llm",
                inputs={"messages": [{"role": "user", "content": prompt}], "token_ids": token_ids,
                    "configuration": self._configuration, "request_configuration": request_configuration},
                extra={"metadata": {"request_id": request_id, "ls_provider": "vllm"}},
            )
            child.post()
            self._requests[request_id] = child
        except Exception as error:
            self._failed(error)

    def finish_request(self, request_id: str, text: str, token_ids: list[int] | None, error: str | None = None) -> None:
        child = self._requests.pop(request_id, None)
        if child is None:
            return
        try:
            child.end(outputs={"generations": [{"text": text}], "token_ids": token_ids}, error=error)
            child.patch()
        except Exception as failure:
            self._failed(failure)

    def finish(self, status: str, error: str | None = None) -> None:
        for request_id in list(self._requests):
            self.finish_request(request_id, "", None, "Request did not complete before capture ended.")
        try:
            for run in (self._capture, self._root):
                if run is not None:
                    run.end(outputs={"capture_status": status}, error=error)
                    run.patch()
            if self._root is not None:
                self._root.client.flush(timeout=5.0)
                if self._status != "failed":
                    self._status = "queued"
        except Exception as failure:
            self._failed(failure)

    def metadata(self) -> dict[str, Any]:
        return {"provider": "langsmith", "status": self._status,
            "trace_id": str(self._root.id) if self._root is not None else None,
            "error": self._error,
            "code_identity": self._code_identity,
            "verification": "Queued submission does not establish server persistence or human-readable trace inspection."}

    def _failed(self, error: Exception) -> None:
        self._status = "failed"
        # SDK errors can contain request headers or endpoint credentials.
        self._error = f"LangSmith tracing failed ({type(error).__name__})."
