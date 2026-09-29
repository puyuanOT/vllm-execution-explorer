"""Background recording buffer.

The scheduling thread only copies plain values and enqueues them. A writer
thread serializes JSONL. A full buffer records one trace gap, keeps the events
already queued, and stops capture so the engine is not blocked.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from vllm_030.constants import (
    CLOCK_DOMAIN,
    PRODUCER_ID,
    SCHEMA_VERSION,
    STRUCTURAL_FORBIDDEN_FIELDS,
)


class RecordingError(RuntimeError):
    pass


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_freeze(item) for item in value]
    if isinstance(value, list):
        return [_freeze(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _freeze(item) for key, item in value.items()}
    raise RecordingError(
        f"recording values must be JSON scalars, lists, or objects, got {type(value).__name__}"
    )


def _without_nulls(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


class Recording:
    def __init__(
        self,
        directory: Path,
        *,
        run_id: str,
        capacity: int = 8192,
        producer_id: str = PRODUCER_ID,
        autostart: bool = True,
    ) -> None:
        if capacity < 2:
            raise RecordingError("capacity must keep one slot free for a trace gap")
        if not run_id or "/" in run_id or "\\" in run_id:
            raise RecordingError("run_id must be a single path segment")
        self.directory = directory
        self.run_id = run_id
        self.producer_id = producer_id
        self._capacity = capacity
        self._pending: deque[tuple[str, dict[str, Any]]] = deque()
        self._cv = threading.Condition()
        self._capturing = True
        self._stop = False
        self._event_sequence = 0
        self._content_sequence = 0
        self._dropped_records = 0
        self._dropped_events = 0
        self._dropped_content = 0
        self._missing_event_start: int | None = None
        self._missing_event_end: int | None = None
        self._gap_sequence: int | None = None
        self._gap_body: dict[str, Any] | None = None
        self._writer_error: BaseException | None = None
        self._closed = False
        directory.mkdir(parents=True, exist_ok=False)
        self._events = (directory / "events.jsonl").open("w", encoding="utf-8")
        self._content = (directory / "content.jsonl").open("w", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._write_loop,
            name="vllm-explorer-collector",
            daemon=True,
        )
        if autostart:
            self.start()

    @classmethod
    def open(cls, directory: Path, **kwargs: Any) -> Recording:
        return cls(directory, **kwargs)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def event(self, event_type: str, **fields: Any) -> int | None:
        forbidden = STRUCTURAL_FORBIDDEN_FIELDS.intersection(fields)
        if forbidden:
            raise RecordingError(
                "structural events cannot contain content fields: "
                + ", ".join(sorted(forbidden))
            )
        body = _without_nulls(_freeze(fields))
        body["event_type"] = event_type
        return self._offer("event", body)

    def content(self, **fields: Any) -> int | None:
        if "kind" not in fields:
            raise RecordingError("content records need a kind")
        return self._offer("content", _without_nulls(_freeze(fields)))

    def text_emitted(
        self,
        request_id: str,
        *,
        emission_index: int,
        text_delta: str,
        token_range: tuple[int, int] | None,
    ) -> None:
        """Record emitted text in the content attachment and a structural pointer."""
        content_fields: dict[str, Any] = {
            "kind": "text",
            "request_id": request_id,
            "emission_index": emission_index,
            "text_delta": text_delta,
        }
        event_fields: dict[str, Any] = {
            "request_id": request_id,
            "emission_index": emission_index,
            "content_ref": request_id,
        }
        if token_range is not None:
            content_fields["token_range"] = [token_range[0], token_range[1]]
            event_fields["token_range_start"] = token_range[0]
            event_fields["token_range_end"] = token_range[1]
        self.content(**content_fields)
        self.event("text_emitted", **event_fields)

    def close(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RecordingError("recording is already closed")
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if not self._thread.is_alive():
            self.start()
        self._thread.join()
        self._events.close()
        self._content.close()
        self._closed = True
        document = dict(manifest)
        document["schema_version"] = SCHEMA_VERSION
        document["run_id"] = self.run_id
        document["clock_domain"] = CLOCK_DOMAIN
        document["event_completeness"] = self.completeness()
        if document["event_completeness"]["status"] != "complete":
            if document.get("completion_status") == "completed":
                document["completion_status"] = "incomplete"
        (self.directory / "manifest.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return document

    def completeness(self) -> dict[str, Any]:
        overflow = self._gap_sequence is not None or self._dropped_records > 0
        failed = self._writer_error is not None
        status = "incomplete" if overflow or failed else "complete"
        gap = None
        if self._gap_sequence is not None:
            gap = {
                "sequence": self._gap_sequence,
                "reason": "buffer_exhausted",
                "missing_sequence_start": self._missing_event_start,
                "missing_sequence_end": self._missing_event_end,
            }
        result: dict[str, Any] = {
            "status": status,
            "dropped_record_count": self._dropped_records,
            "dropped_event_count": self._dropped_events,
            "dropped_content_count": self._dropped_content,
            "gap": gap,
        }
        if failed:
            result["writer_error"] = str(self._writer_error)
        return result

    def _offer(self, kind: str, fields: dict[str, Any]) -> int | None:
        with self._cv:
            if not self._capturing:
                self._reject(kind)
                return None
            if len(self._pending) >= self._capacity - 1:
                self._capturing = False
                self._event_sequence += 1
                gap_sequence = self._event_sequence
                gap = self._envelope(
                    gap_sequence,
                    {
                        "event_type": "trace_gap",
                        "reason": "buffer_exhausted",
                    },
                )
                self._gap_sequence = gap_sequence
                self._gap_body = gap
                self._pending.append(("event", gap))
                self._reject(kind)
                if self._missing_event_start is not None:
                    gap["missing_sequence_start"] = self._missing_event_start
                self._cv.notify()
                return None
            if kind == "event":
                self._event_sequence += 1
                sequence = self._event_sequence
                event_type = str(fields.pop("event_type"))
                body = self._envelope(sequence, {"event_type": event_type, **fields})
                self._pending.append(("event", body))
                self._cv.notify()
                return sequence
            self._content_sequence += 1
            sequence = self._content_sequence
            body = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "content_sequence": sequence,
                "monotonic_ns": time.monotonic_ns(),
                **fields,
            }
            self._pending.append(("content", body))
            self._cv.notify()
            return sequence

    def _reject(self, kind: str) -> None:
        self._dropped_records += 1
        if kind == "content":
            self._dropped_content += 1
            return
        self._dropped_events += 1
        self._event_sequence += 1
        if self._missing_event_start is None:
            self._missing_event_start = self._event_sequence
        self._missing_event_end = self._event_sequence
        if self._gap_body is not None and "missing_sequence_start" not in self._gap_body:
            self._gap_body["missing_sequence_start"] = self._missing_event_start

    def _envelope(self, sequence: int, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "producer_id": self.producer_id,
            "sequence": sequence,
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }

    def _write_loop(self) -> None:
        try:
            while True:
                with self._cv:
                    while not self._pending and not self._stop:
                        self._cv.wait()
                    if not self._pending:
                        return
                    kind, body = self._pending.popleft()
                line = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
                handle = self._events if kind == "event" else self._content
                handle.write(line + "\n")
                handle.flush()
        except Exception as exc:  # noqa: BLE001 - the recording must report writer failure
            self._writer_error = exc
            with self._cv:
                self._capturing = False
                self._stop = True
