"""Load a finished recording and reject an unsupported schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_030.constants import EVENT_REQUIRED, MANIFEST_REQUIRED, SCHEMA_VERSION


class UnsupportedRecordingError(ValueError):
    pass


@dataclass(frozen=True)
class RecordingData:
    manifest: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    content: tuple[dict[str, Any], ...]
    directory: Path


def load_recording(directory: Path) -> RecordingData:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise UnsupportedRecordingError(f"{directory} has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedRecordingError(
            f"unsupported recording schema {version!r}; this explorer reads {SCHEMA_VERSION}"
        )
    missing = [key for key in MANIFEST_REQUIRED if key not in manifest]
    if missing:
        raise UnsupportedRecordingError(f"manifest is missing {', '.join(missing)}")
    events = tuple(_load_jsonl(directory / "events.jsonl"))
    for event in events:
        event_missing = [key for key in EVENT_REQUIRED if key not in event]
        if event_missing:
            raise UnsupportedRecordingError(
                f"event {event.get('sequence')} is missing {', '.join(event_missing)}"
            )
        if event.get("schema_version") != SCHEMA_VERSION:
            raise UnsupportedRecordingError(
                f"unsupported event schema {event.get('schema_version')!r}"
            )
    content_path = directory / "content.jsonl"
    content = tuple(_load_jsonl(content_path)) if content_path.is_file() else ()
    return RecordingData(manifest, events, content, directory)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise UnsupportedRecordingError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows
