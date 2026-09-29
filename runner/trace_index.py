"""Append-only JSONL indexes for independent trace cursors.

Each caller supplies its own sequence cursor. The index remembers file offsets
and complete records; it does not advance a shared reader position per client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlIndex:
    def __init__(self, path: Path, sequence_field: str) -> None:
        self.path = path
        self.sequence_field = sequence_field
        self.byte_offset = 0
        self.records: list[dict[str, Any]] = []
        self.seen: set[int] = set()

    def reset(self) -> None:
        self.byte_offset = 0
        self.records.clear()
        self.seen.clear()

    def refresh(self) -> None:
        if not self.path.is_file():
            return
        size = self.path.stat().st_size
        if size < self.byte_offset:
            self.reset()
        with self.path.open("rb") as handle:
            handle.seek(self.byte_offset)
            data = handle.read()
        if not data or b"\n" not in data:
            return
        parts = data.split(b"\n")
        tail = parts.pop()
        self.byte_offset += len(data) - len(tail)
        for raw in parts:
            if not raw.strip():
                continue
            record = json.loads(raw.decode("utf-8"))
            if not isinstance(record, dict):
                continue
            sequence = record.get(self.sequence_field)
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                continue
            if sequence in self.seen:
                continue
            self.seen.add(sequence)
            self.records.append(record)

    def page(self, after: int, limit: int) -> tuple[list[dict[str, Any]], int, bool]:
        self.refresh()
        chosen = [record for record in self.records if int(record[self.sequence_field]) > after]
        window = chosen[:limit]
        cursor = int(window[-1][self.sequence_field]) if window else after
        return window, cursor, len(chosen) > limit


class RunTraceIndex:
    def __init__(self, directory: Path) -> None:
        self.events = JsonlIndex(directory / "events.jsonl", "sequence")
        self.content = JsonlIndex(directory / "content.jsonl", "content_sequence")
