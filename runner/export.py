"""Copy a recording without prompt or output content."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vllm_030.constants import STRUCTURAL_FORBIDDEN_FIELDS

from runner.load import load_recording


_MANIFEST_CONTENT_FIELDS = STRUCTURAL_FORBIDDEN_FIELDS | {
    "engine_output_token_ids",
    "generated_text",
    "output_text",
    "prompt_text",
}


def export_structural(source: Path, destination: Path) -> Path:
    data = load_recording(source)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "events.jsonl", destination / "events.jsonl")
    manifest = _without_content(data.manifest)
    manifest["content_retained"] = False
    manifest["structural_export"] = True
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _without_content(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_content(child)
            for key, child in value.items()
            if key not in _MANIFEST_CONTENT_FIELDS
        }
    if isinstance(value, list):
        return [_without_content(child) for child in value]
    return value
