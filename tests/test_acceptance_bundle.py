from __future__ import annotations

import hashlib
from pathlib import Path

from runner.load import load_recording
from runner.report import evaluate


BUNDLE = Path(__file__).resolve().parents[1] / "docs" / "acceptance" / "run-42f3805f07c3"


def test_preserved_gpu_acceptance_bundle_is_structural_and_valid() -> None:
    recording = load_recording(BUNDLE)

    assert recording.content == ()
    assert recording.manifest["content_retained"] is False
    assert recording.manifest["structural_export"] is True
    assert "engine_output_token_ids" not in recording.manifest
    assert evaluate(recording, profile="integrity").ok


def test_preserved_gpu_acceptance_checksums_match() -> None:
    expected = {}
    for line in (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest

    assert expected
    for name, digest in expected.items():
        payload = (BUNDLE / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
