#!/usr/bin/env python3
"""Populate the standard Hugging Face cache with the two pinned public models.

This performs repository metadata and file downloads only. It does not import
torch, load model weights into a device, install packages, or execute inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

MODELS = (
    ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
)
TOKENIZER_FILES = {"merges.txt", "tokenizer.model", "spiece.model", "vocab.txt", "LICENSE", "LICENSE.txt"}


def native_file(name: str) -> bool:
    return "/" not in name and (name.endswith((".safetensors", ".json", ".jinja", ".py")) or name in TOKENIZER_FILES)


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(receipt_path: Path) -> int:
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "huggingface_hub_version": huggingface_hub.__version__, "cache_root": str(HF_HUB_CACHE), "models": [],
               "scope": "Pinned public repository file downloads only; no GPU loading, inference, or package installation."}
    write_receipt(receipt_path, receipt)
    api = HfApi(token=False)
    try:
        for repository, revision in MODELS:
            print(f"Inspecting public files: {repository} at {revision}", flush=True)
            metadata = api.model_info(repository, revision=revision, files_metadata=True)
            if metadata.sha != revision:
                raise ValueError(f"Repository revision mismatch for {repository}: {metadata.sha}")
            selected = [row for row in metadata.siblings if native_file(row.rfilename)]
            names = [row.rfilename for row in selected]
            if "config.json" not in names or "tokenizer.json" not in names or not any(name.endswith(".safetensors") for name in names):
                raise ValueError(f"Missing native configuration, tokenizer or safetensors for {repository}")
            model = {"repository": repository, "revision": revision, "resolved_revision": metadata.sha, "status": "downloading",
                     "public_files": [{"name": row.rfilename, "published_size_bytes": row.size,
                                       "published_lfs_sha256": getattr(row.lfs, "sha256", None) if row.lfs else None} for row in selected],
                     "omitted_public_files": [row.rfilename for row in metadata.siblings if not native_file(row.rfilename)]}
            receipt["models"].append(model)
            write_receipt(receipt_path, receipt)
            print(f"Downloading {repository}: {len(names)} native files", flush=True)
            snapshot = Path(snapshot_download(repository, revision=revision, token=False, allow_patterns=names, max_workers=8))
            for row in model["public_files"]:
                size = (snapshot / row["name"]).stat().st_size
                row["downloaded_size_bytes"] = size
                if row["published_size_bytes"] is not None and size != row["published_size_bytes"]:
                    raise ValueError(f"Downloaded size mismatch: {repository}/{row['name']}")
            model.update(status="completed", snapshot_path=str(snapshot),
                         downloaded_size_bytes=sum(row["downloaded_size_bytes"] for row in model["public_files"]),
                         completed_utc=datetime.now(timezone.utc).isoformat())
            write_receipt(receipt_path, receipt)
            print(f"Ready: {repository} ({model['downloaded_size_bytes']} bytes)", flush=True)
        receipt.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        write_receipt(receipt_path, receipt)
        return 0
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                       completed_utc=datetime.now(timezone.utc).isoformat())
        write_receipt(receipt_path, receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=Path(__file__).with_name("model-download-receipt.json"))
    args = parser.parse_args()
    return download(args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
