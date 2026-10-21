#!/usr/bin/env python3
"""Publish qualified captures into the localhost recording library unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def bundle(root: Path, mapping: Path) -> dict:
    entries = json.loads(mapping.read_text())
    catalog_path = root / "web/public/recordings/index.json"
    catalog = json.loads(catalog_path.read_text())
    receipts = []
    for entry in entries:
        case = root / entry["directory"]
        assessment_path = case / "six-request/evidence-assessment.json"
        if not assessment_path.exists():
            assessment_path = case / "six-request/assessment.json"
        assessment = json.loads(assessment_path.read_text())
        expansion = json.loads((case / "expansion-assessment.json").read_text())
        if not assessment["ok"] or not expansion["ok"]:
            raise ValueError(f"Unqualified case: {case}")
        source = case / "six-request/recording"
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["completion_status"] != "completed" or manifest["gpu_memory"]["status"] != "completed":
            raise ValueError("Capture or memory collection has not finalized.")
        profile = manifest["profiling"]
        if profile["status"] not in {"completed", "partial"} or profile["attachment_status"] != "available":
            raise ValueError("Native profile is not finalized and available.")
        destination = root / "web/public/recordings" / entry["id"]
        destination.mkdir(exist_ok=True)
        paths = ["manifest.json", "events.jsonl", "content.jsonl", profile["normalized_artifact"]["path"]]
        paths.extend(row["path"] for row in profile["raw_artifacts"])
        files = []
        for name in paths:
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError("Unsafe artifact path.")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != path.read_bytes():
                    raise ValueError(f"Refusing to replace a saved recording artifact: {target}")
            else:
                shutil.copyfile(path, target)
            files.append({"path": name, "size_bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        receipts.append({"id": entry["id"], "run_id": manifest["run_id"], "files": files,
                         "assessment": {"path": assessment_path.relative_to(root).as_posix(),
                                        "sha256": hashlib.sha256(assessment_path.read_bytes()).hexdigest()},
                         "capture_source": manifest["observability"]["code_identity"]})
    new_entries = [{key: value for key, value in entry.items() if key != "directory"} for entry in entries]
    replaced_ids = {entry["id"] for entry in new_entries}
    catalog["recordings"] = new_entries + [entry for entry in catalog["recordings"] if entry["id"] not in replaced_ids]
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return {"recordings": receipts, "distribution": "Original manifest, events, content and native profile artifacts; no relabeling of captured settings."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    receipt = bundle(root, args.mapping)
    (args.mapping.parent / "bundle-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Bundled {len(receipt['recordings'])} qualified real captures.")
