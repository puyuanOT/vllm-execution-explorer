"""Author an expanded physical inventory for UI addressability testing only."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--blocks", type=int, default=35000)
args = parser.parse_args()
manifest = json.loads((args.source / "manifest.json").read_text())
events = [json.loads(line) for line in (args.source / "events.jsonl").read_text().splitlines() if line]
inventory = next(event for event in events if event["event_type"] == "cache_inventory")
if len(inventory["pools"]) != 1 or args.blocks < inventory["pools"][0]["num_gpu_blocks"]:
    parser.error("Use one authored fixture pool and expand its capacity.")
if manifest.get("execution_kind") != "fixture":
    parser.error("Only a teaching fixture can be used; never relabel GPU evidence.")
args.output.mkdir(parents=True, exist_ok=False)
run_id = f"authored-expanded-pool-{args.blocks}"
manifest.update(run_id=run_id, demo_fixture=True,
    fixture_lesson="Authored pool expansion for UI addressability; not a measured GPU inventory or performance benchmark.")
pool = inventory["pools"][0]
original = pool["num_gpu_blocks"]
pool["blocks"].extend({"block_id": index, "allocation_generation": 0, "ref_count": 0,
    "has_hash": False, "is_null": False} for index in range(original, args.blocks))
pool.update(num_gpu_blocks=args.blocks, initial_free_blocks=args.blocks, inventory_complete=True)
manifest["cache_geometry"].update(num_gpu_blocks=args.blocks, initial_free_blocks=args.blocks,
    pools=inventory["pools"], groups=inventory["groups"])
for event in events:
    event["run_id"] = run_id
content = [json.loads(line) for line in (args.source / "content.jsonl").read_text().splitlines() if line]
for record in content:
    record["run_id"] = run_id
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for name, records in (("events.jsonl", events), ("content.jsonl", content)):
    (args.output / name).write_text("".join(json.dumps(item) + "\n" for item in records))
print(args.output.resolve())
