#!/usr/bin/env python3
"""Measure prepared literal prompts with pinned tokenizer JSON, without model weights.

Requires tokenizers==0.22.1. This checks local tokenizer feasibility; the engine's
observed prompt IDs/counts still must agree during GPU qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import ssl
from urllib.parse import quote
from urllib.request import urlopen

import tokenizers
from tokenizers import Tokenizer


def common_prefix(sequences: list[list[int]]) -> int:
    count = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        count += 1
    return count


def measure(plans: Path, artifacts: Path, ca_bundle: Path | None = None) -> dict:
    artifacts.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=str(ca_bundle) if ca_bundle else None)
    grouped = {}
    for path in sorted(plans.glob("*-request.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = (payload.get("tokenizer") or payload["model"], payload["tokenizer_revision"])
        grouped.setdefault(identity, []).append((path.name, payload))
    if not grouped:
        raise ValueError("No prepared request plans found.")
    results = []
    for (repository, revision), variants in grouped.items():
        url = f"https://huggingface.co/{quote(repository, safe='/')}/resolve/{quote(revision, safe='')}/tokenizer.json"
        destination = artifacts / f"{repository.replace('/', '--')}-{revision}-tokenizer.json"
        if not destination.exists():
            with urlopen(url, timeout=60, context=context) as response:
                destination.write_bytes(response.read())
        raw = destination.read_bytes()
        tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        counts, sequences = [], {}
        representative = variants[0][1]
        for row in representative["requests"]:
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            sequences[row["request_id"]] = ids
            counts.append({"request_id": row["request_id"], "prompt_tokens": len(ids),
                           "max_output_tokens": row["max_output_tokens"],
                           "prompt_plus_max_output_tokens": len(ids) + row["max_output_tokens"],
                           "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                           "token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()})
        shared = representative["demo_qualification"]["shared_request_ids"]
        shared_count = common_prefix([sequences[id] for id in shared])
        signature = [(row["request_id"], row["prompt"], row["max_output_tokens"]) for row in representative["requests"]]
        checks = {
            "variant_prompts_and_caps_identical": all(
                [(row["request_id"], row["prompt"], row["max_output_tokens"]) for row in payload["requests"]] == signature
                for _, payload in variants),
            "all_prompts_plus_output_fit_context": all(
                row["prompt_plus_max_output_tokens"] <= payload["max_model_len"] for row in counts for _, payload in variants),
            "shared_prefix_has_complete_dense_block": shared_count >= 16,
        }
        if any(payload["model"] == "Qwen/Qwen3.8-27B" for _, payload in variants):
            checks["shared_prefix_has_complete_hybrid_784_token_block"] = shared_count >= 784
        results.append({"repository": repository, "revision": revision, "source_url": url,
                        "artifact_sha256": hashlib.sha256(raw).hexdigest(), "artifact_size_bytes": len(raw),
                        "plan_files": [name for name, _ in variants], "requests": counts,
                        "shared_request_ids": shared, "common_prefix_tokens": shared_count,
                        "complete_16_token_prefix_blocks": shared_count // 16, "checks": checks})
    return {"ok": all(all(row["checks"].values()) for row in results), "tokenizers_version": tokenizers.__version__,
            "add_special_tokens": False, "automatic_chat_template": False, "truncation": False, "padding": False,
            "models": results,
            "scope": "Local pinned tokenizer JSON checks only; no model weights, native vLLM tokenization, GPU memory feasibility, output generation, or observed cache-hit evidence."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=Path(__file__).with_name("plans"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path, help="Existing trusted certificate bundle; TLS verification remains enabled.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.plans, args.artifacts, args.ca_bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for model in result["models"]:
        print(f'{model["repository"]}: prompt counts {[row["prompt_tokens"] for row in model["requests"]]}; common prefix {model["common_prefix_tokens"]}; checks {model["checks"]}')
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
