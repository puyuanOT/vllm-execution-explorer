#!/usr/bin/env python3
"""Read-only real-capture qualification checks. No network, GPU work or text dumps.

Stdlib checks independently compare the recorded contracts; a small Node bridge
also checks the actual frontend replay projections. Unavailable evidence is not
converted into passing GPU qualification. Exit 0=pass, 1=failed, 2=incomplete.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
GEOMETRY = {"num_layers": 36, "hidden_size": 2560, "intermediate_size": 9728,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
CLOCK = {"domain": "torch.profiler.chrome_trace", "unit": "microseconds", "origin": "trace_local"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


class Audit:
    def __init__(self, args):
        self.args = args
        self.directory = args.recording.resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.events = jsonl(self.directory / "events.jsonl")
        self.content = jsonl(self.directory / "content.jsonl")
        self.findings: list[dict] = []
        self.facts: dict = {}

    def check(self, name: str, condition: bool | None, detail: str, *, required=True):
        self.findings.append({"check": name, "status": "unavailable" if condition is None else "pass" if condition else "fail",
                              "required": required, "detail": detail})

    def artifact(self, value: dict | None, label: str) -> bytes | None:
        if not value:
            self.check(label, None, "Artifact was not retained.")
            return None
        relative = value.get("path", "")
        path = (self.directory / relative).resolve()
        safe = (isinstance(relative, str) and relative and not Path(relative).is_absolute()
                and all(part not in {".", "..", ""} for part in relative.split("/"))
                and not any(char in relative for char in "\\%?#:") and path.is_relative_to(self.directory))
        if not safe:
            self.check(label, False, "Artifact reference is not a safe local relative path.")
            return None
        if not path.is_file():
            self.check(label, None, f"Declared local artifact is missing: {relative}")
            return None
        data = path.read_bytes()
        valid = len(data) == value.get("size_bytes") and hashlib.sha256(data).hexdigest() == value.get("sha256")
        self.check(label, valid, f"SHA-256 and size checked for {relative}.")
        return data if valid else None

    def run(self):
        m = self.manifest
        fixture = m.get("execution_kind") == "fixture"
        self.check("gpu_provenance", not fixture and m.get("execution_kind") == "gpu" and m.get("hardware", {}).get("gpu") is not None,
                   "GPU execution kind and reported CUDA hardware are required; fixture evidence never qualifies a GPU.", required=not self.args.allow_fixture)
        if fixture and self.args.allow_fixture:
            self.findings[-1]["status"] = "not_applicable"
        self.check("vllm_version", m.get("engine", {}).get("version") == "0.30.0", "Exact vLLM 0.30.0 engine metadata.")
        self.check("model_identity", m.get("model", {}).get("id") == MODEL, f"Expected {MODEL}.")
        revisions = [m.get("model", {}).get("revision"), m.get("tokenizer", {}).get("revision")]
        self.check("immutable_revisions", revisions == [REVISION, REVISION], "Canonical model and tokenizer must both use the pinned SHA.", required=not fixture)
        if fixture:
            self.findings[-1]["status"] = "not_applicable"
        architecture = m.get("architecture", {})
        dimensions = {key: architecture.get(key) for key in GEOMETRY}
        self.facts["architecture"] = {"source": architecture.get("source"), "dimensions": dimensions}
        self.check("qwen3_geometry", architecture.get("supported") is True and architecture.get("model_type") == "qwen3" and dimensions == GEOMETRY,
                   "Explicit head_dim=128 is required; hidden/query-head division would incorrectly give 80.")
        self.check("qwen3_operator_metadata", architecture.get("activation") == "silu" and architecture.get("norm") == "RMSNorm" and architecture.get("position_encoding") == "RoPE" and architecture.get("layer_names") == [f"model.layers.{index}" for index in range(36)],
                   "Canonical dense Qwen3 operator metadata and complete ordered layer identities.")
        self.check("loaded_architecture", architecture.get("source") == ("pinned_model_config_reference" if fixture else "loaded_model_config"),
                   "GPU geometry must come from initialized model configuration; fixture reference geometry is separately labeled.")
        if not fixture:
            initial = m.get("initial_cache_state", {})
            ready = next((event for event in self.events if event.get("event_type") == "capture_ready"), {})
            self.check("warmup_reset_boundary", initial.get("reset_verified") is True and initial.get("warmup_completed_before_reset") is True and ready.get("cache_reset_verified") is True and ready.get("warmup_completed_before_reset") is True,
                       "Capture readiness follows successful warmup and verified prefix-cache reset.")
            source = m.get("observability", {}).get("code_identity", {})
            digest = hashlib.sha256()
            if self.args.source_ref:
                files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", self.args.source_ref, "--", "runner", "instrumentation"], cwd=ROOT, text=True).splitlines()
                records = [(name, subprocess.check_output(["git", "show", self.args.source_ref + ":" + name], cwd=ROOT)) for name in sorted(files) if Path(name).suffix in {".py", ".json"}]
            else:
                source_root = ROOT if self.args.source_root in {None, "current"} else Path(self.args.source_root).resolve()
                paths = sorted(path for folder in (source_root / "runner", source_root / "instrumentation") for path in folder.rglob("*") if path.is_file() and path.suffix in {".py", ".json"})
                records = [(path.relative_to(source_root).as_posix(), path.read_bytes()) for path in paths]
            for filename, data in records:
                name = filename.encode()
                digest.update(len(name).to_bytes(8, "big")); digest.update(name)
                digest.update(len(data).to_bytes(8, "big")); digest.update(data)
            self.facts["capture_source"] = {"recorded_sha256": source.get("source_sha256"), "comparison_sha256": digest.hexdigest(),
                                            "comparison_ref": self.args.source_ref or self.args.source_root or "working_tree", "declared_revision": source.get("declared_revision")}
            self.check("capture_source_identity", source.get("source_sha256") == digest.hexdigest() if source.get("source_sha256") else None,
                       "Distributed runner/instrumentation fingerprint matches the specified audited source; docs do not affect it.")
        self.check("retained_content", m.get("content_retained") is True and bool(self.content), "Token content sidecars are necessary for ID/provenance checks.")
        gap = next((event for event in self.events if event.get("event_type") == "trace_gap"), None)
        authoritative = self.events[:self.events.index(gap)] if gap else self.events
        self.check("event_chain", bool(self.events) and all(event.get("sequence") == index + 1 and event.get("run_id") == m.get("run_id") for index, event in enumerate(self.events)), "Contiguous per-run event identities.")
        self.check("complete_host_capture", None if gap else m.get("event_completeness", {}).get("status") == "complete",
                   "Trace-gap suffixes are never used to claim current state.", required=not self.args.allow_incomplete)
        self.facts["events"] = dict(Counter(event.get("event_type") for event in authoritative))
        self.integrity()
        self.host(authoritative)
        self.profile(authoritative)
        self.replay()
        required = [finding for finding in self.findings if finding["required"]]
        status = "failed" if any(finding["status"] == "fail" for finding in self.findings) else "incomplete" if any(finding["status"] == "unavailable" for finding in required) else "pass"
        projection_digest = hashlib.sha256()
        for name in ("reduce.ts", "learning.ts", "profiling.ts"):
            projection_digest.update(name.encode()); projection_digest.update((ROOT / "replay" / "src" / name).read_bytes())
        return {"run_id": m.get("run_id"), "scope": "fixture_contract_only" if fixture else "recorded_gpu_contracts",
                "analysis": {"contract_version": 1, "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "replay_projection_sha256": projection_digest.hexdigest()},
                "status": status, "findings": self.findings, "facts": self.facts,
                "limitations": ["No numerical K/V tensors are read; successful result boundaries establish logical full-attention KV writes.",
                                "Packed rows identify batch participants, never individual-token or head duration.",
                                "This script does not qualify GPU/browser performance or inspect actual UI rendering.",
                                "Manifest initialized-config provenance must be supported by the preserved runtime/deployment receipt."]}

    def integrity(self):
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "instrumentation"))
        from runner.load import load_recording
        from runner.report import evaluate

        report = evaluate(load_recording(self.directory), profile="integrity")
        for finding in report.findings:
            self.check("existing_integrity." + finding.code, finding.ok, finding.detail)

    def host(self, events):
        inventories = [event for event in events if event.get("event_type") == "cache_inventory"]
        if len(inventories) != 1:
            self.check("physical_inventory", None if not inventories else False, "Exactly one complete initial physical inventory is required.")
            return
        inventory = inventories[0]
        pools = {pool["pool"]: pool for pool in inventory.get("pools", [])}
        groups = {group["cache_group"]: group for group in inventory.get("groups", [])}
        valid = bool(pools) and len(pools) == len(inventory.get("pools", []))
        slots, snapshot = {}, []
        for name, pool in pools.items():
            capacity = pool.get("num_gpu_blocks")
            blocks = pool.get("blocks", [])
            ids = [block.get("block_id") for block in blocks]
            valid &= isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0 and pool.get("inventory_complete") is True and len(ids) == capacity and set(ids) == set(range(capacity))
            for block in blocks:
                valid &= all(isinstance(block.get(key), int) and not isinstance(block[key], bool) and block[key] >= 0 for key in ("allocation_generation", "ref_count")) and all(isinstance(block.get(key), bool) for key in ("has_hash", "is_null"))
                slots[(name, block["block_id"])] = (block["allocation_generation"], block["is_null"])
            nulls = sum(block.get("is_null", False) for block in blocks)
            free = sum(not block.get("is_null") and block.get("ref_count") == 0 for block in blocks)
            valid &= pool.get("initial_free_blocks") == free
            snapshot.append({"pool": name, "capacity": capacity, "reserved_null": nulls, "allocation_available": free,
                             "cached_available": sum(not block.get("is_null") and block.get("ref_count") == 0 and block.get("has_hash") for block in blocks)})
        self.check("physical_inventory", bool(valid), "Every physical ID appears exactly once; reserved-null capacity is excluded from allocation availability.")
        self.facts["initial_pool"] = snapshot
        expected_mapping = "authored_fixture" if self.manifest.get("execution_kind") == "fixture" else "qualified_full_attention"
        self.check("token_slot_qualification", True if groups and all(group.get("pool") in pools and isinstance(group.get("block_size"), int) and group["block_size"] > 0 and group.get("token_slot_mapping") == expected_mapping for group in groups.values()) else None,
                   "Only exact, unsharded full-attention groups qualify live token-slot claims.")
        layer_indices = {int(match.group(1)) for group in groups.values() for name in group.get("layer_names", []) if (match := re.search(r"model\.layers\.(\d+)", name))}
        self.check("cache_layer_coverage", layer_indices == set(range(36)), "Cache-group layer names cover all initialized Qwen3 layers.")
        scheduled, results, violations, writes, accepted_boundaries, terminals, acceptance_at = {}, {}, [], [], {}, {}, {}
        prompt_ids, accepted_ids, content_keys = {}, defaultdict(list), {}
        for row in sorted(self.content, key=lambda value: value.get("content_sequence", 0)):
            if row.get("kind") == "prompt":
                prompt_ids[row["request_id"]] = row.get("token_ids", [])
            elif row.get("kind") == "accepted_tokens":
                accepted_ids[row["request_id"]].extend(row.get("token_ids", []))
            elif row.get("kind") == "kv_tokens":
                content_keys[self.range_key(row)] = row
        for event in events:
            kind = event["event_type"]
            if kind == "step_scheduled":
                scheduled[event["step_id"]] = event
            elif kind == "step_result_received":
                results[event["step_id"]] = event["sequence"]
            elif kind == "tokens_accepted":
                accepted_boundaries[event["request_id"]] = event["sequence"]
                for position in range(event["output_position_start"], event["output_position_start"] + event["token_count"]):
                    acceptance_at[(event["request_id"], position)] = event["sequence"]
            elif kind in {"completion", "cancellation"}:
                terminals[event["request_id"]] = kind
            elif kind == "block_change" and event.get("operation") == "allocate":
                key = (event["pool"], event["block_id"])
                previous = slots.get(key)
                if previous is None or event["allocation_generation"] != previous[0] + 1:
                    violations.append({"sequence": event["sequence"], "reason": "Unknown physical ID or nonincrementing allocation generation."})
                slots[key] = (event["allocation_generation"], previous[1] if previous else False)
            elif kind in {"kv_block_binding", "kv_block_written"}:
                key = (event["pool"], event["block_id"])
                start, end = event["token_range_start"], event["token_range_end"]
                group = groups.get(event["cache_group"], {})
                step = scheduled.get(event["step_id"])
                if slots.get(key) != (event["allocation_generation"], False) or not step or group.get("pool") != event["pool"] or group.get("token_slot_mapping") != expected_mapping or start < 0 or end <= start or start // group.get("block_size", 1) != (end - 1) // group.get("block_size", 1):
                    violations.append({"sequence": event["sequence"], "reason": "Invalid/stale/null/group-crossing KV binding or write."})
                if kind == "kv_block_written":
                    writes.append(event)
                    participant = next((request for request in (step or {}).get("requests", []) if request["request_id"] == event["request_id"]), None)
                    if event.get("causal_event_sequence") != results.get(event["step_id"]) or results.get(event["step_id"], event["sequence"]) >= event["sequence"] or not participant or start < participant["range_start"] or end > participant["range_start"] + participant["range_length"]:
                        violations.append({"sequence": event["sequence"], "reason": "KV write precedes its observed result or falls outside scheduled inputs."})
                    content = content_keys.get(self.range_key(event))
                    if event.get("token_content_observed") and (not content or len(content.get("token_ids", [])) != end - start):
                        violations.append({"sequence": event["sequence"], "reason": "Declared observed KV IDs do not match their exact physical range."})
                    if content:
                        all_ids = prompt_ids.get(event["request_id"], []) + accepted_ids[event["request_id"]]
                        if content["token_ids"] != all_ids[start:end]:
                            violations.append({"sequence": event["sequence"], "reason": "Written token IDs do not match request prompt/accepted-output positions."})
                    prompt_count = len(prompt_ids.get(event["request_id"], []))
                    if step and any(acceptance_at.get((event["request_id"], position - prompt_count), float("inf")) >= step["sequence"] for position in range(max(start, prompt_count), end)):
                        violations.append({"sequence": event["sequence"], "reason": "An output token was computed before its earlier sampled acceptance was observed."})
        self.check("kv_result_boundaries", not violations and bool(writes), f"{len(writes)} returned KV ranges checked; {len(violations)} violations.")
        self.facts["kv_violations"] = violations[:20]
        self.check("ordered_scheduler_queues", bool(scheduled) and all(all(isinstance(step.get(key), dict) and step[key].get("available") is True and all(isinstance(step[key].get(queue), list) for queue in ("waiting", "running", "skipped_waiting")) for key in ("queue_before", "queue_after")) for step in scheduled.values()), "Before/after schedule snapshots must be actually available, not fabricated empty queues.")
        expected = self.manifest.get("engine_output_token_ids")
        actual = {request: accepted_ids.get(request, []) for request in expected} if isinstance(expected, dict) else {}
        actual.update(accepted_ids)
        self.check("returned_output_ids", actual == expected if isinstance(expected, dict) and expected else None, "Accepted output sidecars match each request's separately retained engine output IDs, including zero-output cancellations.")
        finals, final_errors = [], []
        for request, ids in accepted_ids.items():
            final = len(prompt_ids.get(request, [])) + len(ids) - 1
            covering = [write for write in writes if write["request_id"] == request and write["token_range_start"] <= final < write["token_range_end"]]
            valid = all(scheduled[write["step_id"]]["sequence"] > accepted_boundaries.get(request, float("inf")) for write in covering)
            if terminals.get(request) == "completion" and covering:
                valid = False
            if not valid:
                final_errors.append(request)
            finals.append({"request_id": request, "accepted_count": len(ids), "final_output_position": final,
                           "terminal": terminals.get(request), "write_sequences": [write["sequence"] for write in covering]})
        self.check("final_output_boundary", bool(finals) and not final_errors, "A sampled output becomes KV only in a later scheduled computation; naturally completed final outputs stay uncomputed.")
        self.facts["final_outputs"] = finals
        self.requests(events, prompt_ids, accepted_ids)
        features = {"prefix_reuse": sum(event["event_type"] == "prefix_lookup" and event.get("matched_token_count", 0) > 0 for event in events),
                    "preemption": sum(event["event_type"] == "preemption" for event in events),
                    "cancellation": sum(event["event_type"] == "cancellation" for event in events),
                    "cache_eviction": sum(event["event_type"] == "block_change" and event.get("operation") == "evict" for event in events),
                    "cached_release": sum(event["event_type"] == "block_change" and event.get("operation") == "release" and event.get("cached_at_zero_refs") for event in events)}
        self.facts["exercised_features"] = features
        for feature in ("prefix_reuse", "preemption", "cancellation"):
            if getattr(self.args, f"require_{feature}"):
                self.check(f"scenario_{feature}", features[feature] > 0, "The requested scenario must exercise the actual event.")

    @staticmethod
    def range_key(row):
        return tuple(row.get(key) for key in ("step_id", "request_id", "pool", "block_id", "allocation_generation", "cache_group", "token_range_start", "token_range_end"))

    def requests(self, events, prompt_ids, accepted_ids):
        samples = {sample["id"]: sample for batch in self.manifest.get("workload_plan", {}).get("batches", []) for sample in batch.get("samples", [])}
        request_ids = set(samples) | set(prompt_ids) | set(accepted_ids)
        cases, errors = [], []
        for request in sorted(request_ids):
            rows = [row for row in self.content if row.get("request_id") == request]
            input_text = "".join(row.get("text", "") for row in rows if row.get("kind") == "prompt_text")
            chunks = sorted((row for row in rows if row.get("kind") == "text"), key=lambda row: row.get("emission_index", -1))
            output_text = "".join(row.get("text_delta", "") for row in chunks)
            observed = [event for event in events if event.get("request_id") == request]
            prompt = next((event for event in observed if event["event_type"] == "engine_prompt_observed"), {})
            terminal = [event for event in observed if event["event_type"] in {"completion", "cancellation"}]
            emissions = {event["emission_index"]: event for event in observed if event["event_type"] == "text_emitted"}
            sample = samples.get(request, {})
            ids = accepted_ids.get(request, [])
            prompt_count = len(prompt_ids.get(request, []))
            valid = prompt_count > 0 and prompt.get("prompt_token_count") == prompt_count
            vocabulary = self.manifest.get("architecture", {}).get("vocab_size")
            if vocabulary is not None:
                valid &= all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < vocabulary for token in prompt_ids.get(request, []) + ids)
            valid &= sample.get("input_mode") != "prompt" or bool(input_text)
            valid &= len(ids) <= sample.get("max_output_tokens", float("inf"))
            valid &= bool(terminal) or self.args.allow_incomplete
            valid &= [row.get("emission_index") for row in chunks] == list(range(len(chunks)))
            valid &= len(emissions) == len(chunks)
            for row in chunks:
                start, end = row.get("token_range", [-1, -1])
                event = emissions.get(row["emission_index"], {})
                valid &= 0 <= start <= end <= len(ids) and event.get("token_range_start") == start and event.get("token_range_end") == end
            if not valid:
                errors.append(request)
            settings = {**self.manifest.get("sampling", {}), **(sample.get("sampling_override") or {})}
            cases.append({"request_id": request, "input_mode": sample.get("input_mode"), "prompt_token_count": prompt_count,
                          "accepted_output_token_count": len(ids), "max_output_tokens": sample.get("max_output_tokens"), "priority": sample.get("priority"),
                          "sampling": settings, "terminal": [{"event": event["event_type"], "reason": event.get("reason"), "sequence": event["sequence"]} for event in terminal],
                          "prompt_text_characters": len(input_text), "prompt_text_sha256": hashlib.sha256(input_text.encode()).hexdigest() if input_text else None,
                          "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids.get(request, []), separators=(",", ":")).encode()).hexdigest(),
                          "output_text_characters": len(output_text), "output_text_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
                          "output_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(), "text_emissions": len(chunks)})
        self.check("per_request_streams", bool(cases) and not errors, f"{len(cases)} LLM input/output streams, effective sampling/limits and terminal outcomes inspected; {len(errors)} contradictions.")
        self.facts["requests"] = cases
        self.facts["request_stream_errors"] = errors

    def profile(self, events):
        metadata = self.manifest.get("profiling", {})
        if metadata.get("mode") != "eager_diagnostic":
            self.check("native_profile", None, "Native eager diagnostic profiling was not captured.", required=self.args.require_profile)
            return
        payload = self.artifact(metadata.get("normalized_artifact"), "normalized_profile_integrity")
        if payload is None:
            return
        profile = json.loads(payload)
        self.check("native_clock", profile.get("clock") == metadata.get("clock") == CLOCK and profile.get("run_id") == self.manifest.get("run_id") and profile.get("execution_mode") == metadata.get("execution_mode") == "eager", "Trace-local microseconds and eager mode stay independent of host monotonic_ns.")
        if self.manifest.get("execution_kind") != "fixture":
            sources = metadata.get("hook_sources", {})
            expected_sources = ("GPUModelRunner._update_states", "GPUModelRunner._prepare_inputs", "GPUModelRunner.execute_model")
            self.check("qualified_native_hook_boundaries", self.manifest.get("engine_settings", {}).get("enforce_eager") is True and all(isinstance(sources.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", sources[key]) for key in expected_sources),
                       "Actual eager setting plus source fingerprints establish the installed post-reorder hook boundaries.")
            runtime = metadata.get("runtime", {})
            self.check("native_runtime_versions", runtime.get("vllm") == "0.30.0" and bool(runtime.get("torch")) and bool(runtime.get("cuda")),
                       "Native profiler records actual vLLM/PyTorch/CUDA runtime versions.")
        steps = {step["step_id"]: step for step in events if step["event_type"] == "step_scheduled"}
        mapping_errors, observed = [], 0
        for step in profile.get("steps", []):
            if step.get("mapping_status") != "observed":
                continue
            observed += 1
            rows = step.get("packed_rows", [])
            unpadded, padded = step.get("unpadded_tokens"), step.get("padded_tokens")
            host = steps.get(step["step_id"])
            actual = [(row.get("request_id"), row.get("token_position")) for row in rows if not row.get("is_padding")]
            expected = [(request["request_id"], position) for request in (host or {}).get("requests", []) for position in range(request["range_start"], request["range_start"] + request["range_length"])]
            valid = host is not None and len(rows) == (padded if padded is not None else unpadded) and [row.get("row") for row in rows] == list(range(len(rows))) and Counter(actual) == Counter(expected) and len(actual) == unpadded
            valid &= step.get("mapping_observation_boundary") == "GPUModelRunner._prepare_inputs.returned" and all(row.get("request_id") is None and row.get("token_position") is None for row in rows if row.get("is_padding"))
            if padded is not None:
                valid &= step.get("padding_observation_boundary") == "GPUModelRunner._preprocess.returned"
            if not valid:
                mapping_errors.append(step["step_id"])
        self.check("post_reorder_packed_rows", False if mapping_errors else True if observed else None, f"{observed} observed prepared batches agree with scheduler positions; order comes from packed rows, not request arrival order.")
        self.facts["packed_mapping_errors"] = mapping_errors
        activities = profile.get("activities", [])
        cuda = [activity for activity in activities if activity.get("kind", "").startswith("cuda_")]
        mapped = [activity for activity in cuda if activity.get("scope", {}).get("correlation") == "runtime_flow"]
        cpu_layers = {activity.get("scope", {}).get("layer_index") for activity in activities if activity.get("kind") == "cpu_scope" and activity.get("scope", {}).get("layer_index") is not None}
        cuda_layers = {activity.get("scope", {}).get("layer_index") for activity in mapped if activity.get("scope", {}).get("layer_index") is not None}
        step_layers = []
        for step in profile.get("steps", []):
            cpu_at_step = {activity.get("scope", {}).get("layer_index") for activity in activities if activity.get("kind") == "cpu_scope" and activity.get("scope", {}).get("step_id") == step["step_id"] and activity.get("scope", {}).get("layer_index") is not None}
            cuda_at_step = {activity.get("scope", {}).get("layer_index") for activity in mapped if activity.get("scope", {}).get("step_id") == step["step_id"] and activity.get("scope", {}).get("layer_index") is not None}
            step_layers.append({"step_id": step["step_id"], "cpu_layers": sorted(cpu_at_step), "cuda_layers": sorted(cuda_at_step), "complete_layer_coverage": cpu_at_step == cuda_at_step == set(range(36))})
        self.facts["profiler"] = {"status": metadata.get("status"), "qualification": metadata.get("qualification"), "runtime": metadata.get("runtime"),
                                  "captured_steps": len(profile.get("steps", [])), "observed_row_steps": observed,
                                  "cuda_activities": len(cuda), "correlated_cuda_activities": len(mapped), "unmapped_cuda_activities": len(cuda) - len(mapped),
                                  "cpu_layers": sorted(cpu_layers), "cuda_layers": sorted(cuda_layers), "step_layer_coverage": step_layers,
                                  "mapped_module_layers": {component: sorted({activity["scope"]["layer_index"] for activity in mapped if activity["scope"].get("component") == component}) for component in ("attention", "ffn")}}
        self.check("native_cpu_layer_coverage", True if cpu_layers == set(range(36)) else None, "Explicit Qwen3 CPU scopes cover all actual layers; this does not establish GPU layer durations.", required=self.args.require_profile)
        self.check("native_device_identity", True if mapped and all(activity.get("device") is not None and activity.get("stream") is not None for activity in mapped) else None,
                   "Mapped device timelines need actual device and stream identity; unknown device groups never get GPU busy-time claims.", required=self.args.require_profile)
        self.check("measured_cuda_layer_coverage", True if cuda_layers == set(range(36)) else None, "All 36 layer GPU timelines require observed native CUDA correlations; CPU scopes alone are insufficient.", required=self.args.require_profile)
        self.check("native_step_layer_coverage", True if step_layers and all(step["complete_layer_coverage"] for step in step_layers) else None,
                   "Each retained profiled batch has both explicit CPU scopes and correlated CUDA activity for all 36 layers; partial batches remain partial.", required=self.args.require_profile)
        raw_entries = metadata.get("raw_artifacts", [])
        if len(raw_entries) != 1:
            self.check("raw_profile", None if not raw_entries else False, "One retained native trace is required for independent correlation checks.")
            return
        raw = self.artifact(raw_entries[0], "raw_profile_integrity")
        if raw is None:
            return
        raw_trace = json.loads(gzip.decompress(raw) if raw_entries[0]["path"].endswith(".gz") else raw)
        raw_events = raw_trace.get("traceEvents", [])
        by_id = {activity["activity_id"]: activity for activity in activities}
        explicit = defaultdict(list)
        for event in raw_events:
            if event.get("ph") == "X" and str(event.get("name", "")).startswith("explorer."):
                explicit[(event.get("pid"), event.get("tid"))].append(event)
        errors = []
        for activity in cuda:
            try:
                original = raw_events[int(activity["activity_id"].split("-")[-1])]
                assert activity["start"] == original["ts"] and activity["duration"] == original["dur"]
                scope = activity["scope"]
                if scope["correlation"] == "unmapped":
                    assert all(scope.get(key) is None for key in ("step_id", "layer_index", "component"))
                    continue
                assert scope["correlation"] == "runtime_flow" and scope.get("step_id") in steps
                source = scope.get("correlation_source")
                key = "correlation" if source == "native_correlation_id" else "External id" if source == "native_external_id" else None
                assert key is not None and activity.get("related_activity_ids")
                for related_id in activity["related_activity_ids"]:
                    related = by_id[related_id]
                    raw_cpu = raw_events[int(related_id.split("-")[-1])]
                    assert original.get("args", {}).get(key) is not None and str(original["args"][key]) == str(raw_cpu.get("args", {}).get(key))
                    assert all(related["scope"].get(field) == scope.get(field) for field in ("step_id", "layer_index", "component"))
                    ancestors = [event["name"] for event in explicit[(raw_cpu.get("pid"), raw_cpu.get("tid"))] if event["ts"] <= raw_cpu["ts"] and event["ts"] + event["dur"] >= raw_cpu["ts"] + raw_cpu["dur"]]
                    assert f"explorer.step/{scope['step_id']}" in ancestors
                    if scope.get("layer_index") is not None:
                        assert f"explorer.layer/{scope['layer_index']}" in ancestors
                    if scope.get("component") is not None:
                        assert f"explorer.component/{scope['layer_index']}/{scope['component']}" in ancestors
            except (AssertionError, KeyError, IndexError, TypeError, ValueError):
                errors.append(activity.get("activity_id"))
        self.check("raw_cuda_correlations", not errors if cuda else None, f"{len(cuda)} normalized CUDA intervals checked against raw timestamps, native IDs and containing explicit scopes; {len(errors)} errors.")
        self.facts["raw_correlation_errors"] = errors[:20]

    def replay(self):
        node = shutil.which("node")
        if not node:
            self.check("frontend_replay", None, "Node is unavailable for replay projections.")
            return
        code = """
import fs from 'node:fs';
const root=JSON.parse(process.argv[1]), directory=JSON.parse(process.argv[2]);
const {createPlayer,attachContent}=await import(root+'/replay/src/reduce.ts');
const {projectKvCache,projectRequestTokens}=await import(root+'/replay/src/learning.ts');
const {readProfileArtifact,parseProfileAttachment,projectMeasuredProfile}=await import(root+'/replay/src/profiling.ts');
const read=name=>fs.readFileSync(directory+'/'+name,'utf8');
const lines=text=>text.trim()?text.trim().split(/\\n/).map(JSON.parse):[];
const manifest=JSON.parse(read('manifest.json')),events=lines(read('events.jsonl'));
const content=fs.existsSync(directory+'/content.jsonl')?lines(read('content.jsonl')):[];
const player=createPlayer(events,manifest),state=attachContent(player.at(player.maxSequence),content);
const pool=projectKvCache(state,manifest);
const finals=Object.values(state.requests).filter(r=>r.acceptedTokenCount>0).map(r=>{
 const tokens=projectRequestTokens(state,r.id),last=tokens[r.promptTokenCount+r.acceptedTokenCount-1];
 return {requestId:r.id,status:r.status,position:last?.position,kvState:last?.kvState,computedEver:last?.computedEver};
});
const binding=events.find(e=>e.event_type==='kv_block_binding');
let boundary=null;
if(binding){const at=attachContent(player.at(binding.sequence),content),request=at.requests[binding.request_id];
 const scheduled=at.steps.find(s=>s.id===binding.step_id)?.requests.find(r=>r.requestId===binding.request_id);
 const token=scheduled?projectRequestTokens(at,request.id)[scheduled.rangeStart]:null;
 boundary={sequence:binding.sequence,scheduledPosition:token?.position,kvState:token?.kvState,computedEver:token?.computedEver};}
const digest=JSON.stringify(state);player.at(Math.floor(player.maxSequence/2));
let nativeProfile=null;
const artifact=readProfileArtifact(manifest);
if(artifact&&fs.existsSync(directory+'/'+artifact.path)){
 const attachment=parseProfileAttachment(read(artifact.path),manifest),projected=projectMeasuredProfile(manifest,attachment);
 nativeProfile={status:projected.status,clock:projected.clock,evidenceKind:projected.evidenceKind,steps:projected.steps.length,layers:projected.layers.length};
}
console.log(JSON.stringify({trust:state.trust,inventoryStatus:pool.status,
 pools:pool.pools.map(p=>({pool:p.pool,capacity:p.capacity,cells:p.blocks.length,lastId:p.blocks.at(-1)?.blockId,tokenOccupancyStatus:p.tokenOccupancyStatus})),
 finals,boundary,nativeProfile,backwardSeekStable:JSON.stringify(attachContent(player.at(player.maxSequence),content))===digest}));
"""
        result = subprocess.run([node, "--experimental-strip-types", "--input-type=module", "--eval", code, json.dumps(str(ROOT)), json.dumps(str(self.directory))], capture_output=True, text=True, timeout=120)
        if result.returncode:
            self.check("frontend_replay", False, "Actual replay parser/projection failed: " + result.stderr[-1200:])
            return
        replay = json.loads(result.stdout)
        self.facts["replay"] = replay
        if self.args.require_profile:
            self.check("frontend_native_profile", True if replay["nativeProfile"] and replay["nativeProfile"]["status"] == "observed" else None, "Actual native profile loader/parser/projection accepts the retained attachment.")
        self.check("frontend_pool", bool(replay["pools"]) and all(pool["cells"] == pool["capacity"] and pool["lastId"] == pool["capacity"] - 1 for pool in replay["pools"]), "Actual frontend projection includes every physical block, including final ID.")
        known = replay["trust"] == "known"
        self.check("frontend_slot_qualification", True if known and replay["pools"] and all(pool["tokenOccupancyStatus"] == "observed" for pool in replay["pools"]) else None,
                   "Actual token occupancy projection is observed, not merely a fully enumerated physical grid.", required=not self.args.allow_incomplete)
        observed_finals = {token["request_id"]: bool(token["write_sequences"]) for token in self.facts.get("final_outputs", [])}
        self.check("frontend_final_outputs", all(token["computedEver"] == observed_finals.get(token["requestId"]) and (observed_finals.get(token["requestId"]) or token["kvState"] in {"not_computed", "reserved"}) for token in replay["finals"]) if known and replay["finals"] else None,
                   "Historical final-token computation matches actual later writes; acceptance alone never computes KV.", required=not self.args.allow_incomplete)
        self.check("backward_seek", replay["backwardSeekStable"], "Final projection is stable after backward seeking.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, help="Save the audit JSON outside the recording; defaults to stdout.")
    parser.add_argument("--require-profile", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-ref", help="Compare distributed source to a preserved local Git ref, instead of the changing working tree.")
    source.add_argument("--source-root", help="Compare source to this preserved tree path, or 'current' for the working tree.")
    parser.add_argument("--require-prefix-reuse", action="store_true")
    parser.add_argument("--require-preemption", action="store_true")
    parser.add_argument("--require-cancellation", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit the authoritative prefix without claiming full-run qualification.")
    parser.add_argument("--allow-fixture", action="store_true", help="Smoke-test contracts only; this never qualifies GPU evidence.")
    args = parser.parse_args()
    try:
        report = Audit(args).run()
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.resolve().is_relative_to(args.recording.resolve()):
            parser.error("--output must be outside the recording (read-only audit).")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return {"pass": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
