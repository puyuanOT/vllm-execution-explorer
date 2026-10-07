#!/usr/bin/env python3
"""Independently compare native Qwen3 K/V inputs with production cache snapshots.

This is an observer qualification run, not the unmodified API demo recording.
It uses the production VllmSession/recorder. Additional source-input and address
clones run before, and outside, each production cache-write profiler scope.
Readback and comparisons happen only after the native profiler has stopped.

Without --execute, print the validated plan without importing torch/vLLM.
On the authorized CUDA host:
  python docs/qualification/kv-live-gpu/native_reference.py --execute \
      --output recordings/native-reference-2026-10-07
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "instrumentation"))

from runner.configuration import apply_legacy_fields, parse_run_configuration
from runner.load import load_recording
from runner.manifest import manifest_skeleton
from runner.observability import source_identity
from runner.report import evaluate
from runner.serve import RunRequest
from runner.vllm_session import VllmSession
from runner.workload import parse_run_workload
from vllm_030.collector import Recording

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
REQUEST_ID = "native-reference"
LAYERS = [0, 18, 35]
POSITIONS = [0, 1, 2, 3]
SOURCE_URL = "https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/flash_attn.py"


def payload(dtype: str) -> dict[str, Any]:
    return {
        "engine": "vllm", "model": MODEL, "model_revision": REVISION,
        "tokenizer_revision": REVISION, "dtype": dtype, "quantization": "none",
        "max_model_len": 512, "max_num_seqs": 1, "token_budget": 64,
        "prefix_caching": True, "chunked_prefill": True,
        "gpu_memory_utilization": 0.8, "kv_cache_memory_bytes": 256 * 1024 * 1024,
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "stop": []},
        "requests": [{"request_id": REQUEST_ID, "arrival_offset_ms": 0,
                      "max_output_tokens": 4,
                      "prompt": "<|im_start|>user\nExplain how vLLM stores a token's key and value in the KV cache. Use a short, concrete example.<|im_end|>\n<|im_start|>assistant\n"}],
        "profiling_configuration": {
            "mode": "eager_diagnostic", "max_steps": 8, "start_step": 1,
            "tensor_capture": {"enabled": True, "preset": "custom", "request_id": REQUEST_ID,
                               "token_positions": POSITIONS, "layer_indices": LAYERS,
                               "head_indices": [0], "dimension_start": 0,
                               "dimension_count": 16, "max_bytes": 1048576},
        },
    }


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def float32_bits(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


class NativeReference:
    """Observe source axes and native addresses without indexing the cache."""

    def __init__(self, session: VllmSession) -> None:
        self.session = session
        self.hooks = session._profile.hooks
        self.references: list[dict[str, Any]] = []
        self.installed_layers: list[int] = []
        self.installed_sources: dict[str, str] = {}

    def install(self) -> None:
        import torch

        capture = self.hooks.cache_capture
        for layer_name, item in capture.records.items():
            record = item["record"]
            if record["layer_index"] not in LAYERS:
                continue
            if record["layout_status"] != "qualified" or record["dtype"] not in {"bfloat16", "float16"}:
                raise RuntimeError(f"Layer {record['layer_index']} lacks the qualified native BF16/FP16 layout.")
            impl = item["module"].impl
            source = inspect.getsource(type(impl).do_kv_cache_update)
            self.installed_sources[type(impl).__name__ + ".do_kv_cache_update"] = hashlib.sha256(source.encode()).hexdigest()
            production_update = impl.do_kv_cache_update
            self._install_layer(torch, capture, impl, production_update, layer_name, dict(record))
            self.installed_layers.append(record["layer_index"])
        if sorted(self.installed_layers) != LAYERS:
            raise RuntimeError(f"Expected selected layers {LAYERS}; observed {sorted(self.installed_layers)}.")

    def _install_layer(self, torch: Any, capture: Any, impl: Any, production_update: Any,
                       layer_name: str, record: dict[str, Any]) -> None:
        def observed_update(layer: Any, key: Any, value: Any, cache: Any, slot_mapping: Any,
                            *args: Any, **kwargs: Any) -> Any:
            mapping = self.hooks._current_mapping or {}
            rows = [row for row in mapping.get("packed_rows", [])
                    if not row.get("is_padding") and row.get("request_id") == REQUEST_ID
                    and row.get("token_position") in POSITIONS]
            if self.hooks.current_step and mapping.get("mapping_status") == "observed" and rows:
                # The production operation ID is assigned on entry to the inner
                # hook. No source clone runs within explorer.cache_write.
                operation_id = f"cache-write-{len(capture.writes)}"
                scope_name = f"qualification.reference_inputs/{record['layer_index']}/{operation_id}"
                generation_receipt = dict(self.session.adapter._generations._generations)
                with torch.profiler.record_function(scope_name):
                    for row in rows:
                        native_address = slot_mapping[row["row"]:row["row"] + 1].detach().clone()
                        for kind, tensor in (("K", key), ("V", value)):
                            if tensor.ndim != 3 or tensor.shape[1] <= 0 or tensor.shape[2] < 16:
                                raise RuntimeError("Native K/V source tensors do not expose token, KV-head, dimension axes.")
                            selected = tensor[row["row"], 0, :16].detach().clone()
                            self.references.append({
                                "operation_id": operation_id, "step_id": self.hooks.current_step,
                                "request_id": row["request_id"], "token_position": row["token_position"],
                                "packed_row": row["row"], "layer_index": record["layer_index"],
                                "layer_name": layer_name, "cache_group": record["cache_group"], "pool": record["pool"],
                                "kv_head": 0, "kind": kind, "dimension_start": 0, "shape": [16],
                                "dtype": str(tensor.dtype).removeprefix("torch."), "device": str(tensor.device),
                                "manager_block_size": record["manager_block_size"],
                                "kernel_block_size": record["kernel_block_size"],
                                "source_shape": list(tensor.shape), "observer_scope": scope_name,
                                "source_values": selected, "native_address": native_address,
                                "generation_receipt": generation_receipt,
                            })
            return production_update(layer, key, value, cache, slot_mapping, *args, **kwargs)

        # The production profile owns restoration, including this outer wrapper.
        self.hooks._patch(impl, "do_kv_cache_update", observed_update)

    def compare(self, normalized: dict[str, Any]) -> list[dict[str, Any]]:
        snapshots = normalized.get("tensor_snapshots", [])
        identity_fields = ("operation_id", "step_id", "request_id", "token_position", "layer_index", "kv_head", "kind")
        comparisons = []
        for reference in self.references:
            source_values = reference.pop("source_values").float().cpu().tolist()
            address = int(reference.pop("native_address").cpu().item())
            generation_receipt = reference.pop("generation_receipt")
            block, slot = divmod(address, reference["manager_block_size"])
            kernel_block, kernel_slot = divmod(address, reference["kernel_block_size"])
            expected = {**reference, "native_address": address, "block_id": block, "slot": slot,
                        "kernel_block_id": kernel_block, "kernel_slot": kernel_slot,
                        "allocation_generation": generation_receipt.get((reference["pool"], block), 0)}
            matching = [snapshot for snapshot in snapshots
                        if all(snapshot.get(field) == reference[field] for field in identity_fields)]
            snapshot = matching[0] if len(matching) == 1 else {}
            values = snapshot.get("values", [])
            finite = len(source_values) == 16 and len(values) == 16 and all(math.isfinite(number) for number in source_values + values)
            bitwise = finite and float32_bits(source_values) == float32_bits(values)
            addressing = address >= 0 and all(snapshot.get(field) == expected[field] for field in
                ("pool", "cache_group", "block_id", "slot", "allocation_generation", "dtype", "device", "shape", "dimension_start"))
            operation = next((write for write in normalized.get("cache_writes", [])
                              if write.get("operation_id") == reference["operation_id"]), {})
            participants = [row for row in operation.get("participants", [])
                            if row.get("row") == reference["packed_row"]]
            participant_ok = len(participants) == 1 and all(participants[0].get(field) == expected[field]
                for field in ("request_id", "token_position", "pool", "cache_group", "block_id", "slot",
                              "allocation_generation", "kernel_block_id", "kernel_slot"))
            comparisons.append({
                "expected": expected, "snapshot_id": snapshot.get("snapshot_id"),
                "source_values_float32": [number if math.isfinite(number) else str(number) for number in source_values],
                "cache_values_float32": [number if math.isfinite(number) else str(number) for number in values],
                "unique_snapshot": len(matching) == 1, "finite": finite,
                "float32_bitwise_equal": bitwise, "numeric_equal": finite and source_values == values,
                "addressing_equal": addressing, "native_participant_equal": participant_ok,
                "maximum_absolute_error": max((abs(a - b) for a, b in zip(source_values, values)), default=None) if finite else None,
                "ok": len(matching) == 1 and bitwise and addressing and participant_ok,
            })
        return comparisons


def assess(directory: Path, observer: NativeReference, manifest: dict[str, Any]) -> dict[str, Any]:
    profile = manifest.get("profiling", {})
    artifact = profile.get("normalized_artifact") or {}
    normalized_path = directory / "profiling" / "activities.json"
    normalized_bytes = normalized_path.read_bytes()
    normalized = json.loads(normalized_bytes)
    comparisons = observer.compare(normalized)
    scopes = [activity for activity in normalized["activities"]
              if activity["name"].startswith("qualification.reference_inputs/")]
    writes = [activity for activity in normalized["activities"]
              if activity["name"].startswith("explorer.cache_write/")]
    separate = bool(scopes) and all(
        scope["scope"].get("operation_id") is None and all(
            scope["start"] + scope["duration"] <= write["start"] or write["start"] + write["duration"] <= scope["start"]
            for write in writes if (write["process_id"], write["thread_id"]) == (scope["process_id"], scope["thread_id"]))
        for scope in scopes)
    launches_outside_write = all(activity["scope"].get("operation_id") is None
        for scope in scopes for activity in normalized["activities"]
        if activity["name"].startswith(("cudaLaunch", "cuLaunch"))
        and (activity["process_id"], activity["thread_id"]) == (scope["process_id"], scope["thread_id"])
        and scope["start"] <= activity["start"] and activity["start"] + activity["duration"] <= scope["start"] + scope["duration"])
    integrity = evaluate(load_recording(directory))
    expected_identities = {(layer, position, kind) for layer in LAYERS for position in POSITIONS for kind in ("K", "V")}
    actual_identities = {(row["expected"]["layer_index"], row["expected"]["token_position"], row["expected"]["kind"]) for row in comparisons}
    checks = {
        "recording_integrity": integrity.ok,
        "completed_gpu_request": manifest["completion_status"] == "completed" and bool(manifest["engine_output_token_ids"].get(REQUEST_ID)),
        "completed_native_profile": profile.get("status") == "completed" and profile.get("attachment_status") == "available",
        "normalized_artifact_integrity": len(normalized_bytes) == artifact.get("size_bytes") and hashlib.sha256(normalized_bytes).hexdigest() == artifact.get("sha256"),
        "exact_selected_slice_coverage": len(comparisons) == 24 and actual_identities == expected_identities and len(normalized.get("tensor_snapshots", [])) == 24,
        "all_selected_values_and_addresses_equal": bool(comparisons) and all(row["ok"] for row in comparisons),
        "observer_cpu_scopes_outside_native_write_scopes": separate,
        "observer_gpu_launches_outside_native_write_scopes": launches_outside_write,
        "all_selected_writes_have_gpu_evidence": all(write.get("status") == "observed" and bool(write.get("gpu_activity_ids"))
            for write in normalized.get("cache_writes", []) if write.get("operation_id") in {row["expected"]["operation_id"] for row in comparisons}),
        "exact_production_byte_accounting": normalized.get("tensor_capture", {}).get("retained_bytes") == 24 * 16 * 2,
    }
    return {
        "contract_version": 1, "status": "pass" if all(checks.values()) else "failed",
        "kind": "native_source_kv_reference_observer", "run_id": manifest["run_id"], "real_gpu": True,
        "separate_from_unmodified_api_demo": True, "checks": checks, "comparisons": comparisons,
        "integrity_findings": [vars(finding) for finding in integrity.findings],
        "observer_scope_count": len(scopes), "production_tensor_capture": normalized.get("tensor_capture"),
        "source_refs": {"upstream_cache_update": SOURCE_URL, "installed_source_sha256": observer.installed_sources,
                        "production_hook_sources": profile.get("hook_sources"), "capture_source": source_identity(ROOT),
                        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "limitations": ["Extra source/address clones add observer overhead; these timings are not the unmodified API demo or a benchmark.",
                        "This verifies the selected native BF16/FP16 slices, not every tensor element or other cache layouts.",
                        "A cache-write CUDA operation describes batch participants; no individual-token or head latency is measured.",
                        "This direct-engine run does not verify explorer HTTP boundaries or browser presentation."],
    }


def execute(args: argparse.Namespace, body: RunRequest, requests: list[Any], configuration: dict[str, Any], plan: dict[str, Any]) -> int:
    directory = args.output.resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    recording = Recording.open(directory, run_id=directory.name)
    session, observer = None, None
    tokens = {REQUEST_ID: []}
    status, failure, cleanup_failure = "failed", None, None
    try:
        session = VllmSession.open(recording, body)
        observer = NativeReference(session)
        observer.install()
        request = requests[0]
        recording.content(kind="prompt_text", request_id=request.request_id, text=request.prompt)
        session.add_work_request(request)
        deadline = time.monotonic() + args.timeout
        steps, emission = 0, 0
        while session.has_unfinished():
            if time.monotonic() > deadline or steps >= 32:
                raise TimeoutError("The bounded reference workload did not finish.")
            steps += 1
            for chunk in session.step():
                start = len(tokens[chunk.request_id])
                tokens[chunk.request_id].extend(chunk.new_token_ids)
                if chunk.text_delta:
                    recording.text_emitted(chunk.request_id, emission_index=emission, text_delta=chunk.text_delta,
                                           token_range=(start, len(tokens[chunk.request_id])) if chunk.new_token_ids else None)
                    emission += 1
        session._profile.finish()
        status = "completed"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if session is not None:
            try:
                if status != "completed" and session._profile is not None:
                    session._profile.finish(failed=True)
                session.close()
            except Exception as error:
                cleanup_failure = {"type": type(error).__name__, "message": str(error)}
                status = "failed"
        manifest = manifest_skeleton(directory.name, execution_kind="gpu",
            execution_target=(session.execution_target or {}) if session is not None else {}, completion_status=status)
        manifest.update(run_configuration=configuration, workload_plan=plan, engine_output_token_ids=tokens,
                        observability={"code_identity": source_identity(ROOT), "langsmith": "waived_by_user"},
                        qualification_observer={"kind": "native_source_kv_reference", "separate_from_unmodified_api_demo": True,
                                                "source_input_clone_boundary": "before_production_cache_write_scope", "failure": failure,
                                                "cleanup_failure": cleanup_failure})
        if session is not None:
            for attribute, field in (("engine_record", "engine"), ("model_record", "model"), ("tokenizer_record", "tokenizer"),
                ("hardware_record", "hardware"), ("engine_settings", "engine_settings"), ("sampling_record", "sampling"),
                ("initial_cache_state", "initial_cache_state"), ("architecture_record", "architecture"), ("cache_geometry", "cache_geometry"),
                ("kv_cache_record", "kv_cache"), ("profiling_record", "profiling"), ("revision_resolution", "revision_resolution")):
                value = getattr(session, attribute, None)
                if value is not None:
                    manifest[field] = value
        manifest["provenance"].update(real_gpu=bool(session is not None and session.hardware_record), gpu_acceptance="required",
            claim="Separate direct-engine observer run; source K/V clones add overhead. Numerical acceptance is recorded in native-reference-assessment.json.")
        manifest = recording.close(manifest)
    try:
        if failure or cleanup_failure or observer is None:
            result = {"status": "failed", "kind": "native_source_kv_reference_observer", "failure": failure,
                      "cleanup_failure": cleanup_failure, "separate_from_unmodified_api_demo": True}
        else:
            result = assess(directory, observer, manifest)
    except Exception as error:
        result = {"status": "failed", "kind": "native_source_kv_reference_observer",
                  "failure": {"type": type(error).__name__, "message": str(error)}, "separate_from_unmodified_api_demo": True}
    save(directory / "native-reference-assessment.json", result)
    print(json.dumps({"recording": str(directory), "status": result["status"], "checks": result.get("checks"),
                      "failure": result.get("failure")}, indent=2), flush=True)
    return 0 if result["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Run the actual CUDA observer workload.")
    parser.add_argument("--output", type=Path, default=Path("recordings/native-reference"), help="Fresh output directory; existing recordings are never overwritten.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--timeout", type=float, default=120, help="Workload timeout after engine initialization, in seconds.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    data = payload(args.dtype)
    requests, plan = parse_run_workload(data)
    configuration = parse_run_configuration(data, requests, plan)
    body = RunRequest(**data)
    apply_legacy_fields(body, configuration)
    if not args.execute:
        print(json.dumps({"kind": "native_reference_plan_only", "executes_gpu": False, "output": str(args.output),
                          "payload": data, "configuration": configuration}, indent=2))
        return 0
    if args.output.exists():
        parser.error("--output must name a fresh directory")
    return execute(args, body, requests, configuration, plan)


if __name__ == "__main__":
    raise SystemExit(main())
