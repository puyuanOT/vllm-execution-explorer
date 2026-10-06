"""Build deliberately authored UI evidence using the production recorder/normalizer.

No vLLM engine, torch tensor, GPU, tokenizer, or numerical cache is executed.
Run from the repository root: python docs/qualification/kv-diagnostics/create_authored_fixture.py
"""

from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from runner.configuration import parse_run_configuration
from runner.load import load_recording
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.profile_trace import normalize_trace
from runner.profiling import _artifact, profile_record
from runner.workload import parse_run_workload
from vllm_030.collector import Recording


def build_recording(destination: Path) -> Path:
    run_id = "authored-kv-diagnostics"
    layer_names = [f"model.layers.{index}.self_attn.attn" for index in range(2)]
    groups = [{"cache_group": 0, "pool": "pool-0", "block_size": 4, "layer_names": layer_names,
               "cache_type": "authored_full_attention", "token_slot_mapping": "authored_fixture",
               "token_slot_mapping_reason": "Authored token/block relationships; no native memory layout was inspected."}]
    pools = [{"pool": "pool-0", "num_gpu_blocks": 2, "block_size": 4, "initial_free_blocks": 2, "inventory_complete": True,
              "blocks": [{"block_id": block, "allocation_generation": 0, "ref_count": 0, "has_hash": False, "is_null": False} for block in range(2)]}]
    payload = {
        "requests": [
            {"request_id": "prefix-writer", "prompt": "The shared prefix ends", "arrival_offset_ms": 0, "max_output_tokens": 1},
            {"request_id": "prefix-reader", "prompt": "The shared prefix ends with reuse", "arrival_offset_ms": 10, "max_output_tokens": 2},
        ],
        "model": "Qwen/Qwen3-4B-Instruct-2507", "dtype": "float16",
        "vllm_configuration": {"max_model_len": 64, "max_num_seqs": 2, "max_num_scheduled_tokens": 8,
                               "enable_prefix_caching": True, "enable_chunked_prefill": True},
        "profiling_configuration": {"mode": "eager_diagnostic", "max_steps": 3, "start_step": 1,
                                    "tensor_capture": {"enabled": True, "preset": "custom", "request_id": "prefix-writer", "token_positions": [0, 1, 2, 3],
                                                       "layer_indices": [0, 1], "head_indices": [0], "dimension_start": 0, "dimension_count": 16, "max_bytes": 256}},
    }
    requests, plan = parse_run_workload(payload)
    configuration = parse_run_configuration(payload, requests, plan)
    # The controlled clock makes regeneration deterministic. It is explicitly
    # synthetic, as are the native-trace-shaped intervals below.
    clock = itertools.count(1_000_000_000, 1_000_000)
    bookmarks = {}
    with patch("vllm_030.collector.time.monotonic_ns", side_effect=lambda: next(clock)):
        recording = Recording.open(destination, run_id=run_id, producer_id="authored-kv-diagnostic-fixture")
        recording.event("cache_inventory", pools=pools, groups=groups, observation_boundary="authored_fixture")

        def admit(request_id, ids, pieces, text):
            recording.event("request_received", request_id=request_id, arrival_clock="authored_fixture")
            recording.content(kind="prompt_text", request_id=request_id, text=text)
            recording.content(kind="prompt", request_id=request_id, token_ids=ids, token_pieces=pieces, token_count=len(ids),
                              token_piece_semantics="Authored readable fragments, not tokenizer observations.")
            recording.event("engine_prompt_observed", request_id=request_id, prompt_token_count=len(ids), content_ref=request_id, observation_boundary="authored_fixture")
            recording.event("engine_admitted", request_id=request_id, status="scheduler_added", observation_boundary="authored_fixture")

        def block(block_id, operation, before, after, hash_before=False, hash_after=False):
            return recording.event("block_change", pool="pool-0", block_id=block_id, allocation_generation=1,
                                   operation=operation, ref_count_before=before, ref_count_after=after,
                                   has_hash_before=hash_before, has_hash_after=hash_after, cached_at_zero_refs=after == 0 and hash_after)

        def binding(request_id, step_id, block_id, start, end):
            return {"request_id": request_id, "step_id": step_id, "pool": "pool-0", "block_id": block_id, "allocation_generation": 1,
                    "cache_group": 0, "token_range_start": start, "token_range_end": end, "content_ref": request_id}

        def schedule(step, request_id, start, length, prompt_count, tables):
            step_id = f"step-{step}"
            event = recording.event("step_scheduled", step_id=step_id, token_budget=8, total_scheduled_tokens=length,
                                    requests=[{"request_id": request_id, "range_start": start, "range_length": length,
                                               "prompt_tokens": min(length, max(0, prompt_count - start)), "generation_tokens": max(0, start + length - prompt_count), "num_prompt_tokens": prompt_count}],
                                    preempted_request_ids=[], block_table_changes=[{"request_id": request_id, "change": "replace", "block_ids": [tables]}])
            recording.event("step_submitted", step_id=step_id, causal_event_sequence=event, observation="authored_fixture")
            return step_id

        def write(request_id, step_id, block_id, start, ids, pieces):
            result = recording.event("step_result_received", step_id=step_id, observation="authored_fixture")
            fields = binding(request_id, step_id, block_id, start, start + len(ids))
            recording.content(kind="kv_tokens", **fields, token_ids=ids, token_pieces=pieces, token_piece_semantics="Authored readable fragments.")
            return recording.event("kv_block_written", **fields, token_content_observed=True, causal_event_sequence=result, observation_boundary="authored_fixture")

        def sampled(request_id, step_id, position, token_id, piece):
            recording.content(kind="accepted_tokens", request_id=request_id, step_id=step_id, output_position_start=position,
                              token_ids=[token_id], token_pieces=[piece], token_piece_semantics="Authored readable fragment.")
            recording.event("tokens_accepted", request_id=request_id, step_id=step_id, output_position_start=position, token_count=1, content_ref=request_id)

        admit("prefix-writer", [10, 11, 12, 13], ["The", " shared", " prefix", " ends"], payload["requests"][0]["prompt"])
        block(0, "allocate", 0, 1)
        step = schedule(1, "prefix-writer", 0, 4, 4, [0])
        recording.event("kv_block_binding", **binding("prefix-writer", step, 0, 0, 4))
        bookmarks["writer_computed"] = write("prefix-writer", step, 0, 0, [10, 11, 12, 13], ["The", " shared", " prefix", " ends"])
        sampled("prefix-writer", step, 0, 100, ".")
        block(0, "cache", 1, 1, False, True)
        block(0, "release", 1, 0, True, True)
        recording.event("completion", request_id="prefix-writer", step_id=step, status="stop", reason="stop")

        admit("prefix-reader", [10, 11, 12, 13, 14, 15], ["The", " shared", " prefix", " ends", " with", " reuse"], payload["requests"][1]["prompt"])
        recording.event("prefix_lookup", request_id="prefix-reader", matched_token_count=4, block_ids=[[0]], shared_prefix_boundary=4)
        block(0, "touch", 0, 1, True, True)
        block(1, "allocate", 0, 1)
        step = schedule(2, "prefix-reader", 4, 2, 6, [0, 1])
        recording.event("kv_block_binding", **binding("prefix-reader", step, 0, 0, 4))
        bookmarks["reader_scheduled"] = recording.event("kv_block_binding", **binding("prefix-reader", step, 1, 4, 6))
        bookmarks["reader_suffix_computed"] = write("prefix-reader", step, 1, 4, [14, 15], [" with", " reuse"])
        sampled("prefix-reader", step, 0, 101, "!")
        step = schedule(3, "prefix-reader", 6, 1, 6, [0, 1])
        recording.event("kv_block_binding", **binding("prefix-reader", step, 1, 4, 7))
        bookmarks["reader_decode_computed"] = write("prefix-reader", step, 1, 6, [101], ["!"])
        sampled("prefix-reader", step, 1, 102, " Done")
        # Final sampled output102 is intentionally not processed or written to KV.
        block(0, "release", 1, 0, True, True)
        block(1, "release", 1, 0)
        recording.event("completion", request_id="prefix-reader", step_id=step, status="length", reason="length")

        native_events, steps, writes = [], [], []
        for step_index, (request_id, positions) in enumerate((("prefix-writer", range(4)), ("prefix-reader", range(4, 6)), ("prefix-reader", range(6, 7))), 1):
            step_id, origin = f"step-{step_index}", (step_index - 1) * 300 + 100
            rows = [{"row": row, "request_id": request_id, "token_position": position, "is_padding": False} for row, position in enumerate(positions)]
            steps.append({"step_id": step_id, "mapping_status": "observed", "mapping_observation_boundary": "authored_packed_rows", "unpadded_tokens": len(rows), "padded_tokens": len(rows), "padding_status": "observed", "padding_observation_boundary": "authored_padded_dimensions", "packed_rows": rows})
            def cpu(name, start, duration, **args):
                return {"ph": "X", "cat": "user_annotation" if name.startswith("explorer.") else "cuda_runtime", "name": name,
                        "ts": start, "dur": duration, "pid": 1, "tid": 1, "args": args}
            native_events.append(cpu(f"explorer.step/{step_id}", origin, 100))
            for layer_index in range(2):
                operation_id = f"cache-write-{len(writes)}"
                start = origin + layer_index * 40 + 10
                correlation = len(writes) + 10
                native_events.extend([cpu(f"explorer.layer/{layer_index}", start, 30), cpu(f"explorer.component/{layer_index}/attention", start, 25),
                                      cpu(f"explorer.cache_write/{layer_index}/{operation_id}", start + 2, 8), cpu("cudaLaunchKernel", start + 3, 1, correlation=correlation),
                                      {"ph": "X", "cat": "kernel", "name": "AUTHORED reshape_and_cache example", "ts": start + 15, "dur": 20,
                                       "pid": 2, "tid": 7, "args": {"device": 0, "stream": 7, "correlation": correlation}}])
                writes.append({"operation_id": operation_id, "step_id": step_id, "layer_index": layer_index, "layer_name": layer_names[layer_index], "cache_group": 0,
                               "pool": "pool-0", "device": "cuda:0", "status": "observed", "observation_boundary": "authored_cache_update_call", "gpu_activity_ids": [],
                               "participants": [{**row, "pool": "pool-0", "cache_group": 0, "block_id": row["token_position"] // 4, "allocation_generation": 1,
                                                 "slot": row["token_position"] % 4, "kernel_block_id": row["token_position"] // 4, "kernel_slot": row["token_position"] % 4} for row in rows]})
        raw_trace = {"traceEvents": native_events, "authored_fixture": True, "warning": "No profiler or GPU was used."}
        normalized = normalize_trace(raw_trace, run_id=run_id, steps=steps)
        for operation in writes:
            operation["gpu_activity_ids"] = [activity["activity_id"] for activity in normalized["activities"]
                                             if activity["kind"] == "cuda_kernel" and activity["scope"].get("operation_id") == operation["operation_id"]]
        snapshots = []
        for participant in writes[0]["participants"]:
            for kind in ("K", "V"):
                snapshots.append({"snapshot_id": f"snapshot-{len(snapshots)}", "operation_id": "cache-write-0", "step_id": "step-1", "request_id": "prefix-writer",
                                  "token_position": participant["token_position"], "layer_index": 0, "cache_group": 0, "pool": "pool-0", "block_id": 0, "allocation_generation": 1,
                                  "slot": participant["slot"], "device": "cuda:0", "kv_head": 0, "kind": kind, "dtype": "float16", "shape": [16], "dimension_start": 0,
                                  "values": [round((dimension - 8) * 0.125 + participant["slot"] + (0.5 if kind == "V" else 0), 3) for dimension in range(16)],
                                  "byte_count": 32, "observation_boundary": "authored_numeric_example_no_tensor_read"})
        summary = {"enabled": True, "status": "partial", "reason": "Authored example of the 256-byte budget: first-layer slices retained; second-layer values omitted. No numerical tensor was read.",
                   "max_bytes": 256, "retained_bytes": 256, "selection": {"request_id": "prefix-writer", "token_positions": [0, 1, 2, 3], "layer_indices": [0, 1], "head_indices": [0], "dimension_start": 0, "dimension_count": 16},
                   "observation_boundary": "authored_numeric_example_no_tensor_read"}
        normalized.update(cache_writes=writes, tensor_snapshots=snapshots, tensor_capture=summary, evidence="Authored synthetic diagnostic contract example. Native normalization code was used; no native activities or numerical tensors were measured.")
        profile_path = destination / "profiling" / "activities.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
        raw_path = destination / "profiling" / "raw" / "authored.pt.trace.json.gz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(raw_trace).encode("utf-8"))

        manifest = manifest_skeleton(run_id, execution_kind="fixture", execution_target=LOCAL_FIXTURE_TARGET)
        manifest.update(run_configuration=configuration, workload_plan=plan, cache_geometry={"block_size": 4, "num_gpu_blocks": 2, "initial_free_blocks": 2, "pools": pools, "groups": groups},
                        engine_settings={**configuration["vllm"], "kv_cache_dtype": "auto", "enforce_eager": True, "demo_fixture": True},
                        model={"id": "Authored 2-layer dense decoder", "revision": None, "note": "Synthetic small geometry; no model or tokenizer is loaded. Rerun controls retain the reference Qwen3 configuration only."},
                        engine_output_token_ids={"prefix-writer": [100], "prefix-reader": [101, 102]},
                        architecture={"supported": True, "model_type": "qwen3", "architectures": ["AuthoredDenseDecoder"], "source": "authored_fixture", "evidence": "All dimensions are authored UI examples, not Qwen3 model dimensions.",
                                      "num_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
                                      "activation": "silu", "norm": "RMSNorm", "norm_epsilon": 1e-6, "position_encoding": "RoPE", "layer_names": layer_names, "kv_groups": groups},
                        fixture_lesson={"title": "Authored KV diagnostic contract", "claim": "Entirely synthetic fixture: timings, device names, block mappings, token IDs, tensor values and architecture are authored. No vLLM/GPU execution was measured."},
                        qualification_bookmarks=bookmarks)
        manifest["kv_cache"] = {"contract_version": 1, "status": "observed", "unique_storage_bytes": 2048, "observation_boundary": "authored_storage_geometry",
                                "storages": [{"storage_id": f"storage-{index}", "device": "cuda:0", "dtype": "float16", "byte_count": 1024} for index in range(2)],
                                "layers": [{"layer_index": index, "layer_name": layer_names[index], "pool": "pool-0", "cache_group": 0, "device": "cuda:0", "dtype": "float16",
                                            "shape": [2, 2, 4, 32], "strides": [256, 128, 32, 1], "storage_id": f"storage-{index}", "storage_offset": 0, "logical_bytes": 1024,
                                            "backend": "AUTHORED FlashAttention layout example", "layout": "kernel_block,kv_head,token_slot,concatenated_K_V", "layout_status": "qualified",
                                            "manager_block_size": 4, "kernel_block_size": 4, "kernel_blocks_per_manager_block": 1, "num_kernel_blocks": 2, "num_kv_heads": 2, "head_dim": 16,
                                            "quantization": {"cache_dtype": "auto", "mode": "NONE", "scales": {kind: {"status": "observed", "shape": [], "dtype": "float32", "device": "cuda:0"} for kind in ("K", "V")}}} for index in range(2)]}
        manifest["profiling"] = profile_record(configuration["profiling"])
        manifest["profiling"].update(status="completed", qualification="authored_fixture_not_gpu_qualified", completeness="bounded", attachment_status="available", captured_steps=3,
                                     raw_artifacts=[_artifact(raw_path, destination)], normalized_artifact=_artifact(profile_path, destination), normalization=normalized["normalization"],
                                     tensor_capture=summary, evidence="Authored profiler-shaped fixture; no profiling, CUDA execution, or numerical tensor inspection was performed.")
        recording.close(manifest)
    loaded = load_recording(destination)
    assert loaded.manifest["execution_kind"] == "fixture" and loaded.manifest["provenance"]["real_gpu"] is False
    return destination


def build() -> Path:
    target = Path(__file__).parent / "authored-recording"
    with TemporaryDirectory(prefix="authored-fixture-", dir=target.parent) as temporary:
        source = build_recording(Path(temporary) / "recording")
        shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Authored fixture only: {target}; 36 structural events, 8 numerical examples, 6 cache-update examples.")
    return target


if __name__ == "__main__":
    build()
