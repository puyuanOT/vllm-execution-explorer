"""Loaded cache views and bounded snapshots at native cache-update boundaries.

The qualified layout follows vLLM 0.30 FlashAttentionImpl.do_kv_cache_update:
https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/flash_attn.py
No pointer values are retained; aliased views identify one backing allocation.
"""

from __future__ import annotations

import inspect
import hashlib
import math
import re
from typing import Any

MAX_WRITE_PARTICIPANTS = 262_144


def loaded_cache_tensors(runner: Any, inventory: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = (inventory or {}).get("groups", [])
    if not groups:
        groups = [{"cache_group": index, "pool": "pool-0", "block_size": getattr(group.kv_cache_spec, "block_size", None), "layer_names": list(group.layer_names)}
                  for index, group in enumerate(getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", []))]
    storages, layers, records, allocation_ids = [], [], {}, {}
    runner_groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", [])
    for name, module in runner.model.named_modules():
        tensor = getattr(module, "kv_cache", None)
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if tensor is None or not match or not callable(getattr(tensor, "untyped_storage", None)) or tensor.numel() == 0:
            continue
        layer_name = str(getattr(module, "layer_name", None) or name)
        group = next((item for item in groups if layer_name in item.get("layer_names", [])), {})
        device, dtype = str(tensor.device), str(tensor.dtype).removeprefix("torch.")
        storage = tensor.untyped_storage()
        identity = (device, storage.data_ptr())
        if identity not in allocation_ids:
            allocation_ids[identity] = f"storage-{len(storages)}"
            storages.append({"storage_id": allocation_ids[identity], "device": device, "dtype": dtype, "byte_count": int(storage.nbytes())})
        else:
            allocation = next(item for item in storages if item["storage_id"] == allocation_ids[identity])
            if allocation["dtype"] != dtype:
                allocation["dtype"] = "mixed"
        impl = getattr(module, "impl", None)
        backend = type(impl).__name__
        shape, strides = list(tensor.shape), list(tensor.stride())
        manager_size = group.get("block_size")
        index = group.get("cache_group")
        spec = getattr(runner_groups[index], "kv_cache_spec", None) if isinstance(index, int) and index < len(runner_groups) else None
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        if isinstance(per_layer_specs, dict):
            spec = per_layer_specs.get(layer_name)
        kernel_sizes = getattr(runner, "_kernel_block_sizes", [])
        kernel_size = kernel_sizes[index] if isinstance(index, int) and index < len(kernel_sizes) else None
        qualified = False
        reason = "Numerical cache addressing is qualified only for native BF16/FP16 vLLM 0.30 FlashAttention."
        if backend == "FlashAttentionImpl" and dtype in {"bfloat16", "float16"} and len(shape) == 4:
            try:
                source = inspect.getsource(impl.do_kv_cache_update)
                qualified = all(snippet in source for snippet in ("kv_cache.transpose(1, 2).split(self.head_size, dim=-1)", "reshape_and_cache_flash(", "slot_mapping,"))
            except (OSError, TypeError):
                reason = "The installed cache-update implementation could not be inspected."
            if qualified and (not isinstance(kernel_size, int) or kernel_size <= 0 or shape[2] != kernel_size or shape[3] != 2 * int(impl.head_size)
                              or not isinstance(manager_size, int) or manager_size <= 0 or manager_size % kernel_size):
                qualified, reason = False, "Observed manager/kernel cache dimensions do not match the qualified layout."
            if qualified and group.get("token_slot_mapping") != "qualified_full_attention":
                qualified, reason = False, "Manager/token addressing requires a qualified single-GPU full-attention cache group."
        record = {
            "layer_index": int(match[1]), "layer_name": layer_name, "pool": group.get("pool"), "cache_group": index,
            "device": device, "dtype": dtype, "shape": shape, "strides": strides,
            "storage_id": allocation_ids[identity], "storage_offset": int(tensor.storage_offset()), "logical_bytes": int(tensor.numel() * tensor.element_size()),
            "backend": backend, "layout": "kernel_block,kv_head,token_slot,concatenated_K_V" if qualified else "unavailable",
            "layout_status": "qualified" if qualified else "unavailable", "manager_block_size": manager_size,
            "kernel_block_size": kernel_size, "kernel_blocks_per_manager_block": manager_size // kernel_size if qualified else None,
            "num_kernel_blocks": shape[0] if qualified else None, "num_kv_heads": shape[1] if qualified else None,
            "head_dim": shape[3] // 2 if qualified else None,
            "quantization": cache_quantization(module, impl, spec),
        }
        if not qualified:
            record["layout_reason"] = reason
        layers.append(record)
        records[layer_name] = {"record": record, "module": module}
    metadata = {"contract_version": 1, "status": "observed" if layers else "unavailable", "storages": storages, "layers": layers,
                "unique_storage_bytes": sum(item["byte_count"] for item in storages) if layers else None,
                "observation_boundary": "loaded_attention_cache_tensors"}
    if not layers:
        metadata["reason"] = "Loaded attention cache tensors are not exposed by this model runner."
    return metadata, records


def cache_quantization(module: Any, impl: Any, spec: Any = None) -> dict[str, Any]:
    cache_dtype = getattr(impl, "kv_cache_dtype", getattr(module, "kv_cache_dtype", None))
    mode = getattr(spec, "kv_quant_mode", getattr(impl, "kv_quant_mode", getattr(module, "kv_quant_mode", None)))
    scales = {}
    for kind, name in (("K", "_k_scale"), ("V", "_v_scale")):
        scale = getattr(module, name, None)
        observed = scale is not None and hasattr(scale, "shape") and hasattr(scale, "dtype") and hasattr(scale, "device")
        scales[kind] = {"status": "observed" if observed else "unavailable", "shape": list(scale.shape) if observed else [],
                        "dtype": str(scale.dtype).removeprefix("torch.") if observed else None, "device": str(scale.device) if observed else None}
    return {"cache_dtype": str(cache_dtype) if cache_dtype is not None else None,
            "mode": str(getattr(mode, "name", mode)) if mode is not None else None, "scales": scales}


class CacheWriteCapture:
    """Clone selected data on the update stream, then materialize after capture."""

    def __init__(self, hooks: Any, records: dict[str, Any], selection: dict[str, Any] | None = None, generation_lookup: Any = None) -> None:
        self.hooks, self.records = hooks, records
        self.configuration = dict(selection or {})
        self.generation_lookup = generation_lookup
        self.writes: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self._slot_copies: dict[tuple[str, int | None], Any] = {}
        self._pending_values: list[tuple[dict[str, Any], Any, int]] = []
        self._generations: dict[str, dict[int, int]] = {}
        self._retained_bytes = 0
        self._participant_count = 0
        self._errors: list[str] = []
        self._finalized = False
        self.selection = {
            "request_id": self.configuration.get("request_id"), "token_positions": list(self.configuration.get("token_positions", [])),
            "layer_indices": list(self.configuration.get("layer_indices", [])), "head_indices": list(self.configuration.get("head_indices", [0])),
            "dimension_start": self.configuration.get("dimension_start", 0), "dimension_count": self.configuration.get("dimension_count", 16),
        }
        indices = sorted({item["record"]["layer_index"] for item in records.values()})
        if not self.selection["layer_indices"] and indices:
            self.selection["layer_indices"] = sorted({indices[0], indices[len(indices) // 2], indices[-1]})

    def install(self) -> None:
        for layer_name, item in self.records.items():
            impl = getattr(item["module"], "impl", None)
            original = getattr(impl, "do_kv_cache_update", None)
            if callable(original):
                try:
                    source = inspect.getsource(original)
                    self.hooks.source_fingerprints[f"{type(impl).__name__}.do_kv_cache_update"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
                except (OSError, TypeError):
                    pass  # Invocation remains explicit; unsupported numerical layouts remain unavailable.
                self._install_update(impl, original, layer_name, item["record"])

    def _install_update(self, impl: Any, original: Any, layer_name: str, record: dict[str, Any]) -> None:
        def update(layer: Any, key: Any, value: Any, cache: Any, slot_mapping: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.hooks.current_step:
                return original(layer, key, value, cache, slot_mapping, *args, **kwargs)
            operation_id = f"cache-write-{len(self.writes)}"
            mapping = self.hooks._current_mapping
            rows = [dict(row) for row in (mapping or {}).get("packed_rows", []) if not row.get("is_padding")]
            observed = (mapping or {}).get("mapping_status") == "observed" and self._participant_count + len(rows) <= MAX_WRITE_PARTICIPANTS
            write = {
                "operation_id": operation_id, "step_id": self.hooks.current_step, "layer_index": record["layer_index"], "layer_name": layer_name,
                "pool": record["pool"], "cache_group": record["cache_group"], "device": record["device"],
                "status": "observed" if observed else "unavailable", "participants": [], "gpu_activity_ids": [],
                "observation_boundary": "AttentionImpl.do_kv_cache_update.returned",
            }
            if not observed:
                write["reason"] = "Packed identities were unavailable or the bounded cache-write participant limit was reached."
            # Only the native update call belongs to this operation scope. Snapshot
            # gather/copy kernels run afterward and cannot become cache-write timing.
            with self.hooks.record_function(f"explorer.cache_write/{record['layer_index']}/{operation_id}"):
                result = original(layer, key, value, cache, slot_mapping, *args, **kwargs)
            self.writes.append(write)
            if not observed:
                return result
            self._participant_count += len(rows)
            slot_key = (self.hooks.current_step, record["cache_group"])
            try:
                if slot_key not in self._slot_copies:
                    self._slot_copies[slot_key] = slot_mapping[:len(rows)].detach().clone()
                for row in rows:
                    write["participants"].append({**row, "pool": record["pool"], "cache_group": record["cache_group"], "block_id": None,
                                                   "allocation_generation": None, "slot": None, "kernel_block_id": None, "kernel_slot": None})
                if self.generation_lookup is not None and record["layout_status"] == "qualified":
                    table = self.hooks.runner.input_batch.block_table[record["cache_group"]].block_table.np
                    request_indices = self.hooks.runner.input_batch.req_id_to_index
                    generations = {}
                    for row in rows:
                        kernel_block = int(table[request_indices[row["request_id"]], row["token_position"] // record["kernel_block_size"]])
                        block = kernel_block // record["kernel_blocks_per_manager_block"]
                        generations[block] = self.generation_lookup(record["pool"], block)
                    self._generations[operation_id] = generations
                if self.configuration.get("enabled"):
                    self._capture_values(write, record, cache, slot_mapping, rows)
            except Exception as exc:
                write.update(status="unavailable", reason=f"Cache slot/snapshot capture failed ({type(exc).__name__}); no layout claim is made.")
                self._errors.append(write["reason"])
            return result
        self.hooks._patch(impl, "do_kv_cache_update", update)

    def _capture_values(self, write: dict[str, Any], record: dict[str, Any], cache: Any, slot_mapping: Any, rows: list[dict[str, Any]]) -> None:
        if record["layer_index"] not in self.selection["layer_indices"]:
            return
        if record["layout_status"] != "qualified":
            self._errors.append(record.get("layout_reason", "Numerical tensor layout is unavailable."))
            return
        if self.selection["request_id"] is None and rows:
            self.selection["request_id"] = rows[0]["request_id"]
        matching = [row for row in rows if row["request_id"] == self.selection["request_id"]]
        if not self.selection["token_positions"] and matching:
            self.selection["token_positions"] = [row["token_position"] for row in matching[:4]]
        matching = [row for row in matching if row["token_position"] in self.selection["token_positions"]]
        if not matching:
            return
        start, count = self.selection["dimension_start"], self.selection["dimension_count"]
        heads = self.selection["head_indices"]
        if start + count > record["head_dim"] or any(head >= record["num_kv_heads"] for head in heads):
            self._errors.append("Selected KV head or dimension slice exceeds the observed tensor dimensions.")
            return
        for row in matching:
            # Index tensors and cache gathering stay on the same CUDA stream as
            # the native update. Clone owns the contents before block reuse.
            address = slot_mapping[row["row"]].clamp(min=0, max=cache.shape[0] * record["kernel_block_size"] - 1)
            kernel_block = (address // record["kernel_block_size"]).reshape(1)
            kernel_slot = (address % record["kernel_block_size"]).reshape(1)
            for head in heads:
                for kind, offset in (("K", 0), ("V", record["head_dim"])):
                    size = count * cache.element_size()
                    if self._retained_bytes + size > self.configuration.get("max_bytes", 1048576):
                        self._errors.append("The numerical tensor capture byte limit was reached.")
                        return
                    values = cache[kernel_block, head, kernel_slot, offset + start:offset + start + count].reshape(-1).detach().clone()
                    metadata = {"snapshot_id": f"snapshot-{len(self._pending_values)}", "operation_id": write["operation_id"], "step_id": write["step_id"],
                                "request_id": row["request_id"], "token_position": row["token_position"], "layer_index": record["layer_index"],
                                "cache_group": record["cache_group"], "pool": record["pool"], "block_id": None, "allocation_generation": None, "slot": None,
                                "device": record["device"], "kv_head": head, "kind": kind, "dtype": record["dtype"], "shape": [count],
                                "dimension_start": start, "byte_count": size, "observation_boundary": "same_cuda_stream_after_native_cache_update_before_reuse"}
                    self._pending_values.append((metadata, values, row["row"]))
                    self._retained_bytes += size

    def finalize(self) -> None:
        if self._finalized:
            return
        slots = {key: value.cpu().tolist() for key, value in self._slot_copies.items()}
        writes = {item["operation_id"]: item for item in self.writes}
        for write in self.writes:
            record = self.records[write["layer_name"]]["record"]
            addresses = slots.get((write["step_id"], write["cache_group"]), [])
            retained = []
            for row in write["participants"]:
                if row["row"] >= len(addresses) or addresses[row["row"]] < 0 or (record["layout_status"] == "qualified" and addresses[row["row"]] >= record["num_kernel_blocks"] * record["kernel_block_size"]):
                    write["reason"] = "Native slot mapping skipped or invalidated one or more packed rows; those rows are excluded from cache-write participants."
                    continue
                retained.append(row)
                if record["layout_status"] != "qualified":
                    continue
                address = int(addresses[row["row"]])
                kernel_block, kernel_slot = divmod(address, record["kernel_block_size"])
                block, slot = divmod(address, record["manager_block_size"])
                row.update(block_id=block, slot=slot, kernel_block_id=kernel_block, kernel_slot=kernel_slot)
                row["allocation_generation"] = self._generations.get(write["operation_id"], {}).get(block)
            write["participants"] = retained
        for metadata, values, row_index in self._pending_values:
            write = writes[metadata["operation_id"]]
            participant = next((row for row in write["participants"] if row["row"] == row_index), None)
            if participant is None:
                self._retained_bytes -= metadata["byte_count"]
                self._errors.append("A selected row was skipped by the native cache slot mapping; no numerical slice was retained.")
                continue
            metadata.update({key: participant[key] for key in ("block_id", "allocation_generation", "slot")})
            metadata["values"] = values.float().cpu().tolist()
            if any(not math.isfinite(value) for value in metadata["values"]):
                self._retained_bytes -= metadata["byte_count"]
                self._errors.append("A selected numerical slice contains non-finite values and was not retained as JSON numbers.")
                continue
            self.snapshots.append(metadata)
        self._pending_values.clear()
        self._slot_copies.clear()
        self._finalized = True

    def capture_record(self) -> dict[str, Any]:
        enabled = bool(self.configuration.get("enabled"))
        status = "disabled" if not enabled else ("partial" if self._errors and self.snapshots else "unavailable" if not self.snapshots else "completed")
        result = {"enabled": enabled, "status": status, "max_bytes": self.configuration.get("max_bytes", 1048576), "retained_bytes": self._retained_bytes,
                  "selection": self.selection, "observation_boundary": "same_cuda_stream_after_native_cache_update_before_reuse"}
        if enabled and (self._errors or not self.snapshots):
            result["reason"] = "; ".join(dict.fromkeys(self._errors)) or "Selected request/tokens/layers were not observed in the capture window; a targeted rerun is required."
        return result
