"""Copy physical cache geometry and structural queue observations."""

from __future__ import annotations

from typing import Any

from vllm_030.compatibility import CompatibilityError, discover_block_pools


def queue_snapshot(scheduler: Any) -> dict[str, Any]:
    def copy(queue: Any) -> list[dict[str, Any]]:
        result = []
        for item in queue or ():
            request = scheduler.requests.get(item) if isinstance(item, str) else item
            if request is None:
                continue
            status = getattr(request, "status", None)
            result.append({
                "request_id": str(request.request_id), "priority": int(getattr(request, "priority", 0)),
                "num_computed_tokens": int(request.num_computed_tokens), "num_prompt_tokens": int(request.num_prompt_tokens),
                "status": str(getattr(status, "name", status) or "unobserved"),
            })
        return result

    # Scripted fixtures expose a single synthetic queue rather than vLLM's queues.
    known = hasattr(scheduler, "waiting") and hasattr(scheduler, "running")
    return {
        "waiting": copy(getattr(scheduler, "waiting", None)),
        "running": copy(getattr(scheduler, "running", None)),
        "skipped_waiting": copy(getattr(scheduler, "skipped_waiting", None)),
        "available": known,
        "fixture_queue": copy(getattr(scheduler, "queue", None)) if not known else [],
    }


def cache_inventory(scheduler: Any, tracker: Any, *, authored_fixture: bool = False) -> dict[str, Any]:
    pools = discover_block_pools(scheduler)
    if not pools:
        raise CompatibilityError("Initial physical KV inventory is unavailable: no block pools were exposed.")
    raw_groups = getattr(getattr(scheduler, "kv_cache_config", None), "kv_cache_groups", None) or []
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(raw_groups):
        spec = getattr(group, "kv_cache_spec", None)
        size = getattr(spec, "block_size", None)
        status, reason = token_slot_mapping(scheduler, spec, size, authored_fixture=authored_fixture, pool_count=len(pools))
        groups.append({
            "cache_group": index, "pool": "pool-0" if len(pools) == 1 else None,
            "block_size": int(size) if isinstance(size, int) and size > 0 else None,
            "layer_names": list(getattr(group, "layer_names", None) or []),
            "cache_type": type(spec).__name__,
            "token_slot_mapping": status, "token_slot_mapping_reason": reason,
        })
    if not groups and len(pools) == 1 and isinstance(getattr(scheduler, "block_size", None), int):
        status, reason = token_slot_mapping(scheduler, None, scheduler.block_size, authored_fixture=authored_fixture, pool_count=len(pools))
        groups = [{"cache_group": 0, "pool": "pool-0", "block_size": scheduler.block_size, "layer_names": [], "cache_type": "unreported",
                   "token_slot_mapping": status, "token_slot_mapping_reason": reason}]
    snapshots = []
    for index, pool in enumerate(pools):
        name = f"pool-{index}"
        blocks = list(getattr(pool, "blocks", None) or [])
        _validate_block_ids(name, getattr(pool, "num_gpu_blocks", None), [getattr(block, "block_id", None) for block in blocks])
        sizes = {group["block_size"] for group in groups if group["pool"] == name and group["block_size"] is not None}
        snapshots.append({
            "pool": name, "num_gpu_blocks": int(pool.num_gpu_blocks),
            "block_size": next(iter(sizes)) if len(sizes) == 1 else None,
            "initial_free_blocks": int(pool.get_num_free_blocks()),
            "inventory_complete": True,
            "blocks": [{
                "block_id": int(block.block_id), "allocation_generation": tracker.current(name, int(block.block_id)),
                "ref_count": block.ref_cnt, "has_hash": block.block_hash is not None,
                "is_null": bool(getattr(block, "is_null", False)),
            } for block in blocks],
        })
    inventory = {"pools": snapshots, "groups": groups, "observation_boundary": "instrumentation_install"}
    validate_cache_inventory(inventory)
    return inventory


def _validate_block_ids(name: str, capacity: Any, ids: list[Any]) -> None:
    if not _positive_int(capacity) or len(ids) != capacity or any(not isinstance(block_id, int) or isinstance(block_id, bool) for block_id in ids) or len(set(ids)) != capacity or set(ids) != set(range(capacity)):
        raise CompatibilityError(f"Initial physical KV inventory for {name} must enumerate every physical block exactly once (IDs 0 through capacity - 1).")


def validate_cache_inventory(inventory: dict[str, Any]) -> None:
    """A full initial baseline is required even when omitted cells stay untouched."""
    pools = inventory.get("pools")
    if not isinstance(pools, list) or not pools:
        raise CompatibilityError("Initial physical KV inventory is unavailable: no pools were enumerated.")
    seen = set()
    for pool in pools:
        if not isinstance(pool, dict) or not isinstance(pool.get("pool"), str) or not pool["pool"].strip() or pool["pool"] in seen:
            raise CompatibilityError("Initial physical KV inventory contains an absent or duplicate pool identity.")
        seen.add(pool["pool"])
        blocks = pool.get("blocks")
        if pool.get("inventory_complete") is not True or not isinstance(blocks, list):
            raise CompatibilityError("Initial physical KV inventory is incomplete; every physical block must be enumerated exactly once.")
        capacity = pool.get("num_gpu_blocks")
        _validate_block_ids(pool["pool"], capacity, [block.get("block_id") if isinstance(block, dict) else None for block in blocks])
        for block in blocks:
            for field in ("allocation_generation", "ref_count"):
                value = block.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise CompatibilityError(f"Initial physical KV inventory contains an invalid {field}.")
            if not isinstance(block.get("has_hash"), bool) or not isinstance(block.get("is_null"), bool):
                raise CompatibilityError("Initial physical KV inventory must expose each block's hash and reserved-null state.")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _spec_types() -> tuple[type, type]:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

    return FullAttentionSpec, UniformTypeKVCacheSpecs


def token_slot_mapping(scheduler: Any, spec: Any, size: Any, *, authored_fixture: bool, pool_count: int) -> tuple[str, str]:
    if not _positive_int(size):
        return "unavailable", "The cache group does not expose a positive integer block size."
    if pool_count != 1:
        return "unavailable", "The cache group's physical pool mapping is not qualified."
    if authored_fixture:
        return "authored_fixture", "Explicitly authored teaching-fixture token mapper; not a vLLM cache-layout observation."
    parallel = getattr(scheduler, "parallel_config", None) or getattr(getattr(scheduler, "vllm_config", None), "parallel_config", None)
    if parallel is None or any(not _positive_int(getattr(parallel, field, None)) or getattr(parallel, field) != 1 for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")):
        return "unavailable", "A single-GPU runtime configuration has not been established for token-slot mapping."
    for owner, fields in ((scheduler, ("dcp_world_size", "pcp_world_size")),
                          (parallel, ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size", "world_size"))):
        for field in fields:
            if owner is not None and hasattr(owner, field) and (not _positive_int(getattr(owner, field)) or getattr(owner, field) != 1):
                return "unavailable", f"Per-token slot mapping requires a single GPU and unsharded context ({field}=1)."
    try:
        full_type, uniform_type = _spec_types()
    except (ImportError, AttributeError):
        return "unavailable", "Exact vLLM 0.30 cache spec types are unavailable for layout qualification."
    specs = [spec]
    if type(spec) is uniform_type:
        inner = getattr(spec, "kv_cache_specs", None)
        if not isinstance(inner, dict) or not inner:
            return "unavailable", "UniformTypeKVCacheSpecs does not expose nonempty per-layer cache specs."
        specs = list(inner.values())
    for inner in specs:
        if type(inner) is not full_type:
            return "unavailable", "Token cells currently require exact FullAttentionSpec; custom or specialized subclasses are unsupported."
        if getattr(inner, "block_size", None) != size or not _positive_int(getattr(inner, "block_size", None)):
            return "unavailable", "Per-layer block sizes do not match the cache-group block size."
        if isinstance(getattr(inner, "tokens_per_state", None), bool) or getattr(inner, "tokens_per_state", None) != 1 or getattr(inner, "uses_slot_mapping", None) is not True:
            return "unavailable", "Token cells require one stored state per token and an explicit per-token slot mapping."
        if getattr(inner, "sliding_window", None) is not None or getattr(inner, "attention_chunk_size", None) is not None or getattr(inner, "non_causal", None) is not False:
            return "unavailable", "Windowed, chunk-local, or non-causal attention is not qualified for token-cell mapping."
    return "qualified_full_attention", "Exact full-attention specs expose one token per slot with unsharded causal block addressing."


def scheduled_bindings(adapter: Any, scheduler: Any, snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate reserved physical mappings from ranges actually submitted for computation."""
    manager = scheduler.kv_cache_manager
    pools = discover_block_pools(scheduler)
    pool_names = {id(pool): f"pool-{index}" for index, pool in enumerate(pools)}
    groups = {item["cache_group"]: item for item in adapter.cache_inventory["groups"]}
    for change in snapshot["block_table_changes"]:
        request_id = change["request_id"]
        incoming = change["block_ids"]
        previous = adapter._block_tables.get(request_id, [])
        if change["change"] == "append" and previous:
            incoming = [(previous[index] if index < len(previous) else []) + values for index, values in enumerate(incoming)]
        adapter._block_tables[request_id] = incoming
    bindings: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    for scheduled in snapshot["requests"]:
        request_id = scheduled["request_id"]
        request = scheduler.requests.get(request_id)
        token_ids = list(getattr(request, "all_token_ids", None) or [])
        if not token_ids:
            token_ids = list(getattr(request, "prompt_token_ids", None) or []) + list(adapter._accepted_token_ids.get(request_id, []))
        tables = adapter._block_tables.get(request_id, [])
        physical_groups = None
        get_blocks = getattr(manager, "get_blocks", None)
        if callable(get_blocks):
            observed = get_blocks(request_id)
            tables = [list(values) for values in observed.get_block_ids()]
            physical_groups = getattr(observed, "blocks", None)
            adapter._block_tables[request_id] = tables
        start = scheduled["range_start"]
        end = start + scheduled["range_length"]
        for group_index, block_ids in enumerate(tables):
            group = groups.get(group_index, {})
            if group.get("token_slot_mapping") not in {"qualified_full_attention", "authored_fixture"}:
                continue
            size = group.get("block_size")
            if not isinstance(size, int) or size <= 0:
                continue
            for logical_block, block_id in enumerate(block_ids):
                block = None
                if physical_groups is not None and group_index < len(physical_groups) and logical_block < len(physical_groups[group_index]):
                    block = physical_groups[group_index][logical_block]
                pool_name = pool_names.get(id(getattr(block, "pool", None))) if block is not None else None
                pool_name = pool_name or group.get("pool")
                if pool_name is None:
                    continue  # No invented mapping for unsupported multi-pool layouts.
                pool_index = int(pool_name.split("-")[-1])
                if block is None:
                    block = next((item for item in getattr(pools[pool_index], "blocks", ()) if int(item.block_id) == int(block_id)), None)
                if block is None or getattr(block, "is_null", False):
                    continue
                logical_start = logical_block * size
                logical_end = min(logical_start + size, max(end, len(token_ids)))
                if logical_end <= logical_start:
                    continue
                binding = {
                    "request_id": request_id, "pool": pool_name, "block_id": int(block_id),
                    "allocation_generation": adapter._generations.current(pool_name, int(block_id)),
                    "cache_group": group_index, "token_range_start": logical_start,
                    "token_range_end": logical_end, "content_ref": request_id,
                }
                bindings.append(binding)
                write_start, write_end = max(start, logical_start), min(end, logical_start + size)
                if write_end > write_start:
                    ids = token_ids[write_start:write_end]
                    writes.append({**binding, "token_range_start": write_start, "token_range_end": write_end,
                                   "_token_ids": ids if len(ids) == write_end - write_start else []})
    return bindings, writes
