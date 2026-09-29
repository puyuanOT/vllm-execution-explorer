"""Pin the adapter to the vLLM 0.30 block-pool method it replaces."""

from __future__ import annotations

import inspect
from typing import Any

from vllm_030.constants import PINNED_VLLM_PREFIX


class CompatibilityError(RuntimeError):
    pass


_GET_NEW_BLOCKS_SNIPPETS = (
    "popleft_n",
    "_maybe_evict_cached_block",
    "ref_cnt += 1",
    "on_block_allocated",
)


def assert_vllm_version(version: str) -> None:
    if not version.startswith(PINNED_VLLM_PREFIX):
        raise CompatibilityError(
            f"this adapter is pinned to vLLM {PINNED_VLLM_PREFIX}x, found {version}"
        )


def assert_engine_shape(engine_core: Any) -> None:
    scheduler = getattr(engine_core, "scheduler", None)
    executor = getattr(engine_core, "model_executor", None)
    if scheduler is None or executor is None:
        raise CompatibilityError("engine core is missing a scheduler or model executor")
    for name in (
        "schedule",
        "update_from_output",
        "_get_local_prefix_cache_hit",
        "_preempt_request",
        "add_request",
    ):
        if not hasattr(scheduler, name):
            raise CompatibilityError(f"scheduler is missing {name}")
    if not hasattr(scheduler, "max_num_scheduled_tokens"):
        raise CompatibilityError("scheduler is missing max_num_scheduled_tokens")
    if not hasattr(executor, "execute_model"):
        raise CompatibilityError("model executor is missing execute_model")


def assert_block_pool_source(pool: Any) -> None:
    try:
        source = inspect.getsource(pool.get_new_blocks)
    except (OSError, TypeError) as exc:
        raise CompatibilityError("cannot read BlockPool.get_new_blocks") from exc
    missing = [snippet for snippet in _GET_NEW_BLOCKS_SNIPPETS if snippet not in source]
    if missing:
        raise CompatibilityError(
            "BlockPool.get_new_blocks does not match the pinned vLLM 0.30.0 method; "
            f"missing {missing}"
        )


def discover_block_pools(scheduler: Any) -> list[Any]:
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is None:
        return []
    pools: list[Any] = []
    primary = getattr(manager, "block_pool", None)
    if primary is not None:
        pools.append(primary)
    extra = getattr(manager, "block_pools", None) or ()
    for pool in extra:
        if pool not in pools:
            pools.append(pool)
    return pools
