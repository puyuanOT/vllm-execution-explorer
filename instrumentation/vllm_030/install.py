"""Install the vLLM 0.30.0 recording adapter.

``get_new_blocks`` below is adapted from vLLM 0.30.0
(https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py),
Copyright contributors to the vLLM project, SPDX-License-Identifier: Apache-2.0.
The adapter copies plain values at the call and does not write to disk or
synchronize CUDA on the scheduling path.
"""

from __future__ import annotations

from typing import Any

from vllm_030.blocks import GenerationTracker
from vllm_030.capture import copy_block_groups, snapshot_scheduled_step
from vllm_030.collector import Recording
from vllm_030.inventory import cache_inventory, queue_snapshot, scheduled_bindings
from vllm_030.compatibility import (
    assert_block_pool_source,
    assert_engine_shape,
    assert_vllm_version,
    discover_block_pools,
)


class StepIds:
    def __init__(self) -> None:
        self._next = 1

    def create(self) -> str:
        step_id = f"step-{self._next}"
        self._next += 1
        return step_id


class Instrumentation:
    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self._ids = StepIds()
        self._scheduled: dict[str, int] = {}
        self._results: dict[str, int] = {}
        self._result_seen: set[str] = set()
        self._positions: dict[str, int] = {}
        self._accepted_token_ids: dict[str, list[int]] = {}
        self._block_tables: dict[str, list[list[int]]] = {}
        self._step_writes: dict[str, list[dict[str, Any]]] = {}
        self.cache_inventory: dict[str, Any] = {"pools": [], "groups": []}
        self.token_pieces: Any = None
        self.token_pieces_by_request: dict[str, Any] = {}
        self._open_step: str | None = None
        self._generations = GenerationTracker()
        self._restore: list[tuple[Any, str, Any]] = []

    def uninstall(self) -> None:
        for obj, name, previous in reversed(self._restore):
            setattr(obj, name, previous)
        self._restore.clear()

    def _patch(self, obj: Any, name: str, replacement: Any) -> None:
        self._restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def _emit_result(self, step_id: str, observation: str) -> None:
        if step_id in self._result_seen:
            return
        self._result_seen.add(step_id)
        sequence = self.recording.event(
            "step_result_received",
            step_id=step_id,
            causal_event_sequence=self._scheduled.get(step_id),
            observation=observation,
        )
        if sequence is not None:
            self._results[step_id] = sequence
        for write in self._step_writes.pop(step_id, []):
            values = dict(write)
            token_ids = values.pop("_token_ids")
            if token_ids:
                self.recording.content(kind="kv_tokens", step_id=step_id, **values, token_ids=token_ids, **self._pieces(token_ids, values["request_id"]))
            self.recording.event("kv_block_written", step_id=step_id, **values,
                                 token_content_observed=bool(token_ids), causal_event_sequence=sequence,
                                 observation_boundary=observation)

    def _pieces(self, token_ids: list[int], request_id: str | None = None) -> dict[str, Any]:
        decoder = self.token_pieces_by_request.get(request_id, self.token_pieces)
        if decoder is None:
            return {}
        try:
            pieces = decoder(token_ids)
            if isinstance(pieces, list) and len(pieces) == len(token_ids) and all(isinstance(item, str) for item in pieces):
                return {"token_pieces": pieces, "token_piece_semantics": "individually decoded tokenizer fragments; not original-text slices, and concatenation is not guaranteed to equal complete decoded text"}
        except Exception:
            pass  # IDs remain observed; text labels are optional, never invented.
        return {}

    def _emit_accepted(self, step_id: str, result: Any) -> None:
        for output in _iter_outputs(result):
            request_id = str(output.request_id)
            token_ids = [int(token) for token in (getattr(output, "new_token_ids", None) or [])]
            if token_ids:
                start = self._positions.get(request_id, 0)
                self.recording.content(
                    kind="accepted_tokens",
                    request_id=request_id,
                    step_id=step_id,
                    output_position_start=start,
                    token_ids=token_ids,
                    **self._pieces(token_ids, request_id),
                )
                self.recording.event(
                    "tokens_accepted",
                    request_id=request_id,
                    step_id=step_id,
                    output_position_start=start,
                    token_count=len(token_ids),
                    content_ref=request_id,
                    causal_event_sequence=self._results.get(step_id),
                )
                self._positions[request_id] = start + len(token_ids)
                self._accepted_token_ids.setdefault(request_id, []).extend(token_ids)
            reason = getattr(output, "finish_reason", None)
            if reason:
                observed = str(reason)
                event_type = "cancellation" if observed in {"abort", "cancelled"} else "completion"
                self.recording.event(
                    event_type,
                    request_id=request_id,
                    step_id=step_id,
                    status=observed,
                    reason=observed,
                )


def install(
    engine_core: Any,
    recording: Recording,
    *,
    vllm_version: str | None = None,
    check_sources: bool = True,
    token_pieces: Any = None,
) -> Instrumentation:
    if vllm_version is not None:
        assert_vllm_version(vllm_version)
    assert_engine_shape(engine_core)
    pools = discover_block_pools(engine_core.scheduler)
    if check_sources:
        for pool in pools:
            assert_block_pool_source(pool)
    adapter = Instrumentation(recording)
    adapter.token_pieces = token_pieces
    adapter.cache_inventory = cache_inventory(engine_core.scheduler, adapter._generations,
                                             authored_fixture=getattr(engine_core, "explorer_token_slot_mapper", None) == "authored_fixture_v1")
    recording.event("cache_inventory", **adapter.cache_inventory)
    try:
        _install_scheduler(adapter, engine_core.scheduler)
        _install_executor(adapter, engine_core.model_executor)
        for index, pool in enumerate(pools):
            _install_pool(adapter, pool, f"pool-{index}")
    except Exception:
        adapter.uninstall()
        raise
    return adapter


def _install_scheduler(adapter: Instrumentation, scheduler: Any) -> None:
    original_schedule = scheduler.schedule
    original_update = scheduler.update_from_output
    original_prefix = scheduler._get_local_prefix_cache_hit
    original_preempt = scheduler._preempt_request
    original_add = scheduler.add_request

    def schedule(*args: Any, **kwargs: Any) -> Any:
        before = queue_snapshot(scheduler)
        output = original_schedule(*args, **kwargs)
        step_id = adapter._ids.create()
        setattr(output, "explorer_step_id", step_id)
        snapshot = snapshot_scheduled_step(
            output,
            dict(scheduler.requests),
            int(scheduler.max_num_scheduled_tokens),
        )
        snapshot["queue_before"] = before
        snapshot["queue_after"] = queue_snapshot(scheduler)
        policy = getattr(scheduler, "policy", None)
        snapshot["policy"] = str(getattr(policy, "value", policy) or "unreported")
        snapshot["queue_observation_boundary"] = "before_and_after_scheduler.schedule"
        sequence = adapter.recording.event("step_scheduled", step_id=step_id, **snapshot)
        if sequence is not None:
            adapter._scheduled[step_id] = sequence
        bindings, writes = scheduled_bindings(adapter, scheduler, snapshot)
        adapter._step_writes[step_id] = writes
        for binding in bindings:
            adapter.recording.event("kv_block_binding", step_id=step_id, **binding, causal_event_sequence=sequence,
                                   observation_boundary="scheduler.schedule.returned")
        return output

    def update_from_output(scheduler_output: Any, model_runner_output: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_update(scheduler_output, model_runner_output, *args, **kwargs)
        step_id = getattr(scheduler_output, "explorer_step_id", None)
        if step_id:
            adapter._emit_result(str(step_id), "scheduler_update")
            adapter._emit_accepted(str(step_id), result)
        return result

    def _get_local_prefix_cache_hit(request: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_prefix(request, *args, **kwargs)
        blocks, matched, boundary = result[0], result[1], result[2]
        block_ids: list[list[int]] = []
        if hasattr(blocks, "get_block_ids"):
            block_ids = copy_block_groups(blocks.get_block_ids())
        adapter.recording.event(
            "prefix_lookup",
            request_id=str(request.request_id),
            matched_token_count=int(matched),
            block_ids=block_ids,
            shared_prefix_boundary=None if boundary is None else int(boundary),
        )
        return result

    def _preempt_request(request: Any, timestamp: float, *args: Any, **kwargs: Any) -> Any:
        computed_before = int(request.num_computed_tokens)
        result = original_preempt(request, timestamp, *args, **kwargs)
        adapter.recording.event(
            "preemption",
            request_id=str(request.request_id),
            status="preempted",
            computed_tokens_before=computed_before,
            computed_tokens_after=int(request.num_computed_tokens),
            preemption_count=int(request.num_preemptions),
        )
        return result

    def add_request(request: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_add(request, *args, **kwargs)
        token_ids = [int(token) for token in (getattr(request, "prompt_token_ids", None) or [])]
        count = len(token_ids) or int(getattr(request, "num_prompt_tokens", 0))
        request_id = str(request.request_id)
        if token_ids:
            adapter.recording.content(
                kind="prompt",
                request_id=request_id,
                token_count=count,
                token_ids=token_ids,
                **adapter._pieces(token_ids, request_id),
            )
        adapter.recording.event(
            "engine_prompt_observed",
            request_id=request_id,
            prompt_token_count=count,
            content_ref=request_id if token_ids else None,
            observation_boundary="scheduler.add_request",
        )
        adapter.recording.event(
            "engine_admitted",
            request_id=request_id,
            status="scheduler_added",
            observation_boundary="scheduler.add_request.returned",
            priority=int(getattr(request, "priority", 0)),
        )
        return result

    adapter._patch(scheduler, "schedule", schedule)
    adapter._patch(scheduler, "update_from_output", update_from_output)
    adapter._patch(scheduler, "_get_local_prefix_cache_hit", _get_local_prefix_cache_hit)
    adapter._patch(scheduler, "_preempt_request", _preempt_request)
    adapter._patch(scheduler, "add_request", add_request)


def _install_executor(adapter: Instrumentation, executor: Any) -> None:
    original_execute = executor.execute_model

    def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        step_id = getattr(scheduler_output, "explorer_step_id", None)
        if step_id:
            step_id = str(step_id)
            adapter.recording.event(
                "step_submitted",
                step_id=step_id,
                causal_event_sequence=adapter._scheduled.get(step_id),
                observation="before_execute_model",
            )
            adapter._open_step = step_id
        future = original_execute(scheduler_output, *args, **kwargs)
        if step_id:
            _observe_result(future, step_id, adapter)
        return future

    adapter._patch(executor, "execute_model", execute_model)
    if hasattr(executor, "sample_tokens"):
        original_sample = executor.sample_tokens

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            value = original_sample(*args, **kwargs)
            if adapter._open_step:
                _observe_result(value, adapter._open_step, adapter)
            return value

        adapter._patch(executor, "sample_tokens", sample_tokens)


def _observe_result(future: Any, step_id: str, adapter: Instrumentation) -> None:
    if future is None or not hasattr(future, "result") or getattr(future, "_explorer_observed", False):
        return
    original_result = future.result

    def result(*args: Any, **kwargs: Any) -> Any:
        value = original_result(*args, **kwargs)
        adapter._emit_result(step_id, "future.result")
        return value

    try:
        future.result = result
        future._explorer_observed = True
    except Exception:
        return


def _install_pool(adapter: Instrumentation, pool: Any, pool_name: str) -> None:
    recording = adapter.recording
    tracker = adapter._generations
    original_evict = pool._maybe_evict_cached_block
    original_touch = pool.touch
    original_free = pool.free_blocks
    original_unpin = pool.unpin_blocks
    original_cache = getattr(pool, "cache_full_blocks", None)

    def cache_full_blocks(*args: Any, **kwargs: Any) -> Any:
        # Observe hash publication, which may happen while a block is referenced.
        blocks = kwargs.get("blocks") if "blocks" in kwargs else (args[1] if len(args) > 1 else [])
        before = [(block, int(block.ref_cnt), block.block_hash is not None) for block in blocks]
        result = original_cache(*args, **kwargs)
        for block, refs, had_hash in before:
            if had_hash != (block.block_hash is not None):
                _emit_block(recording, tracker, pool_name, block, operation="cache", ref_before=refs,
                            ref_after=int(block.ref_cnt), has_hash_before=had_hash,
                            has_hash_after=block.block_hash is not None, new_occupant=False)
        return result

    def _maybe_evict_cached_block(block: Any, _original: Any = original_evict) -> Any:
        ref_before = int(block.ref_cnt)
        had_hash = block.block_hash is not None
        evicted = _original(block)
        if evicted:
            _emit_block(
                recording,
                tracker,
                pool_name,
                block,
                operation="evict",
                ref_before=ref_before,
                ref_after=int(block.ref_cnt),
                has_hash_before=had_hash,
                has_hash_after=block.block_hash is not None,
                new_occupant=False,
            )
        return evicted

    def touch(blocks: Any, _original: Any = original_touch) -> Any:
        listed = list(blocks)
        before = [(block, int(block.ref_cnt), block.block_hash is not None) for block in listed]
        result = _original(listed)
        for block, ref_before, had_hash in before:
            _emit_block(
                recording,
                tracker,
                pool_name,
                block,
                operation="reuse",
                ref_before=ref_before,
                ref_after=int(block.ref_cnt),
                has_hash_before=had_hash,
                has_hash_after=block.block_hash is not None,
                new_occupant=False,
            )
        return result

    def free_blocks(ordered_blocks: Any, _original: Any = original_free, _pool: Any = pool) -> Any:
        blocks = list(ordered_blocks)
        local = [
            (block, int(block.ref_cnt), block.block_hash is not None)
            for block in blocks
            if _is_local(_pool, block)
        ]
        result = _original(blocks)
        for block, ref_before, had_hash in local:
            _emit_block(
                recording,
                tracker,
                pool_name,
                block,
                operation="release",
                ref_before=ref_before,
                ref_after=int(block.ref_cnt),
                has_hash_before=had_hash,
                has_hash_after=block.block_hash is not None,
                new_occupant=False,
                release_source="free_blocks",
            )
        return result

    def unpin_blocks(blocks: Any, on_reuse: Any, _original: Any = original_unpin, _pool: Any = pool) -> Any:
        listed = list(blocks)
        local = [
            (block, int(block.ref_cnt), block.block_hash is not None)
            for block in listed
            if _is_local(_pool, block)
        ]
        result = _original(listed, on_reuse)
        for block, ref_before, had_hash in local:
            _emit_block(
                recording,
                tracker,
                pool_name,
                block,
                operation="release",
                ref_before=ref_before,
                ref_after=int(block.ref_cnt),
                has_hash_before=had_hash,
                has_hash_after=block.block_hash is not None,
                new_occupant=False,
                release_source="unpin_blocks",
            )
        return result

    def get_new_blocks(num_blocks: int, _pool: Any = pool) -> list[Any]:
        # Adapted from vLLM 0.30.0 BlockPool.get_new_blocks. Reference counts are
        # copied around the same increment the upstream method performs.
        if num_blocks > _pool.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        ret = _pool.free_block_queue.popleft_n(num_blocks)
        if _pool._reuse_watchers:
            _pool._notify_reuse(ret)
        for block in ret:
            had_hash = block.block_hash is not None
            if _pool.enable_caching:
                _pool._maybe_evict_cached_block(block)
            ref_before = int(block.ref_cnt)
            assert ref_before == 0
            block.ref_cnt = ref_before + 1
            metrics = getattr(_pool, "metrics_collector", None)
            if metrics:
                metrics.on_block_allocated(block)
            _emit_block(
                recording,
                tracker,
                pool_name,
                block,
                operation="allocate",
                ref_before=ref_before,
                ref_after=int(block.ref_cnt),
                has_hash_before=had_hash,
                has_hash_after=block.block_hash is not None,
                new_occupant=True,
            )
        return ret

    adapter._patch(pool, "_maybe_evict_cached_block", _maybe_evict_cached_block)
    adapter._patch(pool, "touch", touch)
    adapter._patch(pool, "free_blocks", free_blocks)
    adapter._patch(pool, "unpin_blocks", unpin_blocks)
    adapter._patch(pool, "get_new_blocks", get_new_blocks)
    if callable(original_cache):
        adapter._patch(pool, "cache_full_blocks", cache_full_blocks)


def _emit_block(
    recording: Recording,
    tracker: GenerationTracker,
    pool_name: str,
    block: Any,
    *,
    operation: str,
    ref_before: int,
    ref_after: int,
    has_hash_before: bool,
    has_hash_after: bool,
    new_occupant: bool,
    release_source: str | None = None,
) -> None:
    if getattr(block, "is_null", False):
        return
    block_id = int(block.block_id)
    generation = (
        tracker.new_occupant(pool_name, block_id)
        if new_occupant
        else tracker.current(pool_name, block_id)
    )
    recording.event(
        "block_change",
        pool=pool_name,
        block_id=block_id,
        allocation_generation=generation,
        operation=operation,
        ref_count_before=ref_before,
        ref_count_after=ref_after,
        has_hash_before=has_hash_before,
        has_hash_after=has_hash_after,
        cached_at_zero_refs=ref_after == 0 and has_hash_after,
        release_source=release_source,
    )


def _is_local(pool: Any, block: Any) -> bool:
    owner = getattr(block, "pool", None)
    return owner is None or owner is pool


def _iter_outputs(result: Any) -> list[Any]:
    if result is None:
        return []
    groups = result.values() if isinstance(result, dict) else [result]
    outputs: list[Any] = []
    for group in groups:
        outputs.extend(getattr(group, "outputs", None) or [])
    return outputs
