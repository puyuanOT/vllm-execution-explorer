"""A scripted engine that drives the real vLLM 0.30 adapter.

It is a fixture source for tests and bundled lessons. Replay does not use it,
and a recording it writes is marked as a fixture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class ObservedChunk:
    request_id: str
    new_token_ids: tuple[int, ...]
    text_delta: str
    finish_reason: str | None


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: object | None = None
        self.is_null = False


class _FreeQueue:
    def __init__(self, blocks: list[_Block]) -> None:
        self._blocks = list(blocks)

    def popleft_n(self, count: int) -> list[_Block]:
        chosen = self._blocks[:count]
        self._blocks = self._blocks[count:]
        return chosen

    def append_n(self, blocks: list[_Block]) -> None:
        self._blocks.extend(blocks)

    def prepend_n(self, blocks: list[_Block]) -> None:
        self._blocks = list(blocks) + self._blocks

    def remove(self, block: _Block) -> None:
        self._blocks.remove(block)


class FixturePool:
    def __init__(self, count: int) -> None:
        self.blocks = [_Block(index) for index in range(count)]
        self.free_block_queue = _FreeQueue(self.blocks)
        self.enable_caching = True
        self._reuse_watchers: dict[int, Any] = {}
        self.metrics_collector = None
        self.num_gpu_blocks = count

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_queue._blocks)

    def get_new_blocks(self, num_blocks: int) -> list[_Block]:
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        chosen = self.free_block_queue.popleft_n(num_blocks)
        for block in chosen:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return chosen

    def _notify_reuse(self, blocks: list[_Block]) -> None:
        return None

    def _maybe_evict_cached_block(self, block: _Block) -> bool:
        if block.block_hash is None:
            return False
        block.block_hash = None
        return True

    def touch(self, blocks: list[_Block]) -> None:
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: list[_Block]) -> None:
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.append_n([block])

    def unpin_blocks(self, blocks: list[_Block], on_reuse: Any) -> None:
        for block in blocks:
            block.ref_cnt -= 1
            self._reuse_watchers[block.block_id] = on_reuse
            if block.ref_cnt == 0:
                self.free_block_queue.append_n([block])


class _Request:
    def __init__(self, request_id: str, prompt_token_ids: list[int], max_output_tokens: int | None = None) -> None:
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(prompt_token_ids)
        self.num_computed_tokens = 0
        self.num_preemptions = 0
        self.arrival_time = time.time()
        self.max_output_tokens = max_output_tokens


class _NewRequest:
    def __init__(self, request: _Request, block_ids: tuple[tuple[int, ...], ...]) -> None:
        self.req_id = request.request_id
        self.num_computed_tokens = request.num_computed_tokens
        self.prompt_token_ids = request.prompt_token_ids
        self.block_ids = block_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


class _Cached:
    def __init__(self) -> None:
        self.req_ids: list[str] = []
        self.resumed_req_ids: set[str] = set()
        self.num_computed_tokens: list[int] = []
        self.new_block_ids: list[Any] = []


class FixtureScheduler:
    def __init__(
        self,
        pool: FixturePool,
        *,
        authored: bool = False,
        max_num_seqs: int = 2,
        max_num_scheduled_tokens: int = 32,
    ) -> None:
        self.requests: dict[str, _Request] = {}
        self.queue: list[str] = []
        self.seen: set[str] = set()
        self.authored = authored
        self.max_num_seqs = max_num_seqs
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.block_size = 16
        self.kv_cache_manager = SimpleNamespace(block_pool=pool)
        self._pool = pool

    def add_request(self, request: _Request) -> None:
        self.requests[request.request_id] = request
        self.queue.append(request.request_id)

    def _get_local_prefix_cache_hit(self, request: _Request) -> tuple[Any, int, int, bool]:
        return SimpleNamespace(get_block_ids=lambda: ()), 0, 0, False

    def _preempt_request(self, request: _Request, timestamp: float, drop_stale_output: bool = False) -> None:
        del timestamp, drop_stale_output
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        if request.request_id not in self.queue:
            self.queue.insert(0, request.request_id)

    def schedule(self) -> Any:
        if self.authored:
            return self._schedule_authored()
        return self._schedule_lesson()

    def _schedule_lesson(self) -> Any:
        chosen: list[str] = []
        for request_id in self.queue:
            if len(chosen) == 2:
                break
            chosen.append(request_id)
        new_requests: list[_NewRequest] = []
        cached = _Cached()
        num_scheduled: dict[str, int] = {}
        tokens: dict[str, list[int]] = {}
        finishing: set[str] = set()
        for request_id in chosen:
            request = self.requests[request_id]
            remaining_prompt = request.num_prompt_tokens - request.num_computed_tokens
            length = remaining_prompt if remaining_prompt > 0 else 1
            if request_id not in self.seen:
                block = self._pool.get_new_blocks(1)[0]
                new_requests.append(_NewRequest(request, ((block.block_id,),)))
                self._get_local_prefix_cache_hit(request)
                self.seen.add(request_id)
            else:
                cached.req_ids.append(request_id)
                cached.num_computed_tokens.append(request.num_computed_tokens)
                cached.new_block_ids.append(None)
            start = request.num_computed_tokens
            num_scheduled[request_id] = length
            if start + length >= request.num_prompt_tokens:
                tokens[request_id] = [1000 + start]
            else:
                tokens[request_id] = []
            request.num_computed_tokens = start + length
            if request.num_computed_tokens >= request.num_prompt_tokens + 2:
                finishing.add(request_id)
        for request_id in finishing:
            self.queue.remove(request_id)
        output = SimpleNamespace(
            scheduled_new_reqs=new_requests,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_scheduled,
            total_num_scheduled_tokens=sum(num_scheduled.values()),
            preempted_req_ids=set(),
            explorer_tokens=tokens,
            explorer_finishing=finishing,
        )
        return output

    def _schedule_authored(self) -> Any:
        """Honor output limits, concurrency, and the token budget. This path is the demo fixture only."""
        budget = self.max_num_scheduled_tokens
        chosen: list[tuple[str, int]] = []
        for request_id in list(self.queue):
            if len(chosen) >= self.max_num_seqs or budget <= 0:
                break
            request = self.requests[request_id]
            remaining_prompt = request.num_prompt_tokens - request.num_computed_tokens
            generated = max(0, request.num_computed_tokens - request.num_prompt_tokens)
            limit = request.max_output_tokens if request.max_output_tokens is not None else 0
            remaining_output = limit - generated
            if remaining_prompt <= 0 and remaining_output <= 0:
                self.queue.remove(request_id)
                continue
            if remaining_prompt > 0:
                length = min(remaining_prompt, budget)
            else:
                length = 1 if remaining_output > 0 else 0
            if length <= 0:
                break
            budget -= length
            chosen.append((request_id, length))
        new_requests: list[_NewRequest] = []
        cached = _Cached()
        num_scheduled: dict[str, int] = {}
        tokens: dict[str, list[int]] = {}
        finishing: set[str] = set()
        for request_id, length in chosen:
            request = self.requests[request_id]
            if request_id not in self.seen:
                block = self._pool.get_new_blocks(1)[0]
                new_requests.append(_NewRequest(request, ((block.block_id,),)))
                self._get_local_prefix_cache_hit(request)
                self.seen.add(request_id)
            else:
                cached.req_ids.append(request_id)
                cached.num_computed_tokens.append(request.num_computed_tokens)
                cached.new_block_ids.append(None)
            start = request.num_computed_tokens
            num_scheduled[request_id] = length
            crosses_or_decodes = start + length > request.num_prompt_tokens
            if crosses_or_decodes:
                tokens[request_id] = [1000 + start]
            else:
                tokens[request_id] = []
            request.num_computed_tokens = start + length
            generated = max(0, request.num_computed_tokens - request.num_prompt_tokens)
            limit = request.max_output_tokens if request.max_output_tokens is not None else 0
            if generated >= limit and request.num_computed_tokens >= request.num_prompt_tokens:
                finishing.add(request_id)
        for request_id in finishing:
            if request_id in self.queue:
                self.queue.remove(request_id)
        return SimpleNamespace(
            scheduled_new_reqs=new_requests,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_scheduled,
            total_num_scheduled_tokens=sum(num_scheduled.values()),
            preempted_req_ids=set(),
            explorer_tokens=tokens,
            explorer_finishing=finishing,
        )

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> dict[int, Any]:
        del model_runner_output
        outputs = []
        for request_id, token_ids in scheduler_output.explorer_tokens.items():
            finish = "stop" if request_id in scheduler_output.explorer_finishing else None
            if token_ids or finish:
                outputs.append(
                    SimpleNamespace(
                        request_id=request_id,
                        new_token_ids=list(token_ids),
                        finish_reason=finish,
                    )
                )
        return {0: SimpleNamespace(outputs=outputs)}


class _Future:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


class _Executor:
    def execute_model(self, scheduler_output: Any, non_block: bool = True) -> _Future:
        del scheduler_output, non_block
        return _Future(SimpleNamespace())

    def sample_tokens(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("sample_tokens is unused by the fixture engine")


class FixtureSession:
    """Demo fixture session for authored runs.

    It honors output limits and the requested concurrency and token budget.
    A fixed teaching delay separates scheduler steps so later arrivals can
    queue. That delay is not a GPU measurement. Scripted lessons construct
    FixtureEngine directly and do not use this pacing.
    """

    dispatch_transport = "fixture_direct"

    def __init__(
        self,
        recording: Any,
        *,
        blocks: int = 8,
        max_num_seqs: int = 4,
        token_budget: int = 2048,
        step_interval_s: float = 0.04,
        sleeper: Any = time.sleep,
    ) -> None:
        from vllm_030.install import install

        self.engine = FixtureEngine(
            blocks=blocks,
            authored=True,
            max_num_seqs=max_num_seqs,
            max_num_scheduled_tokens=token_budget,
        )
        self.engine.pool.enable_caching = False
        self.adapter = install(self.engine, recording, check_sources=False)
        self._cancelled = False
        self._sleeper = sleeper
        self._interval = step_interval_s
        self.execution_kind = "fixture"
        self.initial_cache_state = {
            "kind": "not_applicable",
            "reset_required": False,
            "reset_verified": False,
            "warmup_completed_before_reset": False,
            "capture_ready_after_reset": False,
            "evidence": "fixture_has_no_vllm_prefix_cache",
        }
        self.sampling_record = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "stop": []}
        self.engine_settings = {
            "async_scheduling": False,
            "enable_prefix_caching": False,
            "prefix_caching_applied": False,
            "max_num_seqs": max_num_seqs,
            "max_num_scheduled_tokens": token_budget,
            "teaching_pace_ms": int(round(step_interval_s * 1000)),
            "demo_fixture": True,
            "settings_note": (
                "Demo fixture. Teaching pace is synthetic and is not GPU time. "
                "Prefix caching is not simulated, so the recorded prefix-caching flag is false."
            ),
        }

    def add_request(self, request_id: str, prompt: str, max_output_tokens: int = 16) -> None:
        tokens = [index + 1 for index, _word in enumerate(prompt.split())] or [1]
        self.engine.add_request(request_id, tokens, max_output_tokens=max_output_tokens)

    def step(self) -> list[ObservedChunk]:
        if self._cancelled:
            self.engine.scheduler.queue.clear()
            return []
        if self._interval > 0:
            self._sleeper(self._interval)
        return self.engine.step()

    def has_unfinished(self) -> bool:
        return self.engine.has_unfinished()

    def cancel(self) -> None:
        self._cancelled = True
        self.engine.scheduler.queue.clear()

    def close(self) -> None:
        self.adapter.uninstall()


class FixtureEngine:
    """In-process stand-in whose scheduler methods match the adapter's hooks."""

    def __init__(
        self,
        *,
        blocks: int = 8,
        authored: bool = False,
        max_num_seqs: int = 2,
        max_num_scheduled_tokens: int = 32,
    ) -> None:
        self.authored = authored
        self.pool = FixturePool(blocks)
        self.scheduler = FixtureScheduler(
            self.pool,
            authored=authored,
            max_num_seqs=max_num_seqs,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
        )
        self.model_executor = _Executor()
        self.batch_queue = None
        self.async_scheduling = False

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_output_tokens: int | None = None,
    ) -> None:
        self.scheduler.add_request(_Request(request_id, prompt_token_ids, max_output_tokens))

    def step(self) -> list[ObservedChunk]:
        output = self.scheduler.schedule()
        future = self.model_executor.execute_model(output, non_block=True)
        model_output = future.result()
        result = self.scheduler.update_from_output(output, model_output)
        chunks: list[ObservedChunk] = []
        for item in result[0].outputs:
            token_ids = tuple(int(token) for token in item.new_token_ids)
            if self.authored:
                text = "".join(f" t{token}" for token in token_ids)
            else:
                text = "●" * len(token_ids)
            chunks.append(ObservedChunk(item.request_id, token_ids, text, item.finish_reason))
        return chunks

    def has_unfinished(self) -> bool:
        return bool(self.scheduler.queue)
