"""Pure snapshots of scheduler output.

These functions copy the pre-advance cursor from the scheduler output. After
``schedule()`` returns, the live request has already reserved the new tokens.
"""

from __future__ import annotations

from typing import Any


def classify_token_range(start: int, length: int, prompt_length: int) -> tuple[int, int]:
    """Split a scheduled range into prompt processing and generation counts."""
    if start < 0 or length < 0 or prompt_length < 0:
        raise ValueError("token ranges must be non-negative")
    prompt_tokens = max(0, min(start + length, prompt_length) - start)
    return prompt_tokens, length - prompt_tokens


def copy_block_groups(block_ids: Any) -> list[list[int]]:
    if block_ids is None:
        return []
    return [[int(block_id) for block_id in group] for group in block_ids]


def snapshot_scheduled_step(
    scheduler_output: Any,
    live_requests: dict[str, Any],
    token_budget: int,
) -> dict[str, Any]:
    """Copy one scheduled step. Prefers cursors stored on the scheduler output."""
    cursors: dict[str, int] = {}
    prompt_lengths: dict[str, int] = {}
    changes: list[dict[str, Any]] = []

    for new_request in scheduler_output.scheduled_new_reqs:
        request_id = new_request.req_id
        cursors[request_id] = int(new_request.num_computed_tokens)
        prompt_lengths[request_id] = int(new_request.prompt_len)
        changes.append(
            {
                "request_id": request_id,
                "change": "replace",
                "block_ids": copy_block_groups(new_request.block_ids),
            }
        )

    cached = scheduler_output.scheduled_cached_reqs
    resumed = set(cached.resumed_req_ids)
    for index, request_id in enumerate(cached.req_ids):
        cursors[request_id] = int(cached.num_computed_tokens[index])
        live = live_requests.get(request_id)
        if live is not None:
            prompt_lengths[request_id] = int(live.num_prompt_tokens)
        new_blocks = cached.new_block_ids[index] if index < len(cached.new_block_ids) else None
        if new_blocks is not None:
            changes.append(
                {
                    "request_id": request_id,
                    "change": "replace" if request_id in resumed else "append",
                    "block_ids": copy_block_groups(new_blocks),
                }
            )

    requests: list[dict[str, Any]] = []
    for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        length = int(scheduled)
        if request_id not in cursors:
            live = live_requests[request_id]
            # The output omitted the pre-advance cursor. Recover it from the
            # reservation ``schedule()`` already applied.
            cursors[request_id] = int(live.num_computed_tokens) - length
            prompt_lengths[request_id] = int(live.num_prompt_tokens)
        start = cursors[request_id]
        prompt_length = prompt_lengths.get(request_id, 0)
        prompt_tokens, generation_tokens = classify_token_range(start, length, prompt_length)
        requests.append(
            {
                "request_id": request_id,
                "range_start": start,
                "range_length": length,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "num_prompt_tokens": prompt_length,
                "priority": int(getattr(live_requests.get(request_id), "priority", 0)),
            }
        )

    preempted = scheduler_output.preempted_req_ids or ()
    return {
        "token_budget": int(token_budget),
        "total_scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens),
        "requests": requests,
        "preempted_request_ids": sorted(str(request_id) for request_id in preempted),
        "block_table_changes": changes,
    }
