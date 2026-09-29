"""Allocation generations for physical KV blocks.

A later occupant of the same block id gets a new generation. Reuse, release,
and hash eviction keep the current generation.
"""

from __future__ import annotations


class GenerationTracker:
    def __init__(self) -> None:
        self._generations: dict[tuple[str, int], int] = {}

    def current(self, pool: str, block_id: int) -> int:
        return self._generations.get((pool, block_id), 0)

    def new_occupant(self, pool: str, block_id: int) -> int:
        key = (pool, block_id)
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        return generation
