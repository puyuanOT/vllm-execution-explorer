"""Explicit eager Qwen3 scopes and copied post-reorder packed row identities.

Hooks match the installed vLLM 0.30.0 GPUModelRunner: _update_states condenses
and reorders the batch before _prepare_inputs writes req_indices/query_pos.
Packed identities use CPU metadata. Optional cache snapshots clone on the native
update stream and are materialized only after native profiling has stopped.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import hashlib
from typing import Any, Callable


MAX_PACKED_ROWS = 65_536


class DiagnosticHooks:
    def __init__(self, runner: Any, record_function: Callable[[str], Any], max_steps: int, *, check_sources: bool = True,
                 cache_records: dict[str, Any] | None = None, tensor_capture: dict[str, Any] | None = None, generation_lookup: Any = None) -> None:
        self.runner = runner
        self.record_function = record_function
        self.max_steps = max_steps
        self.steps: list[dict[str, Any]] = []
        self.current_step: str | None = None
        self._restore: list[tuple[Any, str, Any]] = []
        self._current_mapping: dict[str, Any] | None = None
        self.active = False
        self.check_sources = check_sources
        self._pending_sample_step: str | None = None
        self.source_fingerprints: dict[str, str] = {}
        from vllm_030.cache_tensors import CacheWriteCapture
        self.cache_capture = CacheWriteCapture(self, cache_records or {}, tensor_capture, generation_lookup)

    def install(self) -> None:
        runner = self.runner
        for name in ("execute_model", "_prepare_inputs", "_preprocess"):
            if not callable(getattr(runner, name, None)):
                raise RuntimeError(f"The vLLM 0.30 profiling hook requires GPUModelRunner.{name}.")
        if self.check_sources:
            self.source_fingerprints = assert_profiler_runner_source(runner)
        modules = list(runner.model.named_modules())
        layers = [(name, module) for name, module in modules if type(module).__name__ in {"Qwen3DecoderLayer", "Qwen3_5DecoderLayer"}]
        if not layers:
            raise RuntimeError("Diagnostic layer scopes require loaded dense Qwen3 decoder layers.")
        for name, layer in layers:
            try:
                index = int(name.split("layers.", 1)[1].split(".", 1)[0])
            except (IndexError, ValueError) as exc:
                raise RuntimeError("Loaded Qwen3 layer identity is unavailable for diagnostic scopes.") from exc
            self._scope_forward(layer, f"explorer.layer/{index}")
            if type(layer).__name__ == "Qwen3_5DecoderLayer" and getattr(layer, "layer_type", None) == "linear_attention":
                self._scope_forward(layer.linear_attn, f"explorer.component/{index}/recurrent")
            else:
                self._scope_forward(layer.self_attn, f"explorer.component/{index}/attention")
            self._scope_forward(layer.mlp, f"explorer.component/{index}/ffn")

        original_execute = runner.execute_model
        original_prepare = runner._prepare_inputs
        original_preprocess = runner._preprocess

        def execute(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            step_id = getattr(scheduler_output, "explorer_step_id", None)
            capture = self.active and step_id is not None and len(self.steps) < self.max_steps
            self._pending_sample_step = None
            self.current_step = str(step_id) if capture else None
            self._current_mapping = None
            context = self.record_function(f"explorer.step/{step_id}") if capture else nullcontext()
            try:
                with context:
                    result = original_execute(scheduler_output, *args, **kwargs)
                    self._pending_sample_step = str(step_id) if capture and result is None else None
                    return result
            finally:
                if capture:
                    if self._current_mapping is None:
                        self._current_mapping = unavailable_mapping(str(step_id), "No prepared packed input rows were observed for this execution.")
                    elif self._current_mapping["mapping_status"] == "observed" and self._current_mapping["padded_tokens"] is None:
                        self._current_mapping.update(padding_status="unavailable", reason="Real packed rows were observed, but the padded model input size was not observed.")
                    self.steps.append(self._current_mapping)
                self.current_step = None
                self._current_mapping = None

        def prepare(scheduler_output: Any, counts: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_prepare(scheduler_output, counts, *args, **kwargs)
            if self.current_step:
                self._current_mapping = copy_packed_rows(runner, scheduler_output, self.current_step)
            return result

        def preprocess(scheduler_output: Any, num_tokens_padded: int, *args: Any, **kwargs: Any) -> Any:
            result = original_preprocess(scheduler_output, num_tokens_padded, *args, **kwargs)
            mapping = self._current_mapping
            if self.current_step and mapping is not None:
                padded = int(num_tokens_padded)
                mapping["padded_tokens"] = padded
                mapping["padding_status"] = "observed"
                if mapping["mapping_status"] == "observed":
                    unpadded = mapping["unpadded_tokens"]
                    if padded < unpadded or padded > MAX_PACKED_ROWS:
                        self._current_mapping = unavailable_mapping(self.current_step, "Observed padded input dimensions exceed the diagnostic mapping bound.", unpadded, padded)
                    else:
                        mapping["packed_rows"].extend({"row": row, "request_id": None, "token_position": None, "is_padding": True} for row in range(unpadded, padded))
                mapping["padding_observation_boundary"] = "GPUModelRunner._preprocess.returned"
            return result

        self._patch(runner, "execute_model", execute)
        self._patch(runner, "_prepare_inputs", prepare)
        self._patch(runner, "_preprocess", preprocess)
        self.cache_capture.install()
        if callable(getattr(runner, "sample_tokens", None)):
            original_sample = runner.sample_tokens

            def sample(*args: Any, **kwargs: Any) -> Any:
                step_id = self._pending_sample_step if self.active else None
                context = self.record_function(f"explorer.step/{step_id}") if step_id else nullcontext()
                try:
                    with context:
                        return original_sample(*args, **kwargs)
                finally:
                    self._pending_sample_step = None

            self._patch(runner, "sample_tokens", sample)

    def _scope_forward(self, module: Any, name: str) -> None:
        original = module.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            context = self.record_function(name) if self.current_step else nullcontext()
            with context:
                return original(*args, **kwargs)

        self._patch(module, "forward", forward)

    def _patch(self, obj: Any, name: str, replacement: Any) -> None:
        self._restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def uninstall(self) -> None:
        self.active = False
        for obj, name, previous in reversed(self._restore):
            setattr(obj, name, previous)
        self._restore.clear()


def unavailable_mapping(step_id: str, reason: str, unpadded: int | None = None, padded: int | None = None) -> dict[str, Any]:
    return {"step_id": step_id, "packed_rows": [], "unpadded_tokens": unpadded, "padded_tokens": padded,
            "mapping_status": "unavailable", "reason": reason,
            "mapping_observation_boundary": "GPUModelRunner._prepare_inputs.returned"}


def copy_packed_rows(runner: Any, scheduler_output: Any, step_id: str) -> dict[str, Any]:
    total = int(scheduler_output.total_num_scheduled_tokens)
    if total < 0 or total > MAX_PACKED_ROWS:
        return unavailable_mapping(step_id, "Packed row count exceeds the diagnostic mapping bound.", total)
    try:
        batch = runner.input_batch
        request_ids = list(batch.req_ids[: batch.num_reqs])
        request_indices = runner.req_indices.np[:total].tolist()
        query_positions = runner.query_pos.np[:total].tolist()
        starts = batch.num_computed_tokens_cpu[: batch.num_reqs].tolist()
        boundaries = runner.query_start_loc.np[: batch.num_reqs + 1].tolist()
        if len(request_indices) != total or len(query_positions) != total or boundaries[0] != 0 or boundaries[-1] != total:
            raise ValueError("invalid observed packed dimensions")
        rows = []
        for row, (request_index, query_position) in enumerate(zip(request_indices, query_positions)):
            index, offset = int(request_index), int(query_position)
            if index < 0 or index >= len(request_ids) or offset < 0 or not boundaries[index] <= row < boundaries[index + 1] or row - boundaries[index] != offset:
                raise ValueError("inconsistent observed packed row identity")
            position = int(starts[index]) + offset
            if position < 0:
                raise ValueError("invalid logical token position")
            rows.append({"row": row, "request_id": str(request_ids[index]), "token_position": position, "is_padding": False})
        return {"step_id": step_id, "packed_rows": rows, "unpadded_tokens": total, "padded_tokens": None,
                "mapping_status": "observed", "mapping_observation_boundary": "GPUModelRunner._prepare_inputs.returned"}
    except (AttributeError, IndexError, TypeError, ValueError):
        return unavailable_mapping(step_id, "Post-reorder packed CPU metadata could not be validated; token-row identity is unknown.", total)


def assert_profiler_runner_source(runner: Any) -> dict[str, str]:
    expected = {
        "_update_states": ("self.input_batch.condense()", "self._may_reorder_batch(scheduler_output)"),
        "_prepare_inputs": ("req_indices", "self.input_batch.num_computed_tokens_cpu", "self.query_pos.np", "self.query_start_loc.np"),
        "execute_model": ("self._update_states(scheduler_output)", "self._prepare_inputs(scheduler_output", "self._preprocess("),
    }
    fingerprints = {}
    for method, snippets in expected.items():
        try:
            source = inspect.getsource(getattr(runner, method))
        except (AttributeError, OSError, TypeError) as exc:
            raise RuntimeError(f"Cannot inspect the installed GPUModelRunner.{method} profiling boundary.") from exc
        if any(snippet not in source for snippet in snippets):
            raise RuntimeError(f"Installed GPUModelRunner.{method} does not match the vLLM 0.30 diagnostic mapping boundary.")
        fingerprints[f"GPUModelRunner.{method}"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return fingerprints
