"""Bounded, opt-in vLLM native eager diagnostic profiling attachments."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from runner.profile_trace import PROFILE_CLOCK, normalize_trace
from vllm_030.profiling import DiagnosticHooks


MAX_TRACE_BYTES = 64 * 1024 * 1024


def profile_record(configuration: dict[str, Any] | None) -> dict[str, Any]:
    config = configuration or {"mode": "off", "max_steps": 8}
    enabled = config.get("mode") == "eager_diagnostic"
    return {
        "contract_version": 1, "mode": config.get("mode", "off"), "enabled": enabled,
        "execution_mode": "eager" if enabled else None, "backend": "torch.profiler" if enabled else None,
        "max_steps": config.get("max_steps", 8), "status": "unavailable" if enabled else "disabled",
        "completeness": "unknown", "qualification": "unverified",
        "clock": dict(PROFILE_CLOCK) if enabled else None, "runtime": {},
        "raw_artifacts": [], "normalized_artifact": None, "attachment_status": "unavailable",
        "evidence": "Diagnostic profiling changes execution to eager mode. It records batch/layer CPU scopes and CUDA activities, with no per-token or per-head duration claims.",
    }


def engine_profiler_config(directory: Path, max_steps: int) -> Any:
    from vllm.config import ProfilerConfig

    return ProfilerConfig(
        profiler="torch", torch_profiler_dir=str((directory / "profiling" / "raw").resolve()),
        torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_with_flops=False,
        torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=False,
        capture_torch_profiler=False, delay_iterations=0, max_iterations=max_steps,
    )


class DiagnosticProfile:
    def __init__(self, engine: Any, core: Any, recording: Any, configuration: dict[str, Any], architecture: dict[str, Any]) -> None:
        self.engine = engine
        self.recording = recording
        self.record = profile_record(configuration)
        self.hooks: DiagnosticHooks | None = None
        self._worker: Any = None
        self._started = False
        self._finished = False
        if not self.record["enabled"]:
            return
        if not architecture.get("supported") or architecture.get("model_type") != "qwen3":
            raise RuntimeError("Eager diagnostic profiling currently requires a supported dense Qwen3 model.")
        config = engine.vllm_config
        if not getattr(config.model_config, "enforce_eager", False):
            raise RuntimeError("Diagnostic profiling requires verified enforce_eager execution.")
        parallel = config.parallel_config
        if any(int(getattr(parallel, field, 1)) != 1 for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")) or getattr(parallel, "use_ubatching", False):
            raise RuntimeError("Diagnostic row/layer correlation currently supports one GPU without microbatching.")
        worker_wrapper = getattr(core.model_executor, "driver_worker", None)
        self._worker = getattr(worker_wrapper, "worker", None)
        runner = getattr(self._worker, "model_runner", None)
        if runner is None or getattr(self._worker, "use_v2_model_runner", False):
            raise RuntimeError("The in-process vLLM 0.30 GPUModelRunner is required for diagnostic row/layer correlation.")
        if getattr(runner, "use_async_scheduling", False) or getattr(runner, "uses_mrope", False) or getattr(runner, "speculative_config", None) is not None:
            raise RuntimeError("Diagnostic packed rows do not support asynchronous/speculative or multimodal positions.")
        import torch

        self.record["runtime"] = {"vllm": importlib.metadata.version("vllm"), "torch": torch.__version__, "cuda": torch.version.cuda}
        self.hooks = DiagnosticHooks(runner, torch.profiler.record_function, int(self.record["max_steps"]))
        try:
            self.hooks.install()
            self.record["hook_sources"] = self.hooks.source_fingerprints
        except Exception:
            self.hooks.uninstall()
            raise

    def start(self) -> None:
        if not self.record["enabled"]:
            return
        try:
            self.engine.start_profile(profile_prefix=f"explorer-{self.recording.run_id}")
            profiler = getattr(self._worker, "profiler", None)
            if profiler is None or not getattr(profiler, "is_running", False):
                raise RuntimeError("The native CUDA/CPU profiler did not start; no measured activities are claimed.")
            self._started = True
            self.hooks.active = True
            self.record["status"] = "recording"
        except Exception:
            self.hooks.uninstall()
            raise

    def after_step(self) -> None:
        if self._started and not self._finished and len(self.hooks.steps) >= self.record["max_steps"]:
            self.finish(bounded=True)

    def finish(self, *, bounded: bool = False, cancelled: bool = False, failed: bool = False) -> None:
        if not self._started or self._finished:
            return
        self._finished = True
        try:
            self.engine.stop_profile()
            self._save_attachment(bounded=bounded, cancelled=cancelled, failed=failed)
        except Exception as exc:
            self.record.update(status="failed", completeness="unknown", reason=f"Profiler finalization failed ({type(exc).__name__}); retained raw files may be incomplete.")
            self.record["raw_artifacts"] = self._raw_artifacts()
        finally:
            self.hooks.uninstall()

    def _raw_artifacts(self) -> list[dict[str, Any]]:
        root = self.recording.directory / "profiling" / "raw"
        return [_artifact(path, self.recording.directory) for path in sorted(root.glob("*.pt.trace.json*")) if path.is_file()]

    def _save_attachment(self, *, bounded: bool, cancelled: bool, failed: bool) -> None:
        raw = self._raw_artifacts()
        self.record["raw_artifacts"] = raw
        if len(raw) != 1:
            raise ValueError("Single-GPU diagnostic profiling did not produce exactly one native trace.")
        path = self.recording.directory / raw[0]["path"]
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            payload = stream.read(MAX_TRACE_BYTES + 1)
        if len(payload) > MAX_TRACE_BYTES:
            self.record.update(status="partial", completeness="bounded", reason="Native trace exceeds the 64 MiB normalization bound; raw evidence is retained.")
            return
        normalized = normalize_trace(json.loads(payload), run_id=self.recording.run_id, steps=self.hooks.steps)
        destination = self.recording.directory / "profiling" / "activities.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(normalized, separators=(",", ":")) + "\n", encoding="utf-8")
        partial = failed or cancelled or not normalized["normalization"]["gpu_activity_count"] or normalized["normalization"]["status"] != "completed" or any(step["mapping_status"] != "observed" or step["padded_tokens"] is None for step in self.hooks.steps)
        self.record.update(status="partial" if partial else "completed", completeness="bounded" if bounded else ("unknown" if partial else "complete"),
                           normalized_artifact=_artifact(destination, self.recording.directory), attachment_status="available",
                           captured_steps=len(self.hooks.steps), normalization=normalized["normalization"])
        if cancelled or failed:
            self.record["reason"] = "The diagnostic run was cancelled or failed; only returned profiler observations are retained."
        elif not normalized["normalization"]["gpu_activity_count"]:
            self.record["reason"] = "No CUDA activities were returned by the native profiler; GPU durations are unavailable."


def _artifact(path: Path, directory: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(directory).as_posix(), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
